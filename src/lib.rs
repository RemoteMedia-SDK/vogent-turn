//! VogentTurnOnnxNode as a standalone Path 3 loadable plugin.
//!
//! Wraps the Vogent-Turn-80M classifier (Whisper-Tiny encoder + SmolLM-135M
//! + binary head) for conversational turn-end detection. Originally lived
//! in `remotemedia-core` under `crates/core/src/nodes/onnx/vogent_turn.rs`;
//! extracted here so the host crate doesn't drag in `ort` / `tokenizers` /
//! `rustfft` just for this single classifier.
//!
//! ## Wiring
//!
//! `VogentTurnOnnxNode` requires **16 kHz mono f32 PCM** on its audio input.
//! Always wire `FastResampleNode` upstream unless your source is already
//! 16 kHz mono.
//!
//! Optional text context (`prev_line`, `curr_line`) is provided by
//! emitting `RuntimeData::Json({"prev": "...", "curr": "..."})` on the
//! same input port — typically wired from an STT node's transcript
//! stream. Text updates do not trigger inference.
//!
//! ## Node types exported
//!
//!   VogentTurnOnnxNode — Audio → Json{turn_classification, turn_end, ...}

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use half::f16;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::types::{Error, RuntimeData};
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};

use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProviderDispatch,
};
use ort::logging::LogLevel;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{DynValue, Tensor as OrtTensor};

type VtResult<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VogentTurnPrecision {
    Fp32,
    Fp16,
}

impl Default for VogentTurnPrecision {
    fn default() -> Self {
        Self::Fp16
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VogentTurnOnnxConfig {
    /// Directory containing the unpacked Vogent-Turn bundle.
    /// Layout (verbatim HF repo):
    ///   <model_dir>/onnx-fp16/whisper-smol-lm-smaller-fp16.onnx
    ///   <model_dir>/onnx-fp32/whisper-smol-lm-smaller.onnx
    ///   <model_dir>/tokenizer.json
    pub model_dir: PathBuf,

    pub precision: VogentTurnPrecision,

    /// "cuda:0" | "cpu" — falls back to CPU if CUDA init fails.
    pub device: String,

    /// Endpoint prob threshold (default 0.5).
    pub threshold: f32,

    /// Minimum interval between inference runs in ms (default 100).
    pub throttle_ms: u64,

    /// Audio window fed to the model in ms. Currently FIXED at 8000.
    pub window_ms: u32,

    /// Suppress per-inference Json unless prob_endpoint moves by >= this
    /// delta. Edge events ignore this. Default 0.0.
    pub min_prob_delta: f32,

    /// Hysteresis applied to threshold for falling edge (default 0.1).
    pub endpoint_hysteresis: f32,

    /// Veto threshold for forwarded VAD `is_speech_end` events.
    /// Defaults to 0.0 which means "never veto".
    pub veto_below_prob: f32,
}

impl Default for VogentTurnOnnxConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            precision: VogentTurnPrecision::Fp16,
            device: "cuda:0".to_string(),
            threshold: 0.5,
            throttle_ms: 100,
            window_ms: 8000,
            min_prob_delta: 0.0,
            endpoint_hysteresis: 0.1,
            veto_below_prob: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ONNX session loader
// ---------------------------------------------------------------------------

fn resolve_onnx_path(
    model_dir: &std::path::Path,
    precision: VogentTurnPrecision,
) -> VtResult<PathBuf> {
    let (sub, file) = match precision {
        VogentTurnPrecision::Fp32 => ("onnx-fp32", "whisper-smol-lm-smaller.onnx"),
        VogentTurnPrecision::Fp16 => ("onnx-fp16", "whisper-smol-lm-smaller-fp16.onnx"),
    };
    let path = model_dir.join(sub).join(file);
    if !path.exists() {
        return Err(Error::InvalidData(format!(
            "VogentTurnOnnxNode: model file not found at {}",
            path.display()
        )));
    }
    Ok(path)
}

fn load_session(
    model_dir: &std::path::Path,
    precision: VogentTurnPrecision,
    device: &str,
) -> VtResult<Session> {
    let path = resolve_onnx_path(model_dir, precision)?;
    let use_cuda = device.starts_with("cuda");

    if use_cuda {
        let device_id = device
            .strip_prefix("cuda:")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        tracing::info!(
            path = %path.display(),
            device_id,
            "VogentTurnOnnxNode: loading with CUDA EP"
        );
        // Do NOT enable with_tf32 or with_conv1d_pad_to_nc1d — known
        // regression on Whisper-family graphs.
        let cuda_providers: Vec<ExecutionProviderDispatch> = vec![
            CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build(),
            CPUExecutionProvider::default().build(),
        ];
        let mut builder = Session::builder()
            .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT opt: {e}")))?
            // Mute the per-call buffer-reuse WARN. ORT's CUDA EP emits
            // `Shape mismatch attempting to re-use buffer {1,401} != {1,1}`
            // on every inference because our `input_ids` length grows
            // across calls. It's a runtime hint with no graph-side fix.
            .with_log_level(LogLevel::Error)
            .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT log level: {e}")))?
            .with_execution_providers(cuda_providers)
            .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT providers: {e}")))?;
        match builder.commit_from_file(&path) {
            Ok(session) => {
                tracing::info!(ep = "CUDA", "VogentTurnOnnxNode: session loaded");
                return Ok(session);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "VogentTurnOnnxNode: CUDA EP failed, falling back to CPU"
                );
            }
        }
    }

    Session::builder()
        .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT opt: {e}")))?
        .with_log_level(LogLevel::Error)
        .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT log level: {e}")))?
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT providers: {e}")))?
        .commit_from_file(&path)
        .map_err(|e| {
            Error::Execution(format!(
                "VogentTurnOnnxNode: failed to load {}: {e}",
                path.display()
            ))
        })
}

/// Build the `audio_features` tensor as fp32 regardless of graph
/// precision. Both ONNX variants declare `audio_features` as fp32
/// at the input boundary — the fp16 variant casts internally after entry.
fn audio_tensor_from_mel(
    mel_flat: Vec<f32>,
    _precision: VogentTurnPrecision,
) -> VtResult<DynValue> {
    let shape = [1_usize, mel::N_MELS, mel::N_FRAMES];
    OrtTensor::from_array((shape, mel_flat))
        .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: audio_features fp32: {e}")))
        .map(|t| t.into_dyn())
}

/// Number of "audio token" positions consumed from the Whisper-Tiny
/// encoder side. Matches `feats.size(2) / 2` in the Python reference.
const AUDIO_TOKENS: usize = 400;
const _: () = assert!(
    AUDIO_TOKENS == mel::N_FRAMES / 2,
    "AUDIO_TOKENS must equal N_FRAMES / 2 (Whisper encoder downsamples 2x)"
);

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

pub struct Backend {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    mel: mel::MelExtractor,
    precision: VogentTurnPrecision,
}

impl Backend {
    fn try_new(
        model_dir: &std::path::Path,
        precision: VogentTurnPrecision,
        device: &str,
    ) -> VtResult<Self> {
        let tokenizer = text::load_tokenizer(model_dir)?;
        let session = load_session(model_dir, precision, device)?;
        let mel = mel::MelExtractor::new();
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            mel,
            precision,
        })
    }

    /// Run inference with already-tokenized text input_ids.
    /// Returns `(prob_continue, prob_endpoint)`.
    pub fn predict_probs_with_ids(
        &self,
        audio_samples: &[f32],
        input_ids: &[i64],
    ) -> VtResult<(f32, f32)> {
        assert_eq!(audio_samples.len(), mel::WINDOW_SAMPLES);

        // 1) Mel
        let mel_flat = self.mel.extract(audio_samples);
        debug_assert_eq!(mel_flat.len(), mel::N_MELS * mel::N_FRAMES);

        // 2) attention_mask = ones(AUDIO_TOKENS + text_len)
        // ORT rejects zero-dimension tensors. Guard by inserting
        // `<|im_start|>` (token id 1 in SmolLM-135M-Instruct) when
        // input_ids is empty.
        let fallback_ids: [i64; 1] = [text::IM_START_TOKEN_ID];
        let effective_ids: &[i64] = if input_ids.is_empty() {
            &fallback_ids
        } else {
            input_ids
        };
        let text_len = effective_ids.len();
        let mask_len = AUDIO_TOKENS + text_len;

        // 3) Build ORT tensors (int64 per local re-export).
        let input_ids_vec: Vec<i64> = effective_ids.to_vec();
        let attention_mask: Vec<i64> = vec![1i64; mask_len];
        let input_ids_tensor = OrtTensor::from_array(([1usize, text_len], input_ids_vec))
            .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: input_ids tensor: {e}")))?;
        let attention_mask_tensor =
            OrtTensor::from_array(([1usize, mask_len], attention_mask)).map_err(|e| {
                Error::Execution(format!("VogentTurnOnnxNode: attention_mask tensor: {e}"))
            })?;
        let audio_features_tensor = audio_tensor_from_mel(mel_flat, self.precision)?;

        // 4) Run.
        let mut sess = self.session.lock();
        let outputs = sess
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "audio_features" => audio_features_tensor,
            ])
            .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: ORT run: {e}")))?;

        // Extract logits — fp16 graph emits fp16; fp32 graph emits fp32.
        let (lc, le) = match self.precision {
            VogentTurnPrecision::Fp32 => {
                let (_shape, data) =
                    outputs["logits"].try_extract_tensor::<f32>().map_err(|e| {
                        Error::Execution(format!("VogentTurnOnnxNode: extract f32 logits: {e}"))
                    })?;
                if data.len() != 2 {
                    return Err(Error::Execution(format!(
                        "VogentTurnOnnxNode: unexpected logits len {} (expected 2)",
                        data.len()
                    )));
                }
                (data[0], data[1])
            }
            VogentTurnPrecision::Fp16 => {
                let (_shape, data) =
                    outputs["logits"].try_extract_tensor::<f16>().map_err(|e| {
                        Error::Execution(format!("VogentTurnOnnxNode: extract f16 logits: {e}"))
                    })?;
                if data.len() != 2 {
                    return Err(Error::Execution(format!(
                        "VogentTurnOnnxNode: unexpected logits len {} (expected 2)",
                        data.len()
                    )));
                }
                (data[0].to_f32(), data[1].to_f32())
            }
        };
        if !lc.is_finite() || !le.is_finite() {
            return Err(Error::Execution(format!(
                "VogentTurnOnnxNode: non-finite logits ({lc}, {le})"
            )));
        }
        // softmax over [continue, endpoint]
        let m = lc.max(le);
        let e0 = (lc - m).exp();
        let e1 = (le - m).exp();
        let z = e0 + e1;
        Ok((e0 / z, e1 / z))
    }

    /// Run inference with text strings (builds prompt + tokenizes).
    pub fn predict_probs(
        &self,
        audio_samples: &[f32],
        prev: &str,
        curr: &str,
    ) -> VtResult<(f32, f32)> {
        let prompt = text::build_prompt(prev, curr);
        let ids = text::encode(&self.tokenizer, &prompt)?;
        self.predict_probs_with_ids(audio_samples, &ids)
    }
}

// ---------------------------------------------------------------------------
// Per-session state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SessionState {
    /// Fixed-size ring buffer of size `mel::WINDOW_SAMPLES`.
    ring: Vec<f32>,
    /// Write cursor.
    head: usize,
    /// Number of samples ever written; capped at WINDOW_SAMPLES.
    filled: usize,
    /// Last inference time.
    last_inference_at: Option<Instant>,
    /// Last emitted prob_endpoint (for `min_prob_delta` gating).
    #[allow(dead_code)]
    last_emitted_prob: Option<f32>,
    /// Most recent `prob_endpoint`, updated every inference.
    last_p_endpoint: Option<f32>,
    /// True while is_endpoint sticky (hysteresis edge state).
    inside_endpoint: bool,
    /// Gates turn_end emission to the window between speech_start/end.
    armed: bool,
    /// Latest text context.
    prev_line: String,
    curr_line: String,
    /// Tokenization cache.
    cached_tokens: Option<(String, String, Vec<i64>)>,
}

impl SessionState {
    fn new() -> Self {
        Self {
            ring: vec![0.0; mel::WINDOW_SAMPLES],
            ..Self::default()
        }
    }

    fn append_audio(&mut self, samples: &[f32]) {
        let n = mel::WINDOW_SAMPLES;
        let mut remaining = samples;
        while !remaining.is_empty() {
            let space = n - self.head;
            let write = remaining.len().min(space);
            self.ring[self.head..self.head + write].copy_from_slice(&remaining[..write]);
            self.head = (self.head + write) % n;
            remaining = &remaining[write..];
        }
        self.filled = (self.filled + samples.len()).min(n);
    }

    fn snapshot(&self) -> Option<Vec<f32>> {
        if self.filled == 0 {
            return None;
        }
        let n = mel::WINDOW_SAMPLES;
        let mut out = vec![0.0_f32; n];
        if self.filled < n {
            debug_assert_eq!(self.head, self.filled);
            let src = &self.ring[..self.filled];
            out[n - self.filled..].copy_from_slice(src);
        } else {
            out[..n - self.head].copy_from_slice(&self.ring[self.head..]);
            out[n - self.head..].copy_from_slice(&self.ring[..self.head]);
        }
        Some(out)
    }

    fn throttle_passed(&self, throttle_ms: u64) -> bool {
        match self.last_inference_at {
            None => true,
            Some(t) => t.elapsed().as_millis() as u64 >= throttle_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

pub struct VogentTurnOnnxNode {
    config: VogentTurnOnnxConfig,
    backend: OnceCell<Arc<Backend>>,
    sessions: Mutex<HashMap<String, SessionState>>,
}

impl VogentTurnOnnxNode {
    /// Lazy constructor — used by the FFI factory which can't surface
    /// errors at construction time.
    pub fn new(config: VogentTurnOnnxConfig) -> Self {
        Self {
            config,
            backend: OnceCell::new(),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Eager constructor — validates config AND loads the ONNX
    /// backend up front. Kept for tests/benches.
    pub fn try_new(config: VogentTurnOnnxConfig) -> VtResult<Self> {
        if config.endpoint_hysteresis >= config.threshold {
            return Err(Error::ConfigError(format!(
                "VogentTurnOnnxNode: endpoint_hysteresis ({}) must be < threshold ({})",
                config.endpoint_hysteresis, config.threshold
            )));
        }
        if config.window_ms != 8000 {
            tracing::warn!(
                requested_window_ms = config.window_ms,
                actual_window_ms = 8000,
                "VogentTurnOnnxNode: windowMs is currently fixed at 8000 \
                 (the model is trained on the last 8 s of audio); requested \
                 value will be ignored"
            );
        }
        let backend = Arc::new(Backend::try_new(
            &config.model_dir,
            config.precision,
            &config.device,
        )?);
        let node = Self::new(config);
        node.backend
            .set(backend)
            .map_err(|_| Error::Execution("VogentTurnOnnxNode: backend cell already set".into()))?;
        Ok(node)
    }

    fn session_key(session_id: Option<String>) -> String {
        session_id.unwrap_or_else(|| "default".to_string())
    }

    /// Lazy backend accessor.
    async fn ensure_backend(&self) -> VtResult<&Arc<Backend>> {
        self.backend
            .get_or_try_init(|| {
                let model_dir = self.config.model_dir.clone();
                let precision = self.config.precision;
                let device = self.config.device.clone();
                let endpoint_hysteresis = self.config.endpoint_hysteresis;
                let threshold = self.config.threshold;
                let window_ms = self.config.window_ms;
                async move {
                    if endpoint_hysteresis >= threshold {
                        return Err(Error::ConfigError(format!(
                            "VogentTurnOnnxNode: endpoint_hysteresis ({}) must be < threshold ({})",
                            endpoint_hysteresis, threshold
                        )));
                    }
                    if window_ms != 8000 {
                        tracing::warn!(
                            requested_window_ms = window_ms,
                            actual_window_ms = 8000,
                            "VogentTurnOnnxNode: windowMs is currently fixed at 8000 \
                             (the model is trained on the last 8 s of audio); requested \
                             value will be ignored"
                        );
                    }
                    let backend = Backend::try_new(&model_dir, precision, &device)?;
                    Ok(Arc::new(backend))
                }
            })
            .await
    }

    /// Process an audio chunk: append to ring, throttle-gate, run inference,
    /// emit Json events.
    async fn handle_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
        channels: u32,
        session_key: &str,
        callback: &mut (dyn FnMut(RuntimeData) -> VtResult<()> + Send),
    ) -> VtResult<usize> {
        if sample_rate != mel::SAMPLE_RATE || channels != 1 {
            tracing::warn!(
                sample_rate,
                channels,
                "VogentTurnOnnxNode: dropping chunk; expected 16 kHz mono. \
                 Wire FastResampleNode upstream."
            );
            return Ok(0);
        }

        let backend = self.ensure_backend().await?;

        // Append + throttle check (single lock acquisition).
        let snapshot;
        let input_ids: Vec<i64>;
        {
            let mut sessions = self.sessions.lock();
            let state = sessions
                .entry(session_key.to_string())
                .or_insert_with(SessionState::new);
            state.append_audio(samples);
            if !state.throttle_passed(self.config.throttle_ms) {
                return Ok(0);
            }
            snapshot = match state.snapshot() {
                Some(s) => s,
                None => return Ok(0),
            };
            input_ids = match &state.cached_tokens {
                Some((p, c, ids)) if p == &state.prev_line && c == &state.curr_line => ids.clone(),
                _ => {
                    let prompt = text::build_prompt(&state.prev_line, &state.curr_line);
                    let ids = text::encode(&backend.tokenizer, &prompt)?;
                    state.cached_tokens =
                        Some((state.prev_line.clone(), state.curr_line.clone(), ids.clone()));
                    ids
                }
            };
        }

        // Run inference outside the lock.
        let started = Instant::now();
        let (p_cont, p_end) = backend.predict_probs_with_ids(&snapshot, &input_ids)?;
        let latency_us = started.elapsed().as_micros() as u64;

        let threshold = self.config.threshold;
        let hyst = self.config.endpoint_hysteresis;
        let is_endpoint = p_end >= threshold;

        // Update edge/throttle state + capture deltas to emit.
        let (emit_per_inference, emit_turn_end) = {
            let mut sessions = self.sessions.lock();
            let state = sessions.get_mut(session_key).expect("session entry exists");
            state.last_inference_at = Some(Instant::now());

            let (per_inf_ok, turn_end_edge, new_inside) = edge_decision(
                p_end,
                state.inside_endpoint,
                state.last_emitted_prob,
                threshold,
                hyst,
                self.config.min_prob_delta,
            );
            if per_inf_ok {
                state.last_emitted_prob = Some(p_end);
            }
            state.last_p_endpoint = Some(p_end);
            state.inside_endpoint = new_inside;
            let emit_turn_end = turn_end_edge && state.armed;
            if emit_turn_end {
                state.armed = false;
            }
            (per_inf_ok, emit_turn_end)
        };

        let mut emitted = 0;
        if emit_per_inference {
            callback(RuntimeData::Json(serde_json::json!({
                "type": "turn_classification",
                "is_endpoint": is_endpoint,
                "prob_endpoint": p_end,
                "prob_continue": p_cont,
                "latency_us": latency_us,
            })))?;
            emitted += 1;
        }
        if emit_turn_end {
            callback(RuntimeData::Json(serde_json::json!({
                "type": "turn_end",
                "is_speech_end": true,
                "prob_endpoint": p_end,
            })))?;
            emitted += 1;
        }
        Ok(emitted)
    }

    /// Update state from a structured Json blob and emit gated/forwarded
    /// events.
    async fn handle_text_state_update(
        &self,
        value: &serde_json::Value,
        session_key: &str,
        callback: &mut (dyn FnMut(RuntimeData) -> VtResult<()> + Send),
    ) -> VtResult<usize> {
        let mut forward_start = false;
        struct VetoInputs {
            audio: Vec<f32>,
            input_ids: Vec<i64>,
        }
        let veto_inputs: Option<VetoInputs>;
        let speech_end_force_drop: bool;

        {
            let mut sessions = self.sessions.lock();
            let state = sessions
                .entry(session_key.to_string())
                .or_insert_with(SessionState::new);

            if value.get("reset").and_then(|v| v.as_bool()) == Some(true) {
                *state = SessionState::new();
            }
            if let Some(p) = value.get("prev").and_then(|v| v.as_str()) {
                if p != state.prev_line {
                    state.prev_line = p.to_string();
                    state.cached_tokens = None;
                }
            }
            if let Some(c) = value.get("curr").and_then(|v| v.as_str()) {
                if c != state.curr_line {
                    state.curr_line = c.to_string();
                    state.cached_tokens = None;
                }
            }

            let speech_start =
                value.get("is_speech_start").and_then(|v| v.as_bool()) == Some(true);
            let speech_end =
                value.get("is_speech_end").and_then(|v| v.as_bool()) == Some(true);

            if speech_start {
                state.armed = true;
                state.last_emitted_prob = None;
                forward_start = true;
            }

            if speech_end {
                if !state.armed {
                    veto_inputs = None;
                    speech_end_force_drop = true;
                } else if self.config.veto_below_prob <= 0.0 {
                    veto_inputs = None;
                    speech_end_force_drop = false;
                    state.armed = false;
                } else {
                    let audio = state.snapshot().unwrap_or_default();
                    let cached_ids = match &state.cached_tokens {
                        Some((p, c, ids))
                            if p == &state.prev_line && c == &state.curr_line =>
                        {
                            Some(ids.clone())
                        }
                        _ => None,
                    };
                    let prev_line = state.prev_line.clone();
                    let curr_line = state.curr_line.clone();
                    veto_inputs = if audio.len() == mel::WINDOW_SAMPLES {
                        Some(VetoInputs {
                            audio,
                            input_ids: cached_ids.unwrap_or_default(),
                        })
                    } else {
                        None
                    };
                    let _ = (prev_line, curr_line);
                    speech_end_force_drop = false;
                    state.armed = false;
                }
            } else {
                veto_inputs = None;
                speech_end_force_drop = false;
            }
        }

        // PASS 2 (no lock): run fresh inference if veto is in play.
        let fresh_p_endpoint: Option<f32> = if let Some(VetoInputs { audio, mut input_ids }) =
            veto_inputs
        {
            'veto_block: {
                let backend = match self.ensure_backend().await {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!(
                            "VogentTurnOnnxNode: backend init for veto check failed, \
                             forwarding without veto: {e}"
                        );
                        break 'veto_block None;
                    }
                };
                if input_ids.is_empty() {
                    let (prev, curr) = {
                        let sessions = self.sessions.lock();
                        sessions
                            .get(session_key)
                            .map(|s| (s.prev_line.clone(), s.curr_line.clone()))
                            .unwrap_or_default()
                    };
                    let prompt = text::build_prompt(&prev, &curr);
                    match text::encode(&backend.tokenizer, &prompt) {
                        Ok(ids) => {
                            let mut sessions = self.sessions.lock();
                            if let Some(state) = sessions.get_mut(session_key) {
                                state.cached_tokens = Some((prev, curr, ids.clone()));
                            }
                            input_ids = ids;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "VogentTurnOnnxNode: tokenize for veto check failed, \
                                 forwarding without veto: {e}"
                            );
                            break 'veto_block None;
                        }
                    }
                }
                if input_ids.is_empty() {
                    break 'veto_block None;
                }
                match backend.predict_probs_with_ids(&audio, &input_ids) {
                    Ok((_p_cont, p_end)) => {
                        let mut sessions = self.sessions.lock();
                        if let Some(state) = sessions.get_mut(session_key) {
                            state.last_p_endpoint = Some(p_end);
                        }
                        Some(p_end)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "VogentTurnOnnxNode: fresh inference on speech_end failed, \
                             forwarding without veto: {e}"
                        );
                        None
                    }
                }
            }
        } else {
            None
        };

        let mut forward_end = false;
        let mut vetoed_end = speech_end_force_drop;
        if value.get("is_speech_end").and_then(|v| v.as_bool()) == Some(true)
            && !speech_end_force_drop
        {
            let veto_floor = self.config.veto_below_prob;
            if veto_floor <= 0.0 {
                forward_end = true;
            } else if let Some(p) = fresh_p_endpoint {
                if p < veto_floor {
                    vetoed_end = true;
                } else {
                    forward_end = true;
                }
            } else {
                forward_end = true;
            }
        }

        let mut emitted = 0;
        if forward_start {
            callback(RuntimeData::Json(value.clone()))?;
            emitted += 1;
        }
        if forward_end {
            callback(RuntimeData::Json(value.clone()))?;
            emitted += 1;
        }
        if vetoed_end {
            tracing::debug!(
                target: "vogent_turn",
                p_endpoint = ?fresh_p_endpoint,
                "VogentTurnOnnxNode: vetoed VAD speech_end (model disagrees)"
            );
        }
        Ok(emitted)
    }

    /// Update curr_line from a bare RuntimeData::Text.
    fn handle_text_curr(&self, curr: &str, session_key: &str) {
        let mut sessions = self.sessions.lock();
        let state = sessions
            .entry(session_key.to_string())
            .or_insert_with(SessionState::new);
        if curr != state.curr_line {
            state.curr_line = curr.to_string();
            state.cached_tokens = None;
        }
    }
}

#[async_trait]
impl AsyncStreamingNode for VogentTurnOnnxNode {
    fn node_type(&self) -> &str {
        "VogentTurnOnnxNode"
    }

    async fn process(&self, _data: RuntimeData) -> Result<RuntimeData, Error> {
        Err(Error::Execution(
            "VogentTurnOnnxNode: unary process not supported; use streaming".into(),
        ))
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let key = Self::session_key(session_id);
        match data {
            RuntimeData::Audio { .. } => {
                // Passthrough: forward the audio chunk BEFORE running
                // inference, so `vad → vogent_turn → accumulator` keeps
                // the audio path transparent.
                let audio_clone = data.clone();
                let (samples, sample_rate, channels) = match &data {
                    RuntimeData::Audio {
                        samples,
                        sample_rate,
                        channels,
                        ..
                    } => (samples.clone(), *sample_rate, *channels),
                    _ => unreachable!("matched on Audio above"),
                };
                callback(audio_clone)?;
                let inference_emits = self
                    .handle_audio(
                        samples.as_ref(),
                        sample_rate,
                        channels,
                        &key,
                        &mut callback,
                    )
                    .await?;
                Ok(1 + inference_emits)
            }
            RuntimeData::Text(t) => {
                self.handle_text_curr(&t, &key);
                Ok(0)
            }
            RuntimeData::Json(v) => self.handle_text_state_update(&v, &key, &mut callback).await,
            _ => Ok(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Edge-decision helper (pure function — extracted for testability).
// ---------------------------------------------------------------------------

fn edge_decision(
    p_end: f32,
    inside_endpoint: bool,
    last_emitted_prob: Option<f32>,
    threshold: f32,
    hysteresis: f32,
    min_prob_delta: f32,
) -> (bool, bool, bool) {
    let per_inf_ok = match last_emitted_prob {
        None => true,
        Some(prev_p) => (p_end - prev_p).abs() >= min_prob_delta,
    };
    let (turn_end_edge, new_inside) = if inside_endpoint {
        if p_end < (threshold - hysteresis) {
            (false, false)
        } else {
            (false, true)
        }
    } else if p_end >= threshold {
        (true, true)
    } else {
        (false, false)
    };
    (per_inf_ok, turn_end_edge, new_inside)
}

// ---------------------------------------------------------------------------
// Text helpers (SmolLM prompt + tokenizer)
// ---------------------------------------------------------------------------

mod text {
    use std::path::Path;

    use remotemedia_plugin_sdk::types::Error;
    use tokenizers::Tokenizer;

    /// SmolLM-135M-Instruct token id for `<|im_start|>`. Used as a single
    /// fallback token when an inference call has no text context.
    pub(super) const IM_START_TOKEN_ID: i64 = 1;

    /// Lowercase, strip whitespace, remove punctuation `.!,?`.
    pub(super) fn clean(txt: &str) -> String {
        let trimmed = txt.trim().to_lowercase();
        trimmed
            .chars()
            .filter(|c| !matches!(c, '.' | '!' | ',' | '?'))
            .collect()
    }

    /// Build the prompt string fed to the SmolLM tokenizer.
    pub(super) fn build_prompt(prev: &str, curr: &str) -> String {
        if prev.is_empty() && curr.is_empty() {
            return String::new();
        }
        let p = clean(prev);
        let c = clean(curr);
        format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{c}")
    }

    pub(super) fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer, Error> {
        let path = model_dir.join("tokenizer.json");
        Tokenizer::from_file(&path).map_err(|e| {
            Error::Execution(format!(
                "VogentTurnOnnxNode: failed to load tokenizer at {}: {e}",
                path.display()
            ))
        })
    }

    pub(super) fn encode(tok: &Tokenizer, prompt: &str) -> Result<Vec<i64>, Error> {
        let enc = tok
            .encode(prompt, true)
            .map_err(|e| Error::Execution(format!("VogentTurnOnnxNode: encode failed: {e}")))?;
        Ok(enc.get_ids().iter().map(|&id| id as i64).collect())
    }
}

// ---------------------------------------------------------------------------
// Whisper-Tiny mel filterbank (Slaney + log10 + Whisper clip/rescale)
// ---------------------------------------------------------------------------

mod mel {
    use std::sync::Arc;

    use rustfft::{num_complex::Complex32, Fft, FftPlanner};

    pub const SAMPLE_RATE: u32 = 16_000;
    pub const N_FFT: usize = 400;
    pub const HOP_LENGTH: usize = 160;
    pub const WIN_LENGTH: usize = 400;
    pub const N_MELS: usize = 80;
    pub const N_FREQS: usize = N_FFT / 2 + 1; // 201
    pub const WINDOW_SECS: f32 = 8.0;
    pub const WINDOW_SAMPLES: usize = (SAMPLE_RATE as f32 * WINDOW_SECS) as usize; // 128_000
    /// HF's WhisperFeatureExtractor with max_length=WINDOW_SAMPLES emits
    /// exactly 800 frames.
    pub const N_FRAMES: usize = 800;

    const _: () = assert!(
        N_FRAMES * HOP_LENGTH == WINDOW_SAMPLES,
        "N_FRAMES must equal WINDOW_SAMPLES / HOP_LENGTH"
    );

    pub struct MelExtractor {
        filterbank: Arc<[f32]>,
        hann: Arc<[f32]>,
        fft: Arc<dyn Fft<f32>>,
    }

    impl MelExtractor {
        pub fn new() -> Self {
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(N_FFT);
            let hann = hann_window(WIN_LENGTH);
            let filterbank = slaney_mel_filterbank(N_MELS, N_FREQS, SAMPLE_RATE as f32);
            Self {
                filterbank: filterbank.into(),
                hann: hann.into(),
                fft,
            }
        }

        /// Compute log-mel spectrogram for exactly `WINDOW_SAMPLES` of audio.
        /// Output: `[N_MELS × N_FRAMES]` row-major.
        pub fn extract(&self, samples: &[f32]) -> Vec<f32> {
            assert_eq!(samples.len(), WINDOW_SAMPLES);

            // 1) STFT with center=True (reflect pad by N_FFT/2 on each side,
            //    skipping boundary element — matches numpy.pad mode='reflect').
            let pad = N_FFT / 2;
            let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
            for i in 0..pad {
                padded.push(samples[pad - i]);
            }
            padded.extend_from_slice(samples);
            let n = samples.len();
            for i in 0..pad {
                padded.push(samples[n - 2 - i]);
            }

            // 2) Allocate scratch
            let mut frame = vec![Complex32::new(0.0, 0.0); N_FFT];
            let mut power = vec![0.0_f32; N_FREQS * N_FRAMES];

            for frame_idx in 0..N_FRAMES {
                let start = frame_idx * HOP_LENGTH;
                for i in 0..WIN_LENGTH {
                    frame[i] = Complex32::new(padded[start + i] * self.hann[i], 0.0);
                }
                for i in WIN_LENGTH..N_FFT {
                    frame[i] = Complex32::new(0.0, 0.0);
                }
                self.fft.process(&mut frame);
                for f in 0..N_FREQS {
                    let c = frame[f];
                    power[f * N_FRAMES + frame_idx] = c.re * c.re + c.im * c.im;
                }
            }

            // 3) mel = filterbank @ power → [N_MELS × N_FRAMES]
            let mut mel = vec![0.0_f32; N_MELS * N_FRAMES];
            for m in 0..N_MELS {
                for f in 0..N_FREQS {
                    let w = self.filterbank[m * N_FREQS + f];
                    if w == 0.0 {
                        continue;
                    }
                    for t in 0..N_FRAMES {
                        mel[m * N_FRAMES + t] += w * power[f * N_FRAMES + t];
                    }
                }
            }

            // 4) Log + clip + rescale (Whisper formula)
            for v in mel.iter_mut() {
                *v = v.max(1e-10).log10();
            }
            let max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let floor = max - 8.0;
            for v in mel.iter_mut() {
                if *v < floor {
                    *v = floor;
                }
                *v = (*v + 4.0) / 4.0;
            }
            mel
        }
    }

    fn hann_window(n: usize) -> Vec<f32> {
        // Periodic hann: matches torch.hann_window(n, periodic=True).
        let mut w = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as f32) / (n as f32);
            w.push(0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos());
        }
        w
    }

    fn hz_to_mel(hz: f32) -> f32 {
        // Slaney mel scale (HF Whisper default).
        let f_min = 0.0;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4_f32).ln() / 27.0;
        if hz < min_log_hz {
            (hz - f_min) / f_sp
        } else {
            min_log_mel + ((hz / min_log_hz).ln()) / logstep
        }
    }

    fn mel_to_hz(mel: f32) -> f32 {
        let f_min = 0.0;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4_f32).ln() / 27.0;
        if mel < min_log_mel {
            f_min + f_sp * mel
        } else {
            min_log_hz * ((mel - min_log_mel) * logstep).exp()
        }
    }

    fn slaney_mel_filterbank(n_mels: usize, n_freqs: usize, sr: f32) -> Vec<f32> {
        let fmin = 0.0;
        let fmax = sr / 2.0;
        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);
        let mut mel_points = Vec::with_capacity(n_mels + 2);
        for i in 0..(n_mels + 2) {
            let t = i as f32 / (n_mels + 1) as f32;
            mel_points.push(mel_min + (mel_max - mel_min) * t);
        }
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * (sr / 2.0) / (n_freqs - 1) as f32)
            .collect();

        let mut fb = vec![0.0_f32; n_mels * n_freqs];
        for m in 0..n_mels {
            let left = hz_points[m];
            let center = hz_points[m + 1];
            let right = hz_points[m + 2];
            for (i, &hz) in bin_freqs.iter().enumerate() {
                let lower = (hz - left) / (center - left);
                let upper = (right - hz) / (right - center);
                let v = lower.min(upper).max(0.0);
                let enorm = 2.0 / (right - left);
                fb[m * n_freqs + i] = v * enorm;
            }
        }
        fb
    }
}

// ---------------------------------------------------------------------------
// Factory + plugin registration
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct VogentTurnOnnxNodeFactory;

impl FfiNodeFactory for VogentTurnOnnxNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("VogentTurnOnnxNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let cfg: VogentTurnOnnxConfig =
            serde_json::from_str(params.as_str()).unwrap_or_default();
        // Lazy construction: validation + ONNX load happen on first audio
        // chunk (in `ensure_backend`). The FFI factory has no fallible
        // constructor surface, so we can't return errors here.
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(VogentTurnOnnxNode::new(cfg)),
            TD_Opaque,
        ))
    }
}

remotemedia_plugin_sdk::plugin_export!(VogentTurnOnnxNodeFactory);

// ---------------------------------------------------------------------------
// Unit tests — restricted to pure helpers (no model bundle / ORT needed).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod text_tests {
    use super::text::*;

    #[test]
    fn clean_lowercases_and_strips_punctuation() {
        assert_eq!(clean("  How are you?!  "), "how are you");
        assert_eq!(clean("Hello, World."), "hello world");
        assert_eq!(clean("Don't worry"), "don't worry");
    }

    #[test]
    fn build_prompt_empty_returns_empty() {
        assert_eq!(build_prompt("", ""), "");
    }

    #[test]
    fn build_prompt_matches_python_reference() {
        let got = build_prompt("How are you?", "I am fine.");
        assert_eq!(
            got,
            "<|im_start|>user\nhow are you<|im_end|>\n<|im_start|>assistant\ni am fine"
        );
    }
}

#[cfg(test)]
mod edge_tests {
    use super::edge_decision;

    #[test]
    fn rising_edge_fires_once() {
        let (_, edge1, inside1) = edge_decision(0.7, false, None, 0.5, 0.1, 0.0);
        assert!(edge1);
        assert!(inside1);
        let (_, edge2, inside2) = edge_decision(0.8, inside1, Some(0.7), 0.5, 0.1, 0.0);
        assert!(!edge2);
        assert!(inside2);
    }

    #[test]
    fn falling_below_hysteresis_resets() {
        let (_, _, inside1) = edge_decision(0.7, false, None, 0.5, 0.1, 0.0);
        assert!(inside1);
        let (_, _, inside2) = edge_decision(0.45, inside1, Some(0.7), 0.5, 0.1, 0.0);
        assert!(inside2);
        let (_, _, inside3) = edge_decision(0.35, inside2, Some(0.45), 0.5, 0.1, 0.0);
        assert!(!inside3);
        let (_, edge4, inside4) = edge_decision(0.6, inside3, Some(0.35), 0.5, 0.1, 0.0);
        assert!(edge4);
        assert!(inside4);
    }
}
