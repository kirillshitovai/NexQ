/// Field-for-field copy of NexQ's `stt::provider::TranscriptResult` (without the serde derives).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub text: String,
    pub is_final: bool,
    pub confidence: f32,
    pub timestamp_ms: u64,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub segment_id: Option<String>,
    pub duration_ms: Option<u64>,
    pub speaker_tag: Option<String>,
}
