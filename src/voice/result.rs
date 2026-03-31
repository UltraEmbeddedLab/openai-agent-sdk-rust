//! Result types for voice/realtime sessions.
//!
//! Provides [`StreamedAudioResult`], which accumulates audio chunks produced by
//! a TTS model during a voice pipeline run.  This mirrors the Python SDK's
//! `voice/result.py` (simplified -- the full TTS streaming pipeline is not yet
//! implemented).

use super::utils::pcm16_duration_seconds;

// ---------------------------------------------------------------------------
// StreamedAudioResult
// ---------------------------------------------------------------------------

/// Accumulated audio output from a voice pipeline session.
///
/// As the TTS model produces audio, individual chunks are appended here.
/// After the session completes, you can access the combined audio data,
/// the transcript, and the total duration.
///
/// # Example
///
/// ```
/// use openai_agents::voice::result::StreamedAudioResult;
///
/// let mut result = StreamedAudioResult::new();
/// result.push_chunk(vec![0u8; 48000]); // 1 second at 24 kHz PCM16
/// assert!(!result.is_complete());
///
/// result.set_complete();
/// assert!(result.is_complete());
/// assert!((result.duration_seconds() - 1.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct StreamedAudioResult {
    /// Individual audio data chunks, in the order they were received.
    pub audio_chunks: Vec<Vec<u8>>,
    /// The transcript of the generated audio, if available.
    pub transcript: Option<String>,
    /// The total text that was sent to the TTS model.
    pub total_output_text: String,
    /// Whether all audio has been received.
    is_complete: bool,
    /// The sample rate used for duration calculations (default: 24000).
    sample_rate: u32,
}

impl StreamedAudioResult {
    /// Create a new empty result with the default sample rate of 24 kHz.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            audio_chunks: Vec::new(),
            transcript: None,
            total_output_text: String::new(),
            is_complete: false,
            sample_rate: 24000,
        }
    }

    /// Create a new empty result with the given sample rate.
    #[must_use]
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            ..Self::new()
        }
    }

    /// Append an audio data chunk to the result.
    pub fn push_chunk(&mut self, chunk: Vec<u8>) {
        self.audio_chunks.push(chunk);
    }

    /// Set the transcript text for the audio.
    pub fn set_transcript(&mut self, transcript: impl Into<String>) {
        self.transcript = Some(transcript.into());
    }

    /// Mark the audio result as complete (all chunks have been received).
    pub const fn set_complete(&mut self) {
        self.is_complete = true;
    }

    /// Whether all audio data has been received.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Get the total audio data as a single contiguous buffer.
    #[must_use]
    pub fn combined_audio(&self) -> Vec<u8> {
        self.audio_chunks
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect()
    }

    /// Get the total audio duration in seconds.
    ///
    /// Assumes PCM16 encoding at the configured sample rate.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        let combined = self.combined_audio();
        pcm16_duration_seconds(&combined, self.sample_rate)
    }

    /// Get the total number of audio bytes across all chunks.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.audio_chunks.iter().map(Vec::len).sum()
    }

    /// Get the number of audio chunks received so far.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.audio_chunks.len()
    }

    /// Get the configured sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Construction ----

    #[test]
    fn new_result_is_empty() {
        let result = StreamedAudioResult::new();
        assert!(result.audio_chunks.is_empty());
        assert!(result.transcript.is_none());
        assert!(result.total_output_text.is_empty());
        assert!(!result.is_complete());
        assert_eq!(result.sample_rate(), 24000);
    }

    #[test]
    fn with_sample_rate() {
        let result = StreamedAudioResult::with_sample_rate(16000);
        assert_eq!(result.sample_rate(), 16000);
        assert!(result.audio_chunks.is_empty());
    }

    // ---- Push chunks ----

    #[test]
    fn push_chunk_adds_data() {
        let mut result = StreamedAudioResult::new();
        result.push_chunk(vec![1, 2, 3, 4]);
        result.push_chunk(vec![5, 6]);
        assert_eq!(result.chunk_count(), 2);
        assert_eq!(result.total_bytes(), 6);
    }

    // ---- Combined audio ----

    #[test]
    fn combined_audio_concatenates() {
        let mut result = StreamedAudioResult::new();
        result.push_chunk(vec![0x01, 0x02]);
        result.push_chunk(vec![0x03, 0x04]);
        assert_eq!(result.combined_audio(), vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn combined_audio_empty() {
        let result = StreamedAudioResult::new();
        assert!(result.combined_audio().is_empty());
    }

    // ---- Duration ----

    #[test]
    fn duration_one_second() {
        let mut result = StreamedAudioResult::new();
        // 24000 samples * 2 bytes = 48000 bytes at 24 kHz = 1 second.
        result.push_chunk(vec![0u8; 48000]);
        assert!((result.duration_seconds() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn duration_custom_sample_rate() {
        let mut result = StreamedAudioResult::with_sample_rate(16000);
        // 16000 samples * 2 bytes = 32000 bytes at 16 kHz = 1 second.
        result.push_chunk(vec![0u8; 32000]);
        assert!((result.duration_seconds() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn duration_empty() {
        let result = StreamedAudioResult::new();
        assert!((result.duration_seconds() - 0.0).abs() < f64::EPSILON);
    }

    // ---- Complete ----

    #[test]
    fn complete_flag() {
        let mut result = StreamedAudioResult::new();
        assert!(!result.is_complete());
        result.set_complete();
        assert!(result.is_complete());
    }

    // ---- Transcript ----

    #[test]
    fn set_transcript() {
        let mut result = StreamedAudioResult::new();
        assert!(result.transcript.is_none());
        result.set_transcript("Hello world");
        assert_eq!(result.transcript.as_deref(), Some("Hello world"));
    }

    #[test]
    fn set_transcript_from_string() {
        let mut result = StreamedAudioResult::new();
        result.set_transcript(String::from("From owned"));
        assert_eq!(result.transcript.as_deref(), Some("From owned"));
    }

    // ---- Total bytes and chunk count ----

    #[test]
    fn total_bytes_empty() {
        let result = StreamedAudioResult::new();
        assert_eq!(result.total_bytes(), 0);
    }

    #[test]
    fn chunk_count_empty() {
        let result = StreamedAudioResult::new();
        assert_eq!(result.chunk_count(), 0);
    }

    // ---- Debug + Clone ----

    #[test]
    fn debug_impl() {
        let result = StreamedAudioResult::new();
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("StreamedAudioResult"));
    }

    #[test]
    fn clone_impl() {
        let mut result = StreamedAudioResult::new();
        result.push_chunk(vec![1, 2]);
        result.set_transcript("test");
        result.set_complete();

        let cloned = result.clone();
        assert_eq!(cloned.chunk_count(), 1);
        assert_eq!(cloned.transcript.as_deref(), Some("test"));
        assert!(cloned.is_complete());
    }

    // ---- Send + Sync ----

    #[test]
    fn result_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StreamedAudioResult>();
    }
}
