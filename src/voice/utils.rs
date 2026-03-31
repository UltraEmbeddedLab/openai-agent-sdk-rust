//! Audio utility functions for voice processing.
//!
//! Provides helpers for splitting text at sentence boundaries (for TTS chunking)
//! and basic PCM16 audio manipulation.  These mirror the Python SDK's
//! `voice/utils.py`.

// ---------------------------------------------------------------------------
// SentenceSplitter
// ---------------------------------------------------------------------------

/// Splits a text buffer at sentence boundaries for TTS streaming.
///
/// In a voice pipeline, text is generated incrementally.  Rather than waiting
/// for the entire response, a `SentenceSplitter` examines the accumulated text
/// buffer and extracts complete sentences that can be sent to the TTS model
/// immediately while keeping the incomplete trailing fragment for the next
/// iteration.
///
/// This mirrors the Python SDK's `get_sentence_based_splitter` factory.
///
/// # Example
///
/// ```
/// use openai_agents::voice::utils::SentenceSplitter;
///
/// let splitter = SentenceSplitter::new(20);
/// let (ready, remaining) = splitter.split_buffer("Hello there! How are you doing today? Fine.");
/// assert_eq!(ready, "Hello there! How are you doing today?");
/// assert_eq!(remaining, "Fine.");
/// ```
#[derive(Debug, Clone)]
pub struct SentenceSplitter {
    min_sentence_length: usize,
}

impl SentenceSplitter {
    /// Create a new splitter with the given minimum sentence length.
    ///
    /// The `min_sentence_length` controls how many characters must be
    /// accumulated before the splitter will return a chunk.
    #[must_use]
    pub const fn new(min_sentence_length: usize) -> Self {
        Self {
            min_sentence_length,
        }
    }

    /// Split the text buffer into a ready-to-speak portion and a remainder.
    ///
    /// Returns a tuple `(ready, remaining)` where `ready` is the text that
    /// should be sent to the TTS model and `remaining` is the leftover buffer.
    /// If no split point is found, `ready` will be empty and `remaining` will
    /// be the entire input.
    #[must_use]
    pub fn split_buffer<'a>(&self, text_buffer: &'a str) -> (&'a str, &'a str) {
        let trimmed = text_buffer.trim();
        if trimmed.is_empty() {
            return ("", text_buffer);
        }

        // Split on sentence-ending punctuation followed by whitespace.
        // We collect sentence ranges by finding split points.
        let sentences: Vec<&str> = split_sentences(trimmed);

        if sentences.len() >= 2 {
            // Join all sentences except the last.
            let last = sentences[sentences.len() - 1];
            // Find where the last sentence starts in the trimmed text.
            let last_start = trimmed.len() - last.len();
            let combined = trimmed[..last_start].trim_end();

            if combined.len() >= self.min_sentence_length {
                // Find the remaining portion in the original (untrimmed) buffer.
                // The remaining text is the last sentence.
                return (combined, last);
            }
        }

        ("", text_buffer)
    }
}

impl Default for SentenceSplitter {
    fn default() -> Self {
        Self::new(20)
    }
}

/// Create a [`SentenceSplitter`] with the default minimum sentence length of 20
/// characters.
///
/// This is a convenience function matching the Python SDK's
/// `get_sentence_based_splitter()`.
#[must_use]
pub fn get_sentence_based_splitter() -> SentenceSplitter {
    SentenceSplitter::default()
}

/// Split text on sentence-ending punctuation followed by whitespace.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Check for sentence-ending punctuation.
        if bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?' {
            // Look ahead for whitespace.
            let mut j = i + 1;
            while j < len && (bytes[j] == b' ' || bytes[j] == b'\t' || bytes[j] == b'\n') {
                j += 1;
            }
            if j > i + 1 {
                // We found punctuation followed by whitespace.
                sentences.push(&text[start..=i]);
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }

    // The remaining text is the last "sentence" (possibly incomplete).
    if start < len {
        sentences.push(&text[start..]);
    }

    sentences
}

// ---------------------------------------------------------------------------
// PCM16 audio utilities
// ---------------------------------------------------------------------------

/// Calculate the duration of PCM16 audio data in seconds.
///
/// PCM16 uses 2 bytes per sample, so the number of samples is `data.len() / 2`.
///
/// # Example
///
/// ```
/// use openai_agents::voice::utils::pcm16_duration_seconds;
///
/// // 48000 bytes at 24kHz = 1 second (24000 samples * 2 bytes).
/// let duration = pcm16_duration_seconds(&vec![0u8; 48000], 24000);
/// assert!((duration - 1.0).abs() < f64::EPSILON);
/// ```
#[must_use]
pub fn pcm16_duration_seconds(data: &[u8], sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        return 0.0;
    }
    let num_samples = data.len() / 2;
    #[allow(clippy::cast_precision_loss)]
    let result = num_samples as f64 / f64::from(sample_rate);
    result
}

/// Resample PCM16 audio data using nearest-neighbor interpolation.
///
/// This is a basic resampling suitable for non-critical audio processing.  For
/// production-quality resampling, consider using a dedicated audio library.
///
/// # Panics
///
/// Panics if `from_rate` is zero while `data` is non-empty.
#[must_use]
pub fn resample_pcm16(data: &[u8], from_rate: u32, to_rate: u32) -> Vec<u8> {
    if from_rate == to_rate || data.is_empty() {
        return data.to_vec();
    }

    assert!(from_rate > 0, "from_rate must be greater than zero");

    let num_samples = data.len() / 2;
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let new_num_samples = (num_samples as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(new_num_samples * 2);

    for i in 0..new_num_samples {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let src_idx = ((i as f64 / ratio) as usize).min(num_samples.saturating_sub(1));
        output.push(data[src_idx * 2]);
        output.push(data[src_idx * 2 + 1]);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SentenceSplitter ----

    #[test]
    fn splitter_default_min_length() {
        let splitter = SentenceSplitter::default();
        assert_eq!(splitter.min_sentence_length, 20);
    }

    #[test]
    fn splitter_new_custom_length() {
        let splitter = SentenceSplitter::new(50);
        assert_eq!(splitter.min_sentence_length, 50);
    }

    #[test]
    fn split_buffer_empty() {
        let splitter = get_sentence_based_splitter();
        let (ready, remaining) = splitter.split_buffer("");
        assert_eq!(ready, "");
        assert_eq!(remaining, "");
    }

    #[test]
    fn split_buffer_whitespace_only() {
        let splitter = get_sentence_based_splitter();
        let (ready, remaining) = splitter.split_buffer("   ");
        assert_eq!(ready, "");
        assert_eq!(remaining, "   ");
    }

    #[test]
    fn split_buffer_single_short_sentence() {
        let splitter = get_sentence_based_splitter();
        let (ready, remaining) = splitter.split_buffer("Hi.");
        assert_eq!(ready, "");
        assert_eq!(remaining, "Hi.");
    }

    #[test]
    fn split_buffer_two_sentences_long_enough() {
        let splitter = SentenceSplitter::new(5);
        let (ready, remaining) = splitter.split_buffer("Hello there! How are you?");
        assert_eq!(ready, "Hello there!");
        assert_eq!(remaining, "How are you?");
    }

    #[test]
    fn split_buffer_multiple_sentences() {
        let splitter = SentenceSplitter::new(10);
        let input = "First sentence. Second sentence. Third.";
        let (ready, remaining) = splitter.split_buffer(input);
        assert_eq!(ready, "First sentence. Second sentence.");
        assert_eq!(remaining, "Third.");
    }

    #[test]
    fn split_buffer_no_split_when_too_short() {
        let splitter = SentenceSplitter::new(100);
        let input = "Short. Also.";
        let (ready, remaining) = splitter.split_buffer(input);
        assert_eq!(ready, "");
        assert_eq!(remaining, input);
    }

    #[test]
    fn split_buffer_exclamation_and_question() {
        let splitter = SentenceSplitter::new(5);
        let (ready, remaining) = splitter.split_buffer("Wow! Really? Yes.");
        assert_eq!(ready, "Wow! Really?");
        assert_eq!(remaining, "Yes.");
    }

    #[test]
    fn split_buffer_no_trailing_space_after_last_punct() {
        // No whitespace after the last period means it is one incomplete sentence.
        let splitter = SentenceSplitter::new(5);
        let (ready, remaining) = splitter.split_buffer("Hello world. Goodbye");
        assert_eq!(ready, "Hello world.");
        assert_eq!(remaining, "Goodbye");
    }

    #[test]
    fn get_sentence_based_splitter_returns_default() {
        let splitter = get_sentence_based_splitter();
        assert_eq!(splitter.min_sentence_length, 20);
    }

    // ---- pcm16_duration_seconds ----

    #[test]
    fn duration_of_empty_data() {
        assert!((pcm16_duration_seconds(&[], 24000) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn duration_one_second_at_24k() {
        // 24000 samples * 2 bytes = 48000 bytes.
        let data = vec![0u8; 48000];
        let dur = pcm16_duration_seconds(&data, 24000);
        assert!((dur - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn duration_half_second_at_16k() {
        // 8000 samples * 2 bytes = 16000 bytes.
        let data = vec![0u8; 16000];
        let dur = pcm16_duration_seconds(&data, 16000);
        assert!((dur - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn duration_zero_sample_rate() {
        assert!((pcm16_duration_seconds(&[0, 0], 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn duration_odd_byte_count() {
        // 3 bytes = 1 sample (odd byte dropped).
        let data = vec![0u8; 3];
        let dur = pcm16_duration_seconds(&data, 1);
        assert!((dur - 1.0).abs() < f64::EPSILON);
    }

    // ---- resample_pcm16 ----

    #[test]
    fn resample_same_rate() {
        let data = vec![1, 2, 3, 4];
        let result = resample_pcm16(&data, 16000, 16000);
        assert_eq!(result, data);
    }

    #[test]
    fn resample_empty() {
        let result = resample_pcm16(&[], 16000, 24000);
        assert!(result.is_empty());
    }

    #[test]
    fn resample_upsample_doubles() {
        // 2 samples at rate 1 -> rate 2 should give 4 samples.
        let data = vec![0x01, 0x00, 0x02, 0x00];
        let result = resample_pcm16(&data, 1, 2);
        assert_eq!(result.len(), 8); // 4 samples * 2 bytes.
    }

    #[test]
    fn resample_downsample_halves() {
        // 4 samples at rate 2 -> rate 1 should give 2 samples.
        let data = vec![0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00];
        let result = resample_pcm16(&data, 2, 1);
        assert_eq!(result.len(), 4); // 2 samples * 2 bytes.
    }

    #[test]
    #[should_panic(expected = "from_rate must be greater than zero")]
    fn resample_zero_from_rate_panics() {
        let _ = resample_pcm16(&[0, 0], 0, 16000);
    }

    // ---- Send + Sync ----

    #[test]
    fn splitter_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SentenceSplitter>();
    }
}
