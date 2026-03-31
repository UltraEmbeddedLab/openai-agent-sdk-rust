# Voice / Realtime

The voice module lets you build agents that communicate via audio using OpenAI's Realtime
API over WebSocket. It mirrors the Python SDK's `realtime/` package.

> **Feature flag.** Voice support requires the `voice` feature:
>
> ```toml
> [dependencies]
> openai-agents = { version = "0.1", features = ["voice"] }
> ```

## Architecture Overview

The voice module is organized in layers:

```
VoicePipeline          ← high-level: handles audio I/O + WebSocket lifecycle
  └── VoiceWorkflow    ← orchestrates one or more agents
        └── RealtimeSession    ← raw WebSocket connection to the Realtime API
              └── RealtimeAgent      ← specialized agent for voice interactions
```

For most applications `VoicePipeline` + `SingleAgentVoiceWorkflow` is the right starting
point. For finer-grained control, use `RealtimeSession` directly.

---

## RealtimeAgent

`RealtimeAgent<C>` is a specialized agent for voice interactions. Unlike the standard
`Agent<C>`, it does not support per-agent model selection or structured output — those are
controlled at the session level.

### Building a RealtimeAgent

```rust,no_run
use openai_agents::voice::{RealtimeAgent, RealtimeInstructions};
use openai_agents::tool::Tool;

let agent = RealtimeAgent::<()>::builder("voice-assistant")
    .instructions("You are a friendly voice assistant. Keep answers short.")
    .build();
```

Dynamic instructions (generated from context at runtime):

```rust,no_run
use openai_agents::voice::RealtimeAgent;
use openai_agents::context::RunContextWrapper;

let agent = RealtimeAgent::<String>::builder("voice-assistant")
    .dynamic_instructions(|ctx: &RunContextWrapper<String>, _agent| {
        let user_name = ctx.context.clone();
        Box::pin(async move {
            Ok(format!("You are a voice assistant. The user's name is {user_name}."))
        })
    })
    .build();
```

### Differences from `Agent`

| Feature | `Agent<C>` | `RealtimeAgent<C>` |
|---|---|---|
| Model selection | Per-agent | Session-level |
| Model settings | Per-agent | Session-level |
| Structured output | Yes | No |
| Input guardrails | Yes | No |
| Output guardrails | Yes | Yes |
| Handoffs | Yes | Yes |
| Tools | Yes | Yes |

---

## RealtimeConfig

`RealtimeConfig` controls the session-level model, audio, and tracing settings.

```rust,no_run
use openai_agents::voice::{
    RealtimeConfig, RealtimeModelSettings, RealtimeAudioConfig,
    RealtimeAudioInputConfig, RealtimeAudioOutputConfig,
    AudioFormat, Voice, TurnDetection,
};

let config = RealtimeConfig {
    model: Some("gpt-4o-realtime-preview".to_owned()),
    model_settings: Some(RealtimeModelSettings {
        temperature: Some(0.8),
        max_response_output_tokens: Some(800),
        ..Default::default()
    }),
    audio: Some(RealtimeAudioConfig {
        input: Some(RealtimeAudioInputConfig {
            format: Some(AudioFormat::Pcm16),
            turn_detection: Some(TurnDetection::ServerVad {
                threshold: 0.6,
                prefix_padding_ms: 300,
                silence_duration_ms: 600,
                create_response: Some(true),
                interrupt_response: Some(true),
            }),
            transcription_language: Some("en".to_owned()),
            transcription_model: Some("whisper-1".to_owned()),
        }),
        output: Some(RealtimeAudioOutputConfig {
            format: Some(AudioFormat::Pcm16),
            voice: Some(Voice::Alloy),
            speed: Some(1.0),
        }),
    }),
    ..Default::default()
};
```

### AudioFormat

| Variant | Wire name | Description |
|---|---|---|
| `AudioFormat::Pcm16` | `"pcm16"` | PCM 16-bit at 24 kHz (default) |
| `AudioFormat::G711Ulaw` | `"g711_ulaw"` | G.711 mu-law |
| `AudioFormat::G711Alaw` | `"g711_alaw"` | G.711 a-law |

### Voice

Available TTS voices:

| Variant | Wire name |
|---|---|
| `Voice::Alloy` | `"alloy"` (default) |
| `Voice::Ash` | `"ash"` |
| `Voice::Ballad` | `"ballad"` |
| `Voice::Coral` | `"coral"` |
| `Voice::Echo` | `"echo"` |
| `Voice::Sage` | `"sage"` |
| `Voice::Shimmer` | `"shimmer"` |
| `Voice::Verse` | `"verse"` |
| `Voice::Custom(name)` | Any string |

### TurnDetection

Controls how the model detects end-of-speech:

```rust
use openai_agents::voice::TurnDetection;

// Server-side VAD (default).
let vad = TurnDetection::ServerVad {
    threshold: 0.5,          // activation threshold 0.0–1.0
    prefix_padding_ms: 300,  // padding before detected speech
    silence_duration_ms: 500, // silence needed to trigger turn end
    create_response: None,
    interrupt_response: None,
};

// Semantic VAD (OpenAI's improved VAD).
let semantic_vad = TurnDetection::SemanticVad {
    eagerness: Some("medium".to_owned()),
    create_response: Some(true),
    interrupt_response: Some(true),
};

// Push-to-talk mode.
let manual = TurnDetection::Disabled;
```

---

## RealtimeSession

`RealtimeSession` manages the raw WebSocket connection to the OpenAI Realtime API. Use it
for low-level control over the session lifecycle.

```rust,no_run
use openai_agents::voice::{RealtimeSession, RealtimeConfig, RealtimeAgent};

let agent = RealtimeAgent::<()>::builder("assistant")
    .instructions("You are a voice assistant.")
    .build();

let config = RealtimeConfig::default();
let session = RealtimeSession::new(agent, config);
```

The session is the bridge between your application's audio I/O and the Realtime API
WebSocket stream.

---

## VoicePipeline

`VoicePipeline` is the highest-level entry point. It orchestrates audio capture, the
WebSocket connection, tool execution, and handoffs.

```rust,no_run
use openai_agents::voice::{
    VoicePipeline, VoicePipelineConfig, AudioInput, AudioOutput, RealtimeConfig,
};

let config = VoicePipelineConfig {
    audio_input: AudioInput::Microphone,
    audio_output: AudioOutput::Speaker,
    realtime_config: RealtimeConfig::default(),
};

// Attach a workflow and start the pipeline.
// (Full WebSocket transport is coming — the API is stable for forward-compatibility.)
let pipeline = VoicePipeline::new(config);
```

### AudioInput / AudioOutput

```rust
use openai_agents::voice::{AudioInput, AudioOutput};

// Microphone and speaker (default).
let input = AudioInput::Microphone;
let output = AudioOutput::Speaker;

// File-based (useful for testing).
let input = AudioInput::File("/tmp/recording.pcm".to_owned());
let output = AudioOutput::File("/tmp/response.pcm".to_owned());

// Programmatic byte stream.
let input = AudioInput::Stream;
let output = AudioOutput::Stream;
```

---

## VoiceWorkflows

A voice workflow ties agents to the session's input/output pipeline. Implement
`VoiceWorkflowBase` for full control, or use `SingleAgentVoiceWorkflow` for simple cases.

### SingleAgentVoiceWorkflow

Feeds each user transcription to a single `Agent<C>` and streams back text that the TTS
model converts to speech:

```rust,no_run
use openai_agents::voice::SingleAgentVoiceWorkflow;
use openai_agents::Agent;

let agent = Agent::<()>::builder("voice-bot")
    .instructions("Answer questions concisely. You are speaking to a user.")
    .build();

let workflow = SingleAgentVoiceWorkflow::new(agent);
```

### Implementing VoiceWorkflowBase

For multi-agent voice conversations or custom routing:

```rust,no_run
use openai_agents::voice::VoiceWorkflowBase;
use openai_agents::error::Result;
use async_trait::async_trait;
use tokio_stream::Stream;

struct MyVoiceWorkflow;

#[async_trait]
impl VoiceWorkflowBase for MyVoiceWorkflow {
    async fn run(
        &self,
        transcription: &str,
    ) -> Result<Box<dyn Stream<Item = String> + Send + Unpin>> {
        // Process the transcription and return a stream of response text chunks.
        let response = format!("You said: {transcription}");
        Ok(Box::new(tokio_stream::once(response)))
    }

    async fn on_start(&self) -> Result<Box<dyn Stream<Item = String> + Send + Unpin>> {
        // Optional greeting spoken when the session starts.
        Ok(Box::new(tokio_stream::once(
            "Hello! How can I help you today?".to_owned()
        )))
    }
}
```

---

## Audio Utilities

The `utils` module provides helpers for working with PCM16 audio.

### SentenceSplitter

Splits a stream of text deltas into complete sentences suitable for TTS streaming. Flushing
partial sentences avoids choppy audio at turn boundaries.

```rust,no_run
use openai_agents::voice::SentenceSplitter;

let mut splitter = SentenceSplitter::new();
splitter.push("The quick brown fox");
splitter.push(" jumps over the lazy");
splitter.push(" dog. And then it ran away.");

while let Some(sentence) = splitter.next_sentence() {
    println!("Sentence: {}", sentence);
    // => "The quick brown fox jumps over the lazy dog."
    // => (partial "And then it ran away." flushed on reset)
}
```

Use `get_sentence_based_splitter()` for a ready-made splitter suitable for most use cases:

```rust,no_run
use openai_agents::voice::get_sentence_based_splitter;

let splitter = get_sentence_based_splitter();
```

### PCM16 utilities

```rust
use openai_agents::voice::{pcm16_duration_seconds, resample_pcm16};

// Calculate the duration of a PCM16 buffer at 24 kHz mono.
let samples: &[i16] = &[0i16; 24000]; // 1 second of silence
let duration = pcm16_duration_seconds(samples, 24000, 1);
assert!((duration - 1.0).abs() < 0.001);

// Resample from 8 kHz to 24 kHz.
let input_8k: Vec<i16> = vec![0i16; 8000];
let output_24k = resample_pcm16(&input_8k, 8000, 24000);
assert_eq!(output_24k.len(), 24000);
```

---

## Events and Transcripts

`RealtimeSessionEvent` is the top-level event emitted by a session. Key variants include:

| Event | Description |
|---|---|
| `SessionCreatedEvent` | WebSocket session opened |
| `SpeechStartedEvent` | User started speaking |
| `SpeechStoppedEvent` | User stopped speaking |
| `TranscriptDeltaEvent` | Incremental transcript text |
| `ResponseDoneEvent` | Model response complete |
| `AudioDeltaEvent` | Incremental audio data |
| `RealtimeToolStartEvent` | Agent tool call began |
| `RealtimeToolEndEvent` | Agent tool call completed |
| `RealtimeHandoffEvent` | Agent handoff occurred |
| `RealtimeErrorEvent` | Session error |

`TranscriptEntry` captures the full transcript of a conversation turn:

```rust,no_run
use openai_agents::voice::TranscriptEntry;

// TranscriptEntry { role: TranscriptRole::User, text: "What's the weather?" }
// TranscriptEntry { role: TranscriptRole::Assistant, text: "It's sunny today." }
```

---

## Complete Example

```rust,no_run
use openai_agents::voice::{
    SingleAgentVoiceWorkflow, VoicePipeline, VoicePipelineConfig,
    AudioInput, AudioOutput, RealtimeConfig, AudioFormat, Voice, TurnDetection,
    RealtimeAudioConfig, RealtimeAudioOutputConfig,
};
use openai_agents::Agent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::<()>::builder("voice-assistant")
        .instructions(
            "You are a friendly, concise voice assistant. \
             Keep answers to one or two sentences."
        )
        .build();

    let workflow = SingleAgentVoiceWorkflow::new(agent);

    let config = VoicePipelineConfig {
        audio_input: AudioInput::Microphone,
        audio_output: AudioOutput::Speaker,
        realtime_config: RealtimeConfig {
            model: Some("gpt-4o-realtime-preview".to_owned()),
            audio: Some(RealtimeAudioConfig {
                output: Some(RealtimeAudioOutputConfig {
                    voice: Some(Voice::Alloy),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
    };

    let mut pipeline = VoicePipeline::new(config);
    pipeline.set_workflow(Box::new(workflow));
    pipeline.run().await?;

    Ok(())
}
```

---

## See Also

- [Agents](./agents.md) — the standard `Agent<C>` type.
- [Streaming](./streaming.md) — text-based streaming.
- [Extensions](./extensions.md) — Codex and other experimental features.
