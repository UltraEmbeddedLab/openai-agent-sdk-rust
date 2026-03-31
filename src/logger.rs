//! Logging configuration helpers.
//!
//! Provides convenience functions for setting up `tracing-subscriber` output
//! during development and debugging. This mirrors the Python SDK's `logger.py`
//! module.

use tracing_subscriber::EnvFilter;

/// Enable verbose stdout logging for debugging.
///
/// Sets up a `tracing-subscriber` with:
/// - `RUST_LOG` environment variable support.
/// - Default level of `debug` for the SDK, `info` for dependencies.
/// - Human-readable output format.
///
/// Call this at the start of your program for development logging. This
/// function should only be called once; subsequent calls will be silently
/// ignored if a global subscriber has already been set.
pub fn enable_verbose_stdout_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("openai_agents=debug,info"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .try_init();
}

/// Set up logging with a specific filter string.
///
/// Example filter strings:
/// - `"debug"` - Everything at debug level.
/// - `"openai_agents=trace"` - Trace level for SDK only.
/// - `"openai_agents::runner=debug"` - Debug level for runner module.
///
/// This function should only be called once; subsequent calls will be
/// silently ignored if a global subscriber has already been set.
pub fn setup_logging(filter: &str) {
    let env_filter = EnvFilter::new(filter);

    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_logging_does_not_panic() {
        // Just verify the function does not panic. Since a global subscriber
        // may already be set by other tests, we use try_init internally.
        setup_logging("warn");
    }

    #[test]
    fn enable_verbose_does_not_panic() {
        // Just verify the function does not panic.
        enable_verbose_stdout_logging();
    }
}
