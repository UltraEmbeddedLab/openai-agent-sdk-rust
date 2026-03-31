//! Retry policy and execution logic for API calls.
//!
//! This module provides [`RetryPolicy`] for configuring retry behavior with
//! exponential backoff and optional jitter, as well as supporting types
//! [`RetryContext`] and [`RetryDecision`].
//!
//! The design mirrors the Python SDK's `retry.py` while adapting to Rust's type
//! system and async patterns.

use std::time::Duration;

use rand::Rng;

use crate::error::{AgentError, Result};

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

/// A retry policy for API calls.
///
/// Controls how many times an operation is retried, how long to wait between
/// attempts (exponential backoff), and whether random jitter is applied.
/// Use the builder methods for ergonomic construction.
///
/// # Examples
///
/// ```
/// use openai_agents::retry::RetryPolicy;
/// use std::time::Duration;
///
/// let policy = RetryPolicy::new(5)
///     .with_initial_delay(Duration::from_millis(500))
///     .with_backoff_factor(3.0)
///     .with_jitter(false);
///
/// assert_eq!(policy.max_retries, 5);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RetryPolicy {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay before the first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Backoff multiplier applied to the delay after each retry.
    pub backoff_factor: f64,
    /// Whether to add random jitter to the delay.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy with the given maximum number of retries.
    ///
    /// All other fields are set to their defaults.
    #[must_use]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Self::default()
        }
    }

    /// Create a retry policy that disables retries entirely.
    #[must_use]
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Set the maximum number of retries and return `self` (fluent builder).
    #[must_use]
    pub const fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set the initial delay before the first retry and return `self` (fluent builder).
    #[must_use]
    pub const fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the maximum delay between retries and return `self` (fluent builder).
    #[must_use]
    pub const fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set the backoff multiplier and return `self` (fluent builder).
    #[must_use]
    pub const fn with_backoff_factor(mut self, factor: f64) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Set whether to add random jitter and return `self` (fluent builder).
    #[must_use]
    pub const fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// Calculate the delay for the given attempt number (0-based).
    ///
    /// Uses exponential backoff: `min(initial_delay * backoff_factor^attempt, max_delay)`.
    /// When jitter is enabled, the delay is randomized uniformly between zero and the
    /// computed value.
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        #[allow(clippy::cast_possible_wrap)]
        let base_secs = self.initial_delay.as_secs_f64() * self.backoff_factor.powi(attempt as i32);
        let capped_secs = base_secs.min(self.max_delay.as_secs_f64());

        if self.jitter {
            let jittered = rand::rng().random_range(0.0..=capped_secs);
            Duration::from_secs_f64(jittered)
        } else {
            Duration::from_secs_f64(capped_secs)
        }
    }

    /// Execute an async operation with retry logic.
    ///
    /// The `operation` closure is called up to `max_retries + 1` times. If the
    /// operation returns an error that is considered retryable (HTTP or model
    /// behavior errors), the policy waits for the computed backoff delay before
    /// the next attempt. Non-retryable errors are returned immediately.
    ///
    /// # Errors
    ///
    /// Returns the last error if all attempts fail or if a non-retryable error
    /// is encountered.
    ///
    /// # Panics
    ///
    /// Panics if `max_retries` is zero and no attempt was made, which should be
    /// unreachable in practice.
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            match operation().await {
                Ok(value) => return Ok(value),
                Err(e) => {
                    if !Self::is_retryable(&e) || attempt == self.max_retries {
                        return Err(e);
                    }
                    last_error = Some(e);
                    let delay = self.delay_for_attempt(attempt);
                    tokio::time::sleep(delay).await;
                }
            }
        }

        // This branch is only reachable if max_retries is 0 and the loop body
        // already returned on attempt 0. In practice, the for loop handles all
        // cases, but we satisfy the compiler here.
        Err(last_error.expect("at least one attempt must have been made"))
    }

    /// Determine if an error is retryable.
    ///
    /// Currently, HTTP errors and model behavior errors are considered retryable.
    /// All other error variants are treated as non-retryable.
    const fn is_retryable(error: &AgentError) -> bool {
        matches!(
            error,
            AgentError::Http(_) | AgentError::ModelBehavior { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// RetryContext
// ---------------------------------------------------------------------------

/// Context provided to retry advice methods.
///
/// Contains information about the current retry state that can be used by
/// custom retry policies or hooks to make decisions.
#[derive(Debug)]
#[non_exhaustive]
pub struct RetryContext {
    /// The current attempt number (0-based).
    pub attempt: u32,
    /// The error message from the last attempt.
    pub last_error: String,
}

impl RetryContext {
    /// Create a new retry context.
    #[must_use]
    pub fn new(attempt: u32, last_error: impl Into<String>) -> Self {
        Self {
            attempt,
            last_error: last_error.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// RetryDecision
// ---------------------------------------------------------------------------

/// Advice from a model or policy about how to retry a failed request.
///
/// This enum allows retry hooks to communicate whether an operation should be
/// retried and, if so, with what delay.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryDecision {
    /// Retry the request using the policy's default delay.
    Retry,
    /// Do not retry; fail immediately.
    Fail,
    /// Retry with a specific delay.
    RetryAfter(Duration),
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    // -- Default policy values --

    #[test]
    fn default_policy_values() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.initial_delay, Duration::from_secs(1));
        assert_eq!(policy.max_delay, Duration::from_secs(30));
        assert!((policy.backoff_factor - 2.0).abs() < f64::EPSILON);
        assert!(policy.jitter);
    }

    // -- delay_for_attempt with exponential backoff (no jitter) --

    #[test]
    fn delay_exponential_backoff_no_jitter() {
        let policy = RetryPolicy::new(5)
            .with_initial_delay(Duration::from_secs(1))
            .with_backoff_factor(2.0)
            .with_jitter(false);

        // attempt 0: 1 * 2^0 = 1s
        assert_eq!(policy.delay_for_attempt(0), Duration::from_secs(1));
        // attempt 1: 1 * 2^1 = 2s
        assert_eq!(policy.delay_for_attempt(1), Duration::from_secs(2));
        // attempt 2: 1 * 2^2 = 4s
        assert_eq!(policy.delay_for_attempt(2), Duration::from_secs(4));
        // attempt 3: 1 * 2^3 = 8s
        assert_eq!(policy.delay_for_attempt(3), Duration::from_secs(8));
    }

    // -- delay respects max_delay cap --

    #[test]
    fn delay_respects_max_delay_cap() {
        let policy = RetryPolicy::new(10)
            .with_initial_delay(Duration::from_secs(1))
            .with_backoff_factor(10.0)
            .with_max_delay(Duration::from_secs(5))
            .with_jitter(false);

        // attempt 0: 1 * 10^0 = 1s (under cap)
        assert_eq!(policy.delay_for_attempt(0), Duration::from_secs(1));
        // attempt 1: 1 * 10^1 = 10s -> capped to 5s
        assert_eq!(policy.delay_for_attempt(1), Duration::from_secs(5));
        // attempt 2: 1 * 10^2 = 100s -> capped to 5s
        assert_eq!(policy.delay_for_attempt(2), Duration::from_secs(5));
    }

    // -- delay with jitter is bounded --

    #[test]
    fn delay_with_jitter_is_bounded() {
        let policy = RetryPolicy::new(3)
            .with_initial_delay(Duration::from_secs(2))
            .with_backoff_factor(2.0)
            .with_jitter(true);

        // Run multiple times to verify jitter stays within bounds.
        for _ in 0..100 {
            let delay = policy.delay_for_attempt(0);
            // Should be between 0 and 2s (initial_delay * 2^0).
            assert!(delay <= Duration::from_secs(2));
        }
    }

    // -- execute succeeds on first try --

    #[tokio::test]
    async fn execute_succeeds_on_first_try() {
        let policy = RetryPolicy::new(3).with_jitter(false);
        let result = policy.execute(|| async { Ok::<_, AgentError>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    // -- execute retries on retryable error --

    #[tokio::test]
    async fn execute_retries_on_retryable_error() {
        let policy = RetryPolicy::new(3)
            .with_initial_delay(Duration::from_millis(1))
            .with_jitter(false);

        let call_count = AtomicU32::new(0);

        let result = policy
            .execute(|| {
                let count = call_count.fetch_add(1, Ordering::SeqCst);
                async move {
                    if count < 2 {
                        Err(AgentError::ModelBehavior {
                            message: "transient".to_string(),
                        })
                    } else {
                        Ok(99)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 99);
        assert_eq!(call_count.load(Ordering::SeqCst), 3); // initial + 2 retries
    }

    // -- execute fails on non-retryable error --

    #[tokio::test]
    async fn execute_fails_on_non_retryable_error() {
        let policy = RetryPolicy::new(3)
            .with_initial_delay(Duration::from_millis(1))
            .with_jitter(false);

        let call_count = AtomicU32::new(0);

        let result: Result<i32> = policy
            .execute(|| {
                call_count.fetch_add(1, Ordering::SeqCst);
                async {
                    Err(AgentError::UserError {
                        message: "bad config".to_string(),
                    })
                }
            })
            .await;

        assert!(result.is_err());
        // Should not retry non-retryable errors.
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        assert!(
            matches!(result.unwrap_err(), AgentError::UserError { message } if message == "bad config")
        );
    }

    // -- execute exhausts retries --

    #[tokio::test]
    async fn execute_exhausts_retries() {
        let policy = RetryPolicy::new(2)
            .with_initial_delay(Duration::from_millis(1))
            .with_jitter(false);

        let call_count = AtomicU32::new(0);

        let result: Result<i32> = policy
            .execute(|| {
                call_count.fetch_add(1, Ordering::SeqCst);
                async {
                    Err(AgentError::ModelBehavior {
                        message: "always fails".to_string(),
                    })
                }
            })
            .await;

        assert!(result.is_err());
        // 1 initial + 2 retries = 3 total calls.
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    // -- RetryPolicy::none() disables retries --

    #[tokio::test]
    async fn none_policy_disables_retries() {
        let policy = RetryPolicy::none();
        assert_eq!(policy.max_retries, 0);

        let call_count = AtomicU32::new(0);

        let result: Result<i32> = policy
            .execute(|| {
                call_count.fetch_add(1, Ordering::SeqCst);
                async {
                    Err(AgentError::ModelBehavior {
                        message: "fail".to_string(),
                    })
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    // -- Builder methods --

    #[test]
    fn builder_methods() {
        let policy = RetryPolicy::default()
            .with_max_retries(5)
            .with_initial_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(60))
            .with_backoff_factor(3.0)
            .with_jitter(false);

        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.initial_delay, Duration::from_millis(500));
        assert_eq!(policy.max_delay, Duration::from_secs(60));
        assert!((policy.backoff_factor - 3.0).abs() < f64::EPSILON);
        assert!(!policy.jitter);
    }

    // -- is_retryable for different error types --

    #[test]
    fn is_retryable_http_error() {
        // We cannot easily construct a reqwest::Error, so we test via execute behavior.
        // Instead, test the model behavior variant directly.
        let err = AgentError::ModelBehavior {
            message: "oops".to_string(),
        };
        assert!(RetryPolicy::is_retryable(&err));
    }

    #[test]
    fn is_retryable_model_behavior() {
        let err = AgentError::ModelBehavior {
            message: "bad tool call".to_string(),
        };
        assert!(RetryPolicy::is_retryable(&err));
    }

    #[test]
    fn is_not_retryable_user_error() {
        let err = AgentError::UserError {
            message: "invalid config".to_string(),
        };
        assert!(!RetryPolicy::is_retryable(&err));
    }

    #[test]
    fn is_not_retryable_max_turns() {
        let err = AgentError::MaxTurnsExceeded { max_turns: 10 };
        assert!(!RetryPolicy::is_retryable(&err));
    }

    #[test]
    fn is_not_retryable_guardrail_tripwire() {
        let err = AgentError::InputGuardrailTripwire {
            guardrail_name: "test".to_string(),
        };
        assert!(!RetryPolicy::is_retryable(&err));
    }

    #[test]
    fn is_not_retryable_tool_timeout() {
        let err = AgentError::ToolTimeout {
            tool_name: "search".to_string(),
            timeout_seconds: 10.0,
        };
        assert!(!RetryPolicy::is_retryable(&err));
    }

    #[test]
    fn is_not_retryable_serialization() {
        let json_err =
            serde_json::from_str::<serde_json::Value>("{{bad}}").expect_err("should fail");
        let err = AgentError::Serialization(json_err);
        assert!(!RetryPolicy::is_retryable(&err));
    }

    // -- RetryContext --

    #[test]
    fn retry_context_creation() {
        let ctx = RetryContext::new(2, "something failed");
        assert_eq!(ctx.attempt, 2);
        assert_eq!(ctx.last_error, "something failed");
    }

    // -- RetryDecision --

    #[test]
    fn retry_decision_variants() {
        let retry = RetryDecision::Retry;
        let fail = RetryDecision::Fail;
        let after = RetryDecision::RetryAfter(Duration::from_secs(5));

        assert_eq!(retry, RetryDecision::Retry);
        assert_eq!(fail, RetryDecision::Fail);
        assert_eq!(after, RetryDecision::RetryAfter(Duration::from_secs(5)));
    }
}
