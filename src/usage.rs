//! Token usage tracking for LLM API requests.
//!
//! This module provides types for tracking token consumption across individual API
//! requests and aggregated across an entire agent run. It mirrors the Python SDK's
//! `usage.py` module.

use serde::{Deserialize, Serialize};

/// Token usage details for input tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct InputTokensDetails {
    /// Number of cached tokens used from the input.
    pub cached_tokens: u64,
}

/// Token usage details for output tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct OutputTokensDetails {
    /// Number of reasoning tokens produced in the output.
    pub reasoning_tokens: u64,
}

/// Usage details for a single API request.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
#[non_exhaustive]
pub struct RequestUsage {
    /// Input tokens for this individual request.
    pub input_tokens: u64,
    /// Output tokens for this individual request.
    pub output_tokens: u64,
    /// Total tokens (input + output) for this individual request.
    pub total_tokens: u64,
    /// Details about the input tokens for this individual request.
    pub input_tokens_details: InputTokensDetails,
    /// Details about the output tokens for this individual request.
    pub output_tokens_details: OutputTokensDetails,
}

/// Aggregated token usage across all requests in an agent run.
///
/// Each call to [`Usage::add`] accumulates the token counts and, when appropriate,
/// preserves per-request breakdowns in [`request_usage_entries`](Usage::request_usage_entries)
/// for detailed cost calculation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
#[non_exhaustive]
pub struct Usage {
    /// Total number of requests made to the LLM API.
    pub requests: u64,
    /// Total input tokens sent, across all requests.
    pub input_tokens: u64,
    /// Total output tokens received, across all requests.
    pub output_tokens: u64,
    /// Total tokens sent and received, across all requests.
    pub total_tokens: u64,
    /// Details about the input tokens, matching the responses API usage details.
    pub input_tokens_details: InputTokensDetails,
    /// Details about the output tokens, matching the responses API usage details.
    pub output_tokens_details: OutputTokensDetails,
    /// Per-request usage entries for accurate cost calculation.
    ///
    /// Each call to [`add`](Usage::add) automatically creates an entry in this list
    /// if the added usage represents a single new request with non-zero tokens.
    /// When the added usage already contains individual request breakdowns, those
    /// entries are merged instead.
    pub request_usage_entries: Vec<RequestUsage>,
}

impl Usage {
    /// Add another `Usage` to this one, aggregating all fields.
    ///
    /// Scalar token counts are summed. Nested detail fields (`cached_tokens`,
    /// `reasoning_tokens`) are summed as well. Per-request entries are preserved:
    ///
    /// - If `other` represents a single request (`requests == 1`) with non-zero
    ///   tokens, a new [`RequestUsage`] entry is appended.
    /// - If `other` already contains individual request breakdowns, those entries
    ///   are appended in order.
    pub fn add(&mut self, other: &Self) {
        self.requests += other.requests;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;

        self.input_tokens_details.cached_tokens += other.input_tokens_details.cached_tokens;
        self.output_tokens_details.reasoning_tokens += other.output_tokens_details.reasoning_tokens;

        // Preserve per-request breakdowns.
        if other.requests == 1 && other.total_tokens > 0 {
            self.request_usage_entries.push(RequestUsage {
                input_tokens: other.input_tokens,
                output_tokens: other.output_tokens,
                total_tokens: other.total_tokens,
                input_tokens_details: other.input_tokens_details.clone(),
                output_tokens_details: other.output_tokens_details.clone(),
            });
        } else if !other.request_usage_entries.is_empty() {
            self.request_usage_entries
                .extend(other.request_usage_entries.iter().cloned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_zeros() {
        let usage = Usage::default();
        assert_eq!(usage.requests, 0);
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
        assert!(usage.request_usage_entries.is_empty());
    }

    #[test]
    fn add_single_request_usage() {
        let mut usage = Usage::default();
        let single = Usage {
            requests: 1,
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            input_tokens_details: InputTokensDetails { cached_tokens: 20 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 10,
            },
            request_usage_entries: Vec::new(),
        };

        usage.add(&single);

        assert_eq!(usage.requests, 1);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.input_tokens_details.cached_tokens, 20);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 10);
        assert_eq!(usage.request_usage_entries.len(), 1);
        assert_eq!(usage.request_usage_entries[0].input_tokens, 100);
        assert_eq!(usage.request_usage_entries[0].total_tokens, 150);
    }

    #[test]
    fn add_multiple_single_requests() {
        let mut usage = Usage::default();

        let first = Usage {
            requests: 1,
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            input_tokens_details: InputTokensDetails { cached_tokens: 10 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 5,
            },
            request_usage_entries: Vec::new(),
        };

        let second = Usage {
            requests: 1,
            input_tokens: 200,
            output_tokens: 80,
            total_tokens: 280,
            input_tokens_details: InputTokensDetails { cached_tokens: 30 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 15,
            },
            request_usage_entries: Vec::new(),
        };

        usage.add(&first);
        usage.add(&second);

        assert_eq!(usage.requests, 2);
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 130);
        assert_eq!(usage.total_tokens, 430);
        assert_eq!(usage.input_tokens_details.cached_tokens, 40);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 20);
        assert_eq!(usage.request_usage_entries.len(), 2);
        assert_eq!(usage.request_usage_entries[0].input_tokens, 100);
        assert_eq!(usage.request_usage_entries[1].input_tokens, 200);
    }

    #[test]
    fn add_merges_existing_request_entries() {
        let mut usage = Usage::default();

        // Simulate an aggregated usage that already has per-request breakdowns.
        let aggregated = Usage {
            requests: 3,
            input_tokens: 500,
            output_tokens: 200,
            total_tokens: 700,
            input_tokens_details: InputTokensDetails { cached_tokens: 50 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 25,
            },
            request_usage_entries: vec![
                RequestUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                    total_tokens: 150,
                    input_tokens_details: InputTokensDetails { cached_tokens: 10 },
                    output_tokens_details: OutputTokensDetails {
                        reasoning_tokens: 5,
                    },
                },
                RequestUsage {
                    input_tokens: 200,
                    output_tokens: 80,
                    total_tokens: 280,
                    input_tokens_details: InputTokensDetails { cached_tokens: 20 },
                    output_tokens_details: OutputTokensDetails {
                        reasoning_tokens: 10,
                    },
                },
                RequestUsage {
                    input_tokens: 200,
                    output_tokens: 70,
                    total_tokens: 270,
                    input_tokens_details: InputTokensDetails { cached_tokens: 20 },
                    output_tokens_details: OutputTokensDetails {
                        reasoning_tokens: 10,
                    },
                },
            ],
        };

        usage.add(&aggregated);

        assert_eq!(usage.requests, 3);
        assert_eq!(usage.total_tokens, 700);
        assert_eq!(usage.request_usage_entries.len(), 3);
    }

    #[test]
    fn add_single_request_with_zero_tokens_does_not_create_entry() {
        let mut usage = Usage::default();
        let empty = Usage {
            requests: 1,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            ..Usage::default()
        };

        usage.add(&empty);

        assert_eq!(usage.requests, 1);
        assert!(usage.request_usage_entries.is_empty());
    }

    #[test]
    fn serde_round_trip_usage() {
        let usage = Usage {
            requests: 2,
            input_tokens: 300,
            output_tokens: 130,
            total_tokens: 430,
            input_tokens_details: InputTokensDetails { cached_tokens: 40 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 20,
            },
            request_usage_entries: vec![RequestUsage {
                input_tokens: 100,
                output_tokens: 50,
                total_tokens: 150,
                input_tokens_details: InputTokensDetails { cached_tokens: 10 },
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 5,
                },
            }],
        };

        let json = serde_json::to_string(&usage).expect("serialize");
        let deserialized: Usage = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(usage, deserialized);
    }

    #[test]
    fn serde_round_trip_request_usage() {
        let req = RequestUsage {
            input_tokens: 42,
            output_tokens: 17,
            total_tokens: 59,
            input_tokens_details: InputTokensDetails { cached_tokens: 5 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 3,
            },
        };

        let json = serde_json::to_string(&req).expect("serialize");
        let deserialized: RequestUsage = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(req, deserialized);
    }

    #[test]
    fn input_details_partial_eq() {
        let same_a = InputTokensDetails { cached_tokens: 10 };
        let same_b = InputTokensDetails { cached_tokens: 10 };
        let different = InputTokensDetails { cached_tokens: 20 };
        assert_eq!(same_a, same_b);
        assert_ne!(same_a, different);
    }

    #[test]
    fn output_details_partial_eq() {
        let same_a = OutputTokensDetails {
            reasoning_tokens: 5,
        };
        let same_b = OutputTokensDetails {
            reasoning_tokens: 5,
        };
        let different = OutputTokensDetails {
            reasoning_tokens: 0,
        };
        assert_eq!(same_a, same_b);
        assert_ne!(same_a, different);
    }

    #[test]
    fn deserialize_from_json_with_missing_fields_uses_defaults() {
        let json = r#"{"requests":1,"input_tokens":50,"output_tokens":25,"total_tokens":75}"#;
        let usage: Usage = serde_json::from_str(json).expect("deserialize");

        assert_eq!(usage.requests, 1);
        assert_eq!(usage.input_tokens, 50);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
        assert!(usage.request_usage_entries.is_empty());
    }
}
