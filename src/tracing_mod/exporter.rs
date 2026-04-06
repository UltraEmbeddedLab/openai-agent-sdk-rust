//! OpenTelemetry OTLP trace exporter.
//!
//! This module is available with the `tracing-otlp` feature flag.
//! It configures the `tracing` ecosystem to export spans to an
//! OTLP-compatible backend (for example, Jaeger, Zipkin, or Grafana Tempo).
//!
//! # Usage
//!
//! ```no_run
//! use openai_agents::tracing_support::exporter::{OtlpExporterConfig, init_otlp_tracing};
//!
//! # fn main() -> openai_agents::Result<()> {
//! let config = OtlpExporterConfig::new()
//!     .with_endpoint("http://localhost:4317")
//!     .with_service_name("my-agent-app");
//!
//! // Keep the guard alive for the lifetime of the application.
//! let _guard = init_otlp_tracing(config)?;
//!
//! // ... run your agents ...
//! # Ok(())
//! # }
//! ```

use crate::error::{AgentError, Result};

/// Configuration for the OTLP exporter.
///
/// Use the builder methods to customise the endpoint and service name.
/// The defaults target a local OTLP collector on `http://localhost:4317`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OtlpExporterConfig {
    /// The OTLP endpoint URL (default: `"http://localhost:4317"`).
    pub endpoint: String,
    /// Service name reported in traces (default: `"openai-agents"`).
    pub service_name: String,
}

impl Default for OtlpExporterConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:4317".to_string(),
            service_name: "openai-agents".to_string(),
        }
    }
}

impl OtlpExporterConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the OTLP endpoint URL.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Set the service name reported in traces.
    #[must_use]
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }
}

/// Initialize the global tracing subscriber with OTLP export.
///
/// This sets up:
/// 1. A `tracing-subscriber` registry with an OpenTelemetry layer.
/// 2. An OTLP exporter that sends spans to the configured endpoint via gRPC.
/// 3. A console logging layer filtered by the `RUST_LOG` environment variable.
///
/// Call this once at the start of your application. The returned [`OtlpGuard`]
/// must be kept alive for the duration of the process; dropping it flushes
/// remaining spans and shuts down the trace provider.
///
/// # Errors
///
/// Returns an error if the OTLP exporter or the tracing subscriber cannot be
/// initialised (for example, if the endpoint is unreachable at setup time or
/// a global subscriber has already been set).
#[cfg(feature = "tracing-otlp")]
pub fn init_otlp_tracing(config: OtlpExporterConfig) -> Result<OtlpGuard> {
    use opentelemetry::KeyValue;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::{SpanExporter, WithExportConfig as _};
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_subscriber::Layer as _;
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.endpoint)
        .build()
        .map_err(|e| AgentError::UserError {
            message: format!("Failed to create OTLP exporter: {e}"),
        })?;

    let resource = Resource::builder_empty()
        .with_attributes([KeyValue::new("service.name", config.service_name.clone())])
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer(config.service_name);
    let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(telemetry_layer)
        .with(
            tracing_subscriber::fmt::layer()
                .with_filter(tracing_subscriber::EnvFilter::from_default_env()),
        )
        .try_init()
        .map_err(|e| AgentError::UserError {
            message: format!("Failed to initialize tracing subscriber: {e}"),
        })?;

    Ok(OtlpGuard {
        provider: Some(provider),
    })
}

/// Stub for when the `tracing-otlp` feature is not enabled.
///
/// Always returns an error directing the caller to enable the feature flag.
///
/// # Errors
///
/// Always returns [`AgentError::UserError`].
#[cfg(not(feature = "tracing-otlp"))]
pub fn init_otlp_tracing(_config: OtlpExporterConfig) -> Result<()> {
    Err(AgentError::UserError {
        message: "OTLP tracing requires the 'tracing-otlp' feature flag".into(),
    })
}

/// A guard that shuts down the OTLP trace provider when dropped.
///
/// Keep this value alive for the duration of your application. When it is
/// dropped the provider flushes any buffered spans and releases resources.
#[cfg(feature = "tracing-otlp")]
pub struct OtlpGuard {
    provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

#[cfg(feature = "tracing-otlp")]
impl OtlpGuard {
    /// Force-flush all buffered trace spans to the configured exporter.
    ///
    /// This ensures any pending spans are exported immediately rather than
    /// waiting for the next scheduled batch export. Useful for short-lived
    /// processes or after completing a unit of work.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying trace provider fails to flush.
    pub fn flush(&self) -> crate::error::Result<()> {
        if let Some(provider) = &self.provider {
            provider
                .force_flush()
                .map_err(|e| crate::error::AgentError::UserError {
                    message: format!("Failed to flush traces: {e}"),
                })?;
        }
        Ok(())
    }
}

#[cfg(feature = "tracing-otlp")]
impl Drop for OtlpGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            if let Err(e) = provider.shutdown() {
                eprintln!("Failed to shut down OTLP trace provider: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = OtlpExporterConfig::default();
        assert_eq!(config.endpoint, "http://localhost:4317");
        assert_eq!(config.service_name, "openai-agents");
    }

    #[test]
    fn new_equals_default() {
        let a = OtlpExporterConfig::new();
        let b = OtlpExporterConfig::default();
        assert_eq!(a.endpoint, b.endpoint);
        assert_eq!(a.service_name, b.service_name);
    }

    #[test]
    fn builder_with_endpoint() {
        let config = OtlpExporterConfig::new().with_endpoint("http://otel:4317");
        assert_eq!(config.endpoint, "http://otel:4317");
        // Service name should remain at default.
        assert_eq!(config.service_name, "openai-agents");
    }

    #[test]
    fn builder_with_service_name() {
        let config = OtlpExporterConfig::new().with_service_name("my-app");
        assert_eq!(config.service_name, "my-app");
        // Endpoint should remain at default.
        assert_eq!(config.endpoint, "http://localhost:4317");
    }

    #[test]
    fn builder_chaining() {
        let config = OtlpExporterConfig::new()
            .with_endpoint("http://collector:4317")
            .with_service_name("agent-service");
        assert_eq!(config.endpoint, "http://collector:4317");
        assert_eq!(config.service_name, "agent-service");
    }

    #[test]
    fn config_is_clone() {
        let config = OtlpExporterConfig::new().with_service_name("test");
        let cloned = config.clone();
        assert_eq!(cloned.endpoint, config.endpoint);
        assert_eq!(cloned.service_name, config.service_name);
    }

    #[test]
    fn config_is_debug() {
        let config = OtlpExporterConfig::new();
        let debug = format!("{config:?}");
        assert!(debug.contains("OtlpExporterConfig"));
        assert!(debug.contains("localhost"));
    }

    #[cfg(not(feature = "tracing-otlp"))]
    #[test]
    fn init_without_feature_returns_error() {
        let result = init_otlp_tracing(OtlpExporterConfig::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("tracing-otlp"),
            "error message should mention the feature flag"
        );
    }

    #[cfg(feature = "tracing-otlp")]
    #[test]
    fn flush_with_none_provider_returns_ok() {
        let guard = OtlpGuard { provider: None };
        assert!(guard.flush().is_ok());
        // Prevent the Drop impl from running shutdown on None (it's fine, but be explicit).
        std::mem::forget(guard);
    }

    #[test]
    fn builder_accepts_string_types() {
        // Verify that both &str and String work via Into<String>.
        let _ = OtlpExporterConfig::new().with_endpoint(String::from("http://host:4317"));
        let _ = OtlpExporterConfig::new().with_service_name(String::from("svc"));
    }
}
