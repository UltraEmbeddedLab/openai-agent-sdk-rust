//! Computer control interface for desktop and browser automation.
//!
//! This module provides the [`Computer`] trait for implementing computer control
//! capabilities (mouse, keyboard, screenshots) and the [`ComputerTool`] struct
//! that wraps a `Computer` implementation as a hosted tool for agents.
//!
//! This module mirrors the Python SDK's `computer.py`.

use async_trait::async_trait;

use crate::error::Result;

/// The type of operating environment for the computer tool.
///
/// Corresponds to the Python SDK's `Environment` literal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Environment {
    /// macOS environment.
    Mac,
    /// Windows environment.
    Windows,
    /// Ubuntu/Linux environment.
    Ubuntu,
    /// Browser-based environment.
    Browser,
}

impl std::fmt::Display for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mac => write!(f, "mac"),
            Self::Windows => write!(f, "windows"),
            Self::Ubuntu => write!(f, "ubuntu"),
            Self::Browser => write!(f, "browser"),
        }
    }
}

/// Mouse button types for click operations.
///
/// Corresponds to the Python SDK's `Button` literal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Button {
    /// Left mouse button.
    Left,
    /// Right mouse button.
    Right,
    /// Mouse wheel button.
    Wheel,
    /// Back navigation button.
    Back,
    /// Forward navigation button.
    Forward,
}

impl std::fmt::Display for Button {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => write!(f, "left"),
            Self::Right => write!(f, "right"),
            Self::Wheel => write!(f, "wheel"),
            Self::Back => write!(f, "back"),
            Self::Forward => write!(f, "forward"),
        }
    }
}

/// A computer environment that can be controlled by an agent.
///
/// Implement this trait to provide desktop or browser automation capabilities.
/// The trait defines the operations needed to control a computer: taking
/// screenshots, clicking, typing, scrolling, and dragging.
///
/// This is the async version that mirrors the Python SDK's `AsyncComputer` class.
/// Since Rust is async-first, there is no separate sync version.
#[async_trait]
pub trait Computer: Send + Sync {
    /// Return the environment type, if applicable.
    ///
    /// Returns `None` if the environment type is not relevant.
    fn environment(&self) -> Option<Environment> {
        None
    }

    /// Return the display dimensions as `(width, height)`, if applicable.
    ///
    /// Returns `None` if the dimensions are not relevant.
    fn dimensions(&self) -> Option<(u32, u32)> {
        None
    }

    /// Take a screenshot and return it as a base64-encoded string.
    async fn screenshot(&self) -> Result<String>;

    /// Click at the given coordinates with the specified button.
    async fn click(&self, x: i32, y: i32, button: Button) -> Result<()>;

    /// Double-click at the given coordinates.
    async fn double_click(&self, x: i32, y: i32) -> Result<()>;

    /// Scroll at the given coordinates by the specified deltas.
    async fn scroll(&self, x: i32, y: i32, scroll_x: i32, scroll_y: i32) -> Result<()>;

    /// Type the given text string.
    async fn type_text(&self, text: &str) -> Result<()>;

    /// Wait for a default duration.
    async fn wait(&self) -> Result<()>;

    /// Move the mouse cursor to the given coordinates.
    async fn move_cursor(&self, x: i32, y: i32) -> Result<()>;

    /// Press one or more keys simultaneously.
    async fn keypress(&self, keys: &[String]) -> Result<()>;

    /// Drag along a path of coordinates.
    ///
    /// The path is a sequence of `(x, y)` points to drag through.
    async fn drag(&self, path: &[(i32, i32)]) -> Result<()>;
}

/// A hosted computer tool that can be given to an agent.
///
/// This represents the metadata for a computer-use tool. The actual
/// computer control is performed by a [`Computer`] trait implementation
/// that is associated with the tool at runtime.
///
/// Corresponds to the Python SDK's `ComputerTool` hosted tool type.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ComputerTool {
    /// Display width for the computer tool.
    pub display_width: u32,
    /// Display height for the computer tool.
    pub display_height: u32,
    /// The environment type for the computer tool.
    pub environment: Environment,
}

impl Default for ComputerTool {
    fn default() -> Self {
        Self {
            display_width: 1920,
            display_height: 1080,
            environment: Environment::Ubuntu,
        }
    }
}

impl ComputerTool {
    /// Create a new computer tool with the specified display dimensions.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            display_width: width,
            display_height: height,
            environment: Environment::Ubuntu,
        }
    }

    /// Set the environment type for this computer tool.
    #[must_use]
    pub const fn with_environment(mut self, environment: Environment) -> Self {
        self.environment = environment;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Computer trait is object-safe ----

    #[test]
    fn computer_trait_is_object_safe() {
        // This compiles only if `Computer` is object-safe.
        fn _assert_object_safe(_: &dyn Computer) {}
    }

    // ---- Computer trait is Send + Sync ----

    #[test]
    fn computer_trait_requires_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // `dyn Computer` must be Send + Sync because the trait has those supertraits.
        assert_send_sync::<Box<dyn Computer>>();
    }

    // ---- ComputerTool defaults ----

    #[test]
    fn computer_tool_default() {
        let tool = ComputerTool::default();
        assert_eq!(tool.display_width, 1920);
        assert_eq!(tool.display_height, 1080);
        assert_eq!(tool.environment, Environment::Ubuntu);
    }

    // ---- ComputerTool::new ----

    #[test]
    fn computer_tool_new() {
        let tool = ComputerTool::new(800, 600);
        assert_eq!(tool.display_width, 800);
        assert_eq!(tool.display_height, 600);
        assert_eq!(tool.environment, Environment::Ubuntu);
    }

    // ---- ComputerTool::with_environment ----

    #[test]
    fn computer_tool_with_environment() {
        let tool = ComputerTool::new(1024, 768).with_environment(Environment::Browser);
        assert_eq!(tool.display_width, 1024);
        assert_eq!(tool.display_height, 768);
        assert_eq!(tool.environment, Environment::Browser);
    }

    // ---- Environment Display ----

    #[test]
    fn environment_display() {
        assert_eq!(Environment::Mac.to_string(), "mac");
        assert_eq!(Environment::Windows.to_string(), "windows");
        assert_eq!(Environment::Ubuntu.to_string(), "ubuntu");
        assert_eq!(Environment::Browser.to_string(), "browser");
    }

    // ---- Button Display ----

    #[test]
    fn button_display() {
        assert_eq!(Button::Left.to_string(), "left");
        assert_eq!(Button::Right.to_string(), "right");
        assert_eq!(Button::Wheel.to_string(), "wheel");
        assert_eq!(Button::Back.to_string(), "back");
        assert_eq!(Button::Forward.to_string(), "forward");
    }

    // ---- Clone and Debug ----

    #[test]
    fn computer_tool_clone_and_debug() {
        let tool = ComputerTool::new(1920, 1080);
        let cloned = tool.clone();
        assert_eq!(cloned.display_width, tool.display_width);
        assert_eq!(cloned.display_height, tool.display_height);

        let debug_str = format!("{tool:?}");
        assert!(debug_str.contains("ComputerTool"));
        assert!(debug_str.contains("1920"));
    }

    // ---- Environment and Button equality ----

    #[test]
    fn environment_equality() {
        assert_eq!(Environment::Mac, Environment::Mac);
        assert_ne!(Environment::Mac, Environment::Windows);
    }

    #[test]
    fn button_equality() {
        assert_eq!(Button::Left, Button::Left);
        assert_ne!(Button::Left, Button::Right);
    }
}
