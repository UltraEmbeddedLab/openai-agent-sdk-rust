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

    /// Click at `(x, y)` with the specified `button` while holding modifier `keys`.
    ///
    /// Drivers that support held modifier keys (for example Ctrl+click) should
    /// override this method. The default implementation discards `keys` and
    /// delegates to [`click`](Self::click), so existing drivers continue to
    /// work unchanged.
    async fn click_with_modifiers(
        &self,
        x: i32,
        y: i32,
        button: Button,
        keys: Option<&[String]>,
    ) -> Result<()> {
        let _ = keys;
        self.click(x, y, button).await
    }

    /// Double-click at `(x, y)` while holding modifier `keys`.
    ///
    /// Default implementation discards `keys` and delegates to
    /// [`double_click`](Self::double_click).
    async fn double_click_with_modifiers(
        &self,
        x: i32,
        y: i32,
        keys: Option<&[String]>,
    ) -> Result<()> {
        let _ = keys;
        self.double_click(x, y).await
    }

    /// Scroll at `(x, y)` by `(scroll_x, scroll_y)` while holding modifier `keys`.
    ///
    /// Default implementation discards `keys` and delegates to
    /// [`scroll`](Self::scroll).
    async fn scroll_with_modifiers(
        &self,
        x: i32,
        y: i32,
        scroll_x: i32,
        scroll_y: i32,
        keys: Option<&[String]>,
    ) -> Result<()> {
        let _ = keys;
        self.scroll(x, y, scroll_x, scroll_y).await
    }

    /// Move the cursor to `(x, y)` while holding modifier `keys`.
    ///
    /// Default implementation discards `keys` and delegates to
    /// [`move_cursor`](Self::move_cursor).
    async fn move_cursor_with_modifiers(
        &self,
        x: i32,
        y: i32,
        keys: Option<&[String]>,
    ) -> Result<()> {
        let _ = keys;
        self.move_cursor(x, y).await
    }

    /// Drag along `path` while holding modifier `keys`.
    ///
    /// Default implementation discards `keys` and delegates to
    /// [`drag`](Self::drag).
    async fn drag_with_modifiers(
        &self,
        path: &[(i32, i32)],
        keys: Option<&[String]>,
    ) -> Result<()> {
        let _ = keys;
        self.drag(path).await
    }
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

    // ---- Computer default modifier-key forwarders ----

    struct RecordingComputer {
        last_click: std::sync::Mutex<Option<(i32, i32, Button)>>,
        last_drag: std::sync::Mutex<Option<Vec<(i32, i32)>>>,
    }

    #[async_trait]
    impl Computer for RecordingComputer {
        async fn screenshot(&self) -> Result<String> {
            Ok(String::new())
        }
        async fn click(&self, x: i32, y: i32, button: Button) -> Result<()> {
            *self.last_click.lock().unwrap() = Some((x, y, button));
            Ok(())
        }
        async fn double_click(&self, _x: i32, _y: i32) -> Result<()> {
            Ok(())
        }
        async fn scroll(&self, _x: i32, _y: i32, _sx: i32, _sy: i32) -> Result<()> {
            Ok(())
        }
        async fn type_text(&self, _text: &str) -> Result<()> {
            Ok(())
        }
        async fn wait(&self) -> Result<()> {
            Ok(())
        }
        async fn move_cursor(&self, _x: i32, _y: i32) -> Result<()> {
            Ok(())
        }
        async fn keypress(&self, _keys: &[String]) -> Result<()> {
            Ok(())
        }
        async fn drag(&self, path: &[(i32, i32)]) -> Result<()> {
            *self.last_drag.lock().unwrap() = Some(path.to_vec());
            Ok(())
        }
    }

    #[tokio::test]
    async fn modifier_key_variants_forward_to_base_methods() {
        let c = RecordingComputer {
            last_click: std::sync::Mutex::new(None),
            last_drag: std::sync::Mutex::new(None),
        };
        let keys = vec!["ctrl".to_owned()];

        c.click_with_modifiers(10, 20, Button::Left, Some(&keys))
            .await
            .unwrap();
        assert_eq!(
            c.last_click.lock().unwrap().as_ref(),
            Some(&(10, 20, Button::Left))
        );

        c.drag_with_modifiers(&[(1, 2), (3, 4)], Some(&keys))
            .await
            .unwrap();
        assert_eq!(
            c.last_drag.lock().unwrap().as_ref(),
            Some(&vec![(1, 2), (3, 4)])
        );

        // None keys also works.
        c.double_click_with_modifiers(5, 6, None).await.unwrap();
        c.scroll_with_modifiers(0, 0, 1, 1, None).await.unwrap();
        c.move_cursor_with_modifiers(0, 0, None).await.unwrap();
    }
}
