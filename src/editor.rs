//! Editor and file manipulation tools.
//!
//! Provides tools for applying patches and editing files, used by agents that
//! need to modify code or text files. This module includes:
//!
//! - [`ApplyPatchTool`] — a hosted tool for applying V4A-format diffs to files.
//! - [`ApplyPatchEditor`] — a trait for custom editor implementations.
//! - [`ApplyPatchOperation`] and [`ApplyPatchResult`] — operation and result types.
//! - [`apply_diff`] — the core diff application algorithm.
//!
//! This module mirrors the Python SDK's `editor.py` and `apply_diff.py`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{AgentError, Result};

// ---------------------------------------------------------------------------
// Types from editor.py
// ---------------------------------------------------------------------------

/// The type of operation for an apply-patch editor request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ApplyPatchOperationType {
    /// Create a new file.
    #[serde(rename = "create_file")]
    CreateFile,
    /// Update an existing file.
    #[serde(rename = "update_file")]
    UpdateFile,
    /// Delete an existing file.
    #[serde(rename = "delete_file")]
    DeleteFile,
}

impl std::fmt::Display for ApplyPatchOperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateFile => write!(f, "create_file"),
            Self::UpdateFile => write!(f, "update_file"),
            Self::DeleteFile => write!(f, "delete_file"),
        }
    }
}

/// Represents a single apply-patch editor operation requested by the model.
///
/// Corresponds to the Python SDK's `ApplyPatchOperation` dataclass.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ApplyPatchOperation {
    /// The type of operation to perform.
    pub operation_type: ApplyPatchOperationType,
    /// Path to the file to modify.
    pub path: String,
    /// The diff content to apply, if applicable.
    pub diff: Option<String>,
}

impl ApplyPatchOperation {
    /// Create a new apply-patch operation.
    #[must_use]
    pub fn new(
        operation_type: ApplyPatchOperationType,
        path: impl Into<String>,
        diff: Option<String>,
    ) -> Self {
        Self {
            operation_type,
            path: path.into(),
            diff,
        }
    }
}

/// The status of an apply-patch operation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ApplyPatchStatus {
    /// The operation completed successfully.
    #[serde(rename = "completed")]
    Completed,
    /// The operation failed.
    #[serde(rename = "failed")]
    Failed,
}

/// Optional metadata returned by editor operations.
///
/// Corresponds to the Python SDK's `ApplyPatchResult` dataclass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ApplyPatchResult {
    /// The status of the operation.
    pub status: Option<ApplyPatchStatus>,
    /// Output message from the operation.
    pub output: Option<String>,
}

impl ApplyPatchResult {
    /// Create a successful result with the given output.
    #[must_use]
    pub fn completed(output: impl Into<String>) -> Self {
        Self {
            status: Some(ApplyPatchStatus::Completed),
            output: Some(output.into()),
        }
    }

    /// Create a failed result with the given error message.
    #[must_use]
    pub fn failed(output: impl Into<String>) -> Self {
        Self {
            status: Some(ApplyPatchStatus::Failed),
            output: Some(output.into()),
        }
    }
}

/// Host-defined editor that applies diffs on disk.
///
/// Implement this trait to provide custom file editing capabilities.
/// Corresponds to the Python SDK's `ApplyPatchEditor` protocol.
#[async_trait]
pub trait ApplyPatchEditor: Send + Sync {
    /// Create a new file based on the operation.
    async fn create_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult>;

    /// Update an existing file based on the operation.
    async fn update_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult>;

    /// Delete a file based on the operation.
    async fn delete_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult>;
}

// ---------------------------------------------------------------------------
// Hosted tool types
// ---------------------------------------------------------------------------

/// A hosted tool for applying patches to files.
///
/// This tool enables agents to apply V4A-format diffs to files
/// using the `OpenAI` Responses API apply-patch tool type.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ApplyPatchTool;

impl ApplyPatchTool {
    /// Create a new apply-patch tool.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// V4A diff application (from apply_diff.py)
// ---------------------------------------------------------------------------

/// The mode for applying a diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyDiffMode {
    /// Default mode: the diff contains context hunks and deletion/insertion lines.
    Default,
    /// Create mode: the diff contains only `+` prefixed lines for a new file.
    Create,
}

/// Apply a V4A diff to the provided text.
///
/// This parser understands both the create-file syntax (only `+` prefixed lines)
/// and the default update syntax that includes context hunks.
///
/// Corresponds to the Python SDK's `apply_diff` function in `apply_diff.py`.
///
/// # Errors
///
/// Returns [`AgentError::UserError`] if the diff is malformed or cannot be applied.
pub fn apply_diff(input: &str, diff: &str, mode: ApplyDiffMode) -> Result<String> {
    let newline = detect_newline(input, diff, mode);
    let diff_lines = normalize_diff_lines(diff);

    if mode == ApplyDiffMode::Create {
        return parse_create_diff(&diff_lines, newline);
    }

    let normalized_input = normalize_text_newlines(input);
    let parsed = parse_update_diff(&diff_lines, &normalized_input)?;
    apply_chunks(&normalized_input, &parsed.chunks, newline)
}

// ---------------------------------------------------------------------------
// Internal diff parsing types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Chunk {
    orig_index: usize,
    del_lines: Vec<String>,
    ins_lines: Vec<String>,
}

#[derive(Debug)]
struct ParserState {
    lines: Vec<String>,
    index: usize,
    fuzz: usize,
}

#[derive(Debug)]
struct ParsedUpdateDiff {
    chunks: Vec<Chunk>,
    #[allow(dead_code)]
    fuzz: usize,
}

#[derive(Debug)]
struct ReadSectionResult {
    next_context: Vec<String>,
    section_chunks: Vec<Chunk>,
    end_index: usize,
    eof: bool,
}

#[derive(Debug)]
struct ContextMatch {
    /// `None` means "not found" (the Python code uses `-1`).
    new_index: Option<usize>,
    fuzz: usize,
}

const END_PATCH: &str = "*** End Patch";
const END_FILE: &str = "*** End of File";

const SECTION_TERMINATORS: &[&str] = &[
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
];

fn end_section_markers() -> Vec<&'static str> {
    let mut markers: Vec<&str> = SECTION_TERMINATORS.to_vec();
    markers.push(END_FILE);
    markers
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn normalize_diff_lines(diff: &str) -> Vec<String> {
    let mut lines: Vec<String> = diff
        .split('\n')
        .map(|line| line.trim_end_matches('\r').to_owned())
        .collect();
    if lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

fn detect_newline_from_text(text: &str) -> &'static str {
    if text.contains("\r\n") { "\r\n" } else { "\n" }
}

fn detect_newline(input: &str, diff: &str, mode: ApplyDiffMode) -> &'static str {
    if mode != ApplyDiffMode::Create && input.contains('\n') {
        return detect_newline_from_text(input);
    }
    detect_newline_from_text(diff)
}

fn normalize_text_newlines(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn is_done(state: &ParserState, prefixes: &[&str]) -> bool {
    if state.index >= state.lines.len() {
        return true;
    }
    prefixes
        .iter()
        .any(|prefix| state.lines[state.index].starts_with(prefix))
}

fn read_str(state: &mut ParserState, prefix: &str) -> String {
    if state.index >= state.lines.len() {
        return String::new();
    }
    let current = &state.lines[state.index];
    if let Some(rest) = current.strip_prefix(prefix) {
        state.index += 1;
        return rest.to_owned();
    }
    String::new()
}

fn parse_create_diff(lines: &[String], newline: &str) -> Result<String> {
    let mut all_lines: Vec<String> = lines.to_vec();
    all_lines.push(END_PATCH.to_owned());

    let mut parser = ParserState {
        lines: all_lines,
        index: 0,
        fuzz: 0,
    };

    let mut output: Vec<String> = Vec::new();

    while !is_done(&parser, SECTION_TERMINATORS) {
        if parser.index >= parser.lines.len() {
            break;
        }
        let line = parser.lines[parser.index].clone();
        parser.index += 1;
        if let Some(rest) = line.strip_prefix('+') {
            output.push(rest.to_owned());
        } else {
            return Err(AgentError::UserError {
                message: format!("Invalid Add File Line: {line}"),
            });
        }
    }

    Ok(output.join(newline))
}

fn parse_update_diff(lines: &[String], input: &str) -> Result<ParsedUpdateDiff> {
    let mut all_lines: Vec<String> = lines.to_vec();
    all_lines.push(END_PATCH.to_owned());

    let mut parser = ParserState {
        lines: all_lines,
        index: 0,
        fuzz: 0,
    };

    let input_lines: Vec<&str> = input.split('\n').collect();
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cursor: usize = 0;
    let markers = end_section_markers();

    while !is_done(&parser, &markers) {
        let anchor = read_str(&mut parser, "@@ ");
        let has_bare_anchor = anchor.is_empty()
            && parser.index < parser.lines.len()
            && parser.lines[parser.index] == "@@";
        if has_bare_anchor {
            parser.index += 1;
        }

        if !(!anchor.is_empty() || has_bare_anchor || cursor == 0) {
            let current_line = if parser.index < parser.lines.len() {
                &parser.lines[parser.index]
            } else {
                ""
            };
            return Err(AgentError::UserError {
                message: format!("Invalid Line:\n{current_line}"),
            });
        }

        let trimmed = anchor.trim();
        if !trimmed.is_empty() {
            cursor = advance_cursor_to_anchor(trimmed, &input_lines, cursor, &mut parser);
        }

        let section = read_section(&parser.lines, parser.index)?;
        let find_result = find_context(&input_lines, &section.next_context, cursor, section.eof);

        let Some(new_idx) = find_result.new_index else {
            let ctx_text = section.next_context.join("\n");
            if section.eof {
                return Err(AgentError::UserError {
                    message: format!("Invalid EOF Context {cursor}:\n{ctx_text}"),
                });
            }
            return Err(AgentError::UserError {
                message: format!("Invalid Context {cursor}:\n{ctx_text}"),
            });
        };
        cursor = new_idx + section.next_context.len();
        parser.fuzz += find_result.fuzz;
        parser.index = section.end_index;

        for ch in &section.section_chunks {
            chunks.push(Chunk {
                orig_index: ch.orig_index + new_idx,
                del_lines: ch.del_lines.clone(),
                ins_lines: ch.ins_lines.clone(),
            });
        }
    }

    Ok(ParsedUpdateDiff {
        chunks,
        fuzz: parser.fuzz,
    })
}

fn advance_cursor_to_anchor(
    anchor: &str,
    input_lines: &[&str],
    mut cursor: usize,
    parser: &mut ParserState,
) -> usize {
    let mut found = false;

    // Exact match search.
    if !input_lines[..cursor].contains(&anchor) {
        if let Some(pos) = input_lines[cursor..]
            .iter()
            .position(|line| *line == anchor)
        {
            cursor = cursor + pos + 1;
            found = true;
        }
    }

    // Fuzzy (trimmed) match search.
    let anchor_trimmed = anchor.trim();
    if !found
        && !input_lines[..cursor]
            .iter()
            .any(|line| line.trim() == anchor_trimmed)
    {
        if let Some(pos) = input_lines[cursor..]
            .iter()
            .position(|line| line.trim() == anchor_trimmed)
        {
            cursor = cursor + pos + 1;
            parser.fuzz += 1;
        }
    }

    cursor
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SectionMode {
    Keep,
    Add,
    Delete,
}

fn read_section(lines: &[String], start_index: usize) -> Result<ReadSectionResult> {
    let mut context: Vec<String> = Vec::new();
    let mut del_lines: Vec<String> = Vec::new();
    let mut ins_lines: Vec<String> = Vec::new();
    let mut section_chunks: Vec<Chunk> = Vec::new();
    let mut mode = SectionMode::Keep;
    let mut index = start_index;

    while index < lines.len() {
        let raw = &lines[index];

        if raw.starts_with("@@")
            || raw.starts_with(END_PATCH)
            || raw.starts_with("*** Update File:")
            || raw.starts_with("*** Delete File:")
            || raw.starts_with("*** Add File:")
            || raw.starts_with(END_FILE)
        {
            break;
        }
        if raw == "***" {
            break;
        }
        if raw.starts_with("***") {
            return Err(AgentError::UserError {
                message: format!("Invalid Line: {raw}"),
            });
        }

        index += 1;
        let last_mode = mode;
        // Empty line is treated as a space-prefixed context line.
        let line = if raw.is_empty() { " " } else { raw.as_str() };
        let prefix = line.as_bytes()[0];

        mode = match prefix {
            b'+' => SectionMode::Add,
            b'-' => SectionMode::Delete,
            b' ' => SectionMode::Keep,
            _ => {
                return Err(AgentError::UserError {
                    message: format!("Invalid Line: {line}"),
                });
            }
        };

        let line_content = &line[1..];
        let switching_to_context = mode == SectionMode::Keep && last_mode != mode;
        if switching_to_context && (!del_lines.is_empty() || !ins_lines.is_empty()) {
            section_chunks.push(Chunk {
                orig_index: context.len() - del_lines.len(),
                del_lines: std::mem::take(&mut del_lines),
                ins_lines: std::mem::take(&mut ins_lines),
            });
        }

        match mode {
            SectionMode::Delete => {
                del_lines.push(line_content.to_owned());
                context.push(line_content.to_owned());
            }
            SectionMode::Add => {
                ins_lines.push(line_content.to_owned());
            }
            SectionMode::Keep => {
                context.push(line_content.to_owned());
            }
        }
    }

    if !del_lines.is_empty() || !ins_lines.is_empty() {
        section_chunks.push(Chunk {
            orig_index: context.len() - del_lines.len(),
            del_lines,
            ins_lines,
        });
    }

    if index < lines.len() && lines[index] == END_FILE {
        return Ok(ReadSectionResult {
            next_context: context,
            section_chunks,
            end_index: index + 1,
            eof: true,
        });
    }

    if index == start_index {
        let next_line = if index < lines.len() {
            &lines[index]
        } else {
            ""
        };
        return Err(AgentError::UserError {
            message: format!("Nothing in this section - index={index} {next_line}"),
        });
    }

    Ok(ReadSectionResult {
        next_context: context,
        section_chunks,
        end_index: index,
        eof: false,
    })
}

fn find_context(lines: &[&str], context: &[String], start: usize, eof: bool) -> ContextMatch {
    if eof {
        let end_start = lines.len().saturating_sub(context.len());
        let end_match = find_context_core(lines, context, end_start);
        if end_match.new_index.is_some() {
            return end_match;
        }
        let fallback = find_context_core(lines, context, start);
        return ContextMatch {
            new_index: fallback.new_index,
            fuzz: fallback.fuzz + 10000,
        };
    }
    find_context_core(lines, context, start)
}

fn find_context_core(lines: &[&str], context: &[String], start: usize) -> ContextMatch {
    if context.is_empty() {
        return ContextMatch {
            new_index: Some(start),
            fuzz: 0,
        };
    }

    // Exact match.
    for i in start..lines.len() {
        if equals_slice(lines, context, i, str::to_owned) {
            return ContextMatch {
                new_index: Some(i),
                fuzz: 0,
            };
        }
    }
    // Trailing-whitespace-trimmed match.
    for i in start..lines.len() {
        if equals_slice(lines, context, i, |v| v.trim_end().to_owned()) {
            return ContextMatch {
                new_index: Some(i),
                fuzz: 1,
            };
        }
    }
    // Fully-stripped match.
    for i in start..lines.len() {
        if equals_slice(lines, context, i, |v| v.trim().to_owned()) {
            return ContextMatch {
                new_index: Some(i),
                fuzz: 100,
            };
        }
    }

    ContextMatch {
        new_index: None,
        fuzz: 0,
    }
}

fn equals_slice(
    source: &[&str],
    target: &[String],
    start: usize,
    map_fn: fn(&str) -> String,
) -> bool {
    if start + target.len() > source.len() {
        return false;
    }
    for (offset, target_value) in target.iter().enumerate() {
        if map_fn(source[start + offset]) != map_fn(target_value) {
            return false;
        }
    }
    true
}

fn apply_chunks(input: &str, chunks: &[Chunk], newline: &str) -> Result<String> {
    let orig_lines: Vec<&str> = input.split('\n').collect();
    let mut dest_lines: Vec<&str> = Vec::new();
    let mut cursor: usize = 0;

    for chunk in chunks {
        if chunk.orig_index > orig_lines.len() {
            return Err(AgentError::UserError {
                message: format!(
                    "applyDiff: chunk.origIndex {} > input length {}",
                    chunk.orig_index,
                    orig_lines.len()
                ),
            });
        }
        if cursor > chunk.orig_index {
            return Err(AgentError::UserError {
                message: format!(
                    "applyDiff: overlapping chunk at {} (cursor {cursor})",
                    chunk.orig_index
                ),
            });
        }

        dest_lines.extend_from_slice(&orig_lines[cursor..chunk.orig_index]);
        cursor = chunk.orig_index;

        for ins in &chunk.ins_lines {
            dest_lines.push(ins);
        }

        cursor += chunk.del_lines.len();
    }

    dest_lines.extend_from_slice(&orig_lines[cursor..]);
    Ok(dest_lines.join(newline))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ApplyPatchEditor trait is object-safe ----

    #[test]
    fn apply_patch_editor_is_object_safe() {
        fn _assert_object_safe(_: &dyn ApplyPatchEditor) {}
    }

    // ---- ApplyPatchTool defaults ----

    #[test]
    fn apply_patch_tool_default() {
        let _tool = ApplyPatchTool::default();
        let _tool2 = ApplyPatchTool::new();
    }

    // ---- ApplyPatchOperationType Display ----

    #[test]
    fn operation_type_display() {
        assert_eq!(
            ApplyPatchOperationType::CreateFile.to_string(),
            "create_file"
        );
        assert_eq!(
            ApplyPatchOperationType::UpdateFile.to_string(),
            "update_file"
        );
        assert_eq!(
            ApplyPatchOperationType::DeleteFile.to_string(),
            "delete_file"
        );
    }

    // ---- ApplyPatchOperation construction ----

    #[test]
    fn apply_patch_operation_new() {
        let op = ApplyPatchOperation::new(
            ApplyPatchOperationType::UpdateFile,
            "/path/to/file.rs",
            Some("diff content".to_owned()),
        );
        assert_eq!(op.operation_type, ApplyPatchOperationType::UpdateFile);
        assert_eq!(op.path, "/path/to/file.rs");
        assert_eq!(op.diff.as_deref(), Some("diff content"));
    }

    // ---- ApplyPatchOperation serialization ----

    #[test]
    fn apply_patch_operation_serialization() {
        let op = ApplyPatchOperation::new(ApplyPatchOperationType::CreateFile, "test.txt", None);
        let json = serde_json::to_string(&op).expect("should serialize");
        assert!(json.contains("\"create_file\""));
        assert!(json.contains("\"test.txt\""));

        let deserialized: ApplyPatchOperation =
            serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(
            deserialized.operation_type,
            ApplyPatchOperationType::CreateFile
        );
        assert_eq!(deserialized.path, "test.txt");
    }

    // ---- ApplyPatchResult construction ----

    #[test]
    fn apply_patch_result_completed() {
        let result = ApplyPatchResult::completed("done");
        assert_eq!(result.status, Some(ApplyPatchStatus::Completed));
        assert_eq!(result.output.as_deref(), Some("done"));
    }

    #[test]
    fn apply_patch_result_failed() {
        let result = ApplyPatchResult::failed("error occurred");
        assert_eq!(result.status, Some(ApplyPatchStatus::Failed));
        assert_eq!(result.output.as_deref(), Some("error occurred"));
    }

    #[test]
    fn apply_patch_result_default() {
        let result = ApplyPatchResult::default();
        assert!(result.status.is_none());
        assert!(result.output.is_none());
    }

    // ---- apply_diff with create mode ----

    #[test]
    fn apply_diff_create_mode() {
        let diff = "+line 1\n+line 2\n+line 3";
        let result = apply_diff("", diff, ApplyDiffMode::Create).expect("should apply");
        assert_eq!(result, "line 1\nline 2\nline 3");
    }

    // ---- apply_diff with simple update ----

    #[test]
    fn apply_diff_simple_update() {
        let input = "line 1\nline 2\nline 3\n";
        let diff = " line 1\n-line 2\n+line 2 modified\n line 3\n";
        let result = apply_diff(input, diff, ApplyDiffMode::Default).expect("should apply");
        assert_eq!(result, "line 1\nline 2 modified\nline 3\n");
    }

    // ---- apply_diff with add-only ----

    #[test]
    fn apply_diff_add_only() {
        let input = "line 1\nline 2\n";
        let diff = " line 1\n+inserted\n line 2\n";
        let result = apply_diff(input, diff, ApplyDiffMode::Default).expect("should apply");
        assert_eq!(result, "line 1\ninserted\nline 2\n");
    }

    // ---- apply_diff with remove-only ----

    #[test]
    fn apply_diff_remove_only() {
        let input = "line 1\nline 2\nline 3\n";
        let diff = " line 1\n-line 2\n line 3\n";
        let result = apply_diff(input, diff, ApplyDiffMode::Default).expect("should apply");
        assert_eq!(result, "line 1\nline 3\n");
    }

    // ---- apply_diff with multi-hunk ----

    #[test]
    fn apply_diff_multi_hunk() {
        let input = "a\nb\nc\nd\ne\n";
        let diff = " a\n-b\n+B\n c\n d\n-e\n+E\n";
        let result = apply_diff(input, diff, ApplyDiffMode::Default).expect("should apply");
        assert_eq!(result, "a\nB\nc\nd\nE\n");
    }

    // ---- apply_diff invalid create line ----

    #[test]
    fn apply_diff_create_invalid_line() {
        let diff = "+line 1\nno prefix\n+line 3";
        let result = apply_diff("", diff, ApplyDiffMode::Create);
        assert!(result.is_err());
    }

    // ---- ApplyPatchStatus serialization ----

    #[test]
    fn apply_patch_status_serialization() {
        let json = serde_json::to_string(&ApplyPatchStatus::Completed).expect("serialize");
        assert_eq!(json, "\"completed\"");

        let json = serde_json::to_string(&ApplyPatchStatus::Failed).expect("serialize");
        assert_eq!(json, "\"failed\"");
    }

    // ---- Send + Sync assertions ----

    #[test]
    fn types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ApplyPatchTool>();
        assert_send_sync::<ApplyPatchOperation>();
        assert_send_sync::<ApplyPatchResult>();
        assert_send_sync::<Box<dyn ApplyPatchEditor>>();
    }
}
