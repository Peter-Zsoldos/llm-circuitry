//! **Program counter and call stack for agents.**
//!
//! A CPU always knows where it is: the program counter points at the current
//! instruction and the stack records which call to return to. Agents on long
//! tasks lose this: they forget they're inside a sub-task, redo finished steps,
//! or "return" to the wrong goal after the context gets compacted.
//!
//! [`ExecStack`] keeps that state outside the context window. Push a frame per
//! goal, record steps (the program counter), pop with a return value that is
//! handed to the parent frame, and [`render`](ExecStack::render) the whole
//! thing back into the context whenever it might have been lost. It serialises
//! to JSON, so it is also a context-switch save/restore format.

use crate::error::{err, Error, ErrorKind, Result};
use crate::util;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrameStatus {
    Active,
    Done,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub id: u64,
    pub goal: String,
    pub parent: Option<u64>,
    pub depth: usize,
    pub status: FrameStatus,
    /// Completed steps; the length is the frame's program counter.
    pub steps: Vec<String>,
    pub notes: Vec<String>,
    pub result: Value,
    pub error: Option<String>,
    pub created_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecStack {
    frames: Vec<Frame>,
    finished: Vec<Frame>,
    next_id: u64,
    max_depth: usize,
}

impl ExecStack {
    pub fn new(max_depth: usize) -> Self {
        ExecStack {
            frames: Vec::new(),
            finished: Vec::new(),
            next_id: 1,
            max_depth: max_depth.max(1),
        }
    }

    /// Enter a sub-goal. Errors with `Capacity` past `max_depth` (stack
    /// overflow: the agent is recursing into sub-sub-sub-tasks).
    pub fn push(&mut self, goal: &str) -> Result<u64> {
        if self.frames.len() >= self.max_depth {
            return err(
                ErrorKind::Capacity,
                format!(
                    "stack overflow: max depth {} reached while pushing '{goal}' ({})",
                    self.max_depth,
                    self.breadcrumb()
                ),
            );
        }
        let id = self.next_id;
        self.next_id += 1;
        self.frames.push(Frame {
            id,
            goal: goal.into(),
            parent: self.frames.last().map(|f| f.id),
            depth: self.frames.len(),
            status: FrameStatus::Active,
            steps: vec![],
            notes: vec![],
            result: Value::Null,
            error: None,
            created_ms: util::now_ms(),
        });
        Ok(id)
    }

    fn top_mut(&mut self) -> Result<&mut Frame> {
        self.frames
            .last_mut()
            .ok_or_else(|| Error::new(ErrorKind::InvalidState, "the execution stack is empty"))
    }

    /// Record a completed step in the current frame. Returns the new program counter.
    pub fn step(&mut self, description: &str) -> Result<usize> {
        let f = self.top_mut()?;
        f.steps.push(description.into());
        Ok(f.steps.len())
    }

    /// Attach a note (a fact learned, a decision made) to the current frame.
    pub fn note(&mut self, text: &str) -> Result<()> {
        self.top_mut()?.notes.push(text.into());
        Ok(())
    }

    /// Finish the current goal and return `result` to the parent frame.
    pub fn pop(&mut self, result: Value) -> Result<Frame> {
        let mut f = self
            .frames
            .pop()
            .ok_or_else(|| Error::new(ErrorKind::InvalidState, "the execution stack is empty"))?;
        f.status = FrameStatus::Done;
        f.result = result;
        if let Some(parent) = self.frames.last_mut() {
            let summary = if f.result.is_null() {
                "done".to_string()
            } else {
                util::truncate(&f.result.to_string(), 120)
            };
            parent.steps.push(format!("sub-goal '{}' returned: {summary}", f.goal));
        }
        self.finished.push(f.clone());
        Ok(f)
    }

    /// Abandon the current goal with an error, reported to the parent frame.
    pub fn fail(&mut self, error: &str) -> Result<Frame> {
        let mut f = self
            .frames
            .pop()
            .ok_or_else(|| Error::new(ErrorKind::InvalidState, "the execution stack is empty"))?;
        f.status = FrameStatus::Failed;
        f.error = Some(error.into());
        if let Some(parent) = self.frames.last_mut() {
            parent.notes.push(format!("sub-goal '{}' FAILED: {error}", f.goal));
        }
        self.finished.push(f.clone());
        Ok(f)
    }

    /// Fail every frame above `frame_id`, leaving it on top (like an exception
    /// propagating up to a handler).
    pub fn unwind_to(&mut self, frame_id: u64, reason: &str) -> Result<Vec<Frame>> {
        if !self.frames.iter().any(|f| f.id == frame_id) {
            return err(ErrorKind::NotFound, format!("frame {frame_id} is not on the stack"));
        }
        let mut out = Vec::new();
        while self.frames.last().map(|f| f.id) != Some(frame_id) {
            out.push(self.fail(reason)?);
        }
        Ok(out)
    }

    pub fn current(&self) -> Option<&Frame> {
        self.frames.last()
    }

    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }

    pub fn finished(&self) -> &[Frame] {
        &self.finished
    }

    /// `Ship release > Fix tests > Investigate test_auth`
    pub fn breadcrumb(&self) -> String {
        self.frames
            .iter()
            .map(|f| f.goal.as_str())
            .collect::<Vec<_>>()
            .join(" > ")
    }

    /// Text block to re-inject into the context so the model knows where it is.
    pub fn render(&self) -> String {
        if self.frames.is_empty() {
            return "EXECUTION STACK: empty (no active goal)".into();
        }
        let mut out = format!("EXECUTION STACK (depth {}, outermost first)\n", self.frames.len());
        let last = self.frames.len() - 1;
        for (i, f) in self.frames.iter().enumerate() {
            let pad = "  ".repeat(i);
            let here = if i == last { "   <-- CURRENT GOAL" } else { "" };
            out.push_str(&format!("{pad}#{} {}{here}\n", f.id, f.goal));
            if !f.steps.is_empty() {
                let shown = &f.steps[f.steps.len().saturating_sub(5)..];
                let more = f.steps.len() - shown.len();
                let prefix = if more > 0 {
                    format!("(+{more} earlier) ")
                } else {
                    String::new()
                };
                out.push_str(&format!("{pad}   done: {prefix}{}\n", shown.join("; ")));
            }
            for n in &f.notes {
                out.push_str(&format!("{pad}   note: {n}\n"));
            }
        }
        out.push_str("When the current goal is finished, return to its parent goal above.");
        out
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

impl Default for ExecStack {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn push_step_pop() {
        let mut s = ExecStack::new(3);
        let root = s.push("Ship v2").unwrap();
        s.step("bumped version").unwrap();
        s.push("Fix failing tests").unwrap();
        s.push("Investigate test_auth").unwrap();
        assert_eq!(s.push("too deep").unwrap_err().kind, ErrorKind::Capacity);
        assert_eq!(s.breadcrumb(), "Ship v2 > Fix failing tests > Investigate test_auth");
        assert!(s.render().contains("<-- CURRENT GOAL"));
        s.pop(json!("token expiry off by one")).unwrap();
        assert!(s.current().unwrap().steps[0].contains("off by one"));

        let restored = ExecStack::from_json(&s.to_json().unwrap()).unwrap();
        assert_eq!(restored.depth(), 2);

        let failed = s.unwind_to(root, "release cancelled").unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(s.current().unwrap().id, root);
        assert!(s.current().unwrap().notes[0].contains("FAILED"));
    }
}
