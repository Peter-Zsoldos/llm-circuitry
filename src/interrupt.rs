//! **Interrupt controller, watchdog timer and real-time clock.**
//!
//! A CPU can be interrupted between instructions, has a timer chip that gives
//! it a sense of time, and a watchdog that resets it when the software hangs.
//! An agent loop has none of these: a user can't say "stop, the requirements
//! changed" mid-task, the model has no idea how long it has been working, and
//! a stuck agent repeats the same tool call until the budget runs out.
//!
//! * [`InterruptController`]: prioritised, maskable events to check between agent steps.
//! * [`Watchdog`]: step, token and time budgets, stall detection and loop/cycle detection.
//! * [`Clock`]: wall-clock time, elapsed time and a deadline as a line of text for the context.

use crate::util;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::time::Instant;

/// Priority 255 is non-maskable.
pub const NMI: u8 = 255;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interrupt {
    pub id: u64,
    pub priority: u8,
    pub source: String,
    pub message: String,
    pub payload: Value,
    pub ts_ms: u64,
}

impl Interrupt {
    /// One line to insert into the model's context.
    pub fn render(&self) -> String {
        format!(
            "[INTERRUPT priority={} source={} at={}] {}",
            self.priority,
            self.source,
            util::iso8601(self.ts_ms),
            self.message
        )
    }
}

#[derive(Debug, Default)]
pub struct InterruptController {
    queue: Vec<Interrupt>,
    mask: u8,
    next_id: u64,
}

impl InterruptController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn raise(&mut self, priority: u8, source: &str, message: &str, payload: Value) -> u64 {
        self.next_id += 1;
        self.queue.push(Interrupt {
            id: self.next_id,
            priority,
            source: source.into(),
            message: message.into(),
            payload,
            ts_ms: util::now_ms(),
        });
        self.next_id
    }

    /// Hold back interrupts with priority below `level` (e.g. during a critical
    /// section). Priority 255 is always delivered.
    pub fn set_mask(&mut self, level: u8) {
        self.mask = level;
    }

    pub fn mask(&self) -> u8 {
        self.mask
    }

    fn deliverable(&self, i: &Interrupt) -> bool {
        i.priority >= self.mask || i.priority == NMI
    }

    fn sorted(mut v: Vec<Interrupt>) -> Vec<Interrupt> {
        v.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.id.cmp(&b.id)));
        v
    }

    /// Deliverable interrupts, most urgent first, without removing them.
    pub fn pending(&self) -> Vec<Interrupt> {
        Self::sorted(self.queue.iter().filter(|i| self.deliverable(i)).cloned().collect())
    }

    /// Interrupts currently held back by the mask.
    pub fn masked(&self) -> Vec<Interrupt> {
        Self::sorted(self.queue.iter().filter(|i| !self.deliverable(i)).cloned().collect())
    }

    /// Cheap check to run between agent steps.
    pub fn has_pending(&self, min_priority: u8) -> bool {
        self.queue
            .iter()
            .any(|i| self.deliverable(i) && i.priority >= min_priority)
    }

    /// Take the most urgent deliverable interrupt.
    pub fn poll(&mut self) -> Option<Interrupt> {
        let first = self.pending().into_iter().next()?;
        self.queue.retain(|i| i.id != first.id);
        Some(first)
    }

    /// Take all deliverable interrupts, most urgent first.
    pub fn drain(&mut self) -> Vec<Interrupt> {
        let out = self.pending();
        self.queue.retain(|i| !(i.priority >= self.mask || i.priority == NMI));
        out
    }

    pub fn clear(&mut self) -> usize {
        let n = self.queue.len();
        self.queue.clear();
        n
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WatchdogConfig {
    pub max_steps: Option<u64>,
    pub max_seconds: Option<f64>,
    pub max_tokens: Option<u64>,
    /// Trip if this many steps pass without a call to [`Watchdog::progress`].
    pub stall_steps: Option<u64>,
    /// Trip when the same action (or cycle of actions) repeats this many times in a row.
    pub loop_repeats: usize,
    /// Longest cycle length to look for (A,B,A,B is period 2).
    pub max_period: usize,
    /// Fraction of a budget at which to start warning.
    pub warn_ratio: f64,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        WatchdogConfig {
            max_steps: None,
            max_seconds: None,
            max_tokens: None,
            stall_steps: None,
            loop_repeats: 3,
            max_period: 3,
            warn_ratio: 0.8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Warn,
    Trip,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub status: Status,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WatchdogStats {
    pub steps: u64,
    pub tokens: u64,
    pub elapsed_seconds: f64,
    pub steps_since_progress: u64,
    pub tripped: Option<String>,
}

#[derive(Debug)]
pub struct Watchdog {
    cfg: WatchdogConfig,
    start: Instant,
    steps: u64,
    tokens: u64,
    last_progress: u64,
    history: VecDeque<u64>,
    tripped: Option<String>,
}

impl Watchdog {
    pub fn new(cfg: WatchdogConfig) -> Self {
        Watchdog {
            cfg,
            start: Instant::now(),
            steps: 0,
            tokens: 0,
            last_progress: 0,
            history: VecDeque::new(),
            tripped: None,
        }
    }

    pub fn config(&self) -> &WatchdogConfig {
        &self.cfg
    }

    /// Record one agent step. `signature` identifies the action, e.g.
    /// `"search(query='x')"`. Identical signatures in a row count as a loop.
    pub fn tick(&mut self, signature: &str, tokens: u64) -> Verdict {
        self.steps += 1;
        self.tokens += tokens;
        let mut h = DefaultHasher::new();
        signature.hash(&mut h);
        self.history.push_back(h.finish());
        let keep = (self.cfg.max_period * self.cfg.loop_repeats).max(1);
        while self.history.len() > keep {
            self.history.pop_front();
        }
        if let Some(reason) = &self.tripped {
            return Verdict {
                status: Status::Trip,
                reason: Some(reason.clone()),
            };
        }
        if let Some(reason) = self.check_trip() {
            self.tripped = Some(reason.clone());
            return Verdict {
                status: Status::Trip,
                reason: Some(reason),
            };
        }
        match self.check_warn() {
            Some(reason) => Verdict {
                status: Status::Warn,
                reason: Some(reason),
            },
            None => Verdict {
                status: Status::Ok,
                reason: None,
            },
        }
    }

    fn check_trip(&self) -> Option<String> {
        let elapsed = self.start.elapsed().as_secs_f64();
        if let Some(m) = self.cfg.max_steps.filter(|m| self.steps > *m) {
            return Some(format!("step budget exhausted ({} > {m})", self.steps));
        }
        if let Some(m) = self.cfg.max_tokens.filter(|m| self.tokens > *m) {
            return Some(format!("token budget exhausted ({} > {m})", self.tokens));
        }
        if let Some(m) = self.cfg.max_seconds.filter(|m| elapsed > *m) {
            return Some(format!(
                "time budget exhausted ({} > {})",
                util::fmt_duration(elapsed),
                util::fmt_duration(m)
            ));
        }
        if let Some(m) = self.cfg.stall_steps.filter(|m| self.steps - self.last_progress >= *m) {
            return Some(format!("no progress reported in {m} steps"));
        }
        self.detect_loop()
    }

    fn detect_loop(&self) -> Option<String> {
        let h: Vec<u64> = self.history.iter().copied().collect();
        let repeats = self.cfg.loop_repeats;
        if repeats < 2 {
            return None;
        }
        for p in 1..=self.cfg.max_period {
            let need = p * repeats;
            if h.len() < need {
                continue;
            }
            let tail = &h[h.len() - need..];
            let unit = &tail[..p];
            if tail.chunks(p).all(|c| c == unit) {
                return Some(if p == 1 {
                    format!("loop: the same action was repeated {repeats} times in a row")
                } else {
                    format!("loop: the same {p}-step cycle of actions was repeated {repeats} times")
                });
            }
        }
        None
    }

    fn check_warn(&self) -> Option<String> {
        let r = self.cfg.warn_ratio;
        let elapsed = self.start.elapsed().as_secs_f64();
        if let Some(m) = self.cfg.max_steps.filter(|m| self.steps as f64 >= *m as f64 * r) {
            return Some(format!("{} of {m} steps used", self.steps));
        }
        if let Some(m) = self.cfg.max_tokens.filter(|m| self.tokens as f64 >= *m as f64 * r) {
            return Some(format!("{} of {m} tokens used", self.tokens));
        }
        if let Some(m) = self.cfg.max_seconds.filter(|m| elapsed >= m * r) {
            return Some(format!(
                "{} of {} used",
                util::fmt_duration(elapsed),
                util::fmt_duration(m)
            ));
        }
        None
    }

    /// Report meaningful progress (resets stall detection and loop history).
    pub fn progress(&mut self) {
        self.last_progress = self.steps;
        self.history.clear();
    }

    pub fn reset(&mut self) {
        *self = Watchdog::new(self.cfg.clone());
    }

    pub fn tripped(&self) -> Option<&str> {
        self.tripped.as_deref()
    }

    pub fn stats(&self) -> WatchdogStats {
        WatchdogStats {
            steps: self.steps,
            tokens: self.tokens,
            elapsed_seconds: self.start.elapsed().as_secs_f64(),
            steps_since_progress: self.steps - self.last_progress,
            tripped: self.tripped.clone(),
        }
    }
}

/// Gives the model a sense of time.
#[derive(Debug)]
pub struct Clock {
    start_ms: u64,
    start: Instant,
    deadline_s: Option<f64>,
}

impl Default for Clock {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Clock {
    pub fn new(deadline_seconds: Option<f64>) -> Self {
        Clock {
            start_ms: util::now_ms(),
            start: Instant::now(),
            deadline_s: deadline_seconds,
        }
    }

    pub fn now_iso(&self) -> String {
        util::iso8601(util::now_ms())
    }

    pub fn started_iso(&self) -> String {
        util::iso8601(self.start_ms)
    }

    pub fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Set a deadline `seconds` after the clock started.
    pub fn set_deadline(&mut self, seconds: Option<f64>) {
        self.deadline_s = seconds;
    }

    pub fn remaining(&self) -> Option<f64> {
        self.deadline_s.map(|d| d - self.elapsed())
    }

    /// e.g. `[clock] now=2026-10-14T09:20:11Z elapsed=12m03s remaining=47m57s`
    pub fn note(&self) -> String {
        let mut s = format!(
            "[clock] now={} elapsed={}",
            self.now_iso(),
            util::fmt_duration(self.elapsed())
        );
        if let Some(r) = self.remaining() {
            if r >= 0.0 {
                s.push_str(&format!(" remaining={}", util::fmt_duration(r)));
            } else {
                s.push_str(&format!(" OVERDUE by {}", util::fmt_duration(-r)));
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupts_respect_priority_and_mask() {
        let mut ic = InterruptController::new();
        ic.raise(1, "ci", "build finished", Value::Null);
        ic.raise(9, "user", "stop, requirements changed", Value::Null);
        ic.set_mask(5);
        assert!(ic.has_pending(0));
        assert_eq!(ic.poll().unwrap().source, "user");
        assert!(ic.poll().is_none());
        assert_eq!(ic.masked().len(), 1);
        ic.raise(NMI, "ops", "shutdown", Value::Null);
        assert_eq!(ic.drain().len(), 1);
        ic.set_mask(0);
        assert_eq!(ic.poll().unwrap().source, "ci");
    }

    #[test]
    fn watchdog_detects_loops_and_cycles() {
        let mut w = Watchdog::new(WatchdogConfig::default());
        assert_eq!(w.tick("search(a)", 0).status, Status::Ok);
        assert_eq!(w.tick("search(a)", 0).status, Status::Ok);
        assert_eq!(w.tick("search(a)", 0).status, Status::Trip);

        let mut w = Watchdog::new(WatchdogConfig::default());
        for s in ["open(x)", "close(x)", "open(x)", "close(x)", "open(x)"] {
            assert_ne!(w.tick(s, 0).status, Status::Trip);
        }
        let v = w.tick("close(x)", 0);
        assert_eq!(v.status, Status::Trip);
        assert!(v.reason.unwrap().contains("2-step"));
    }

    #[test]
    fn watchdog_budgets_and_stalls() {
        let cfg = WatchdogConfig {
            max_steps: Some(5),
            stall_steps: Some(3),
            ..Default::default()
        };
        let mut w = Watchdog::new(cfg);
        w.tick("a", 10);
        w.tick("b", 10);
        w.progress();
        w.tick("c", 10);
        assert_eq!(w.tick("d", 10).status, Status::Warn);
        assert_eq!(w.tick("e", 10).status, Status::Trip);
        assert!(w.tripped().unwrap().contains("no progress"));
        w.reset();
        assert!(w.tripped().is_none());
    }

    #[test]
    fn clock_note() {
        let c = Clock::new(Some(60.0));
        assert!(c.note().contains("remaining="));
        assert!(Clock::new(Some(-5.0)).note().contains("OVERDUE"));
    }
}
