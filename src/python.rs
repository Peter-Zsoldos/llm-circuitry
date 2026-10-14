//! PyO3 bindings. Structured results are returned as plain dicts and lists.

use crate::error::{Error, ErrorKind};
use crate::{address, attest, coherence, dma, ecc, interrupt, pager, protect, replay, stack, txn};
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pythonize::{depythonize, pythonize};
use serde::Serialize;
use serde_json::Value;

pyo3::create_exception!(
    _core,
    CircuitError,
    PyException,
    "Base class for all llm_circuitry errors."
);
pyo3::create_exception!(
    _core,
    NotFoundError,
    CircuitError,
    "A referenced id, key, handle or anchor does not exist."
);
pyo3::create_exception!(
    _core,
    StaleError,
    CircuitError,
    "The caller acted on data that has since changed."
);
pyo3::create_exception!(
    _core,
    ConflictError,
    CircuitError,
    "Another agent holds a conflicting lock."
);
pyo3::create_exception!(
    _core,
    CapacityError,
    CircuitError,
    "A capacity limit (stack depth, token budget) was exceeded."
);
pyo3::create_exception!(
    _core,
    InvalidStateError,
    CircuitError,
    "The operation is not valid in the current state."
);
pyo3::create_exception!(
    _core,
    IntegrityError,
    CircuitError,
    "A signature or hash chain failed verification."
);
pyo3::create_exception!(
    _core,
    DeniedError,
    CircuitError,
    "A permission or capability check failed."
);
pyo3::create_exception!(_core, ValidationError, CircuitError, "An argument was malformed.");

impl From<Error> for PyErr {
    fn from(e: Error) -> PyErr {
        let m = e.message;
        match e.kind {
            ErrorKind::NotFound => NotFoundError::new_err(m),
            ErrorKind::Stale => StaleError::new_err(m),
            ErrorKind::Conflict => ConflictError::new_err(m),
            ErrorKind::Capacity => CapacityError::new_err(m),
            ErrorKind::InvalidState => InvalidStateError::new_err(m),
            ErrorKind::Integrity => IntegrityError::new_err(m),
            ErrorKind::Denied => DeniedError::new_err(m),
            ErrorKind::Invalid => ValidationError::new_err(m),
            ErrorKind::Io => CircuitError::new_err(m),
        }
    }
}

fn py<'py, T: Serialize + ?Sized>(py: Python<'py>, v: &T) -> PyResult<Bound<'py, PyAny>> {
    Ok(pythonize(py, v)?)
}

fn val(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    match obj {
        None => Ok(Value::Null),
        Some(o) => Ok(depythonize(o)?),
    }
}

fn ring_arg(obj: &Bound<'_, PyAny>) -> PyResult<protect::Ring> {
    if let Ok(n) = obj.extract::<u8>() {
        return Ok(protect::Ring::from_level(n)?);
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(protect::Ring::parse(&s)?);
    }
    Err(PyTypeError::new_err(
        "ring must be an int 0-3 or one of 'kernel', 'user', 'tool', 'untrusted'",
    ))
}

// ---------------------------------------------------------------- protect

/// Context window with privilege rings, taint tracking and an action policy.
#[pyclass(name = "ProtectedContext", module = "llm_circuitry._core")]
struct PyProtectedContext {
    inner: protect::ProtectedContext,
}

#[pymethods]
impl PyProtectedContext {
    #[new]
    fn new() -> Self {
        PyProtectedContext {
            inner: protect::ProtectedContext::new(),
        }
    }

    /// Random nonce used in the data boundary markers.
    #[getter]
    fn boundary(&self) -> String {
        self.inner.boundary().to_string()
    }

    /// Add content at a ring (0 kernel, 1 user, 2 tool, 3 untrusted). Returns the segment id.
    fn push(&mut self, ring: &Bound<'_, PyAny>, source: &str, content: &str) -> PyResult<u64> {
        Ok(self.inner.push(ring_arg(ring)?, source, content))
    }

    /// Add content derived from other segments; it inherits their least-trusted ring.
    fn derive(&mut self, source: &str, content: &str, derived_from: Vec<u64>) -> PyResult<u64> {
        Ok(self.inner.derive(source, content, &derived_from)?)
    }

    fn get<'py>(&self, py: Python<'py>, id: u64) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, self.inner.get(id)?)
    }

    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, self.inner.segments())
    }

    fn remove<'py>(&mut self, py: Python<'py>, id: u64) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.remove(id)?)
    }

    /// Least-trusted ring allowed to influence `action`; on_violation is 'require_approval' or 'deny'.
    #[pyo3(signature = (action, max_ring, on_violation = "require_approval"))]
    fn set_rule(&mut self, action: &str, max_ring: &Bound<'_, PyAny>, on_violation: &str) -> PyResult<()> {
        self.inner
            .set_rule(action, ring_arg(max_ring)?, protect::OnViolation::parse(on_violation)?);
        Ok(())
    }

    #[pyo3(signature = (max_ring, on_violation = "require_approval"))]
    fn set_default_rule(&mut self, max_ring: &Bound<'_, PyAny>, on_violation: &str) -> PyResult<()> {
        self.inner
            .set_default_rule(ring_arg(max_ring)?, protect::OnViolation::parse(on_violation)?);
        Ok(())
    }

    fn set_alert_severity(&mut self, severity: u8) {
        self.inner.set_alert_severity(severity);
    }

    /// Decide whether `action` may run given the segments that influenced it.
    /// Returns {"verdict": "allow"|"require_approval"|"deny", "reason": str, "tainted_by": [ids]}.
    fn check<'py>(&self, py: Python<'py>, action: &str, derived_from: Vec<u64>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.check(action, &derived_from))
    }

    fn preamble(&self) -> String {
        self.inner.preamble()
    }

    fn render_segment(&self, id: u64) -> PyResult<String> {
        Ok(self.inner.render_segment(id)?)
    }

    fn render(&self) -> String {
        self.inner.render()
    }

    fn __len__(&self) -> usize {
        self.inner.segments().len()
    }
}

/// Heuristic scan for prompt-injection markers. Returns a list of findings.
#[pyfunction]
fn scan_injection<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    self::py(py, &protect::scan_injection(text))
}

// ---------------------------------------------------------------- txn

/// Write-ahead, hash-chained journal of agent actions with commit and rollback.
#[pyclass(name = "TransactionManager", module = "llm_circuitry._core")]
struct PyTxnManager {
    inner: txn::TxnManager,
}

#[pymethods]
impl PyTxnManager {
    /// In-memory, or persisted to a JSONL journal at `path` (replayed and verified on open).
    #[new]
    #[pyo3(signature = (path = None))]
    fn new(path: Option<String>) -> PyResult<Self> {
        Ok(PyTxnManager {
            inner: match path {
                Some(p) => txn::TxnManager::open(p)?,
                None => txn::TxnManager::new(),
            },
        })
    }

    fn begin(&mut self, label: &str) -> PyResult<u64> {
        Ok(self.inner.begin(label)?)
    }

    /// Record intent to perform a step (call before executing it). Returns the step id.
    #[pyo3(signature = (txn, name, params = None, compensation = None))]
    fn step(
        &mut self,
        txn: u64,
        name: &str,
        params: Option<&Bound<'_, PyAny>>,
        compensation: Option<String>,
    ) -> PyResult<u64> {
        Ok(self.inner.step(txn, name, val(params)?, compensation)?)
    }

    #[pyo3(signature = (txn, step, result = None))]
    fn done(&mut self, txn: u64, step: u64, result: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        Ok(self.inner.done(txn, step, val(result)?)?)
    }

    fn fail(&mut self, txn: u64, step: u64, error: &str) -> PyResult<()> {
        Ok(self.inner.fail(txn, step, error)?)
    }

    fn commit(&mut self, txn: u64) -> PyResult<()> {
        Ok(self.inner.commit(txn)?)
    }

    /// Start a rollback; returns the steps to undo, newest first.
    fn begin_rollback<'py>(&mut self, py: Python<'py>, txn: u64, reason: &str) -> PyResult<Bound<'py, PyAny>> {
        let plan = self.inner.begin_rollback(txn, reason)?;
        self::py(py, &plan)
    }

    fn compensated(&mut self, txn: u64, step: u64) -> PyResult<()> {
        Ok(self.inner.compensated(txn, step)?)
    }

    fn compensation_failed(&mut self, txn: u64, step: u64, error: &str) -> PyResult<()> {
        Ok(self.inner.compensation_failed(txn, step, error)?)
    }

    /// Returns "rolled_back" or "rollback_incomplete".
    fn finish_rollback<'py>(&mut self, py: Python<'py>, txn: u64) -> PyResult<Bound<'py, PyAny>> {
        let status = self.inner.finish_rollback(txn)?;
        self::py(py, &status)
    }

    fn get<'py>(&self, py: Python<'py>, txn: u64) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, self.inner.get(txn)?)
    }

    fn list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.list())
    }

    /// Unfinished transactions with steps to compensate and steps in doubt.
    fn recover<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.recover())
    }

    fn journal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, self.inner.journal())
    }

    fn verify(&self) -> PyResult<()> {
        Ok(self.inner.verify()?)
    }
}

// ---------------------------------------------------------------- coherence

/// Versioned shared state with stale-read detection and invalidations between agents.
#[pyclass(name = "CoherentStore", module = "llm_circuitry._core")]
struct PyCoherentStore {
    inner: coherence::CoherentStore,
}

#[pymethods]
impl PyCoherentStore {
    #[new]
    fn new() -> Self {
        PyCoherentStore {
            inner: coherence::CoherentStore::new(),
        }
    }

    /// Read a key as `agent` (records the version the agent now holds). None if missing.
    fn read<'py>(&mut self, py: Python<'py>, agent: &str, key: &str) -> PyResult<Bound<'py, PyAny>> {
        let r = self.inner.read(agent, key);
        self::py(py, &r)
    }

    /// Peek at a key without registering a cached copy.
    fn get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.get(key))
    }

    /// Write a key; raises StaleError if the agent's view is outdated. Returns the new version.
    #[pyo3(signature = (agent, key, value, expected_version = None))]
    fn write(
        &mut self,
        agent: &str,
        key: &str,
        value: &Bound<'_, PyAny>,
        expected_version: Option<u64>,
    ) -> PyResult<u64> {
        Ok(self.inner.write(agent, key, val(Some(value))?, expected_version)?)
    }

    /// Stale keys for `agent` (all keys it has read if `keys` is None).
    #[pyo3(signature = (agent, keys = None))]
    fn validate<'py>(&self, py: Python<'py>, agent: &str, keys: Option<Vec<String>>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.validate(agent, &keys.unwrap_or_default()))
    }

    /// Raise StaleError if the agent holds outdated data.
    #[pyo3(signature = (agent, keys = None))]
    fn require_fresh(&self, agent: &str, keys: Option<Vec<String>>) -> PyResult<()> {
        Ok(self.inner.require_fresh(agent, &keys.unwrap_or_default())?)
    }

    /// "modified", "exclusive", "shared" or "invalid".
    fn state<'py>(&self, py: Python<'py>, agent: &str, key: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.state(agent, key))
    }

    /// Take the pending invalidation messages for `agent`.
    fn invalidations<'py>(&mut self, py: Python<'py>, agent: &str) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.invalidations(agent);
        self::py(py, &v)
    }

    #[pyo3(signature = (agent, key, ttl_seconds = 30.0))]
    fn lock(&mut self, agent: &str, key: &str, ttl_seconds: f64) -> PyResult<()> {
        Ok(self.inner.lock(agent, key, (ttl_seconds.max(0.0) * 1000.0) as u64)?)
    }

    fn unlock(&mut self, agent: &str, key: &str) -> PyResult<()> {
        Ok(self.inner.unlock(agent, key)?)
    }

    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }
}

// ---------------------------------------------------------------- interrupt

/// Prioritised, maskable interrupts to check between agent steps. Priority 255 is non-maskable.
#[pyclass(name = "InterruptController", module = "llm_circuitry._core")]
struct PyInterruptController {
    inner: interrupt::InterruptController,
}

#[pymethods]
impl PyInterruptController {
    #[new]
    fn new() -> Self {
        PyInterruptController {
            inner: interrupt::InterruptController::new(),
        }
    }

    /// Raise an interrupt (priority 0-255). Returns its id.
    #[pyo3(signature = (priority, source, message, payload = None))]
    fn raise_interrupt(
        &mut self,
        priority: u8,
        source: &str,
        message: &str,
        payload: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        Ok(self.inner.raise(priority, source, message, val(payload)?))
    }

    /// Hold back interrupts below this priority.
    fn set_mask(&mut self, level: u8) {
        self.inner.set_mask(level);
    }

    #[getter]
    fn mask(&self) -> u8 {
        self.inner.mask()
    }

    #[pyo3(signature = (min_priority = 0))]
    fn has_pending(&self, min_priority: u8) -> bool {
        self.inner.has_pending(min_priority)
    }

    fn pending<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.pending())
    }

    fn masked<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.masked())
    }

    /// Take the most urgent deliverable interrupt, or None.
    fn poll<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let i = self.inner.poll();
        self::py(py, &i)
    }

    fn drain<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.drain();
        self::py(py, &v)
    }

    /// Take all deliverable interrupts rendered as context lines.
    fn drain_text(&mut self) -> Vec<String> {
        self.inner.drain().iter().map(|i| i.render()).collect()
    }

    fn clear(&mut self) -> usize {
        self.inner.clear()
    }
}

/// Step/token/time budgets, stall detection and loop detection for agent loops.
#[pyclass(name = "Watchdog", module = "llm_circuitry._core")]
struct PyWatchdog {
    inner: interrupt::Watchdog,
}

#[pymethods]
impl PyWatchdog {
    #[new]
    #[pyo3(signature = (max_steps = None, max_seconds = None, max_tokens = None, stall_steps = None, loop_repeats = 3, max_period = 3, warn_ratio = 0.8))]
    fn new(
        max_steps: Option<u64>,
        max_seconds: Option<f64>,
        max_tokens: Option<u64>,
        stall_steps: Option<u64>,
        loop_repeats: usize,
        max_period: usize,
        warn_ratio: f64,
    ) -> Self {
        PyWatchdog {
            inner: interrupt::Watchdog::new(interrupt::WatchdogConfig {
                max_steps,
                max_seconds,
                max_tokens,
                stall_steps,
                loop_repeats,
                max_period,
                warn_ratio,
            }),
        }
    }

    /// Record a step. Returns {"status": "ok"|"warn"|"trip", "reason": str|None}.
    #[pyo3(signature = (signature, tokens = 0))]
    fn tick<'py>(&mut self, py: Python<'py>, signature: &str, tokens: u64) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.tick(signature, tokens);
        self::py(py, &v)
    }

    /// Report meaningful progress (resets stall and loop detection).
    fn progress(&mut self) {
        self.inner.progress();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn tripped(&self) -> Option<String> {
        self.inner.tripped().map(String::from)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.stats())
    }
}

/// Gives the model a sense of time: wall clock, elapsed time and deadline.
#[pyclass(name = "Clock", module = "llm_circuitry._core")]
struct PyClock {
    inner: interrupt::Clock,
}

#[pymethods]
impl PyClock {
    #[new]
    #[pyo3(signature = (deadline_seconds = None))]
    fn new(deadline_seconds: Option<f64>) -> Self {
        PyClock {
            inner: interrupt::Clock::new(deadline_seconds),
        }
    }

    fn now_iso(&self) -> String {
        self.inner.now_iso()
    }

    fn started_iso(&self) -> String {
        self.inner.started_iso()
    }

    fn elapsed(&self) -> f64 {
        self.inner.elapsed()
    }

    fn remaining(&self) -> Option<f64> {
        self.inner.remaining()
    }

    #[pyo3(signature = (seconds = None))]
    fn set_deadline(&mut self, seconds: Option<f64>) {
        self.inner.set_deadline(seconds);
    }

    /// One line for the context, e.g. "[clock] now=... elapsed=12m03s remaining=47m57s".
    fn note(&self) -> String {
        self.inner.note()
    }
}

// ---------------------------------------------------------------- stack

/// Explicit goal stack (program counter + call stack) that lives outside the context.
#[pyclass(name = "ExecStack", module = "llm_circuitry._core")]
struct PyExecStack {
    inner: stack::ExecStack,
}

#[pymethods]
impl PyExecStack {
    #[new]
    #[pyo3(signature = (max_depth = 16))]
    fn new(max_depth: usize) -> Self {
        PyExecStack {
            inner: stack::ExecStack::new(max_depth),
        }
    }

    /// Enter a sub-goal. Raises CapacityError past max_depth. Returns the frame id.
    fn push(&mut self, goal: &str) -> PyResult<u64> {
        Ok(self.inner.push(goal)?)
    }

    /// Record a completed step; returns the program counter.
    fn step(&mut self, description: &str) -> PyResult<usize> {
        Ok(self.inner.step(description)?)
    }

    fn note(&mut self, text: &str) -> PyResult<()> {
        Ok(self.inner.note(text)?)
    }

    /// Finish the current goal, returning `result` to the parent.
    #[pyo3(signature = (result = None))]
    fn pop<'py>(&mut self, py: Python<'py>, result: Option<&Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        let f = self.inner.pop(val(result)?)?;
        self::py(py, &f)
    }

    fn fail<'py>(&mut self, py: Python<'py>, error: &str) -> PyResult<Bound<'py, PyAny>> {
        let f = self.inner.fail(error)?;
        self::py(py, &f)
    }

    fn unwind_to<'py>(&mut self, py: Python<'py>, frame_id: u64, reason: &str) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.unwind_to(frame_id, reason)?;
        self::py(py, &v)
    }

    fn current<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.current())
    }

    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, self.inner.frames())
    }

    fn finished<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, self.inner.finished())
    }

    fn breadcrumb(&self) -> String {
        self.inner.breadcrumb()
    }

    /// Text block to re-inject into the context.
    fn render(&self) -> String {
        self.inner.render()
    }

    fn to_json(&self) -> PyResult<String> {
        Ok(self.inner.to_json()?)
    }

    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(PyExecStack {
            inner: stack::ExecStack::from_json(s)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.depth()
    }
}

// ---------------------------------------------------------------- ecc

/// Verifies cited quotes and numbers against source documents.
#[pyclass(name = "Verifier", module = "llm_circuitry._core")]
struct PyVerifier {
    inner: ecc::Verifier,
}

#[pymethods]
impl PyVerifier {
    #[new]
    #[pyo3(signature = (fuzzy_threshold = 0.8))]
    fn new(fuzzy_threshold: f64) -> Self {
        PyVerifier {
            inner: ecc::Verifier::new(fuzzy_threshold),
        }
    }

    fn add_source(&mut self, source_id: &str, text: &str) {
        self.inner.add_source(source_id, text);
    }

    fn remove_source(&mut self, source_id: &str) -> bool {
        self.inner.remove_source(source_id)
    }

    fn sources(&self) -> Vec<String> {
        self.inner.sources()
    }

    /// Check one quote (and optionally the claim's numbers) against a source.
    #[pyo3(signature = (source_id, quote, claim = None))]
    fn check<'py>(
        &self,
        py: Python<'py>,
        source_id: &str,
        quote: &str,
        claim: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.check(source_id, quote, claim))
    }

    /// Parse `[source_id: "quote"]` citations in text and check them all.
    fn check_cited<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.check_cited(text))
    }
}

// ---------------------------------------------------------------- dma

/// Content-addressed blob store: pass `blob:` handles instead of copying data through context.
#[pyclass(name = "BlobStore", module = "llm_circuitry._core")]
struct PyBlobStore {
    inner: dma::BlobStore,
}

#[pymethods]
impl PyBlobStore {
    #[new]
    #[pyo3(signature = (capacity_bytes = None))]
    fn new(capacity_bytes: Option<u64>) -> Self {
        PyBlobStore {
            inner: dma::BlobStore::new(capacity_bytes),
        }
    }

    /// Store str or bytes; returns a handle like "blob:3f9a...".
    #[pyo3(signature = (data, mime = "text/plain", label = ""))]
    fn put(&mut self, data: &Bound<'_, PyAny>, mime: &str, label: &str) -> PyResult<String> {
        if let Ok(s) = data.extract::<String>() {
            return Ok(self.inner.put(s.as_bytes(), mime, label));
        }
        if let Ok(b) = data.extract::<Vec<u8>>() {
            return Ok(self.inner.put(&b, mime, label));
        }
        Err(PyTypeError::new_err("data must be str or bytes"))
    }

    fn get<'py>(&mut self, py: Python<'py>, handle: &str) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.inner.get(handle)?))
    }

    fn text(&mut self, handle: &str) -> PyResult<String> {
        Ok(self.inner.text(handle)?)
    }

    #[pyo3(signature = (handle, start = 0, end = None))]
    fn slice<'py>(
        &mut self,
        py: Python<'py>,
        handle: &str,
        start: u64,
        end: Option<u64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.inner.slice(handle, start, end)?))
    }

    /// Lines start..=end (1-based, inclusive).
    #[pyo3(signature = (handle, start = 1, end = None))]
    fn lines(&mut self, handle: &str, start: u64, end: Option<u64>) -> PyResult<String> {
        Ok(self.inner.lines(handle, start, end)?)
    }

    #[pyo3(signature = (handle, needle, max_results = 20))]
    fn grep<'py>(
        &mut self,
        py: Python<'py>,
        handle: &str,
        needle: &str,
        max_results: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.grep(handle, needle, max_results)?;
        self::py(py, &v)
    }

    fn info<'py>(&self, py: Python<'py>, handle: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.info(handle)?)
    }

    /// One-line description with a preview, for the model's context.
    #[pyo3(signature = (handle, preview_chars = 120))]
    fn describe(&mut self, handle: &str, preview_chars: usize) -> PyResult<String> {
        Ok(self.inner.describe(handle, preview_chars)?)
    }

    fn retain(&mut self, handle: &str) -> PyResult<u32> {
        Ok(self.inner.retain(handle)?)
    }

    fn release(&mut self, handle: &str) -> PyResult<u32> {
        Ok(self.inner.release(handle)?)
    }

    fn delete(&mut self, handle: &str) -> PyResult<()> {
        Ok(self.inner.delete(handle)?)
    }

    fn handles(&self) -> Vec<String> {
        self.inner.handles()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.stats())
    }

    fn __contains__(&self, handle: &str) -> bool {
        self.inner.info(handle).is_ok()
    }
}

// ---------------------------------------------------------------- address

/// Exact 1-based line addressing with edit-stable anchors like "L42#9f3a1c2b".
#[pyclass(name = "AddressableDoc", module = "llm_circuitry._core")]
struct PyAddressableDoc {
    inner: address::AddressableDoc,
}

#[pymethods]
impl PyAddressableDoc {
    #[new]
    fn new(text: &str) -> Self {
        PyAddressableDoc {
            inner: address::AddressableDoc::new(text),
        }
    }

    fn update(&mut self, text: &str) {
        self.inner.update(text);
    }

    fn line(&self, n: usize) -> PyResult<String> {
        Ok(self.inner.line(n)?.to_string())
    }

    fn range(&self, start: usize, end: usize) -> PyResult<String> {
        Ok(self.inner.range(start, end)?)
    }

    #[pyo3(signature = (start = 1, end = None))]
    fn numbered(&self, start: usize, end: Option<usize>) -> PyResult<String> {
        Ok(self.inner.numbered(start, end)?)
    }

    fn find<'py>(&self, py: Python<'py>, needle: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.find(needle))
    }

    fn anchor(&self, n: usize) -> PyResult<String> {
        Ok(self.inner.anchor(n)?)
    }

    /// Where an anchored line is now. Raises NotFoundError if it was changed or deleted.
    fn resolve<'py>(&self, py: Python<'py>, anchor: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.resolve(anchor)?)
    }

    #[getter]
    fn line_count(&self) -> usize {
        self.inner.line_count()
    }

    fn __len__(&self) -> usize {
        self.inner.line_count()
    }
}

// ---------------------------------------------------------------- pager

/// CLOCK paging of context pages under a token budget, with swap and dirty tracking.
#[pyclass(name = "Pager", module = "llm_circuitry._core")]
struct PyPager {
    inner: pager::Pager,
}

#[pymethods]
impl PyPager {
    #[new]
    fn new(budget_tokens: u64) -> Self {
        PyPager {
            inner: pager::Pager::new(budget_tokens),
        }
    }

    /// Load or replace a page; returns the pages evicted to make room.
    #[pyo3(signature = (id, content, tokens = None, pinned = false))]
    fn load<'py>(
        &mut self,
        py: Python<'py>,
        id: &str,
        content: &str,
        tokens: Option<u64>,
        pinned: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.load(id, content, tokens, pinned)?;
        self::py(py, &v)
    }

    /// Use a page: {"content", "fault", "evicted"}. Swapped pages are faulted back in.
    fn access<'py>(&mut self, py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyAny>> {
        let a = self.inner.access(id)?;
        self::py(py, &a)
    }

    #[pyo3(signature = (id, content, tokens = None))]
    fn write<'py>(
        &mut self,
        py: Python<'py>,
        id: &str,
        content: &str,
        tokens: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.write(id, content, tokens)?;
        self::py(py, &v)
    }

    fn get<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, self.inner.get(id)?)
    }

    fn pin(&mut self, id: &str) -> PyResult<()> {
        Ok(self.inner.pin(id)?)
    }

    fn unpin(&mut self, id: &str) -> PyResult<()> {
        Ok(self.inner.unpin(id)?)
    }

    fn evict<'py>(&mut self, py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyAny>> {
        let e = self.inner.evict(id)?;
        self::py(py, &e)
    }

    fn drop_page<'py>(&mut self, py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyAny>> {
        let p = self.inner.drop_page(id)?;
        self::py(py, &p)
    }

    fn set_budget<'py>(&mut self, py: Python<'py>, budget_tokens: u64) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.set_budget(budget_tokens);
        self::py(py, &v)
    }

    fn resident(&self) -> Vec<String> {
        self.inner.resident()
    }

    fn swapped(&self) -> Vec<String> {
        self.inner.swapped()
    }

    fn dirty(&self) -> Vec<String> {
        self.inner.dirty()
    }

    fn clean(&mut self, id: &str) -> PyResult<()> {
        Ok(self.inner.clean(id)?)
    }

    #[getter]
    fn used(&self) -> u64 {
        self.inner.used()
    }

    #[getter]
    fn budget(&self) -> u64 {
        self.inner.budget()
    }

    fn render(&self) -> String {
        self.inner.render()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.stats())
    }
}

/// Rough token estimate (~4 chars per token).
#[pyfunction]
fn estimate_tokens(text: &str) -> u64 {
    pager::estimate_tokens(text)
}

// ---------------------------------------------------------------- attest

/// HMAC-SHA256 signed instructions and capability tokens.
#[pyclass(name = "Signer", module = "llm_circuitry._core")]
struct PySigner {
    inner: attest::Signer,
}

#[pymethods]
impl PySigner {
    /// With no key, a random 32-byte key is generated.
    #[new]
    #[pyo3(signature = (issuer, key = None))]
    fn new(issuer: &str, key: Option<Vec<u8>>) -> PyResult<Self> {
        Ok(PySigner {
            inner: match key {
                Some(k) => attest::Signer::new(issuer, &k)?,
                None => attest::Signer::generate(issuer),
            },
        })
    }

    #[getter]
    fn issuer(&self) -> String {
        self.inner.issuer().to_string()
    }

    /// Secret key as hex. Store it in a secrets manager; never put it in a prompt.
    fn key_hex(&self) -> String {
        self.inner.key_hex()
    }

    fn sign<'py>(&self, py: Python<'py>, subject: &str, content: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.sign(subject, content))
    }

    fn verify(&self, envelope: &Bound<'_, PyAny>) -> PyResult<bool> {
        let env: attest::Envelope = depythonize(envelope)?;
        Ok(self.inner.verify(&env))
    }

    /// Issue a capability token for `agent` (wildcards allowed in actions and resources).
    #[pyo3(signature = (agent, actions, resources = None, ttl_seconds = None, on_behalf_of = None))]
    fn grant(
        &self,
        agent: &str,
        actions: Vec<String>,
        resources: Option<Vec<String>>,
        ttl_seconds: Option<f64>,
        on_behalf_of: Option<&str>,
    ) -> PyResult<String> {
        Ok(self.inner.grant(
            agent,
            &actions,
            &resources.unwrap_or_default(),
            ttl_seconds,
            on_behalf_of,
        )?)
    }

    /// Verify a token allows `action` on `resource`; raises DeniedError or IntegrityError otherwise.
    #[pyo3(signature = (token, action, resource = "*"))]
    fn check<'py>(&self, py: Python<'py>, token: &str, action: &str, resource: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.check(token, action, resource)?)
    }

    fn inspect<'py>(&self, py: Python<'py>, token: &str) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.inspect(token)?)
    }
}

/// Append-only, hash-chained audit log.
#[pyclass(name = "AuditLog", module = "llm_circuitry._core")]
struct PyAuditLog {
    inner: attest::AuditLog,
}

#[pymethods]
impl PyAuditLog {
    #[new]
    #[pyo3(signature = (path = None))]
    fn new(path: Option<String>) -> PyResult<Self> {
        Ok(PyAuditLog {
            inner: match path {
                Some(p) => attest::AuditLog::open(p)?,
                None => attest::AuditLog::new(),
            },
        })
    }

    #[pyo3(signature = (actor, action, detail = None, on_behalf_of = None))]
    fn append<'py>(
        &mut self,
        py: Python<'py>,
        actor: &str,
        action: &str,
        detail: Option<&Bound<'_, PyAny>>,
        on_behalf_of: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let e = self.inner.append(actor, action, val(detail)?, on_behalf_of)?;
        self::py(py, &e)
    }

    fn verify(&self) -> PyResult<()> {
        Ok(self.inner.verify()?)
    }

    fn entries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, self.inner.entries())
    }

    #[pyo3(signature = (actor = None, action = None))]
    fn query<'py>(&self, py: Python<'py>, actor: Option<&str>, action: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.query(actor, action))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyfunction]
fn glob_match(pattern: &str, text: &str) -> bool {
    attest::glob_match(pattern, text)
}

// ---------------------------------------------------------------- replay

/// Record/replay cache for model calls, keyed by canonical request JSON.
#[pyclass(name = "ReplayCache", module = "llm_circuitry._core")]
struct PyReplayCache {
    inner: replay::ReplayCache,
}

#[pymethods]
impl PyReplayCache {
    /// mode: "record", "replay" or "auto".
    #[new]
    #[pyo3(signature = (mode = "auto", path = None))]
    fn new(mode: &str, path: Option<String>) -> PyResult<Self> {
        let path = path.map(std::path::PathBuf::from);
        Ok(PyReplayCache {
            inner: replay::ReplayCache::new(replay::ReplayMode::parse(mode)?, path.as_deref())?,
        })
    }

    #[getter]
    fn mode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.mode())
    }

    #[staticmethod]
    fn key(request: &Bound<'_, PyAny>) -> PyResult<String> {
        Ok(replay::ReplayCache::key(&val(Some(request))?))
    }

    /// Recorded response, or None if the real call should be made.
    fn lookup<'py>(&mut self, py: Python<'py>, request: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let r = self.inner.lookup(&val(Some(request))?)?;
        self::py(py, &r)
    }

    fn store(&mut self, request: &Bound<'_, PyAny>, response: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.inner.store(&val(Some(request))?, val(Some(response))?)?)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self::py(py, &self.inner.stats())
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CircuitError", py.get_type::<CircuitError>())?;
    m.add("NotFoundError", py.get_type::<NotFoundError>())?;
    m.add("StaleError", py.get_type::<StaleError>())?;
    m.add("ConflictError", py.get_type::<ConflictError>())?;
    m.add("CapacityError", py.get_type::<CapacityError>())?;
    m.add("InvalidStateError", py.get_type::<InvalidStateError>())?;
    m.add("IntegrityError", py.get_type::<IntegrityError>())?;
    m.add("DeniedError", py.get_type::<DeniedError>())?;
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    m.add_class::<PyProtectedContext>()?;
    m.add_class::<PyTxnManager>()?;
    m.add_class::<PyCoherentStore>()?;
    m.add_class::<PyInterruptController>()?;
    m.add_class::<PyWatchdog>()?;
    m.add_class::<PyClock>()?;
    m.add_class::<PyExecStack>()?;
    m.add_class::<PyVerifier>()?;
    m.add_class::<PyBlobStore>()?;
    m.add_class::<PyAddressableDoc>()?;
    m.add_class::<PyPager>()?;
    m.add_class::<PySigner>()?;
    m.add_class::<PyAuditLog>()?;
    m.add_class::<PyReplayCache>()?;
    m.add_function(wrap_pyfunction!(scan_injection, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(glob_match, m)?)?;
    Ok(())
}
