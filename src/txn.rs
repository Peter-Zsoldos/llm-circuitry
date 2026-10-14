//! **A journaling filesystem for real-world agent actions.**
//!
//! Filesystems survive crashes because they write intent to a journal before
//! touching data, and databases let you group writes into a transaction that
//! commits or rolls back as one unit. Agents now send emails, move money and
//! run migrations, with neither guarantee.
//!
//! [`TxnManager`] is a write-ahead, hash-chained journal of agent actions:
//! record a step *before* executing it, mark it done or failed afterwards, and
//! on failure get the exact list of completed steps to compensate, in reverse
//! order. After a crash, [`TxnManager::open`] plus [`TxnManager::recover`] tells
//! you which transactions were in flight and which steps are *in doubt*
//! (started but never confirmed).
//!
//! The manager doesn't run your code; it records and plans. The Python package
//! wraps it in a `with Transaction(...)` block that runs do/undo callables.

use crate::error::{err, Error, ErrorKind, Result};
use crate::util;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Begin {
        txn: u64,
        label: String,
    },
    Step {
        txn: u64,
        step: u64,
        name: String,
        params: Value,
        compensation: Option<String>,
    },
    StepDone {
        txn: u64,
        step: u64,
        result: Value,
    },
    StepFailed {
        txn: u64,
        step: u64,
        error: String,
    },
    RollbackBegin {
        txn: u64,
        reason: String,
    },
    Compensated {
        txn: u64,
        step: u64,
    },
    CompensationFailed {
        txn: u64,
        step: u64,
        error: String,
    },
    Commit {
        txn: u64,
    },
    RollbackEnd {
        txn: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalEntry {
    pub seq: u64,
    pub ts_ms: u64,
    pub event: Event,
    pub prev_hash: String,
    pub hash: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TxnStatus {
    Active,
    RollingBack,
    Committed,
    RolledBack,
    /// Rollback finished but some steps could not be undone.
    RollbackIncomplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    /// Recorded but not confirmed: after a crash, its outcome is unknown.
    Pending,
    Done,
    Failed,
    Compensated,
    CompensationFailed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: u64,
    pub name: String,
    pub params: Value,
    pub compensation: Option<String>,
    pub status: StepStatus,
    pub result: Value,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TxnRecord {
    pub txn: u64,
    pub label: String,
    pub status: TxnStatus,
    pub steps: Vec<StepRecord>,
}

/// What to do about one unfinished transaction after a crash.
#[derive(Clone, Debug, Serialize)]
pub struct RecoveryItem {
    pub txn: u64,
    pub label: String,
    pub status: TxnStatus,
    /// Completed steps that should be undone, most recent first.
    pub to_compensate: Vec<StepRecord>,
    /// Steps that were started but never confirmed: check the outside world.
    pub in_doubt: Vec<StepRecord>,
}

#[derive(Debug, Default)]
pub struct TxnManager {
    entries: Vec<JournalEntry>,
    txns: BTreeMap<u64, TxnRecord>,
    next_txn: u64,
    path: Option<PathBuf>,
}

impl TxnManager {
    /// In-memory journal.
    pub fn new() -> Self {
        TxnManager {
            next_txn: 1,
            ..Default::default()
        }
    }

    /// Journal persisted to a JSONL file. Existing entries are replayed and the
    /// hash chain is verified; tampering returns an `Integrity` error.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut m = TxnManager::new();
        for line in util::read_jsonl(&path)? {
            let entry: JournalEntry = serde_json::from_str(&line)?;
            m.check_link(&entry)?;
            m.apply(&entry.event);
            m.entries.push(entry);
        }
        m.path = Some(path);
        Ok(m)
    }

    fn check_link(&self, e: &JournalEntry) -> Result<()> {
        let prev = self.entries.last().map(|p| p.hash.as_str()).unwrap_or(util::GENESIS);
        let payload = serde_json::to_string(&e.event)?;
        if e.seq != self.entries.len() as u64
            || e.prev_hash != prev
            || e.hash != util::chain_hash(prev, e.seq, e.ts_ms, &payload)
        {
            return err(
                ErrorKind::Integrity,
                format!("journal entry {} fails hash-chain verification", e.seq),
            );
        }
        Ok(())
    }

    /// Re-verify the whole hash chain.
    pub fn verify(&self) -> Result<()> {
        let mut check = TxnManager::new();
        for e in &self.entries {
            check.check_link(e)?;
            check.entries.push(e.clone());
        }
        Ok(())
    }

    fn append(&mut self, event: Event) -> Result<()> {
        let seq = self.entries.len() as u64;
        let ts_ms = util::now_ms();
        let prev_hash = self
            .entries
            .last()
            .map(|p| p.hash.clone())
            .unwrap_or_else(|| util::GENESIS.into());
        let payload = serde_json::to_string(&event)?;
        let hash = util::chain_hash(&prev_hash, seq, ts_ms, &payload);
        let entry = JournalEntry {
            seq,
            ts_ms,
            event,
            prev_hash,
            hash,
        };
        if let Some(p) = &self.path {
            util::append_jsonl(p, &serde_json::to_string(&entry)?)?;
        }
        self.apply(&entry.event);
        self.entries.push(entry);
        Ok(())
    }

    fn apply(&mut self, event: &Event) {
        match event {
            Event::Begin { txn, label } => {
                self.next_txn = self.next_txn.max(txn + 1);
                self.txns.insert(
                    *txn,
                    TxnRecord {
                        txn: *txn,
                        label: label.clone(),
                        status: TxnStatus::Active,
                        steps: vec![],
                    },
                );
            }
            Event::Step {
                txn,
                step,
                name,
                params,
                compensation,
            } => {
                if let Some(t) = self.txns.get_mut(txn) {
                    t.steps.push(StepRecord {
                        step: *step,
                        name: name.clone(),
                        params: params.clone(),
                        compensation: compensation.clone(),
                        status: StepStatus::Pending,
                        result: Value::Null,
                        error: None,
                    });
                }
            }
            Event::StepDone { txn, step, result } => {
                if let Some(s) = self.step_mut(*txn, *step) {
                    s.status = StepStatus::Done;
                    s.result = result.clone();
                }
            }
            Event::StepFailed { txn, step, error } => {
                if let Some(s) = self.step_mut(*txn, *step) {
                    s.status = StepStatus::Failed;
                    s.error = Some(error.clone());
                }
            }
            Event::RollbackBegin { txn, .. } => {
                if let Some(t) = self.txns.get_mut(txn) {
                    t.status = TxnStatus::RollingBack;
                }
            }
            Event::Compensated { txn, step } => {
                if let Some(s) = self.step_mut(*txn, *step) {
                    s.status = StepStatus::Compensated;
                }
            }
            Event::CompensationFailed { txn, step, error } => {
                if let Some(s) = self.step_mut(*txn, *step) {
                    s.status = StepStatus::CompensationFailed;
                    s.error = Some(error.clone());
                }
            }
            Event::Commit { txn } => {
                if let Some(t) = self.txns.get_mut(txn) {
                    t.status = TxnStatus::Committed;
                }
            }
            Event::RollbackEnd { txn } => {
                if let Some(t) = self.txns.get_mut(txn) {
                    let clean = t
                        .steps
                        .iter()
                        .all(|s| matches!(s.status, StepStatus::Compensated | StepStatus::Failed));
                    t.status = if clean {
                        TxnStatus::RolledBack
                    } else {
                        TxnStatus::RollbackIncomplete
                    };
                }
            }
        }
    }

    fn step_mut(&mut self, txn: u64, step: u64) -> Option<&mut StepRecord> {
        self.txns.get_mut(&txn)?.steps.iter_mut().find(|s| s.step == step)
    }

    fn txn_in(&self, txn: u64, allowed: &[TxnStatus]) -> Result<&TxnRecord> {
        let t = self.get(txn)?;
        if !allowed.contains(&t.status) {
            return err(
                ErrorKind::InvalidState,
                format!("transaction {txn} is {:?}, expected one of {allowed:?}", t.status),
            );
        }
        Ok(t)
    }

    fn step_in(&self, txn: u64, step: u64, allowed: &[StepStatus]) -> Result<&StepRecord> {
        let s = self
            .get(txn)?
            .steps
            .iter()
            .find(|s| s.step == step)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("transaction {txn} has no step {step}")))?;
        if !allowed.contains(&s.status) {
            return err(
                ErrorKind::InvalidState,
                format!(
                    "step {step} of transaction {txn} is {:?}, expected one of {allowed:?}",
                    s.status
                ),
            );
        }
        Ok(s)
    }

    pub fn begin(&mut self, label: &str) -> Result<u64> {
        let txn = self.next_txn;
        self.append(Event::Begin {
            txn,
            label: label.to_string(),
        })?;
        Ok(txn)
    }

    /// Record intent to perform a step. Call this *before* executing it.
    /// `compensation` names how to undo it (for humans and for crash recovery).
    pub fn step(&mut self, txn: u64, name: &str, params: Value, compensation: Option<String>) -> Result<u64> {
        let t = self.txn_in(txn, &[TxnStatus::Active])?;
        if let Some(open) = t.steps.iter().find(|s| s.status == StepStatus::Pending) {
            return err(
                ErrorKind::InvalidState,
                format!(
                    "step {} ('{}') is still pending; mark it done or failed first",
                    open.step, open.name
                ),
            );
        }
        let step = t.steps.len() as u64;
        self.append(Event::Step {
            txn,
            step,
            name: name.to_string(),
            params,
            compensation,
        })?;
        Ok(step)
    }

    pub fn done(&mut self, txn: u64, step: u64, result: Value) -> Result<()> {
        self.txn_in(txn, &[TxnStatus::Active])?;
        self.step_in(txn, step, &[StepStatus::Pending])?;
        self.append(Event::StepDone { txn, step, result })
    }

    pub fn fail(&mut self, txn: u64, step: u64, error: &str) -> Result<()> {
        self.txn_in(txn, &[TxnStatus::Active])?;
        self.step_in(txn, step, &[StepStatus::Pending])?;
        self.append(Event::StepFailed {
            txn,
            step,
            error: error.to_string(),
        })
    }

    /// Commit. Fails if any step is pending or failed: those need a rollback.
    pub fn commit(&mut self, txn: u64) -> Result<()> {
        let t = self.txn_in(txn, &[TxnStatus::Active])?;
        if let Some(s) = t.steps.iter().find(|s| s.status != StepStatus::Done) {
            return err(
                ErrorKind::InvalidState,
                format!("cannot commit: step {} ('{}') is {:?}", s.step, s.name, s.status),
            );
        }
        self.append(Event::Commit { txn })
    }

    /// Start rolling back. Returns the completed steps to undo, newest first.
    pub fn begin_rollback(&mut self, txn: u64, reason: &str) -> Result<Vec<StepRecord>> {
        self.txn_in(
            txn,
            &[TxnStatus::Active, TxnStatus::RollingBack, TxnStatus::RollbackIncomplete],
        )?;
        self.append(Event::RollbackBegin {
            txn,
            reason: reason.to_string(),
        })?;
        Ok(self.rollback_plan(txn))
    }

    fn rollback_plan(&self, txn: u64) -> Vec<StepRecord> {
        self.txns
            .get(&txn)
            .map(|t| {
                t.steps
                    .iter()
                    .rev()
                    .filter(|s| {
                        matches!(
                            s.status,
                            StepStatus::Done | StepStatus::Pending | StepStatus::CompensationFailed
                        )
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn compensated(&mut self, txn: u64, step: u64) -> Result<()> {
        self.txn_in(txn, &[TxnStatus::RollingBack])?;
        self.step_in(
            txn,
            step,
            &[StepStatus::Done, StepStatus::Pending, StepStatus::CompensationFailed],
        )?;
        self.append(Event::Compensated { txn, step })
    }

    pub fn compensation_failed(&mut self, txn: u64, step: u64, error: &str) -> Result<()> {
        self.txn_in(txn, &[TxnStatus::RollingBack])?;
        self.step_in(
            txn,
            step,
            &[StepStatus::Done, StepStatus::Pending, StepStatus::CompensationFailed],
        )?;
        self.append(Event::CompensationFailed {
            txn,
            step,
            error: error.to_string(),
        })
    }

    /// Finish the rollback. Returns `RolledBack`, or `RollbackIncomplete` if
    /// any completed step was not compensated.
    pub fn finish_rollback(&mut self, txn: u64) -> Result<TxnStatus> {
        self.txn_in(txn, &[TxnStatus::RollingBack])?;
        self.append(Event::RollbackEnd { txn })?;
        Ok(self.get(txn)?.status)
    }

    pub fn get(&self, txn: u64) -> Result<&TxnRecord> {
        self.txns
            .get(&txn)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("no transaction {txn}")))
    }

    pub fn list(&self) -> Vec<&TxnRecord> {
        self.txns.values().collect()
    }

    /// Transactions that never reached a clean end state.
    pub fn recover(&self) -> Vec<RecoveryItem> {
        self.txns
            .values()
            .filter(|t| {
                matches!(
                    t.status,
                    TxnStatus::Active | TxnStatus::RollingBack | TxnStatus::RollbackIncomplete
                )
            })
            .map(|t| RecoveryItem {
                txn: t.txn,
                label: t.label.clone(),
                status: t.status,
                to_compensate: t
                    .steps
                    .iter()
                    .rev()
                    .filter(|s| matches!(s.status, StepStatus::Done | StepStatus::CompensationFailed))
                    .cloned()
                    .collect(),
                in_doubt: t
                    .steps
                    .iter()
                    .filter(|s| s.status == StepStatus::Pending)
                    .cloned()
                    .collect(),
            })
            .collect()
    }

    pub fn journal(&self) -> &[JournalEntry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn commit_and_rollback() {
        let mut m = TxnManager::new();
        let t = m.begin("deploy").unwrap();
        let a = m
            .step(t, "create_bucket", json!({"name": "b"}), Some("delete_bucket".into()))
            .unwrap();
        m.done(t, a, json!("ok")).unwrap();
        let b = m.step(t, "upload", json!({}), Some("delete_files".into())).unwrap();
        m.fail(t, b, "timeout").unwrap();
        assert!(m.commit(t).is_err());
        let plan = m.begin_rollback(t, "upload failed").unwrap();
        assert_eq!(plan.iter().map(|s| s.step).collect::<Vec<_>>(), vec![a]);
        m.compensated(t, a).unwrap();
        assert_eq!(m.finish_rollback(t).unwrap(), TxnStatus::RolledBack);
        m.verify().unwrap();

        let t2 = m.begin("ok").unwrap();
        let s = m.step(t2, "x", Value::Null, None).unwrap();
        m.done(t2, s, Value::Null).unwrap();
        m.commit(t2).unwrap();
        assert_eq!(m.get(t2).unwrap().status, TxnStatus::Committed);
    }

    #[test]
    fn crash_recovery_from_file() {
        let dir = std::env::temp_dir().join(format!("llmc-txn-{}", util::to_hex(&util::random_bytes::<6>())));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.jsonl");
        {
            let mut m = TxnManager::open(&path).unwrap();
            let t = m.begin("pay invoices").unwrap();
            let s = m.step(t, "pay", json!({"invoice": 1}), Some("refund".into())).unwrap();
            m.done(t, s, json!({"tx": "abc"})).unwrap();
            m.step(t, "pay", json!({"invoice": 2}), Some("refund".into())).unwrap();
            // process "crashes" here
        }
        let m = TxnManager::open(&path).unwrap();
        let r = m.recover();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].to_compensate.len(), 1);
        assert_eq!(r[0].in_doubt.len(), 1);

        let text = std::fs::read_to_string(&path)
            .unwrap()
            .replace("\"invoice\":1", "\"invoice\":9");
        std::fs::write(&path, text).unwrap();
        assert_eq!(TxnManager::open(&path).unwrap_err().kind, ErrorKind::Integrity);
        std::fs::remove_dir_all(dir).ok();
    }
}
