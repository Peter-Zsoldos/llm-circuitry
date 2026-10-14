//! **Cache coherence for multi-agent systems.**
//!
//! In a multi-core CPU, every core caches memory, and a coherence protocol
//! (MESI) makes sure a core never keeps computing on a value another core has
//! changed. In a multi-agent system, every agent's context window is a private
//! cache of the world, and nothing tells agent B that agent A just changed the
//! file B is reasoning about.
//!
//! [`CoherentStore`] is versioned shared state that remembers which version
//! each agent last read. Writes from stale reads are rejected, other readers get
//! invalidation messages to inject into their context, and
//! [`CoherentStore::validate`] is a cheap guard to run before any agent acts.

use crate::error::{err, ErrorKind, Result};
use crate::util;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Versioned {
    pub key: String,
    pub value: Value,
    pub version: u64,
    pub writer: String,
    pub updated_ms: u64,
}

/// Sent to an agent whose cached copy of `key` is no longer current.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Invalidation {
    pub key: String,
    pub your_version: u64,
    pub new_version: u64,
    pub writer: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaleRead {
    pub key: String,
    /// 0 means the agent never read this key.
    pub cached_version: u64,
    pub current_version: u64,
}

/// MESI-style state of one key from one agent's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LineState {
    /// This agent wrote the current version.
    Modified,
    /// This agent is the only one holding the current version.
    Exclusive,
    /// Several agents hold the current version.
    Shared,
    /// Not cached, or the cached copy is outdated.
    Invalid,
}

#[derive(Clone, Debug)]
struct Lock {
    agent: String,
    expires_ms: u64,
}

#[derive(Debug, Default)]
pub struct CoherentStore {
    lines: BTreeMap<String, Versioned>,
    caches: HashMap<String, HashMap<String, u64>>,
    inbox: HashMap<String, Vec<Invalidation>>,
    locks: HashMap<String, Lock>,
}

impl CoherentStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read `key` as `agent`, recording the version the agent now holds.
    pub fn read(&mut self, agent: &str, key: &str) -> Option<Versioned> {
        let line = self.lines.get(key)?.clone();
        self.caches
            .entry(agent.into())
            .or_default()
            .insert(key.into(), line.version);
        if let Some(inbox) = self.inbox.get_mut(agent) {
            inbox.retain(|i| i.key != key);
        }
        Some(line)
    }

    /// Look at a value without registering a cached copy.
    pub fn get(&self, key: &str) -> Option<&Versioned> {
        self.lines.get(key)
    }

    pub fn keys(&self) -> Vec<String> {
        self.lines.keys().cloned().collect()
    }

    fn check_lock(&self, agent: &str, key: &str) -> Result<()> {
        if let Some(l) = self.locks.get(key) {
            if l.agent != agent && l.expires_ms > util::now_ms() {
                return err(ErrorKind::Conflict, format!("'{key}' is locked by agent '{}'", l.agent));
            }
        }
        Ok(())
    }

    /// Write `key`. With `expected_version`, this is a compare-and-swap. Without
    /// it, the write is rejected if the agent previously read an older version
    /// (it would be acting on stale data); a blind write to a key the agent
    /// never read is allowed. Returns the new version.
    pub fn write(&mut self, agent: &str, key: &str, value: Value, expected_version: Option<u64>) -> Result<u64> {
        self.check_lock(agent, key)?;
        let (current, writer) = self
            .lines
            .get(key)
            .map(|l| (l.version, l.writer.clone()))
            .unwrap_or((0, String::new()));
        match expected_version {
            Some(exp) if exp != current => {
                return err(
                    ErrorKind::Stale,
                    format!("write to '{key}' expected version {exp}, but it is at version {current} (last written by '{writer}')"),
                )
            }
            None => {
                if let Some(cached) = self.caches.get(agent).and_then(|c| c.get(key)) {
                    if *cached != current {
                        return err(
                            ErrorKind::Stale,
                            format!("agent '{agent}' read '{key}' at version {cached}, but it is now version {current} (written by '{writer}'); re-read before writing"),
                        );
                    }
                }
            }
            _ => {}
        }
        let new = current + 1;
        self.lines.insert(
            key.into(),
            Versioned {
                key: key.into(),
                value,
                version: new,
                writer: agent.into(),
                updated_ms: util::now_ms(),
            },
        );
        self.caches.entry(agent.into()).or_default().insert(key.into(), new);
        for (other, cache) in &self.caches {
            if other == agent {
                continue;
            }
            if let Some(&v) = cache.get(key) {
                if v < new {
                    let inbox = self.inbox.entry(other.clone()).or_default();
                    inbox.retain(|i| i.key != key);
                    inbox.push(Invalidation {
                        key: key.into(),
                        your_version: v,
                        new_version: new,
                        writer: agent.into(),
                    });
                }
            }
        }
        Ok(new)
    }

    /// Which of `keys` (or all keys this agent has read, if empty) are out of
    /// date for `agent`. Empty result means safe to act.
    pub fn validate(&self, agent: &str, keys: &[String]) -> Vec<StaleRead> {
        let empty = HashMap::new();
        let cache = self.caches.get(agent).unwrap_or(&empty);
        let keys: Vec<String> = if keys.is_empty() {
            cache.keys().cloned().collect()
        } else {
            keys.to_vec()
        };
        let mut out: Vec<StaleRead> = keys
            .into_iter()
            .filter_map(|k| {
                let current = self.lines.get(&k).map(|l| l.version).unwrap_or(0);
                let cached = cache.get(&k).copied().unwrap_or(0);
                (cached != current).then_some(StaleRead {
                    key: k,
                    cached_version: cached,
                    current_version: current,
                })
            })
            .collect();
        out.sort_by(|a, b| a.key.cmp(&b.key));
        out
    }

    /// Like [`validate`](Self::validate), but returns a `Stale` error naming the
    /// outdated keys.
    pub fn require_fresh(&self, agent: &str, keys: &[String]) -> Result<()> {
        let stale = self.validate(agent, keys);
        if stale.is_empty() {
            return Ok(());
        }
        let list: Vec<String> = stale
            .iter()
            .map(|s| {
                format!(
                    "'{}' (have v{}, current v{})",
                    s.key, s.cached_version, s.current_version
                )
            })
            .collect();
        err(
            ErrorKind::Stale,
            format!("agent '{agent}' is acting on stale data: {}", list.join(", ")),
        )
    }

    pub fn state(&self, agent: &str, key: &str) -> LineState {
        let Some(line) = self.lines.get(key) else {
            return LineState::Invalid;
        };
        let cached = self.caches.get(agent).and_then(|c| c.get(key)).copied();
        if cached != Some(line.version) {
            return LineState::Invalid;
        }
        if line.writer == agent {
            return LineState::Modified;
        }
        let holders = self
            .caches
            .values()
            .filter(|c| c.get(key) == Some(&line.version))
            .count();
        if holders > 1 {
            LineState::Shared
        } else {
            LineState::Exclusive
        }
    }

    /// Take the pending invalidation messages for `agent`.
    pub fn invalidations(&mut self, agent: &str) -> Vec<Invalidation> {
        self.inbox.remove(agent).unwrap_or_default()
    }

    /// Exclusive write lock on `key` for `ttl_ms` milliseconds (re-entrant for the holder).
    pub fn lock(&mut self, agent: &str, key: &str, ttl_ms: u64) -> Result<()> {
        self.check_lock(agent, key)?;
        self.locks.insert(
            key.into(),
            Lock {
                agent: agent.into(),
                expires_ms: util::now_ms() + ttl_ms,
            },
        );
        Ok(())
    }

    pub fn unlock(&mut self, agent: &str, key: &str) -> Result<()> {
        match self.locks.get(key) {
            Some(l) if l.agent == agent => {
                self.locks.remove(key);
                Ok(())
            }
            Some(l) => err(
                ErrorKind::Conflict,
                format!("'{key}' is locked by '{}', not '{agent}'", l.agent),
            ),
            None => err(ErrorKind::NotFound, format!("'{key}' is not locked")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stale_reads_are_caught() {
        let mut s = CoherentStore::new();
        s.write("planner", "plan", json!("v1"), None).unwrap();
        s.read("coder", "plan").unwrap();
        s.read("tester", "plan").unwrap();
        assert_eq!(s.state("coder", "plan"), LineState::Shared);
        assert_eq!(s.state("planner", "plan"), LineState::Modified);

        s.write("planner", "plan", json!("v2"), None).unwrap();
        assert_eq!(s.state("coder", "plan"), LineState::Invalid);
        assert_eq!(s.validate("coder", &[]).len(), 1);
        assert!(s.require_fresh("coder", &[]).is_err());
        let inv = s.invalidations("coder");
        assert_eq!(inv[0].new_version, 2);

        let e = s.write("coder", "plan", json!("coder's idea"), None).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Stale);
        s.read("coder", "plan");
        assert!(s.write("coder", "plan", json!("v3"), Some(2)).is_ok());
        assert_eq!(
            s.write("tester", "plan", json!("x"), Some(2)).unwrap_err().kind,
            ErrorKind::Stale
        );
    }

    #[test]
    fn locks() {
        let mut s = CoherentStore::new();
        s.lock("a", "file.py", 60_000).unwrap();
        assert_eq!(
            s.write("b", "file.py", json!(1), None).unwrap_err().kind,
            ErrorKind::Conflict
        );
        s.write("a", "file.py", json!(1), None).unwrap();
        s.unlock("a", "file.py").unwrap();
        s.write("b", "file.py", json!(2), Some(1)).unwrap();
    }
}
