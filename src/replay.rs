//! **Deterministic replay: the debugger port for model calls.**
//!
//! Hardware debugging relies on being able to run the same thing twice and get
//! the same result. Model calls are non-deterministic and cost money, so a bug
//! seen once in an agent run is often impossible to reproduce.
//!
//! [`ReplayCache`] records each request/response pair keyed by a hash of the
//! canonical JSON request, optionally to a JSONL file. In `replay` mode every
//! call must come from the recording (a miss is an error), which makes agent
//! runs reproducible in tests and debugging sessions.

use crate::error::{err, ErrorKind, Result};
use crate::util;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReplayMode {
    /// Always call the model; record every response (overwriting).
    Record,
    /// Never call the model; every request must have a recording.
    Replay,
    /// Use a recording when there is one, otherwise call and record.
    Auto,
}

impl ReplayMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "record" => Ok(ReplayMode::Record),
            "replay" => Ok(ReplayMode::Replay),
            "auto" => Ok(ReplayMode::Auto),
            other => err(
                ErrorKind::Invalid,
                format!("unknown replay mode '{other}' (use record, replay or auto)"),
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Recording {
    key: String,
    request: Value,
    response: Value,
    ts_ms: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ReplayStats {
    pub recordings: usize,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug)]
pub struct ReplayCache {
    map: HashMap<String, Value>,
    path: Option<PathBuf>,
    mode: ReplayMode,
    hits: u64,
    misses: u64,
}

impl ReplayCache {
    pub fn new(mode: ReplayMode, path: Option<&Path>) -> Result<Self> {
        let mut map = HashMap::new();
        if let Some(p) = path {
            for line in util::read_jsonl(p)? {
                let r: Recording = serde_json::from_str(&line)?;
                map.insert(r.key, r.response);
            }
        }
        Ok(ReplayCache {
            map,
            path: path.map(Path::to_path_buf),
            mode,
            hits: 0,
            misses: 0,
        })
    }

    pub fn mode(&self) -> ReplayMode {
        self.mode
    }

    /// Stable key for a request (object keys are sorted before hashing).
    pub fn key(request: &Value) -> String {
        util::sha256_hex(request.to_string().as_bytes())[..32].to_string()
    }

    /// `Some(response)` if the call should be served from the recording, `None`
    /// if the caller should make the real call (and then [`store`](Self::store) it).
    pub fn lookup(&mut self, request: &Value) -> Result<Option<Value>> {
        if self.mode == ReplayMode::Record {
            self.misses += 1;
            return Ok(None);
        }
        let key = Self::key(request);
        match self.map.get(&key) {
            Some(v) => {
                self.hits += 1;
                Ok(Some(v.clone()))
            }
            None => {
                self.misses += 1;
                if self.mode == ReplayMode::Replay {
                    return err(
                        ErrorKind::NotFound,
                        format!("replay mode: no recording for request {key}"),
                    );
                }
                Ok(None)
            }
        }
    }

    pub fn store(&mut self, request: &Value, response: Value) -> Result<()> {
        if self.mode == ReplayMode::Replay {
            return err(ErrorKind::InvalidState, "cannot record in replay mode");
        }
        let key = Self::key(request);
        if let Some(p) = &self.path {
            let rec = Recording {
                key: key.clone(),
                request: request.clone(),
                response: response.clone(),
                ts_ms: util::now_ms(),
            };
            util::append_jsonl(p, &serde_json::to_string(&rec)?)?;
        }
        self.map.insert(key, response);
        Ok(())
    }

    pub fn stats(&self) -> ReplayStats {
        ReplayStats {
            recordings: self.map.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn record_then_replay() {
        let dir = std::env::temp_dir().join(format!("llmc-replay-{}", util::to_hex(&util::random_bytes::<6>())));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("calls.jsonl");
        let req = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0});
        {
            let mut c = ReplayCache::new(ReplayMode::Auto, Some(&path)).unwrap();
            assert!(c.lookup(&req).unwrap().is_none());
            c.store(&req, json!("hello!")).unwrap();
        }
        let mut c = ReplayCache::new(ReplayMode::Replay, Some(&path)).unwrap();
        let reordered = json!({"temperature": 1.0, "messages": [{"content": "hi", "role": "user"}], "model": "m"});
        assert_eq!(c.lookup(&reordered).unwrap(), Some(json!("hello!")));
        assert_eq!(c.lookup(&json!({"other": 1})).unwrap_err().kind, ErrorKind::NotFound);
        std::fs::remove_dir_all(dir).ok();
    }
}
