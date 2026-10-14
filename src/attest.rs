//! **TPM, secure boot and access control for agents.**
//!
//! A TPM can prove which software booted, an OS knows which user owns each
//! process, and file permissions say who may do what. In agent systems an
//! instruction carries no proof of where it came from, a tool can't tell which
//! agent is calling or on whose behalf, and logs can be edited after the fact.
//!
//! * [`Signer`]: HMAC-SHA256 signed instructions ([`Envelope`]), so a
//!   downstream agent can tell a genuine orchestrator message from text that
//!   merely claims to be one.
//! * Capability tokens ([`Signer::grant`] / [`Signer::check`]): least-privilege,
//!   expiring permissions such as "agent `mailer` may `email.send` to
//!   `*@mycompany.com` on behalf of `alice` for 10 minutes".
//! * [`AuditLog`]: append-only, hash-chained record of who did what for whom;
//!   any edit breaks the chain.

use crate::error::{err, Error, ErrorKind, Result};
use crate::util;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    pub issuer: String,
    pub subject: String,
    pub content: String,
    pub ts_ms: u64,
    pub sig: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Capability {
    pub issuer: String,
    pub agent: String,
    /// Action patterns, `*` wildcards allowed (`email.*`).
    pub actions: Vec<String>,
    /// Resource patterns, `*` wildcards allowed (`/repo/src/*`).
    pub resources: Vec<String>,
    pub on_behalf_of: Option<String>,
    pub issued_ms: u64,
    pub expires_ms: Option<u64>,
}

/// `*` matches any run of characters (including none); everything else is literal.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0, 0);
    let (mut star, mut mark) = (None, 0);
    while ti < t.len() {
        if pi < p.len() && p[pi] != '*' && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

pub struct Signer {
    issuer: String,
    key: Vec<u8>,
}

impl std::fmt::Debug for Signer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Signer")
            .field("issuer", &self.issuer)
            .finish_non_exhaustive()
    }
}

impl Signer {
    /// A signer with an explicit secret key (at least 16 bytes).
    pub fn new(issuer: &str, key: &[u8]) -> Result<Self> {
        if key.len() < 16 {
            return err(ErrorKind::Invalid, "signing key must be at least 16 bytes");
        }
        Ok(Signer {
            issuer: issuer.into(),
            key: key.to_vec(),
        })
    }

    /// A signer with a fresh random 32-byte key.
    pub fn generate(issuer: &str) -> Self {
        Signer {
            issuer: issuer.into(),
            key: util::random_bytes::<32>().to_vec(),
        }
    }

    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// The secret key as hex, for storing in a secrets manager. Never put it in a prompt.
    pub fn key_hex(&self) -> String {
        util::to_hex(&self.key)
    }

    fn mac(&self, parts: &[&str]) -> String {
        // Length-prefix every field so ("ab","c") and ("a","bc") sign differently.
        let mut msg = Vec::new();
        for p in parts {
            msg.extend_from_slice(&(p.len() as u64).to_le_bytes());
            msg.extend_from_slice(p.as_bytes());
        }
        util::to_hex(&util::hmac_sha256(&self.key, &msg))
    }

    /// Sign `content` addressed to `subject`.
    pub fn sign(&self, subject: &str, content: &str) -> Envelope {
        let ts_ms = util::now_ms();
        let sig = self.mac(&[&self.issuer, subject, content, &ts_ms.to_string()]);
        Envelope {
            issuer: self.issuer.clone(),
            subject: subject.into(),
            content: content.into(),
            ts_ms,
            sig,
        }
    }

    /// True if the envelope was signed by this signer's key and not modified.
    pub fn verify(&self, env: &Envelope) -> bool {
        let expected = self.mac(&[&env.issuer, &env.subject, &env.content, &env.ts_ms.to_string()]);
        env.issuer == self.issuer && util::ct_eq(expected.as_bytes(), env.sig.as_bytes())
    }

    /// Issue a capability token string (`cap1.<payload>.<sig>`).
    pub fn grant(
        &self,
        agent: &str,
        actions: &[String],
        resources: &[String],
        ttl_seconds: Option<f64>,
        on_behalf_of: Option<&str>,
    ) -> Result<String> {
        if actions.is_empty() {
            return err(ErrorKind::Invalid, "a capability needs at least one action pattern");
        }
        let now = util::now_ms();
        let cap = Capability {
            issuer: self.issuer.clone(),
            agent: agent.into(),
            actions: actions.to_vec(),
            resources: if resources.is_empty() {
                vec!["*".into()]
            } else {
                resources.to_vec()
            },
            on_behalf_of: on_behalf_of.map(String::from),
            issued_ms: now,
            expires_ms: ttl_seconds.map(|t| (now as f64 + t * 1000.0).max(0.0) as u64),
        };
        let payload = util::to_hex(serde_json::to_string(&cap)?.as_bytes());
        let sig = self.mac(&["cap1", &payload]);
        Ok(format!("cap1.{payload}.{sig}"))
    }

    /// Decode and verify a token without checking a specific action.
    pub fn inspect(&self, token: &str) -> Result<Capability> {
        let mut parts = token.split('.');
        let (Some("cap1"), Some(payload), Some(sig), None) = (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return err(ErrorKind::Invalid, "malformed capability token");
        };
        if !util::ct_eq(self.mac(&["cap1", payload]).as_bytes(), sig.as_bytes()) {
            return err(ErrorKind::Integrity, "capability token signature is invalid");
        }
        let bytes =
            util::from_hex(payload).ok_or_else(|| Error::new(ErrorKind::Invalid, "malformed capability payload"))?;
        let cap: Capability = serde_json::from_slice(&bytes)?;
        if let Some(exp) = cap.expires_ms {
            if util::now_ms() > exp {
                return err(
                    ErrorKind::Denied,
                    format!("capability for '{}' expired at {}", cap.agent, util::iso8601(exp)),
                );
            }
        }
        Ok(cap)
    }

    /// Verify a token and check that it allows `action` on `resource`.
    pub fn check(&self, token: &str, action: &str, resource: &str) -> Result<Capability> {
        let cap = self.inspect(token)?;
        if !cap.actions.iter().any(|p| glob_match(p, action)) {
            return err(
                ErrorKind::Denied,
                format!(
                    "agent '{}' is not allowed to '{action}' (granted: {:?})",
                    cap.agent, cap.actions
                ),
            );
        }
        if !cap.resources.iter().any(|p| glob_match(p, resource)) {
            return err(
                ErrorKind::Denied,
                format!(
                    "agent '{}' may '{action}' but not on '{resource}' (granted: {:?})",
                    cap.agent, cap.resources
                ),
            );
        }
        Ok(cap)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub seq: u64,
    pub ts_ms: u64,
    pub actor: String,
    pub on_behalf_of: Option<String>,
    pub action: String,
    pub detail: Value,
    pub prev_hash: String,
    pub hash: String,
}

impl AuditEntry {
    fn payload(&self) -> String {
        json!({"actor": self.actor, "on_behalf_of": self.on_behalf_of, "action": self.action, "detail": self.detail})
            .to_string()
    }
}

/// Append-only, hash-chained audit log, optionally persisted as JSONL.
#[derive(Debug, Default)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
    path: Option<PathBuf>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load (or create) a log file. Call [`verify`](Self::verify) to check it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let entries = util::read_jsonl(&path)?
            .iter()
            .map(|l| serde_json::from_str(l).map_err(Error::from))
            .collect::<Result<Vec<AuditEntry>>>()?;
        Ok(AuditLog {
            entries,
            path: Some(path),
        })
    }

    pub fn append(
        &mut self,
        actor: &str,
        action: &str,
        detail: Value,
        on_behalf_of: Option<&str>,
    ) -> Result<AuditEntry> {
        let mut e = AuditEntry {
            seq: self.entries.len() as u64,
            ts_ms: util::now_ms(),
            actor: actor.into(),
            on_behalf_of: on_behalf_of.map(String::from),
            action: action.into(),
            detail,
            prev_hash: self
                .entries
                .last()
                .map(|p| p.hash.clone())
                .unwrap_or_else(|| util::GENESIS.into()),
            hash: String::new(),
        };
        e.hash = util::chain_hash(&e.prev_hash, e.seq, e.ts_ms, &e.payload());
        if let Some(p) = &self.path {
            util::append_jsonl(p, &serde_json::to_string(&e)?)?;
        }
        self.entries.push(e.clone());
        Ok(e)
    }

    /// Check the whole chain. Returns an `Integrity` error naming the first bad entry.
    pub fn verify(&self) -> Result<()> {
        let mut prev = util::GENESIS.to_string();
        for (i, e) in self.entries.iter().enumerate() {
            if e.seq != i as u64
                || e.prev_hash != prev
                || e.hash != util::chain_hash(&prev, e.seq, e.ts_ms, &e.payload())
            {
                return err(
                    ErrorKind::Integrity,
                    format!("audit log entry {i} has been modified, removed or reordered"),
                );
            }
            prev = e.hash.clone();
        }
        Ok(())
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    pub fn query(&self, actor: Option<&str>, action: Option<&str>) -> Vec<AuditEntry> {
        self.entries
            .iter()
            .filter(|e| actor.map_or(true, |a| e.actor == a))
            .filter(|e| action.map_or(true, |a| glob_match(a, &e.action)))
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob_match("email.*", "email.send"));
        assert!(glob_match("*@acme.com", "bob@acme.com"));
        assert!(!glob_match("*@acme.com", "bob@evil.com"));
        assert!(glob_match("a*b*c", "aXXbYYc"));
        assert!(!glob_match("a*b*c", "aXXbYY"));
        assert!(glob_match("*", ""));
    }

    #[test]
    fn signed_instructions() {
        let orch = Signer::generate("orchestrator");
        let env = orch.sign("worker-1", "deploy to staging");
        assert!(orch.verify(&env));
        let mut forged = env.clone();
        forged.content = "deploy to production".into();
        assert!(!orch.verify(&forged));
        assert!(!Signer::generate("orchestrator").verify(&env));
    }

    #[test]
    fn capabilities() {
        let s = Signer::new("auth", b"0123456789abcdef0123").unwrap();
        let tok = s
            .grant(
                "mailer",
                &["email.send".into()],
                &["*@acme.com".into()],
                Some(600.0),
                Some("alice"),
            )
            .unwrap();
        assert_eq!(
            s.check(&tok, "email.send", "bob@acme.com")
                .unwrap()
                .on_behalf_of
                .as_deref(),
            Some("alice")
        );
        assert_eq!(
            s.check(&tok, "email.send", "x@evil.com").unwrap_err().kind,
            ErrorKind::Denied
        );
        assert_eq!(
            s.check(&tok, "fs.delete", "bob@acme.com").unwrap_err().kind,
            ErrorKind::Denied
        );
        let tampered = tok.replacen("cap1.7b", "cap1.7c", 1);
        assert_eq!(
            s.check(&tampered, "email.send", "bob@acme.com").unwrap_err().kind,
            ErrorKind::Integrity
        );
        let expired = s.grant("x", &["*".into()], &[], Some(-1.0), None).unwrap();
        assert_eq!(s.check(&expired, "a", "b").unwrap_err().kind, ErrorKind::Denied);
    }

    #[test]
    fn audit_chain_detects_tampering() {
        let mut log = AuditLog::new();
        log.append("agent-1", "email.send", json!({"to": "bob"}), Some("alice"))
            .unwrap();
        log.append("agent-2", "fs.write", json!({"path": "/tmp/x"}), None)
            .unwrap();
        log.verify().unwrap();
        assert_eq!(log.query(None, Some("email.*")).len(), 1);
        log.entries[0].detail = json!({"to": "eve"});
        assert_eq!(log.verify().unwrap_err().kind, ErrorKind::Integrity);
    }
}
