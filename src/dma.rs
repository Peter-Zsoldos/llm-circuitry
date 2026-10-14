//! **DMA: pass large data by reference instead of through the context.**
//!
//! In a computer, a disk controller copies data straight into RAM without
//! routing every byte through the CPU. In agent systems everything goes through
//! the context: if agent A wants to hand agent B a 50 MB CSV, someone has to
//! read or summarise it on the way.
//!
//! [`BlobStore`] is a content-addressed store that hands out short handles
//! (`blob:3f9a…`). Agents pass handles to each other and to tools, and only pull
//! in what they need: a description, a line range, or the lines matching a
//! search. Blobs are reference-counted; unreferenced blobs stay cached until the
//! capacity limit evicts them (least recently used first).

use crate::error::{err, Error, ErrorKind, Result};
use crate::util;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlobInfo {
    pub handle: String,
    pub size: u64,
    pub lines: u64,
    pub mime: String,
    pub label: String,
    pub refs: u32,
    pub created_ms: u64,
    pub last_access_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrepHit {
    /// 1-based line number.
    pub line: u64,
    pub text: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StoreStats {
    pub blobs: usize,
    pub bytes: u64,
    pub capacity_bytes: Option<u64>,
    pub evictions: u64,
    pub dedup_hits: u64,
}

#[derive(Debug)]
struct Blob {
    info: BlobInfo,
    data: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct BlobStore {
    blobs: BTreeMap<String, Blob>,
    capacity: Option<u64>,
    used: u64,
    evictions: u64,
    dedup_hits: u64,
    clock: u64,
}

fn count_lines(data: &[u8]) -> u64 {
    let n = data.iter().filter(|b| **b == b'\n').count() as u64;
    if !data.is_empty() && !data.ends_with(b"\n") {
        n + 1
    } else {
        n
    }
}

impl BlobStore {
    pub fn new(capacity_bytes: Option<u64>) -> Self {
        BlobStore {
            capacity: capacity_bytes,
            ..Default::default()
        }
    }

    fn touch(&mut self) -> u64 {
        // Monotonic counter so LRU order is exact even within one millisecond.
        self.clock += 1;
        self.clock
    }

    /// Store data and return its handle. Storing identical bytes again returns
    /// the same handle and adds a reference.
    pub fn put(&mut self, data: &[u8], mime: &str, label: &str) -> String {
        let handle = format!("blob:{}", &util::sha256_hex(data)[..16]);
        let tick = self.touch();
        if let Some(b) = self.blobs.get_mut(&handle) {
            b.info.refs += 1;
            b.info.last_access_ms = tick;
            self.dedup_hits += 1;
            return handle;
        }
        let info = BlobInfo {
            handle: handle.clone(),
            size: data.len() as u64,
            lines: count_lines(data),
            mime: mime.into(),
            label: label.into(),
            refs: 1,
            created_ms: util::now_ms(),
            last_access_ms: tick,
        };
        self.used += data.len() as u64;
        self.blobs.insert(
            handle.clone(),
            Blob {
                info,
                data: data.to_vec(),
            },
        );
        self.evict_to_capacity(&handle);
        handle
    }

    fn evict_to_capacity(&mut self, keep: &str) {
        let Some(cap) = self.capacity else { return };
        while self.used > cap {
            let victim = self
                .blobs
                .values()
                .filter(|b| b.info.refs == 0 && b.info.handle != keep)
                .min_by_key(|b| b.info.last_access_ms)
                .map(|b| b.info.handle.clone());
            match victim {
                Some(h) => {
                    if let Some(b) = self.blobs.remove(&h) {
                        self.used -= b.info.size;
                        self.evictions += 1;
                    }
                }
                None => break,
            }
        }
    }

    fn blob(&mut self, handle: &str) -> Result<&Blob> {
        let tick = self.touch();
        match self.blobs.get_mut(handle) {
            Some(b) => {
                b.info.last_access_ms = tick;
                Ok(b)
            }
            None => err(ErrorKind::NotFound, format!("unknown or evicted handle '{handle}'")),
        }
    }

    pub fn get(&mut self, handle: &str) -> Result<Vec<u8>> {
        Ok(self.blob(handle)?.data.clone())
    }

    pub fn text(&mut self, handle: &str) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.blob(handle)?.data).into_owned())
    }

    /// Byte range `[start, end)`, clamped to the blob size.
    pub fn slice(&mut self, handle: &str, start: u64, end: Option<u64>) -> Result<Vec<u8>> {
        let b = self.blob(handle)?;
        let len = b.data.len();
        let s = (start as usize).min(len);
        let e = end.map(|e| (e as usize).min(len)).unwrap_or(len).max(s);
        Ok(b.data[s..e].to_vec())
    }

    /// Lines `start..=end` (1-based, inclusive), clamped.
    pub fn lines(&mut self, handle: &str, start: u64, end: Option<u64>) -> Result<String> {
        if start == 0 {
            return err(ErrorKind::Invalid, "line numbers are 1-based");
        }
        let text = self.text(handle)?;
        let end = end.unwrap_or(u64::MAX);
        Ok(text
            .lines()
            .enumerate()
            .filter(|(i, _)| (*i as u64 + 1) >= start && (*i as u64 + 1) <= end)
            .map(|(_, l)| l)
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// Lines containing `needle` (case-insensitive), so an agent can search a
    /// blob without loading it.
    pub fn grep(&mut self, handle: &str, needle: &str, max_results: usize) -> Result<Vec<GrepHit>> {
        let text = self.text(handle)?;
        let n = needle.to_lowercase();
        Ok(text
            .lines()
            .enumerate()
            .filter(|(_, l)| l.to_lowercase().contains(&n))
            .take(max_results)
            .map(|(i, l)| GrepHit {
                line: i as u64 + 1,
                text: util::truncate(l, 400),
            })
            .collect())
    }

    pub fn info(&self, handle: &str) -> Result<BlobInfo> {
        self.blobs
            .get(handle)
            .map(|b| b.info.clone())
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("unknown or evicted handle '{handle}'")))
    }

    /// One line describing the blob, for the model's context.
    pub fn describe(&mut self, handle: &str, preview_chars: usize) -> Result<String> {
        let info = self.info(handle)?;
        let preview = util::truncate(&self.text(handle)?.replace(['\n', '\r'], " ⏎ "), preview_chars);
        let label = if info.label.is_empty() {
            String::new()
        } else {
            format!(" '{}'", info.label)
        };
        Ok(format!(
            "{}{label} ({}, {}, {} lines) preview: {preview}",
            info.handle,
            info.mime,
            util::fmt_bytes(info.size),
            info.lines
        ))
    }

    pub fn retain(&mut self, handle: &str) -> Result<u32> {
        let tick = self.touch();
        let b = self
            .blobs
            .get_mut(handle)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("unknown or evicted handle '{handle}'")))?;
        b.info.refs += 1;
        b.info.last_access_ms = tick;
        Ok(b.info.refs)
    }

    /// Drop a reference. At zero references the blob becomes evictable.
    pub fn release(&mut self, handle: &str) -> Result<u32> {
        let b = self
            .blobs
            .get_mut(handle)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("unknown or evicted handle '{handle}'")))?;
        b.info.refs = b.info.refs.saturating_sub(1);
        let refs = b.info.refs;
        self.evict_to_capacity("");
        Ok(refs)
    }

    /// Remove a blob immediately, whatever its reference count.
    pub fn delete(&mut self, handle: &str) -> Result<()> {
        match self.blobs.remove(handle) {
            Some(b) => {
                self.used -= b.info.size;
                Ok(())
            }
            None => err(ErrorKind::NotFound, format!("unknown or evicted handle '{handle}'")),
        }
    }

    pub fn handles(&self) -> Vec<String> {
        self.blobs.keys().cloned().collect()
    }

    pub fn stats(&self) -> StoreStats {
        StoreStats {
            blobs: self.blobs.len(),
            bytes: self.used,
            capacity_bytes: self.capacity,
            evictions: self.evictions,
            dedup_hits: self.dedup_hits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_slices_and_grep() {
        let mut s = BlobStore::new(None);
        let csv = "id,name\n1,alice\n2,bob\n3,carol\n";
        let h = s.put(csv.as_bytes(), "text/csv", "users");
        assert!(h.starts_with("blob:") && h.len() == 21);
        assert_eq!(s.put(csv.as_bytes(), "text/csv", "users"), h);
        assert_eq!(s.info(&h).unwrap().refs, 2);
        assert_eq!(s.info(&h).unwrap().lines, 4);
        assert_eq!(s.lines(&h, 2, Some(3)).unwrap(), "1,alice\n2,bob");
        assert_eq!(s.grep(&h, "BOB", 10).unwrap()[0].line, 3);
        assert_eq!(s.slice(&h, 0, Some(2)).unwrap(), b"id");
        assert!(s.describe(&h, 40).unwrap().contains("'users' (text/csv"));
    }

    #[test]
    fn lru_eviction_of_unreferenced_blobs() {
        let mut s = BlobStore::new(Some(10));
        let a = s.put(b"aaaaa", "x", "");
        let b = s.put(b"bbbbb", "x", "");
        s.release(&a).unwrap();
        s.release(&b).unwrap();
        s.get(&a).unwrap(); // a is now more recently used than b
        let c = s.put(b"ccccc", "x", "");
        assert!(s.info(&b).is_err());
        assert!(s.info(&a).is_ok() && s.info(&c).is_ok());
        assert_eq!(s.stats().evictions, 1);
    }
}
