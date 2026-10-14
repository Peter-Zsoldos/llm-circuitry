//! **Virtual memory for the context window.**
//!
//! An OS keeps more memory in use than physically fits by paging: pages that
//! haven't been used recently are written to disk and brought back on access
//! (a page fault). The CLOCK algorithm approximates least-recently-used cheaply
//! with one "referenced" bit per page.
//!
//! [`Pager`] does the same for context under a token budget: load pages
//! (documents, tool results, memories), mark them accessed, pin the ones that
//! must stay, and it evicts the rest with CLOCK when the budget is exceeded.
//! Evicted pages go to a swap area and fault back in on access. Modified
//! ("dirty") pages are reported so they can be written back to long-term memory
//! before they're forgotten.

use crate::error::{err, Error, ErrorKind, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Page {
    pub id: String,
    pub content: String,
    pub tokens: u64,
    pub pinned: bool,
    pub referenced: bool,
    pub dirty: bool,
    pub faults: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evicted {
    pub id: String,
    pub tokens: u64,
    /// The page was modified since it was last written back.
    pub dirty: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Access {
    pub content: String,
    /// The page was swapped out and had to be loaded back in.
    pub fault: bool,
    pub evicted: Vec<Evicted>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PagerStats {
    pub budget: u64,
    pub used: u64,
    pub resident: usize,
    pub swapped: usize,
    pub loads: u64,
    pub faults: u64,
    pub evictions: u64,
}

/// Rough token estimate (~4 characters per token) when no count is given.
pub fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4).max(1)
}

#[derive(Debug, Default)]
pub struct Pager {
    resident: Vec<Page>,
    swapped: BTreeMap<String, Page>,
    hand: usize,
    budget: u64,
    loads: u64,
    faults: u64,
    evictions: u64,
}

impl Pager {
    pub fn new(budget_tokens: u64) -> Self {
        Pager {
            budget: budget_tokens,
            ..Default::default()
        }
    }

    pub fn used(&self) -> u64 {
        self.resident.iter().map(|p| p.tokens).sum()
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    pub fn set_budget(&mut self, budget_tokens: u64) -> Vec<Evicted> {
        self.budget = budget_tokens;
        self.make_room("")
    }

    fn pos(&self, id: &str) -> Option<usize> {
        self.resident.iter().position(|p| p.id == id)
    }

    fn pinned_tokens(&self, except: &str) -> u64 {
        self.resident
            .iter()
            .filter(|p| p.pinned && p.id != except)
            .map(|p| p.tokens)
            .sum()
    }

    /// Load (or replace) a page. Replacing an existing page marks it dirty.
    pub fn load(&mut self, id: &str, content: &str, tokens: Option<u64>, pinned: bool) -> Result<Vec<Evicted>> {
        let tokens = tokens.unwrap_or_else(|| estimate_tokens(content));
        if tokens + self.pinned_tokens(id) > self.budget {
            return err(
                ErrorKind::Capacity,
                format!(
                    "page '{id}' ({tokens} tokens) does not fit: budget {}, pinned {}",
                    self.budget,
                    self.pinned_tokens(id)
                ),
            );
        }
        let existed = self.drop_page(id).ok();
        let dirty = existed.is_some();
        let faults = existed.map(|p| p.faults).unwrap_or(0);
        self.resident.push(Page {
            id: id.into(),
            content: content.into(),
            tokens,
            pinned,
            referenced: true,
            dirty,
            faults,
        });
        self.loads += 1;
        Ok(self.make_room(id))
    }

    /// Use a page. If it was swapped out this is a page fault and it is brought back.
    pub fn access(&mut self, id: &str) -> Result<Access> {
        if let Some(i) = self.pos(id) {
            self.resident[i].referenced = true;
            return Ok(Access {
                content: self.resident[i].content.clone(),
                fault: false,
                evicted: vec![],
            });
        }
        let mut page = self
            .swapped
            .remove(id)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("no page '{id}'")))?;
        if page.tokens + self.pinned_tokens(id) > self.budget {
            let msg = format!(
                "page '{id}' ({} tokens) cannot be faulted in: budget {}",
                page.tokens, self.budget
            );
            self.swapped.insert(id.into(), page);
            return err(ErrorKind::Capacity, msg);
        }
        page.referenced = true;
        page.faults += 1;
        self.faults += 1;
        let content = page.content.clone();
        self.resident.push(page);
        let evicted = self.make_room(id);
        Ok(Access {
            content,
            fault: true,
            evicted,
        })
    }

    /// Change a page's content and mark it dirty.
    pub fn write(&mut self, id: &str, content: &str, tokens: Option<u64>) -> Result<Vec<Evicted>> {
        let pinned = self.get(id)?.pinned;
        let faults = self.get(id)?.faults;
        let ev = self.load(id, content, tokens, pinned)?;
        if let Some(i) = self.pos(id) {
            self.resident[i].dirty = true;
            self.resident[i].faults = faults;
        }
        Ok(ev)
    }

    fn page_mut(&mut self, id: &str) -> Result<&mut Page> {
        if let Some(i) = self.pos(id) {
            return Ok(&mut self.resident[i]);
        }
        self.swapped
            .get_mut(id)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("no page '{id}'")))
    }

    pub fn get(&self, id: &str) -> Result<&Page> {
        self.resident
            .iter()
            .find(|p| p.id == id)
            .or_else(|| self.swapped.get(id))
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("no page '{id}'")))
    }

    pub fn pin(&mut self, id: &str) -> Result<()> {
        if self.pos(id).is_none() {
            self.access(id)?;
        }
        self.page_mut(id)?.pinned = true;
        Ok(())
    }

    pub fn unpin(&mut self, id: &str) -> Result<()> {
        self.page_mut(id)?.pinned = false;
        Ok(())
    }

    /// Swap a page out now.
    pub fn evict(&mut self, id: &str) -> Result<Evicted> {
        let i = self
            .pos(id)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("page '{id}' is not resident")))?;
        Ok(self.evict_at(i))
    }

    /// Remove a page entirely (resident or swapped).
    pub fn drop_page(&mut self, id: &str) -> Result<Page> {
        if let Some(i) = self.pos(id) {
            if self.hand > i {
                self.hand -= 1;
            }
            return Ok(self.resident.remove(i));
        }
        self.swapped
            .remove(id)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("no page '{id}'")))
    }

    fn evict_at(&mut self, i: usize) -> Evicted {
        let mut p = self.resident.remove(i);
        if self.hand > i {
            self.hand -= 1;
        }
        p.referenced = false;
        p.pinned = false;
        self.evictions += 1;
        let ev = Evicted {
            id: p.id.clone(),
            tokens: p.tokens,
            dirty: p.dirty,
        };
        self.swapped.insert(p.id.clone(), p);
        ev
    }

    /// CLOCK: sweep the hand; referenced pages get a second chance, pinned pages
    /// and `protect` are skipped, everything else is evicted until we fit.
    fn make_room(&mut self, protect: &str) -> Vec<Evicted> {
        let mut out = Vec::new();
        let mut idle_moves = 0usize;
        while self.used() > self.budget && !self.resident.is_empty() {
            if idle_moves > 2 * self.resident.len() + 2 {
                break;
            }
            if self.hand >= self.resident.len() {
                self.hand = 0;
            }
            let p = &mut self.resident[self.hand];
            if p.pinned || p.id == protect {
                self.hand += 1;
                idle_moves += 1;
            } else if p.referenced {
                p.referenced = false;
                self.hand += 1;
                idle_moves += 1;
            } else {
                let i = self.hand;
                out.push(self.evict_at(i));
                idle_moves = 0;
            }
        }
        out
    }

    pub fn resident(&self) -> Vec<String> {
        self.resident.iter().map(|p| p.id.clone()).collect()
    }

    pub fn swapped(&self) -> Vec<String> {
        self.swapped.keys().cloned().collect()
    }

    /// Pages modified since their last write-back.
    pub fn dirty(&self) -> Vec<String> {
        self.resident
            .iter()
            .chain(self.swapped.values())
            .filter(|p| p.dirty)
            .map(|p| p.id.clone())
            .collect()
    }

    /// Mark a page as written back to long-term storage.
    pub fn clean(&mut self, id: &str) -> Result<()> {
        self.page_mut(id)?.dirty = false;
        Ok(())
    }

    /// Resident pages as one context string, plus an index of swapped pages the
    /// model can ask for by id.
    pub fn render(&self) -> String {
        let mut out: Vec<String> = self
            .resident
            .iter()
            .map(|p| format!("### [page {}]\n{}", p.id, p.content))
            .collect();
        if !self.swapped.is_empty() {
            let ids: Vec<&str> = self.swapped.keys().map(String::as_str).collect();
            out.push(format!("### [swapped out, request by id to load]\n{}", ids.join(", ")));
        }
        out.join("\n\n")
    }

    pub fn stats(&self) -> PagerStats {
        PagerStats {
            budget: self.budget,
            used: self.used(),
            resident: self.resident.len(),
            swapped: self.swapped.len(),
            loads: self.loads,
            faults: self.faults,
            evictions: self.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_eviction_and_faults() {
        let mut p = Pager::new(30);
        p.load("system", "rules", Some(10), true).unwrap();
        p.load("a", "doc a", Some(10), false).unwrap();
        p.load("b", "doc b", Some(10), false).unwrap();
        let ev = p.load("c", "doc c", Some(10), false).unwrap();
        assert_eq!(ev.len(), 1);
        assert!(p.resident().contains(&"system".to_string()));
        assert_eq!(p.used(), 30);

        let gone = ev[0].id.clone();
        let acc = p.access(&gone).unwrap();
        assert!(acc.fault);
        assert_eq!(acc.evicted.len(), 1);
        assert_eq!(p.stats().faults, 1);
        assert!(p.render().contains("[swapped out"));
    }

    #[test]
    fn dirty_writeback_and_capacity() {
        let mut p = Pager::new(20);
        p.load("notes", "v1", Some(5), false).unwrap();
        p.write("notes", "v2", Some(5)).unwrap();
        assert_eq!(p.dirty(), vec!["notes".to_string()]);
        let ev = p.evict("notes").unwrap();
        assert!(ev.dirty);
        p.clean("notes").unwrap();
        assert!(p.dirty().is_empty());
        assert_eq!(
            p.load("huge", "x", Some(50), false).unwrap_err().kind,
            ErrorKind::Capacity
        );
    }
}
