//! **Exact addressing with edit-stable anchors.**
//!
//! RAM is addressed exactly: address 0x4F2A returns the same byte every time.
//! LLM memory is addressed by fuzzy content ("the part about pricing"), and
//! line numbers the model saw earlier go stale as soon as the file is edited.
//!
//! [`AddressableDoc`] gives exact, 1-based line addressing plus *anchors* like
//! `L42#9f3a1c2b`: the line number together with a hash of that line's
//! content. After edits, [`resolve`](AddressableDoc::resolve) finds where the
//! anchored line moved to, or reports that it no longer exists, instead of
//! silently pointing at the wrong line.

use crate::error::{err, Error, ErrorKind, Result};
use crate::util;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Location {
    pub line: usize,
    /// 1-based character column.
    pub column: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Resolved {
    pub line: usize,
    pub moved: bool,
    pub text: String,
    /// The anchor for the line's current position.
    pub anchor: String,
}

#[derive(Clone, Debug, Default)]
pub struct AddressableDoc {
    lines: Vec<String>,
    hashes: Vec<String>,
}

fn line_hash(line: &str) -> String {
    util::sha256_hex(line.trim().as_bytes())[..8].to_string()
}

impl AddressableDoc {
    pub fn new(text: &str) -> Self {
        let mut d = AddressableDoc::default();
        d.update(text);
        d
    }

    /// Replace the content. Anchors made earlier can still be resolved.
    pub fn update(&mut self, text: &str) {
        self.lines = text.lines().map(String::from).collect();
        self.hashes = self.lines.iter().map(|l| line_hash(l)).collect();
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    fn check(&self, n: usize) -> Result<()> {
        if n == 0 || n > self.lines.len() {
            return err(
                ErrorKind::NotFound,
                format!("line {n} is out of range 1..={}", self.lines.len()),
            );
        }
        Ok(())
    }

    pub fn line(&self, n: usize) -> Result<&str> {
        self.check(n)?;
        Ok(&self.lines[n - 1])
    }

    /// Lines `start..=end` joined with newlines.
    pub fn range(&self, start: usize, end: usize) -> Result<String> {
        self.check(start)?;
        let end = end.min(self.lines.len());
        if end < start {
            return err(ErrorKind::Invalid, format!("empty range {start}..={end}"));
        }
        Ok(self.lines[start - 1..end].join("\n"))
    }

    /// Lines with right-aligned numbers, the way a model should see a file it may edit.
    pub fn numbered(&self, start: usize, end: Option<usize>) -> Result<String> {
        if self.lines.is_empty() {
            return Ok(String::new());
        }
        self.check(start)?;
        let end = end.unwrap_or(self.lines.len()).min(self.lines.len());
        let width = end.to_string().len();
        Ok((start..=end)
            .map(|n| format!("{n:>width$} | {}", self.lines[n - 1]))
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// Every exact occurrence of `needle`.
    pub fn find(&self, needle: &str) -> Vec<Location> {
        if needle.is_empty() {
            return vec![];
        }
        let mut out = Vec::new();
        for (i, l) in self.lines.iter().enumerate() {
            for (byte, _) in l.match_indices(needle) {
                out.push(Location {
                    line: i + 1,
                    column: l[..byte].chars().count() + 1,
                });
            }
        }
        out
    }

    pub fn anchor(&self, n: usize) -> Result<String> {
        self.check(n)?;
        Ok(format!("L{n}#{}", self.hashes[n - 1]))
    }

    /// Find the line an anchor refers to. If its content moved, the nearest line
    /// with the same content is returned; if it was deleted or changed, `NotFound`.
    pub fn resolve(&self, anchor: &str) -> Result<Resolved> {
        let bad = || {
            Error::new(
                ErrorKind::Invalid,
                format!("malformed anchor '{anchor}' (expected L<line>#<hash>)"),
            )
        };
        let rest = anchor.strip_prefix('L').ok_or_else(bad)?;
        let (num, hash) = rest.split_once('#').ok_or_else(bad)?;
        let n: usize = num.parse().map_err(|_| bad())?;
        let best = self
            .hashes
            .iter()
            .enumerate()
            .filter(|(_, h)| h.as_str() == hash)
            .map(|(i, _)| i + 1)
            .min_by_key(|line| line.abs_diff(n));
        match best {
            Some(line) => Ok(Resolved {
                line,
                moved: line != n,
                text: self.lines[line - 1].clone(),
                anchor: format!("L{line}#{hash}"),
            }),
            None => err(
                ErrorKind::NotFound,
                format!("anchor {anchor}: that line was changed or deleted"),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_survive_edits() {
        let mut d = AddressableDoc::new("fn main() {\n    let x = 1;\n    println!(\"{x}\");\n}");
        let a = d.anchor(3).unwrap();
        assert_eq!(d.resolve(&a).unwrap().line, 3);
        d.update("// header\n// more\nfn main() {\n    let x = 1;\n    println!(\"{x}\");\n}");
        let r = d.resolve(&a).unwrap();
        assert_eq!((r.line, r.moved), (5, true));
        d.update("fn main() {}");
        assert_eq!(d.resolve(&a).unwrap_err().kind, ErrorKind::NotFound);
    }

    #[test]
    fn addressing() {
        let d = AddressableDoc::new("alpha\nbeta gamma beta\ndelta");
        assert_eq!(d.line(2).unwrap(), "beta gamma beta");
        assert!(d.line(4).is_err());
        assert_eq!(
            d.find("beta"),
            vec![Location { line: 2, column: 1 }, Location { line: 2, column: 12 }]
        );
        assert_eq!(d.numbered(2, Some(3)).unwrap(), "2 | beta gamma beta\n3 | delta");
        assert_eq!(d.range(1, 2).unwrap(), "alpha\nbeta gamma beta");
    }
}
