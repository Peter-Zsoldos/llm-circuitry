//! **ECC memory for model outputs.**
//!
//! ECC RAM checks every word it reads and flags corruption. LLM output has no
//! such check: a hallucinated figure looks exactly like a correct one.
//!
//! [`Verifier`] holds the source documents an answer was supposed to come from
//! and checks each cited quote against them: verbatim, after normalisation
//! (case, punctuation, whitespace), or by fuzzy token overlap. It also flags
//! numbers in a claim that appear nowhere in the source, the most common kind
//! of quiet hallucination.
//!
//! [`Verifier::check_cited`] parses inline citations of the form
//! `[source_id: "exact quote"]`, so you can ask the model to cite that way and
//! check the whole answer in one call.

use crate::util;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    Exact,
    Normalized,
    Fuzzy,
    NotFound,
    UnknownSource,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaimCheck {
    pub source_id: String,
    pub quote: String,
    pub claim: Option<String>,
    pub status: MatchKind,
    /// 1.0 for exact or normalized matches, token overlap for fuzzy ones.
    pub score: f64,
    /// Byte span of an exact match in the source.
    pub span: Option<[usize; 2]>,
    /// Closest passage in the source, for fuzzy matches.
    pub matched_text: Option<String>,
    /// Numbers in the claim or quote that don't appear anywhere in the source.
    pub unsupported_numbers: Vec<String>,
    pub ok: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub checks: Vec<ClaimCheck>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
}

#[derive(Debug, Clone)]
pub struct Verifier {
    sources: BTreeMap<String, String>,
    fuzzy_threshold: f64,
}

/// Numbers as normalised strings ("1,200.50" -> "1200.50").
fn extract_numbers(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let mut num = String::new();
            while i < chars.len() {
                let c = chars[i];
                let sep_between_digits =
                    (c == ',' || c == '.') && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() && !num.is_empty();
                if c.is_ascii_digit() {
                    num.push(c);
                } else if sep_between_digits {
                    if c == '.' {
                        num.push('.');
                    }
                } else {
                    break;
                }
                i += 1;
            }
            out.push(num);
        } else {
            i += 1;
        }
    }
    out
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new(0.8)
    }
}

impl Verifier {
    pub fn new(fuzzy_threshold: f64) -> Self {
        Verifier {
            sources: BTreeMap::new(),
            fuzzy_threshold,
        }
    }

    pub fn add_source(&mut self, id: &str, text: &str) {
        self.sources.insert(id.into(), text.into());
    }

    pub fn remove_source(&mut self, id: &str) -> bool {
        self.sources.remove(id).is_some()
    }

    pub fn sources(&self) -> Vec<String> {
        self.sources.keys().cloned().collect()
    }

    /// Check one quote against one source. If `claim` is given, its numbers are
    /// also checked against the source.
    pub fn check(&self, source_id: &str, quote: &str, claim: Option<&str>) -> ClaimCheck {
        let mut c = ClaimCheck {
            source_id: source_id.into(),
            quote: quote.into(),
            claim: claim.map(String::from),
            status: MatchKind::NotFound,
            score: 0.0,
            span: None,
            matched_text: None,
            unsupported_numbers: vec![],
            ok: false,
        };
        let Some(src) = self.sources.get(source_id) else {
            c.status = MatchKind::UnknownSource;
            return c;
        };
        if let Some(pos) = src.find(quote).filter(|_| !quote.is_empty()) {
            c.status = MatchKind::Exact;
            c.score = 1.0;
            c.span = Some([pos, pos + quote.len()]);
        } else {
            let nq = util::normalize(quote);
            if !nq.is_empty() && util::normalize(src).contains(&nq) {
                c.status = MatchKind::Normalized;
                c.score = 1.0;
            } else {
                let (score, text) = fuzzy_best(src, &nq);
                c.score = score;
                c.matched_text = text;
                if score >= self.fuzzy_threshold {
                    c.status = MatchKind::Fuzzy;
                }
            }
        }
        let src_numbers: HashSet<String> = extract_numbers(src).into_iter().collect();
        let mut seen = HashSet::new();
        for n in extract_numbers(quote)
            .into_iter()
            .chain(extract_numbers(claim.unwrap_or("")))
        {
            if !src_numbers.contains(&n) && seen.insert(n.clone()) {
                c.unsupported_numbers.push(n);
            }
        }
        c.ok =
            c.status != MatchKind::NotFound && c.status != MatchKind::UnknownSource && c.unsupported_numbers.is_empty();
        c
    }

    /// Parse `[source_id: "quote"]` citations in `text` and check each one. The
    /// claim for a citation is the text since the previous sentence end or citation.
    pub fn check_cited(&self, text: &str) -> Report {
        let checks: Vec<ClaimCheck> = parse_citations(text)
            .into_iter()
            .map(|(id, quote, claim)| self.check(&id, &quote, Some(&claim)))
            .collect();
        let passed = checks.iter().filter(|c| c.ok).count();
        Report {
            total: checks.len(),
            passed,
            failed: checks.len() - passed,
            checks,
        }
    }
}

/// Best window of the source by multiset token overlap with the normalised quote.
fn fuzzy_best(src: &str, norm_quote: &str) -> (f64, Option<String>) {
    let q: Vec<&str> = norm_quote.split(' ').filter(|t| !t.is_empty()).collect();
    if q.is_empty() {
        return (0.0, None);
    }
    let norm_src = util::normalize(src);
    let s: Vec<&str> = norm_src.split(' ').filter(|t| !t.is_empty()).collect();
    if s.is_empty() {
        return (0.0, None);
    }
    let n = q.len().min(s.len());
    let mut want: HashMap<&str, i32> = HashMap::new();
    for &t in &q {
        *want.entry(t).or_default() += 1;
    }
    let mut have: HashMap<&str, i32> = HashMap::new();
    let mut overlap = 0i32;
    let (mut best, mut best_at) = (0i32, 0usize);
    for (i, &t) in s.iter().enumerate() {
        let w = *want.get(t).unwrap_or(&0);
        let h = have.entry(t).or_default();
        if *h < w {
            overlap += 1;
        }
        *h += 1;
        if i >= n {
            let old = s[i - n];
            let w = *want.get(old).unwrap_or(&0);
            let h = have.entry(old).or_default();
            *h -= 1;
            if *h < w {
                overlap -= 1;
            }
        }
        if i + 1 >= n && overlap > best {
            best = overlap;
            best_at = i + 1 - n;
        }
    }
    let text = s[best_at..best_at + n].join(" ");
    (best as f64 / q.len() as f64, Some(text))
}

/// (source_id, quote, claim) triples from `[id: "quote"]` citations.
fn parse_citations(text: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut claim_start = 0usize;
    let mut i = 0usize;
    while let Some(rel) = text[i..].find('[') {
        let open = i + rel;
        i = open + 1;
        let rest = &text[open + 1..];
        let Some(colon) = rest.find(':') else { break };
        let id = rest[..colon].trim();
        if id.is_empty() || id.len() > 80 || id.contains(char::is_whitespace) || id.contains(['[', ']']) {
            continue;
        }
        let after = rest[colon + 1..].trim_start();
        let (open_q, close_q) = if after.starts_with('"') {
            ('"', '"')
        } else if after.starts_with('\u{201C}') {
            ('\u{201C}', '\u{201D}')
        } else {
            continue;
        };
        let body = &after[open_q.len_utf8()..];
        let Some(end) = body.find(close_q) else { continue };
        let quote = &body[..end];
        let tail = body[end + close_q.len_utf8()..].trim_start();
        if !tail.starts_with(']') {
            continue;
        }
        let close_idx = text.len() - tail.len() + 1;
        let before = &text[claim_start..open];
        let sentence_start = before
            .rfind(['.', '!', '?', '\n'])
            .map(|p| p + 1)
            .filter(|p| !before[*p..].trim().is_empty())
            .unwrap_or(0);
        let claim = before[sentence_start..].trim().to_string();
        out.push((id.to_string(), quote.to_string(), claim));
        claim_start = close_idx;
        i = close_idx;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verifier() -> Verifier {
        let mut v = Verifier::default();
        v.add_source(
            "q3",
            "In the third quarter, revenue increased 12% to $4,200,000. Headcount was flat at 85 employees.",
        );
        v
    }

    #[test]
    fn match_levels() {
        let v = verifier();
        assert_eq!(v.check("q3", "revenue increased 12%", None).status, MatchKind::Exact);
        assert_eq!(
            v.check("q3", "Revenue  INCREASED 12 %", None).status,
            MatchKind::Normalized
        );
        let f = v.check("q3", "headcount was flat at about 85 employees", None);
        assert_eq!(f.status, MatchKind::Fuzzy, "{f:?}");
        assert_eq!(
            v.check("q3", "profits doubled overnight", None).status,
            MatchKind::NotFound
        );
        assert_eq!(v.check("nope", "x", None).status, MatchKind::UnknownSource);
    }

    #[test]
    fn hallucinated_numbers_are_flagged() {
        let v = verifier();
        let c = v.check("q3", "revenue increased 12%", Some("Revenue grew 15% to $4.2M"));
        assert!(!c.ok);
        assert!(c.unsupported_numbers.contains(&"15".to_string()));
        assert!(
            v.check("q3", "revenue increased 12%", Some("Revenue grew 12% to $4,200,000"))
                .ok
        );
    }

    #[test]
    fn inline_citations() {
        let v = verifier();
        let answer = "Revenue rose 12% [q3: \"revenue increased 12%\"]. Staff grew to 90 [q3: \"Headcount was flat at 85 employees\"]. \
                      Margins doubled [q3: \"margins doubled\"].";
        let r = v.check_cited(answer);
        assert_eq!(r.total, 3);
        assert_eq!(r.passed, 1);
        assert_eq!(r.checks[1].unsupported_numbers, vec!["90".to_string()]);
        assert_eq!(r.checks[1].claim.as_deref(), Some("Staff grew to 90"));
    }
}
