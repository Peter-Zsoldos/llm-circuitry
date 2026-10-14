//! # llm-circuitry
//!
//! If an LLM is a CPU, its context window is RAM, retrieval is the disk and the
//! weights are ROM, then a lot of the *other* hardware a real computer relies on
//! is still missing. This crate implements those missing parts as small,
//! dependency-light building blocks:
//!
//! | Module        | Hardware analogy                    | What it gives an LLM system                          |
//! |---------------|-------------------------------------|------------------------------------------------------|
//! | [`protect`]   | MMU, privilege rings, NX bit        | Privilege-tagged context, taint tracking, injection scanning |
//! | [`txn`]       | Journaling filesystem               | Write-ahead journal, commit/rollback of real-world actions |
//! | [`coherence`] | Cache coherence (MESI)              | Versioned shared state, stale-read detection between agents |
//! | [`interrupt`] | Interrupt controller, watchdog, RTC | Prioritised interrupts, loop/stall detection, a sense of time |
//! | [`stack`]     | Program counter and call stack      | Explicit goal stack that survives context loss       |
//! | [`ecc`]       | ECC memory                          | Verification of cited quotes and numbers against sources |
//! | [`dma`]       | DMA                                 | Pass-by-reference blob handles instead of copying through context |
//! | [`address`]   | Physical addressing                 | Exact line addressing with edit-stable anchors       |
//! | [`pager`]     | Virtual memory / page replacement   | CLOCK paging of context under a token budget         |
//! | [`attest`]    | TPM, secure boot, ACLs              | Signed instructions, capability tokens, tamper-evident audit log |
//! | [`replay`]    | JTAG / deterministic replay         | Record and replay of model calls                     |
//!
//! Everything is plain Rust with `serde` types. Enable the `python` feature to
//! build the `llm_circuitry` Python extension module.

pub mod address;
pub mod attest;
pub mod coherence;
pub mod dma;
pub mod ecc;
pub mod error;
pub mod interrupt;
pub mod pager;
pub mod protect;
pub mod replay;
pub mod stack;
pub mod txn;
pub(crate) mod util;

pub use error::{Error, ErrorKind, Result};

#[cfg(feature = "python")]
mod python;
