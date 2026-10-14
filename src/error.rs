use std::fmt;

/// Broad category of an [`Error`]. Each kind maps to its own Python exception class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// A referenced id, key, handle or anchor does not exist.
    NotFound,
    /// The caller acted on data that has since changed.
    Stale,
    /// Another agent holds a conflicting lock.
    Conflict,
    /// A capacity limit (stack depth, token budget) would be exceeded.
    Capacity,
    /// The operation is not valid in the object's current state.
    InvalidState,
    /// A signature, hash chain or checksum does not verify.
    Integrity,
    /// A permission or capability check failed.
    Denied,
    /// An argument was malformed.
    Invalid,
    /// An I/O error while reading or writing a journal file.
    Io,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::new(ErrorKind::Io, e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::new(ErrorKind::Invalid, e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(kind: ErrorKind, message: impl Into<String>) -> Result<T> {
    Err(Error::new(kind, message))
}
