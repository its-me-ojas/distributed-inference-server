// type aliases for the inference server.

use serde::Deserialize;
use uuid::Uuid;

// unique identifier for an inference request
pub type RequestID = Uuid;
// batches group multiple requests for efficient GPU processing
pub type BatchId = Uuid;
// simple u32 since workers are local to a single server instance
pub type WorkerId = Uuid;
// the key is the token sequence, requests with same prefix can share cache
pub type CacheKey = Vec<u32>;

// priority levels for request scheduling
// higher priority requests are processed before lower priority ones
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}
