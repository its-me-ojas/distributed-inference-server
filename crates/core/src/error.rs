// error types for the inference server.

use thiserror::Error;

// top-level server errors (internal, not exposed to clients)
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model load error: {0}")]
    ModelLoad(String),

    #[error("Worker error: {0}")]
    Worker(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// API-level errors (returned to client as HTTP responses)
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("Queue full , server is overloaded")]
    QueueFull,

    #[error("Request timeout")]
    Timeout,

    #[error("Internal server error: {0}")]
    Internal(String),
}

impl ApiError {
    // returns the HTTPS status code for this error
    pub fn status_code(&self) -> u16 {
        match self {
            ApiError::Validation(_) => 400,
            ApiError::QueueFull => 503,
            ApiError::Timeout => 408,
            ApiError::Internal(_) => 500,
        }
    }

    // returns the error type string for the API response
    pub fn error_type(&self) -> &'static str {
        match self {
            ApiError::Validation(_) => "invalid_request_error",
            ApiError::QueueFull => "rate_limit_error",
            ApiError::Timeout => "timeout_error",
            ApiError::Internal(_) => "server_error",
        }
    }
}

// Validation error for incoming requests
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Invalid JSON: {0}")]
    InvalidJson(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Token limit exceeded: {actual} tokens > {limit} max")]
    TokenLimitExceeded { actual: usize, limit: usize },

    #[error("Invalid parameter '{field}': {reason}")]
    InvalidParameter { field: String, reason: String },

    #[error("Empty prompt not allowed")]
    EmptyPrompt,
}

// errors from the queue manager
#[derive(Debug, Error)]
pub enum QueueError {
    #[error("Queue is full")]
    Full,

    #[error("Request not found: {0}")]
    NotFound(String),

    #[error("Request cancelled")]
    Cancelled,
}

// error from the batcher
#[derive(Debug, Error)]
pub enum BatcherError {
    #[error("Batch timeout")]
    Timeout,

    #[error("Channel closed")]
    ChannelClosed,
}

// errors from the cache
#[derive(Debug, Error)]
pub enum CacheError {
    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Cache full")]
    Full,
}

// errors from scheduler
#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Worker shutdown")]
    Shutdown,

    #[error("Out of memory")]
    OutOfMemory,
}

// errors from the token streamer
#[derive(Debug, Error)]
pub enum StreamError {
    #[error("Client disconnected")]
    ClientDisconnected,

    #[error("Stream not found: {0}")]
    NotFound(String),

    #[error("Send failed")]
    SendFailed,
}
