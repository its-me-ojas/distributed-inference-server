// core types and models for the inference server.

pub mod error;
pub mod models;
pub mod types;

// commonly used types
pub use error::{ApiError, ServerError, ValidationError};
pub use types::{BatchId, CacheKey, Priority, RequestID, WorkerId};

// models
pub use models::{
    ChatMessage, ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse, ErrorResponse,
    FinishReason, GenerateRequest, GenerateResponse, Role, TokenEvent, Usage,
};
