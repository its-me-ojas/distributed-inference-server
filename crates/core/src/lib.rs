// core types and models for the inference server.

pub mod error;
pub mod models;
pub mod types;
pub mod validator;

// commonly used types
pub use error::{ApiError, ServerError, ValidationError};
pub use types::{BatchId, CacheKey, Priority, RequestID, WorkerId};
pub use validator::{RequestValidator, Validated, ValidatorConfig};

// models
pub use models::{
    ChatMessage, ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse, ErrorResponse,
    FinishReason, GenerateRequest, GenerateResponse, Role, TokenEvent, Usage,
};
