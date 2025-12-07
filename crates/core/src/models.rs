// request and response data models.

use serde::{Deserialize, Serialize};

use crate::Priority;

// token usage statistics returned with every query
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,     // no of tokens in input prompt
    pub completion_tokens: usize, // no of tokens in output prompt
    pub total_tokens: usize,      // total tokens = (prompt + completion)
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

// reason why generation stopped
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,         // model generated a stop token
    Length,       // reached max_token limit
    StopSequence, // hit a stop sequence
}

// chat message roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,    // system prompt (sets behaviour)
    User,      // user message
    Assistant, // model response
}

// a single message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

// ============================================================================
// Request Types
// ============================================================================

// request body for POST /generate endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct GenerateRequest {
    // input text to generate from
    pub prompt: String,

    // max tokens to generate (256)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    // sampling temperature (0.0 - 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    // top-p (nuclues) sampling threshold
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    // stop generation when these sequences are encountered
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    // whether to stream tokens are they are generated
    #[serde(default)]
    pub stream: bool,

    // request priority
    #[serde(default)]
    pub priority: Option<Priority>,
}

// request body for POST /chat endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    // conversation history
    pub messages: Vec<ChatMessage>,

    // max tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    // sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    // top-p sampling threshold
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    // stop sequence
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    // whether to stream tokens
    #[serde(default)]
    pub stream: bool,
}

// request body for POST /embeddings endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsRequest {
    // text(s) to embed , can be single string or array
    pub input: EmbeddingsInput,

    // model to use (optional, uses default if not specified)
    #[serde(default)]
    pub model: Option<String>,
}

// input for embeddings , either single text or multiple
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingsInput {
    // returns all inputs as vector
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingsInput::Single(s) => vec![s],
            EmbeddingsInput::Multiple(v) => v,
        }
    }
}

// ============================================================================
// Response Types
// ============================================================================

// response body for POST /generate endpoint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerateResponse {
    // unique response ID
    pub id: String,
    // object type (always "text_completion")
    pub object: String,
    // unix timestamp of creation
    pub created: u64,
    // model used for generation
    pub model: String,
    // generated completions
    pub choices: Vec<GenerateChoice>,
    // token usage statistics
    pub usage: Usage,
}

// a single completion choice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerateChoice {
    /// generated text.
    pub text: String,
    /// index in the choices array.
    pub index: usize,
    /// why generation stopped.
    pub finish_reason: FinishReason,
}

// response body for POST /chat endpoint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatResponse {
    // unique response ID
    pub id: String,
    // object type (always "chat.completion")
    pub object: String,
    // unix timestamp of creation
    pub created: u64,
    // model used for generation
    pub model: String,
    // generated message choices
    pub choices: Vec<ChatChoice>,
    // token usage statistics
    pub usage: Usage,
}

// a single chat completion choice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatChoice {
    // index in choices array
    pub index: usize,
    // generated message
    pub message: ChatMessage,
    // why generation stopped
    pub finish_reason: FinishReason,
}

// response body for POST /embeddings endpoint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingsResponse {
    // object type (always list)
    pub object: String,
    // embedding results
    pub data: Vec<EmbeddingData>,
    // model used for embeddings,
    pub model: String,
    // token usage statistics
    pub usage: Usage,
}

// a single embedding result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingData {
    // object type (always "embedding")
    pub object: String,
    // the embedding vector
    pub embedding: Vec<f32>,
    // index in the input array
    pub index: usize,
}

// ============================================================================
// Error Response
// ============================================================================

// error response body (returned on any error)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

// error details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorDetail {
    // human-readable error message
    pub message: String,
    // error type (eg: "invalid_request_error")
    pub error_type: String,
    // error code (eg: "invalid_json")
    pub code: String,
}

impl ErrorResponse {
    // create a new error response
    pub fn new(
        message: impl Into<String>,
        error_types: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_types.into(),
                code: code.into(),
            },
        }
    }
}

// ============================================================================
// Streaming Types
// ============================================================================

// server-sent event for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TokenEvent {
    // a generated token
    Token {
        token: String,
        index: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprob: Option<f32>,
    },
    // generation complete
    Done {
        finish_reason: FinishReason,
        usage: Usage,
    },
    // error during generation
    Error {
        messages: String,
        code: String,
    },
}

// ============================================================================
// Default Value Functions
// ============================================================================

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

// ============================================================================
// Default for Request Types (useful for testing)
// ============================================================================

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            stop_sequences: Vec::new(),
            stream: false,
            priority: None,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // generators for property testing
    fn arb_usage() -> impl Strategy<Value = Usage> {
        (0usize..10000, 0usize..10000).prop_map(|(p, c)| Usage::new(p, c))
    }

    fn arb_finish_reason() -> impl Strategy<Value = FinishReason> {
        prop_oneof![
            Just(FinishReason::Stop),
            Just(FinishReason::Length),
            Just(FinishReason::StopSequence),
        ]
    }

    fn arb_role() -> impl Strategy<Value = Role> {
        prop_oneof![Just(Role::System), Just(Role::User), Just(Role::Assistant),]
    }

    fn arb_chat_message() -> impl Strategy<Value = ChatMessage> {
        (arb_role(), "[a-zA-Z0-9 ]{0,100}")
            .prop_map(|(role, content)| ChatMessage { role, content })
    }

    fn arb_generate_choice() -> impl Strategy<Value = GenerateChoice> {
        ("[a-zA-Z0-9 ]{0,200}", 0usize..10, arb_finish_reason()).prop_map(
            |(text, index, finish_reason)| GenerateChoice {
                text,
                index,
                finish_reason,
            },
        )
    }

    fn arb_generate_response() -> impl Strategy<Value = GenerateResponse> {
        (
            "[a-zA-Z0-9-]{36}",
            "[a-zA-Z0-9_]{1,50}",
            0u64..u64::MAX,
            prop::collection::vec(arb_generate_choice(), 1..5),
            arb_usage(),
        )
            .prop_map(|(id, model, created, choices, usage)| GenerateResponse {
                id,
                object: "text_completion".to_string(),
                created,
                model,
                choices,
                usage,
            })
    }

    fn arb_chat_choice() -> impl Strategy<Value = ChatChoice> {
        (0usize..10, arb_chat_message(), arb_finish_reason()).prop_map(
            |(index, message, finish_reason)| ChatChoice {
                index,
                message,
                finish_reason,
            },
        )
    }

    fn arb_chat_response() -> impl Strategy<Value = ChatResponse> {
        (
            "[a-zA-Z0-9-]{36}",
            "[a-zA-Z0-9_]{1,50}",
            0u64..u64::MAX,
            prop::collection::vec(arb_chat_choice(), 1..5),
            arb_usage(),
        )
            .prop_map(|(id, model, created, choices, usage)| ChatResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model,
                choices,
                usage,
            })
    }

    fn arb_embedding_data() -> impl Strategy<Value = EmbeddingData> {
        (prop::collection::vec(-1.0f32..1.0f32, 1..100), 0usize..10).prop_map(
            |(embedding, index)| EmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index,
            },
        )
    }

    fn arb_embeddings_response() -> impl Strategy<Value = EmbeddingsResponse> {
        (
            "[a-zA-Z0-9_]{1,50}",
            prop::collection::vec(arb_embedding_data(), 1..5),
            arb_usage(),
        )
            .prop_map(|(model, data, usage)| EmbeddingsResponse {
                object: "list".to_string(),
                data,
                model,
                usage,
            })
    }

    fn arb_error_response() -> impl Strategy<Value = ErrorResponse> {
        ("[a-zA-Z0-9 ]{1,100}", "[a-z_]{1,30}", "[a-z_]{1,30}")
            .prop_map(|(message, error_type, code)| ErrorResponse::new(message, error_type, code))
    }

    // ========================================================================
    // Property Tests
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_generate_response_roundtrip(response in arb_generate_response()) {
            let json = serde_json::to_string(&response).expect("serialize failed");
            let parsed: GenerateResponse = serde_json::from_str(&json).expect("deserialize failed");
            prop_assert_eq!(response, parsed);
        }

        #[test]
        fn prop_chat_response_roundtrip(response in arb_chat_response()) {
            let json = serde_json::to_string(&response).expect("serialize failed");
            let parsed: ChatResponse = serde_json::from_str(&json).expect("deserialize failed");
            prop_assert_eq!(response, parsed);
        }

        #[test]
        fn prop_embeddings_response_roundtrip(response in arb_embeddings_response()) {
            let json = serde_json::to_string(&response).expect("serialize failed");
            let parsed: EmbeddingsResponse = serde_json::from_str(&json).expect("deserialize failed");
            prop_assert_eq!(response, parsed);
        }

        #[test]
        fn prop_error_response_roundtrip(response in arb_error_response()) {
            let json = serde_json::to_string(&response).expect("serialize failed");
            let parsed: ErrorResponse = serde_json::from_str(&json).expect("deserialize failed");
            prop_assert_eq!(response, parsed);
        }

    }
}
