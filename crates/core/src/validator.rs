// Request validation for the inference server
// Validates incoming requests against configurable limits and rules

use crate::{ChatRequest, EmbeddingsRequest, GenerateRequest, ValidationError};

// configuration for the request validator
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    pub max_context_tokens: usize, // max tokens allowed in a prompt (context window)
    pub max_output_tokens: usize,  // max tokens that can be generated
    pub min_temperature: f32,      // min temp value
    pub max_temperature: f32,      // max temp value
    pub min_top_p: f32,            // min top_p value
    pub max_top_p: f32,            // max top_p value
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 8192, // 8K context window
            max_output_tokens: 4096,
            min_temperature: 0.0,
            max_temperature: 2.0,
            min_top_p: 0.0,
            max_top_p: 1.0,
        }
    }
}

// wrapper for a validated request
#[derive(Debug, Clone)]
pub struct Validated<T>(pub T);

impl<T> Validated<T> {
    // unwrap the validated request
    pub fn into_inner(self) -> T {
        self.0
    }
}

// request validator that checks incoming requests against configured limits.
#[derive(Debug, Clone)]
pub struct RequestValidator {
    config: ValidatorConfig,
}

impl RequestValidator {
    // creates a new validator with the given configuration
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    // create a validator with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ValidatorConfig::default())
    }

    // count tokens in a string using simple whitespace tokenization
    // in prod, i will use tiktoken-rs or similar for accurate token counts
    pub fn token_count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        (text.len() + 3) / 4
    }

    // validate a generate request
    pub fn validate_generate(
        &self,
        request: &GenerateRequest,
    ) -> Result<Validated<GenerateRequest>, ValidationError> {
        // check for empty prompt
        if request.prompt.trim().is_empty() {
            return Err(ValidationError::EmptyPrompt);
        }

        // check token count
        let prompt_tokens = self.token_count(&request.prompt);
        if prompt_tokens > self.config.max_context_tokens {
            return Err(ValidationError::TokenLimitExceeded {
                actual: prompt_tokens,
                limit: self.config.max_context_tokens,
            });
        }

        // check max_tokens
        if request.max_tokens > self.config.max_output_tokens {
            return Err(ValidationError::InvalidParameter {
                field: "max_tokens".to_string(),
                reason: format!(
                    "must be <= {}, got {}",
                    self.config.max_output_tokens, request.max_tokens
                ),
            });
        }

        // check temperature
        if request.temperature < self.config.min_temperature
            || request.temperature > self.config.max_temperature
        {
            return Err(ValidationError::InvalidParameter {
                field: "temperature".to_string(),
                reason: format!(
                    "must be between {} and {}, got {}",
                    self.config.min_temperature, self.config.max_temperature, request.temperature
                ),
            });
        }

        // check top_p
        if request.top_p < self.config.min_top_p || request.top_p > self.config.max_top_p {
            return Err(ValidationError::InvalidParameter {
                field: "top_p".to_string(),
                reason: format!(
                    "must be between {} and {}, got {}",
                    self.config.min_top_p, self.config.max_top_p, request.top_p
                ),
            });
        }

        Ok(Validated(request.clone()))
    }

    // validate a chat request
    pub fn validate_chat(
        &self,
        request: &ChatRequest,
    ) -> Result<Validated<ChatRequest>, ValidationError> {
        // check for empty messages
        if request.messages.is_empty() {
            return Err(ValidationError::MissingField("messages".to_string()));
        }

        // check that at least one message has content
        let has_content = request
            .messages
            .iter()
            .any(|m| !m.content.trim().is_empty());
        if !has_content {
            return Err(ValidationError::EmptyPrompt);
        }

        // Count total tokens across all messages
        let total_tokens: usize = request
            .messages
            .iter()
            .map(|m| self.token_count(&m.content))
            .sum();
        if total_tokens > self.config.max_context_tokens {
            return Err(ValidationError::TokenLimitExceeded {
                actual: total_tokens,
                limit: self.config.max_context_tokens,
            });
        }

        // check max_tokens
        if request.max_tokens > self.config.max_output_tokens {
            return Err(ValidationError::InvalidParameter {
                field: "max_tokens".to_string(),
                reason: format!(
                    "must be <= {}, got {}",
                    self.config.max_output_tokens, request.max_tokens
                ),
            });
        }

        // check temperature
        if request.temperature < self.config.min_temperature
            || request.temperature > self.config.max_temperature
        {
            return Err(ValidationError::InvalidParameter {
                field: "temperature".to_string(),
                reason: format!(
                    "must be between {} and {}, got {}",
                    self.config.min_temperature, self.config.max_temperature, request.temperature
                ),
            });
        }

        // check top_p
        if request.top_p < self.config.min_top_p || request.top_p > self.config.max_top_p {
            return Err(ValidationError::InvalidParameter {
                field: "top_p".to_string(),
                reason: format!(
                    "must be between {} and {}, got {}",
                    self.config.min_top_p, self.config.max_top_p, request.top_p
                ),
            });
        }

        Ok(Validated(request.clone()))
    }

    // validate an embeddings request.
    pub fn validate_embeddings(
        &self,
        request: &EmbeddingsRequest,
    ) -> Result<Validated<EmbeddingsRequest>, ValidationError> {
        let inputs = request.input.clone().into_vec();

        // check for empty input
        if inputs.is_empty() {
            return Err(ValidationError::MissingField("input".to_string()));
        }

        // check each input for content and token limits
        for (i, input) in inputs.iter().enumerate() {
            if input.trim().is_empty() {
                return Err(ValidationError::InvalidParameter {
                    field: format!("input[{}]", i),
                    reason: "cannot be empty".to_string(),
                });
            }

            let tokens = self.token_count(input);
            if tokens > self.config.max_context_tokens {
                return Err(ValidationError::TokenLimitExceeded {
                    actual: tokens,
                    limit: self.config.max_context_tokens,
                });
            }
        }

        Ok(Validated(request.clone()))
    }

    // Get the validator configuration
    pub fn config(&self) -> &ValidatorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ChatMessage, Role};
    use proptest::prelude::*;

    // ========================================================================
    // Generators
    // ========================================================================

    fn arb_valid_prompt() -> impl Strategy<Value = String> {
        // non empty string within token limits
        "[a-zA-Z]{1,1000}"
    }

    fn arb_valid_temperature() -> impl Strategy<Value = f32> {
        0.0f32..=2.0f32
    }

    fn arb_valid_top_p() -> impl Strategy<Value = f32> {
        0.0f32..=1.0f32
    }

    fn arb_valid_max_tokens() -> impl Strategy<Value = usize> {
        1usize..=4096usize
    }

    fn arb_valid_generate_request() -> impl Strategy<Value = GenerateRequest> {
        (
            arb_valid_prompt(),
            arb_valid_max_tokens(),
            arb_valid_temperature(),
            arb_valid_top_p(),
        )
            .prop_map(|(prompt, max_tokens, temperature, top_p)| GenerateRequest {
                prompt,
                max_tokens,
                temperature,
                top_p,
                stop_sequences: vec![],
                stream: false,
                priority: None,
            })
    }

    fn arb_role() -> impl Strategy<Value = Role> {
        prop_oneof![Just(Role::System), Just(Role::User), Just(Role::Assistant)]
    }

    fn arb_valid_chat_message() -> impl Strategy<Value = ChatMessage> {
        (arb_role(), "[a-zA-Z0-9 ]{1,200}")
            .prop_map(|(role, content)| ChatMessage { role, content })
    }

    fn arb_valid_chat_request() -> impl Strategy<Value = ChatRequest> {
        (
            prop::collection::vec(arb_valid_chat_message(), 1..5),
            arb_valid_max_tokens(),
            arb_valid_temperature(),
            arb_valid_top_p(),
        )
            .prop_map(|(messages, max_tokens, temperature, top_p)| ChatRequest {
                messages,
                max_tokens,
                temperature,
                top_p,
                stop_sequences: vec![],
                stream: false,
            })
    }

    // invalid request generators
    fn arb_empty_prompt() -> impl Strategy<Value = String> {
        prop_oneof![
            Just(String::new()),
            Just("   ".to_string()),
            Just("\t\n".to_string()),
        ]
    }

    fn arb_invalid_temperature() -> impl Strategy<Value = f32> {
        prop_oneof![
            -10.0f32..-0.01f32, // too low
            2.01f32..10.0f32,   // too high
        ]
    }

    fn arb_invalid_top_p() -> impl Strategy<Value = f32> {
        prop_oneof![
            -10.0f32..-0.01f32, // Too low
            1.01f32..10.0f32,   // Too high
        ]
    }

    fn arb_oversized_prompt() -> impl Strategy<Value = String> {
        // generate a prompt that exceeds token limit (8192 * 4 = ~32768 chars)
        "[a-zA-Z]{35000,40000}"
    }

    // ========================================================================
    // Property Tests
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]


        #[test]
        fn prop_valid_generate_request_accepted(request in arb_valid_generate_request()) {
            let validator = RequestValidator::with_defaults();
            let result = validator.validate_generate(&request);
            prop_assert!(result.is_ok(), "Valid request should be accepted: {:?}", result);
        }

        #[test]
        fn prop_valid_chat_request_accepted(request in arb_valid_chat_request()) {
            let validator = RequestValidator::with_defaults();
            let result = validator.validate_chat(&request);
            prop_assert!(result.is_ok(), "Valid chat request should be accepted: {:?}", result);
        }

        #[test]
        fn prop_empty_prompt_rejected(empty_prompt in arb_empty_prompt()) {
            let validator = RequestValidator::with_defaults();
            let request = GenerateRequest {
                prompt: empty_prompt,
                ..Default::default()
            };
            let result = validator.validate_generate(&request);
            prop_assert!(result.is_err(), "Empty prompt should be rejected");
            prop_assert!(matches!(result, Err(ValidationError::EmptyPrompt)));
        }


        #[test]
        fn prop_invalid_temperature_rejected(temp in arb_invalid_temperature()) {
            let validator = RequestValidator::with_defaults();
            let request = GenerateRequest {
                prompt: "Hello world".to_string(),
                temperature: temp,
                ..Default::default()
            };
            let result = validator.validate_generate(&request);
            prop_assert!(result.is_err(), "Invalid temperature {} should be rejected", temp);
            match result {
                Err(ValidationError::InvalidParameter { field, .. }) => {
                    prop_assert_eq!(field, "temperature");
                }
                other => prop_assert!(false, "Expected InvalidParameter, got {:?}", other),
            }
        }


        #[test]
        fn prop_invalid_top_p_rejected(top_p in arb_invalid_top_p()) {
            let validator = RequestValidator::with_defaults();
            let request = GenerateRequest {
                prompt: "Hello world".to_string(),
                top_p,
                ..Default::default()
            };
            let result = validator.validate_generate(&request);
            prop_assert!(result.is_err(), "Invalid top_p {} should be rejected", top_p);
            match result {
                Err(ValidationError::InvalidParameter { field, .. }) => {
                    prop_assert_eq!(field, "top_p");
                }
                other => prop_assert!(false, "Expected InvalidParameter, got {:?}", other),
            }
        }


        #[test]
        fn prop_oversized_prompt_rejected(prompt in arb_oversized_prompt()) {
            let validator = RequestValidator::with_defaults();
            let request = GenerateRequest {
                prompt,
                ..Default::default()
            };
            let result = validator.validate_generate(&request);
            prop_assert!(result.is_err(), "Oversized prompt should be rejected");
            match result{
                Err(ValidationError::TokenLimitExceeded { actual, .. }) => {
                    prop_assert!(actual > validator.config().max_context_tokens);
                }
                other => prop_assert!(false, "Expected TokenLimitExceeded, got {:?}", other),
            }
        }

        #[test]
        fn prop_token_count_proportional_to_length(text in "[a-zA-Z0-9 ]{0,10000}") {
            let validator = RequestValidator::with_defaults();
            let count = validator.token_count(&text);
            // Token count should be roughly proportional to text length
            // Our approximation is len/4, so count should be <= len
            prop_assert!(count <= text.len(), "Token count {} should be <= text length {}", count, text.len());
            // And for non-empty text, count should be > 0
            if !text.is_empty() {
                prop_assert!(count > 0, "Non-empty text should have positive token count");
            }
        }
    }
}
