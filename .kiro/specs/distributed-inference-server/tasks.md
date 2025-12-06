# Implementation Plan

- [ ] 1. Project Setup and Core Types

  - [ ] 1.1 Initialize Rust project with Cargo workspace structure
    - Create `Cargo.toml` with workspace members: `server`, `core`, `inference`
    - Configure dependencies: tokio, axum, serde, thiserror, tracing, proptest
    - Set up rustfmt and clippy configurations
    - _Requirements: 10.1_
  - [ ] 1.2 Define core type aliases and error types
    - Create `RequestId`, `BatchId`, `WorkerId`, `CacheKey` type aliases
    - Implement `ServerError`, `ApiError`, `ValidationError` with thiserror
    - Implement `ApiError::status_code()` and `ApiError::error_type()` methods
    - _Requirements: 9.1, 9.2, 11.4_
  - [ ] 1.3 Define request and response data models
    - Implement `GenerateRequest`, `ChatRequest`, `EmbeddingsRequest` with serde
    - Implement `GenerateResponse`, `ChatResponse`, `EmbeddingsResponse`
    - Implement `ErrorResponse`, `Usage`, `FinishReason` types
    - _Requirements: 1.1, 1.2, 1.3, 11.1, 11.2, 11.3_
  - [ ] 1.4 Write property test for response serialization round-trip
    - **Property 25: Response Serialization Round-Trip**
    - **Validates: Requirements 11.5**

- [ ] 2. Request Validation

  - [ ] 2.1 Implement RequestValidator trait and struct
    - Create `RequestValidator` with configurable limits (max_tokens, context_window)
    - Implement `validate_generate()`, `validate_chat()`, `validate_embeddings()`
    - Implement `token_count()` using tiktoken-rs or simple whitespace tokenizer
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ] 2.2 Write property test for valid request acceptance
    - **Property 1: Valid Request Acceptance**
    - **Validates: Requirements 1.1**
  - [ ] 2.3 Write property test for invalid request rejection
    - **Property 2: Invalid Request Rejection**
    - **Validates: Requirements 1.4**
  - [ ] 2.4 Write property test for token limit enforcement
    - **Property 3: Token Limit Enforcement**
    - **Validates: Requirements 1.5**

- [ ] 3. Priority Queue Manager

  - [ ] 3.1 Implement QueueManager with priority queues
    - Create `PriorityQueueManager` with separate queues for High/Normal/Low
    - Implement `enqueue()` with priority assignment
    - Implement `dequeue_batch()` respecting priority ordering
    - Implement `queue_depth()` returning `QueueDepth` struct
    - _Requirements: 3.4, 3.5_
  - [ ] 3.2 Implement backpressure with hysteresis
    - Add high/low watermark configuration
    - Implement `is_accepting()` with hysteresis logic
    - Track state transitions only at watermark crossings
    - _Requirements: 3.1, 3.2_
  - [ ] 3.3 Implement request timeout handling
    - Add background task to scan for expired requests
    - Remove expired requests and signal timeout to callers
    - _Requirements: 3.3_
  - [ ] 3.4 Write property test for priority queue ordering
    - **Property 6: Priority Queue Ordering**
    - **Validates: Requirements 3.4, 3.5**
  - [ ] 3.5 Write property test for backpressure hysteresis
    - **Property 7: Backpressure Hysteresis**
    - **Validates: Requirements 3.1, 3.2**
  - [ ] 3.6 Write property test for request timeout enforcement
    - **Property 8: Request Timeout Enforcement**
    - **Validates: Requirements 3.3**

- [ ] 4. Request Batcher

  - [ ] 4.1 Implement RequestBatcher with batching window
    - Create `RequestBatcher` with configurable `batch_timeout` and `max_batch_size`
    - Implement `add_request()` to queue requests for batching
    - Implement `get_batch()` that returns batch when window expires or max size reached
    - _Requirements: 2.1, 2.2_
  - [ ] 4.2 Implement sequence padding logic
    - Pad sequences to max length in batch
    - Track `original_length` for each request
    - Generate attention masks based on padding
    - _Requirements: 2.3_
  - [ ] 4.3 Implement priority-aware batching
    - Include high-priority requests in next available batch
    - _Requirements: 2.4_
  - [ ] 4.4 Write property test for batch formation
    - **Property 4: Batch Formation Within Window**
    - **Validates: Requirements 2.1, 2.2**
  - [ ] 4.5 Write property test for batch padding correctness
    - **Property 5: Batch Padding Correctness**
    - **Validates: Requirements 2.3**

- [ ] 5. Checkpoint - Ensure all tests pass

  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. KV Cache Manager

  - [ ] 6.1 Implement KVCacheManager with LRU eviction
    - Create `LruKVCache` with configurable memory limit
    - Implement `get()`, `put()` with access timestamp tracking
    - Implement `evict_lru()` to free memory when needed
    - _Requirements: 4.2, 4.3, 4.4_
  - [ ] 6.2 Implement prefix matching for cache reuse
    - Implement `get_prefix()` to find longest matching prefix
    - Return cached KV tensors and match length
    - _Requirements: 4.1_
  - [ ] 6.3 Implement cache serialization/deserialization
    - Implement `serialize()` and `deserialize()` for `CacheEntry`
    - Use bincode or similar for efficient binary serialization
    - _Requirements: 4.6_
  - [ ] 6.4 Implement cache statistics
    - Track hit_count, miss_count, eviction_count
    - Implement `stats()` returning `CacheStats`
    - _Requirements: 4.5_
  - [ ] 6.5 Write property test for prefix reuse
    - **Property 9: KV Cache Prefix Reuse**
    - **Validates: Requirements 4.1**
  - [ ] 6.6 Write property test for LRU eviction
    - **Property 10: LRU Cache Eviction**
    - **Validates: Requirements 4.2**
  - [ ] 6.7 Write property test for access timestamp update
    - **Property 11: Cache Access Timestamp Update**
    - **Validates: Requirements 4.3**
  - [ ] 6.8 Write property test for cache serialization round-trip
    - **Property 12: Cache Serialization Round-Trip**
    - **Validates: Requirements 4.6**

- [ ] 7. Token Streamer

  - [ ] 7.1 Implement TokenStreamer with SSE formatting
    - Create `TokenStreamer` managing active streams per request
    - Implement `create_stream()` returning sender/receiver pair
    - Implement SSE event formatting for token events
    - _Requirements: 5.2_
  - [ ] 7.2 Implement stream completion and error handling
    - Implement `close_stream()` with finish_reason
    - Implement error event formatting
    - Handle client disconnection detection
    - _Requirements: 5.3, 5.4, 5.5_
  - [ ] 7.3 Write property test for SSE token event format
    - **Property 13: SSE Token Event Format**
    - **Validates: Requirements 5.2**
  - [ ] 7.4 Write property test for stream completion event
    - **Property 14: Stream Completion Event**
    - **Validates: Requirements 5.3**
  - [ ] 7.5 Write property test for streaming error event format
    - **Property 15: Streaming Error Event Format**
    - **Validates: Requirements 5.5**

- [ ] 8. Checkpoint - Ensure all tests pass

  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Adaptive Scheduler

  - [ ] 9.1 Implement Scheduler with pluggable strategies
    - Create `AdaptiveScheduler` with `SchedulingStrategy` enum
    - Implement `schedule()` dispatching to strategy-specific logic
    - Implement `set_strategy()` for runtime strategy changes
    - _Requirements: 6.1_
  - [ ] 9.2 Implement round-robin scheduling
    - Track last-used worker index
    - Cycle through workers in order
    - _Requirements: 6.1_
  - [ ] 9.3 Implement least-loaded scheduling
    - Track active batch count per worker
    - Route to worker with minimum active batches
    - _Requirements: 6.2_
  - [ ] 9.4 Implement memory-aware scheduling
    - Estimate batch memory requirements
    - Route to worker with sufficient available memory
    - _Requirements: 6.3_
  - [ ] 9.5 Implement worker health checking
    - Background task for periodic health checks
    - Remove unhealthy workers from routing pool
    - Reinstate recovered workers
    - _Requirements: 6.4, 6.5_
  - [ ] 9.6 Write property test for least-loaded routing
    - **Property 16: Least-Loaded Routing**
    - **Validates: Requirements 6.2**
  - [ ] 9.7 Write property test for memory-aware routing
    - **Property 17: Memory-Aware Routing**
    - **Validates: Requirements 6.3**
  - [ ] 9.8 Write property test for unhealthy worker removal
    - **Property 18: Unhealthy Worker Removal**
    - **Validates: Requirements 6.4**
  - [ ] 9.9 Write property test for worker recovery reinstatement
    - **Property 19: Worker Recovery Reinstatement**
    - **Validates: Requirements 6.5**

- [ ] 10. Inference Worker

  - [ ] 10.1 Implement InferenceWorker trait and mock implementation
    - Define `InferenceWorker` trait with `initialize()`, `infer()`, `shutdown()`
    - Create `MockInferenceWorker` for testing
    - Implement `status()` and `model_info()` methods
    - _Requirements: 7.1, 7.2, 7.3_
  - [ ] 10.2 Implement llama-cpp-rs integration
    - Add llama-cpp-rs dependency
    - Implement `LlamaCppWorker` loading model from path
    - Implement batch inference with token generation
    - _Requirements: 7.2, 7.3, 10.2, 10.3_
  - [ ] 10.3 Implement worker error handling and restart
    - Detect unrecoverable errors
    - Notify scheduler of worker failure
    - Implement automatic restart logic
    - _Requirements: 7.4, 9.4_
  - [ ] 10.4 Implement request failure isolation
    - Catch per-request errors during batch inference
    - Return error for failed request, continue others
    - _Requirements: 9.2_
  - [ ] 10.5 Write property test for worker count configuration
    - **Property 20: Worker Count Configuration**
    - **Validates: Requirements 7.1, 7.2**
  - [ ] 10.6 Write property test for batch inference completion
    - **Property 21: Batch Inference Completion**
    - **Validates: Requirements 7.3**
  - [ ] 10.7 Write property test for request failure isolation
    - **Property 22: Request Failure Isolation**
    - **Validates: Requirements 9.2**

- [ ] 11. Checkpoint - Ensure all tests pass

  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Configuration Management

  - [ ] 12.1 Implement configuration loading with precedence
    - Create `Config` struct with all configurable parameters
    - Implement loading from file (TOML/YAML)
    - Implement environment variable overrides
    - Implement CLI argument overrides (using clap)
    - _Requirements: 10.1_
  - [ ] 12.2 Implement configuration validation
    - Validate all parameter ranges and types
    - Return specific validation errors
    - Exit with non-zero status on invalid config
    - _Requirements: 10.4_
  - [ ] 12.3 Implement runtime configuration updates
    - Support hot-reload for batch size, queue thresholds, scheduling strategy
    - Use watch channel for config change notifications
    - _Requirements: 10.5_
  - [ ] 12.4 Write property test for configuration precedence
    - **Property 26: Configuration Precedence**
    - **Validates: Requirements 10.1**
  - [ ] 12.5 Write property test for invalid configuration rejection
    - **Property 27: Invalid Configuration Rejection**
    - **Validates: Requirements 10.4**

- [ ] 13. API Server

  - [ ] 13.1 Implement HTTP server with Axum
    - Set up Axum router with `/generate`, `/chat`, `/embeddings` endpoints
    - Implement request parsing and validation
    - Wire up to queue manager
    - _Requirements: 1.1, 1.2, 1.3_
  - [ ] 13.2 Implement streaming endpoints
    - Add SSE support for streaming responses
    - Handle `stream: true` parameter
    - Wire up token streamer
    - _Requirements: 1.6, 5.1_
  - [ ] 13.3 Implement `/server/stats` endpoint
    - Return JSON with queue depth, worker status, cache stats, throughput
    - _Requirements: 8.1_
  - [ ] 13.4 Implement `/metrics` Prometheus endpoint
    - Expose latency histograms, token throughput, batch sizes, cache hit rate
    - Use prometheus crate for metric formatting
    - _Requirements: 8.2_
  - [ ] 13.5 Implement error response formatting
    - Convert `ApiError` to JSON error responses
    - Set appropriate HTTP status codes
    - _Requirements: 11.4_
  - [ ] 13.6 Write property test for API response format consistency
    - **Property 23: API Response Format Consistency**
    - **Validates: Requirements 11.1, 11.2, 11.3**
  - [ ] 13.7 Write property test for error response format
    - **Property 24: Error Response Format**
    - **Validates: Requirements 11.4**

- [ ] 14. Metrics and Observability

  - [ ] 14.1 Implement MetricsCollector
    - Create `PrometheusMetricsCollector` with histogram and counter metrics
    - Implement `record_request()`, `record_batch()`, `record_inference()`
    - Implement `record_ttft()`, `record_cache_access()`
    - _Requirements: 8.2, 8.3, 8.4_
  - [ ] 14.2 Implement OpenTelemetry tracing
    - Add tracing-opentelemetry dependency
    - Instrument request lifecycle, batching, inference, streaming phases
    - _Requirements: 8.5_
  - [ ] 14.3 Implement metrics snapshot for /server/stats
    - Implement `snapshot()` returning `MetricsSnapshot`
    - Calculate tokens-per-second, average latencies, percentiles
    - _Requirements: 8.1, 8.4_

- [ ] 15. Checkpoint - Ensure all tests pass

  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Server Orchestration

  - [ ] 16.1 Implement server startup and shutdown
    - Create `InferenceServer` orchestrating all components
    - Implement `start()` spawning workers and starting HTTP server
    - Implement graceful `shutdown()` draining requests
    - _Requirements: 7.1, 9.1_
  - [ ] 16.2 Implement worker scaling
    - Support adding workers at runtime
    - Support removing workers with request draining
    - _Requirements: 7.5_
  - [ ] 16.3 Implement graceful degradation
    - Monitor memory pressure
    - Adjust batch sizes and cache eviction based on pressure level
    - _Requirements: 9.3, 9.5_

- [ ] 17. Model Hot-Swapping (v2)

  - [ ] 17.1 Implement model swap API
    - Add admin endpoint for model swap requests
    - Load new model in background while serving with current
    - _Requirements: 13.1_
  - [ ] 17.2 Implement atomic model switch
    - Track in-flight requests per model
    - Atomically switch new requests to new model
    - Wait for in-flight requests to complete before unloading old model
    - _Requirements: 13.2, 13.3_
  - [ ] 17.3 Implement swap failure handling
    - Continue serving with original model on failure
    - Report failure via API and metrics
    - _Requirements: 13.4_
  - [ ] 17.4 Implement post-swap cleanup
    - Clear KV cache after successful swap
    - Reset relevant statistics
    - _Requirements: 13.5_
  - [ ] 17.5 Write property test for model swap continuity
    - **Property 28: Model Swap Continuity**
    - **Validates: Requirements 13.2, 13.3**
  - [ ] 17.6 Write property test for model swap failure recovery
    - **Property 29: Model Swap Failure Recovery**
    - **Validates: Requirements 13.4**

- [ ] 18. Speculative Decoding (v2)

  - [ ] 18.1 Implement draft model loading
    - Support loading secondary draft model
    - Configure speculation parameters (gamma, acceptance threshold)
    - _Requirements: 12.1_
  - [ ] 18.2 Implement speculative token generation
    - Generate candidate tokens with draft model
    - Batch-verify candidates with main model
    - Accept matching tokens, fall back on rejection
    - _Requirements: 12.2, 12.3_
  - [ ] 18.3 Implement adaptive speculation
    - Track acceptance rate per request pattern
    - Disable speculation when acceptance rate below threshold
    - _Requirements: 12.5_

- [ ] 19. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
