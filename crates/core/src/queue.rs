// Priority queue manager for inference requests
// manages three priority queues (High,Normal,Low) with backpressure control using high/ low watermarks (hysteresis)

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use crate::{error::QueueError, Priority, RequestID};

// Configuration for queue manager
#[derive(Debug, Clone)]
pub struct QueueConfig {
    // queue depth that triggers backpressure (stop accepting)
    pub high_watermark: usize,
    // queue depth that triggers backpressure (resume accepting)
    pub low_watermark: usize,
    // max time a request can wait in queue
    pub request_timeout: Duration,
    // max total queue size across all priorities
    pub max_queue_size: usize,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            high_watermark: 1000,
            low_watermark: 500,
            request_timeout: Duration::from_secs(30),
            max_queue_size: 2000,
        }
    }
}

// queue depth statistics by priority
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueueDepth {
    pub high: usize,
    pub normal: usize,
    pub low: usize,
    pub total: usize,
}

// a queued request with metadata
#[derive(Debug, Clone)]
pub struct QueuedRequest<T> {
    pub id: RequestID,
    pub data: T,
    pub priority: Priority,
    pub enqueued_at: Instant,
}

impl<T> QueuedRequest<T> {
    pub fn new(id: RequestID, data: T, priority: Priority) -> Self {
        Self {
            id,
            data,
            priority,
            enqueued_at: Instant::now(),
        }
    }

    // check if this request has exceeded the timeout
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.enqueued_at.elapsed() > timeout
    }
}

// Priority queue manager with backpressure control
//
// uses hysteresis to prevent oscillation:
// - when total depth > high_watermark -> stop accepting
// - when total depth < low_watermark -> resume accepting
#[derive(Debug)]
pub struct PriorityQueueManager<T> {
    high_queue: VecDeque<QueuedRequest<T>>,
    normal_queue: VecDeque<QueuedRequest<T>>,
    low_queue: VecDeque<QueuedRequest<T>>,
    config: QueueConfig,
    // current backpressure state (true = rejecting new requests)
    backpressure_active: bool,
}

impl<T> PriorityQueueManager<T> {
    // create a new queue manager with given configuration
    pub fn new(config: QueueConfig) -> Self {
        Self {
            high_queue: VecDeque::new(),
            normal_queue: VecDeque::new(),
            low_queue: VecDeque::new(),
            config,
            backpressure_active: false,
        }
    }

    // create a queue manager with default configuration
    pub fn with_default() -> Self {
        Self::new(QueueConfig::default())
    }

    // enqueue a request with the given priority
    // returns error if backpressure is active or queue is full
    pub fn enqueue(&mut self, request: QueuedRequest<T>) -> Result<(), QueueError> {
        // check backpressure
        if self.backpressure_active {
            return Err(QueueError::Full);
        }

        // check absolute max size
        let total = self.total_depth();
        if total >= self.config.max_queue_size {
            return Err(QueueError::Full);
        }

        // add to appropriate queue
        match request.priority {
            Priority::High => self.high_queue.push_back(request),
            Priority::Low => self.low_queue.push_back(request),
            Priority::Normal => self.normal_queue.push_back(request),
        }

        // check if we need to activate backpressure
        self.update_backpressure_state();

        Ok(())
    }

    // dequeue up to 'max_count' requests, respecting priority order
    // high priority first, then normal , then low
    pub fn dequeue_batch(&mut self, max_count: usize) -> Vec<QueuedRequest<T>> {
        let mut batch = Vec::with_capacity(max_count);

        // drain from high priority first
        while batch.len() < max_count && !self.high_queue.is_empty() {
            if let Some(req) = self.high_queue.pop_front() {
                batch.push(req);
            }
        }

        // then normal priority
        while batch.len() < max_count && !self.normal_queue.is_empty() {
            if let Some(req) = self.normal_queue.pop_front() {
                batch.push(req);
            }
        }

        // then low priority
        while batch.len() < max_count && !self.low_queue.is_empty() {
            if let Some(req) = self.low_queue.pop_front() {
                batch.push(req);
            }
        }

        // update backpressure state after dequeue
        self.update_backpressure_state();

        batch
    }

    // dequeue a single request (highest priority available)
    pub fn dequeue_one(&mut self) -> Option<QueuedRequest<T>> {
        let result = self
            .high_queue
            .pop_front()
            .or_else(|| self.normal_queue.pop_front())
            .or_else(|| self.low_queue.pop_front());

        self.update_backpressure_state();
        result
    }

    // get current queue depths
    pub fn queue_depth(&self) -> QueueDepth {
        QueueDepth {
            high: self.high_queue.len(),
            normal: self.normal_queue.len(),
            low: self.low_queue.len(),
            total: self.total_depth(),
        }
    }

    // check if the queue is accepting new requests
    pub fn is_accepting(&self) -> bool {
        !self.backpressure_active
    }

    // get total depth across all queues
    pub fn total_depth(&self) -> usize {
        self.high_queue.len() + self.normal_queue.len() + self.low_queue.len()
    }

    // check if all queues are empty
    pub fn is_empty(&self) -> bool {
        self.high_queue.is_empty() && self.normal_queue.is_empty() && self.low_queue.is_empty()
    }

    // remove and return all expired requests
    pub fn remove_expired(&mut self) -> Vec<QueuedRequest<T>> {
        let timeout = self.config.request_timeout;
        let mut expired = Vec::new();

        // helper to drain expired from a queue
        fn drain_expired<T>(
            queue: &mut VecDeque<QueuedRequest<T>>,
            timeout: Duration,
            expired: &mut Vec<QueuedRequest<T>>,
        ) {
            let mut i = 0;
            while i < queue.len() {
                if queue[i].is_expired(timeout) {
                    if let Some(req) = queue.remove(i) {
                        expired.push(req);
                    }
                } else {
                    i += 1;
                }
            }
        }

        drain_expired(&mut self.high_queue, timeout, &mut expired);
        drain_expired(&mut self.normal_queue, timeout, &mut expired);
        drain_expired(&mut self.low_queue, timeout, &mut expired);

        self.update_backpressure_state();
        expired
    }

    // get the queue configuration.
    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    // update backpressure state based on current queue depth.
    // Uses hysteresis: activate at high_watermark, deactivate at low_watermark.
    fn update_backpressure_state(&mut self) {
        let total = self.total_depth();

        if self.backpressure_active {
            // currently in backpressure - release when below low watermark
            if total < self.config.low_watermark {
                self.backpressure_active = false;
            }
        } else {
            // currently accepting - activate backpressure when above high watermark
            if total > self.config.high_watermark {
                self.backpressure_active = true;
            }
        }
    }
}
