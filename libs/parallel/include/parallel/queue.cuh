#ifndef __MAPPIN_PARALLEL_QUEUE__
#define __MAPPIN_PARALLEL_QUEUE__

#include "inet.cuh"
#include <algorithm>
#include <cassert>
#include <cstdint>

template <uint32_t N> __device__ bool ensureEnqueue(uint32_t *count) {
  uint32_t num = *count;
  while (true) {
    if (num >= N)
      return false;
    if (atomicAdd(count, 1) < N)
      return true;
    num = atomicSub(count, 1) - 1;
  }
}

template <uint32_t N> __device__ bool ensureDequeue(uint32_t *count) {
  uint32_t num = *count;
  while (true) {
    if (num <= 0)
      return false;
    if (atomicSub(count, 1) > N)
      return true;
    num = atomicAdd(count, 1) + 1;
  }
}

// The global top-level queue
template <uint32_t N> class InteractionQueue {
  uint32_t enqueing;
  uint32_t dequeing;

  __device__ bool ensureEnqueue(size_t size) {
    uint32_t num = this->count;
    while (true) {
      if (num >= N)
        return false;
      if (atomicAdd(&this->count, size) <= N)
        return true;
      num = atomicSub(&this->count, size) - size;
    }
  }

  __device__ bool ensureDequeue(size_t size) {
    uint32_t num = this->count;
    while (true) {
      if (num <= 0)
        return false;
      if (atomicSub(&this->count, size) >= 0)
        return true;
      num = atomicAdd(&this->count, size) + size;
    }
  }

public:
  Interaction buffer[N];
  unsigned long long head;
  unsigned long long tail;
  int32_t count;

  __host__ InteractionQueue() : head(0), tail(0), count(0), enqueing(0) {}

  __host__ InteractionQueue(Interaction *interactions, size_t size,
                            size_t num_threads)
      : tail(size), count(size), enqueing(0) {
    assert(size < N);
    this->head = std::min(num_threads, size);

    memcpy(this->buffer, interactions, size * sizeof(Interaction));
  }

  __device__ inline bool isEmpty() { return !(head == tail); }

  __device__ void enqueue(int64_t *index, size_t size) {
    if (this->ensureEnqueue(size)) {
      // If full wait until dequing operations are done
      while (this->count + this->dequeing >= N) {
      }
      atomicAdd(&this->enqueing, size);
      *index = atomicAdd(&this->tail, size) % N;
    } else
      *index = -1;
  }

  // Doesn't quite work when there are multiple enqueues/dequeues
  __device__ inline void ackEnqueue(size_t size) {
    atomicSub(&this->enqueing, size);
  }

  __device__ void dequeue(int64_t *index, size_t size) {
    if (this->ensureDequeue(size)) {
      // If empty wait until enqueing operations are done
      while (this->count - this->enqueing <= 0) {
      }
      atomicAdd(&this->dequeing, size);
      *index = atomicAdd(&this->head, size) % N;
    } else {
      *index = -1;
    }
  }
  __device__ inline void ackDequeue(size_t size) {
    atomicSub(&this->dequeing, size);
  }
};

#endif
