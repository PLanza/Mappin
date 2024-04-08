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
// A simplified version of Kerbl, et al's 'Buffer Queue'
template <uint32_t N> class InteractionQueue {
  Interaction buffer[N];
  unsigned long long head;
  unsigned long long tail;
  int32_t count;

  __device__ bool ensureEnqueue(size_t size) {
    uint32_t num = this->count;
    while (true) {
      if (num >= N)
        return false;
      if (atomicAdd(&this->count, size) < N)
        return true;
      num = atomicSub(&this->count, size) - size;
    }
  }

  __device__ bool ensureDequeue(size_t size) {
    uint32_t num = this->count;
    while (true) {
      if (num <= 0)
        return false;
      if (atomicSub(&this->count, size) > N)
        return true;
      num = atomicAdd(&this->count, size) + size;
    }
  }

public:
  __host__ InteractionQueue() : head(0), tail(0), count(0) {}

  __host__ InteractionQueue(Interaction *interactions, size_t size,
                            size_t num_threads)
      : tail(size), count(size) {
    assert(size < N);
    this->head = std::min(num_threads, size);

    memcpy(this->buffer, interactions, size * sizeof(Interaction));
  }

  __device__ inline Interaction getInteraction(size_t pos) {
    return this->buffer[pos];
  }

  __device__ inline void subCount(size_t val) { atomicSub(&this->count, val); }

  __device__ inline void addHead(size_t val) { atomicAdd(&this->head, val); }

  __device__ bool enqueue(Interaction *src, size_t size) {
    if (this->ensureEnqueue(size)) {
      size_t start = atomicAdd(&this->tail, size) % N;

      // Skipping waitForTicket since we don't expect there to be enough
      // enqueues to wrap around

      memcpy(this->buffer + start, src, sizeof(Interaction) * size);

      return true;
    } else {
      return false;
    }
  }

  __device__ bool dequeue(Interaction *dst, size_t size) {
    if (this->ensureDequeue(size)) {
      size_t start = atomicAdd(&this->head, size) % N;

      // Skipping waitForTicket since we don't expect there to be enough
      // enqueues to wrap around

      memcpy(dst, this->buffer + start, sizeof(Interaction) * size);

      return true;
    } else {
      return false;
    }
  }
};

#endif
