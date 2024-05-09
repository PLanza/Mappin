#ifndef __MAPPIN_PARALLEL_QUEUE__
#define __MAPPIN_PARALLEL_QUEUE__

#include "inet.hpp"

#include <cassert>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

template <uint32_t N> __device__ bool ensureEnqueue(int32_t *count) {
  int32_t num = *count;
  while (true) {
    if (num >= N)
      return false;
    if (atomicAdd(count, 1) < N)
      return true;
    num = atomicSub(count, 1) - 1;
  }
}

template <uint32_t N> __device__ bool ensureDequeue(int32_t *count) {
  int32_t num = *count;
  while (true) {
    if (num <= 0)
      return false;
    if (atomicSub(count, 1) > 0)
      return true;
    num = atomicAdd(count, 1) + 1;
  }
}

namespace cg = cooperative_groups;

inline __device__ int atomicAggInc(unsigned long long *ctr) {
  auto g = cg::coalesced_threads();
  int warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

// The global top-level queue
template <uint32_t N> class InteractionQueue {
  // Need to change these into queues of tickets
  int32_t enqueing;
  int32_t dequeing;

  __device__ bool ensureEnqueue(int32_t size) {
    int32_t num = this->count;
    while (true) {
      if (num + size > N)
        return false;
      if (atomicAdd(&this->count, size) < N)
        return true;
      num = atomicSub(&this->count, size) - size;
    }
  }

  __device__ bool ensureDequeue(int32_t size) {
    int32_t num = this->count;
    while (true) {
      if (num - size < 0)
        return false;
      if (atomicSub(&this->count, size) > 0)
        return true;
      num = atomicAdd(&this->count, size) + size;
    }
  }

public:
  Interaction buffer[N];
  unsigned long long head;
  unsigned long long tail;
  int32_t count;

  __host__ InteractionQueue()
      : head(0), tail(0), count(0), enqueing(0), dequeing(0) {}

  __host__ InteractionQueue(Interaction *interactions, size_t size)
      : tail(size), head(0), count(size), enqueing(0), dequeing(0) {
    assert(size < N);
    memcpy(this->buffer, interactions, size * sizeof(Interaction));
  }

  __device__ inline bool isEmpty() { return this->head == this->tail; }

  __device__ bool enqueue(Interaction interaction) {
    if (this->ensureEnqueue(static_cast<int32_t>(1))) {
      unsigned long long index = atomicAggInc(&this->tail);
      this->buffer[index] = interaction;

      return true;
    } else
      return false;
  }

  __device__ void enqueueMany(int64_t *index, size_t size) {
    if (this->ensureEnqueue(static_cast<int32_t>(size))) {
      // If full wait until dequing operations are done
      while (this->count + this->dequeing >= N)
        ;

      *index = atomicAdd(&this->tail, size) % N;
      atomicAdd(&this->enqueing, size);
    } else
      *index = -1;
  }

  // Doesn't quite work when there are multiple enqueues/dequeues
  __device__ inline void ackEnqueue(size_t size) {
    atomicSub(&this->enqueing, size);
  }

  __device__ bool dequeue(Interaction *interaction) {
    if (this->ensureDequeue(static_cast<int32_t>(1))) {

      unsigned long long index = atomicAggInc(&this->head) % N;
      *interaction = this->buffer[index];

      return true;
    } else {
      return false;
    }
  }

  __device__ void dequeueMany(int64_t *index, size_t size) {
    if (this->ensureDequeue(static_cast<int32_t>(size))) {
      // If empty wait until enqueing operations are done
      while (this->count - this->enqueing < 0) {
      }
      *index = atomicAdd(&this->head, size) % N;
      atomicAdd(&this->dequeing, size);
    } else {
      *index = -1;
    }
  }
  __device__ inline void ackDequeue(size_t size) {
    atomicSub(&this->dequeing, size);
  }
};

#endif
