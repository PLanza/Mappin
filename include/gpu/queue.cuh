#ifndef __MAPPIN_PARALLEL_QUEUE__
#define __MAPPIN_PARALLEL_QUEUE__

#include <cassert>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <vector>

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

inline __device__ int atomicAggInc(uint32_t *ctr) {
  auto g = cg::coalesced_threads();
  int warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

inline __device__ int atomicAggInc(int32_t *ctr) {
  auto g = cg::coalesced_threads();
  int warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

// The global top-level queue
template <class T, uint32_t N> class ParallelQueue {

  __device__ bool ensureEnqueue(int32_t size) {
    int32_t num = this->count;
    while (true) {
      if (num + size > N)
        return false;
      if (atomicAdd(&this->count, size) < N - size)
        return true;

      num = atomicSub(&this->count, size) - size;
    }
  }

  __device__ bool ensureDequeue(int32_t size) {
    int32_t num = this->count;
    while (true) {
      if (num - size < 0)
        return false;
      if (atomicSub(&this->count, size) >= size)
        return true;

      num = atomicAdd(&this->count, size) + size;
    }
  }

public:
  T buffer[N];
  unsigned long long head;
  unsigned long long tail;
  int32_t count;

  __host__ ParallelQueue(std::vector<T> init_q)
      : tail(init_q.size()), head(0), count(init_q.size()) {
    assert(init_q.size() < N);
    memcpy(this->buffer, init_q.data(), init_q.size() * sizeof(T));
  }

  __device__ inline bool isEmpty() { return this->head == this->tail; }

  __device__ bool enqueue(T item) {
    if (this->ensureEnqueue(1)) {

      this->buffer[atomicAggInc(&this->tail) % N] = item;
      return true;
    } else
      return false;
  }

  __device__ void enqueueMany(int32_t *index, size_t size) {
    if (this->ensureEnqueue(static_cast<int32_t>(size))) {

      *index = atomicAdd(&this->tail, size) % N;
    } else
      *index = -1;
  }

  __device__ bool dequeue(T *item) {
    if (this->ensureDequeue(1)) {

      *item = this->buffer[atomicAggInc(&this->head) % N];
      return true;
    } else
      return false;
  }

  __device__ void dequeueMany(int32_t *index, size_t size) {
    if (this->ensureDequeue(static_cast<int32_t>(size))) {
      *index = atomicAdd(&this->head, size) % N;
    } else
      *index = -1;
  }
};

template <class T, uint32_t N>
ParallelQueue<T, N> *newParallelQueue(std::vector<T> init_q) {
  ParallelQueue<T, N> *queue_d;
  cudaMalloc((void **)&queue_d, sizeof(ParallelQueue<T, N>));
  // Copy data
  cudaMemcpy(queue_d->buffer, init_q.data(), init_q.size() * sizeof(T),
             cudaMemcpyHostToDevice);

  // Set indices
  unsigned long long size = init_q.size();
  cudaMemcpy(&queue_d->tail, &size, sizeof(unsigned long long),
             cudaMemcpyHostToDevice);
  cudaMemset(&queue_d->head, 0, sizeof(unsigned long long));
  cudaMemcpy(&queue_d->count, &size, sizeof(unsigned long long),
             cudaMemcpyHostToDevice);

  return queue_d;
}

#endif
