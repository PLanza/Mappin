#include "../include/parallel/queue.cuh"

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
