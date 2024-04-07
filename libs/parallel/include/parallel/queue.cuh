#ifndef __MAPPIN_PARALLEL_QUEUE__
#define __MAPPIN_PARALLEL_QUEUE__

template <uint32_t N> bool __device__ ensureEnqueue(uint32_t *);
template <uint32_t N> bool __device__ ensureDequeue(uint32_t *);

#endif
