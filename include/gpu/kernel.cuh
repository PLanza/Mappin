#ifndef __MAPPIN_PARALLEL_KERNEL__
#define __MAPPIN_PARALLEL_KERNEL__

#include "inet.hpp"
#include "queue.cuh"

#define MAX_NETWORK_SIZE (1024 * 1024 * 16) // ~44.7M nodes
#define MAX_INTERACTIONS_SIZE (1024 * 1024)

#define BLOCK_DIM_X 128

template <uint32_t N> class InteractionQueue;

__global__ void copyNetwork(NodeElement *, uint32_t *, NodeElement *,
                            ParallelQueue<uint32_t, MAX_INTERACTIONS_SIZE> *,
                            int32_t, unsigned long long);

__global__ void
reduceNetwork(ParallelQueue<Interaction, MAX_INTERACTIONS_SIZE> *,
              NodeElement *, uint32_t *, int32_t, unsigned long long);
#endif
