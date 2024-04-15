#ifndef __MAPPIN_PARALLEL_KERNEL__
#define __MAPPIN_PARALLEL_KERNEL__

#include "inet.hpp"

#define MAX_NETWORK_SIZE (1024 * 1024 * 1024) // ~44.7M nodes
#define MAX_INTERACTIONS_SIZE (1024 * 1024)

#define GRID_DIM_X 1
#define BLOCK_DIM_X 32

template <uint32_t N> class InteractionQueue;

__global__ void runINet(InteractionQueue<MAX_INTERACTIONS_SIZE> *, size_t,
                        bool *, NodeElement *, NodeElement *, NodeElement *);

#endif
