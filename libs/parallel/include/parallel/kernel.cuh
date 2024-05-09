#ifndef __MAPPIN_PARALLEL_KERNEL__
#define __MAPPIN_PARALLEL_KERNEL__

#include "inet.hpp"

#define MAX_NETWORK_SIZE (1024 * 1024 * 16) // ~44.7M nodes
#define MAX_INTERACTIONS_SIZE (1024 * 1024)

#define GRID_DIM_X 16
#define BLOCK_DIM_X 128

template <uint32_t N> class InteractionQueue;

__global__ void runINet(InteractionQueue<MAX_INTERACTIONS_SIZE> *, bool *,
                        NodeElement *);

__global__ void copyNetwork(NodeElement *, NodeElement *,
                            InteractionQueue<MAX_INTERACTIONS_SIZE> *);
#endif
