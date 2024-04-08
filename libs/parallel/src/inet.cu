#include "../include/parallel/inet.cuh"
#include "../include/parallel/queue.cuh"

const uint8_t NODE_ARITIES_H[NODE_KINDS] = {
    1, 0, 2, 2, 0, 2, 2, 3, 3, 0, 3, 3, 2, 2, 2, 1, 1, 1, 0, 1, 1, 2,
};

__global__ void runINet(NodeElement *network,
                        InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue,
                        size_t inters_size) {

  Interaction local_queue[4];
  if (blockDim.x * blockIdx.x + threadIdx.x < inters_size)
    local_queue[0] =
        globalQueue->getInteraction(blockDim.x * blockIdx.x + threadIdx.x);

  // initialize global queue

  // setup shared queue (buffer, count, head, tail)

  // interact(local_q[0])

  // if shared queue full, deque to global queue

  // at the end copy network back to init_network_d
  // must be smaller than init network so there will be space
}
