#include "../include/parallel/inet.cuh"

const uint8_t NODE_ARITIES_H[NODE_KINDS] = {
    1, 0, 2, 2, 0, 2, 2, 3, 3, 0, 3, 3, 2, 2, 2, 1, 1, 1, 0, 1, 1, 2,
};

__global__ void inet(/* node_arities, action_map, initial interactions*/) {

  // local_q[0] = interactions[block, thread]
  // setup shared queue (buffer, count, head, tail)

  // interact(local_q[0])

  // if shared queue full, deque to global queue

  // at the end copy network back to init_network_d
  // must be smaller than init network so there will be space
}
