#include "../include/parallel/actions.cuh"
#include "../include/parallel/inet.cuh"
#include "../include/parallel/queue.cuh"
#include <cstdint>

const uint8_t NODE_ARITIES_H[NODE_KINDS] = {
    1, 0, 2, 2, 0, 2, 2, 3, 3, 0, 3, 3, 2, 2, 2, 1, 1, 1, 0, 1, 1, 2,
};

#define BLOCK_QUEUE_SIZE 4 * 256
#define MAX_NEW_NODES 3

__device__ inline uint32_t actMapIndex(uint8_t left, uint8_t right) {
  return (left * (2 * NODE_KINDS - left + 1) / 2 + right - left) * 2 *
         MAX_ACTIONS;
}

// Perform connections and return if a new interaction needs to be added
__device__ bool makeConnections(ConnectAction ca, NodeElement *&n1,
                                NodeElement *&n2,
                                NodeElement **const active_pair,
                                NodeElement **const new_nodes) {
  uint64_t p1 = connect_p(ca.c1), p2 = connect_p(ca.c2);

  if (connect_g(ca.c1) == ACTIVE_PAIR) {
    n1 = active_pair[connect_n(ca.c1)];
  } else if (connect_g(ca.c1) == VARS) {
    n1 =
        active_pair[connect_n(ca.c1)][1 + 2 * (connect_p(ca.c1) + 1)].port_node;
    p1 =
        active_pair[connect_n(ca.c1)][1 + 2 * (connect_p(ca.c1) + 1)].port_port;
  } else {
    n1 = new_nodes[connect_n(ca.c1)];
  }

  if (connect_g(ca.c2) == ACTIVE_PAIR) {
    n2 = active_pair[connect_n(ca.c2)];
  } else if (connect_g(ca.c2) == VARS) {
    n2 =
        active_pair[connect_n(ca.c2)][1 + 2 * (connect_p(ca.c2) + 1)].port_node;
    p2 =
        active_pair[connect_n(ca.c2)][1 + 2 * (connect_p(ca.c2) + 1)].port_port;
  } else {
    n2 = new_nodes[connect_n(ca.c2)];
  }

  // Potential contention
  if (connect_g(ca.c1) == VARS) {
    uint64_t old_node =
        reinterpret_cast<uintptr_t>(active_pair[connect_n(ca.c1)]);
    uint64_t old_port = connect_p(ca.c1) + 1;
    unsigned long long assumed_node, assumed_port;
    do {
      assumed_node = old_node;
      assumed_port = old_port;
      old_node = atomicCAS((unsigned long long *)n1 + 1 + 2 * p1, assumed_node,
                           reinterpret_cast<uintptr_t>(n2));
      // Chance of failure here!
      old_port = atomicCAS((unsigned long long *)n1 + 1 + 2 * p1 + 1,
                           assumed_port, p2);
    } while (assumed_node != old_node || assumed_port != old_port);
  } else {
    // We want these assignments to be a single memory write
    ((Port *)(n1 + 1))[p1] = {(NodeElement *)n2, p2};
  }

  if (connect_g(ca.c2) == VARS) {
    uint64_t old_node =
        reinterpret_cast<uintptr_t>(active_pair[connect_n(ca.c2)]);
    uint64_t old_port = connect_p(ca.c2) + 1;
    unsigned long long assumed_node, assumed_port;
    do {
      assumed_node = old_node;
      assumed_port = old_port;
      old_node = atomicCAS((unsigned long long *)n2 + 1 + 2 * p2, assumed_node,
                           reinterpret_cast<uintptr_t>(n1));
      // Chance of failure here!
      old_port = atomicCAS((unsigned long long *)n2 + 1 + 2 * p2 + 1,
                           assumed_port, p1);
    } while (assumed_node != old_node || assumed_port != old_port);
  } else {
    // We want these assignments to be a single memory write
    ((Port *)(n2 + 1))[p2] = {(NodeElement *)n1, p1};
  }

  return p1 == 0 && p2 == 0;
}

__global__ void runINet(NodeElement *network,
                        InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue,
                        size_t inters_size, bool *global_done) {

  Interaction interact_buf[5];
  uint8_t buf_elems = 1;

  __shared__ uint8_t *block_done;

  __shared__ Interaction block_queue[BLOCK_QUEUE_SIZE]; // 16384
  __shared__ uint32_t count, head, tail;

  __shared__ int64_t global_queue_idx;

  // Copy initial interactions
  if (blockDim.x * blockIdx.x + threadIdx.x < inters_size)
    interact_buf[0] =
        globalQueue->buffer[blockDim.x * blockIdx.x + threadIdx.x];

  if (threadIdx.x == 0) {
    count = 0;
    head = 0;
    tail = 0;

    // might want to statically allocate this
    block_done = (uint8_t *)malloc(sizeof(uint8_t) * blockDim.x / 8);
    memset(block_done, 0, sizeof(uint8_t) * blockDim.x / 8);
  }
  __syncthreads();

  while (true) {
    // TODO: optimize this first section so that spinning is more efficient

    // Might need this to be atomic
    block_done[threadIdx.x / 8] |= ((uint8_t)buf_elems == 0)
                                   << (threadIdx.x % 8);

    // Set first bit to 0 if the queues are not empty
    if (threadIdx.x == 0)
      block_done[0] &= ((uint8_t)~0)
                       << (globalQueue->isEmpty() || head == tail);

    // If all threads in block are done
    if (__syncthreads_and(block_done[threadIdx.x / 8] == (uint8_t)~0u)) {
      global_done[blockIdx.x] = true;
      // If all blocks are done
      if (__syncthreads_and(global_done[threadIdx.x % gridDim.x]))
        // Might need to synchronize across grid
        break;

      continue;
    }

    // Attempt to dequeue block_queue if it's full
    if (count == BLOCK_QUEUE_SIZE) {
      if (threadIdx.x == 0)
        globalQueue->enqueue(&global_queue_idx, 3 * blockDim.x);
      __syncthreads();

      // If both the block queue and the global queue are full then spin
      if (global_queue_idx == -1)
        continue;

      // Otherwise copy data from block to global queue
      for (int i = 0; i < 3; i++) {
        globalQueue->buffer[global_queue_idx + i * blockDim.x + threadIdx.x] =
            block_queue[head + i * blockDim.x + threadIdx.x];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        head += 3 * blockDim.x;
        count -= 3 * blockDim.x;
        globalQueue->ackEnqueue(3 * blockDim.x);
      }
    }

    // Attempt to enqueue block_queue from global if nearing empty
    if (count < blockDim.x) {
      if (threadIdx.x == 0)
        globalQueue->enqueue(&global_queue_idx, blockDim.x - count);
      __syncthreads();

      if (global_queue_idx != -1) {
        if (threadIdx.x < blockDim.x - count)
          block_queue[tail + threadIdx.x] =
              globalQueue->buffer[global_queue_idx + threadIdx.x];
        __syncthreads();

        if (threadIdx.x == 0) {
          tail += blockDim.x - count;
          count = blockDim.x;
          globalQueue->ackEnqueue(blockDim.x - count);
        }
      }
    }

    if (buf_elems == 0) {
      if (ensureDequeue<BLOCK_QUEUE_SIZE>(&count)) {
        interact_buf[0] = block_queue[atomicAdd(&head, 1) % BLOCK_QUEUE_SIZE];
      } else
        continue;
    } else {
      buf_elems--;
    }

    bool switch_nodes = interact_buf[buf_elems].n1->header.kind >
                        interact_buf[0].n2->header.kind;
    // If there is enough register space, consider loading into register
    NodeElement *left =
        switch_nodes ? interact_buf[buf_elems].n2 : interact_buf[buf_elems].n1;
    NodeElement *right =
        switch_nodes ? interact_buf[buf_elems].n1 : interact_buf[buf_elems].n2;

    // Load actions
    Action *actions = actions_map +
                      actMapIndex(left->header.kind, right->header.kind) +
                      MAX_ACTIONS * (left->header.value == right->header.value);
    uint8_t next_action = 0;

    NodeElement *active_pair[2] = {left, right};

    NodeElement *new_nodes[MAX_NEW_NODES];
    uint8_t next_new = 0;

    // TODO: Test doing it all in a single loop
    while (next_action < MAX_ACTIONS && actions[next_action].kind == NEW) {
      NewNodeAction nna = actions[next_action].action.new_node;
      uint32_t value;
      if (nna.value == -1)
        value = left->header.value;
      else if (nna.value == -2)
        value = right->header.value;
      else if (nna.value == -3)
        value = reinterpret_cast<std::uintptr_t>(left);
      else
        value = nna.value;

      new_nodes[next_new] = (NodeElement *)malloc(
          sizeof(NodeElement) * (1 + 2 * (NODE_ARITIES[nna.kind] + 1)));
      // Should do this in one memory operation
      new_nodes[next_new][0] = {{nna.kind, value}};

      next_action++;
      next_new++;
    }

    // Perform connect actions
    while (next_action < MAX_ACTIONS && actions[next_action].kind == CONNECT) {
      ConnectAction ca = actions[next_action].action.connect;
      NodeElement *n1, *n2;

      // Add any new interactions
      if (makeConnections(ca, n1, n2, active_pair, new_nodes)) {
        if (buf_elems < 5) {
          interact_buf[buf_elems] = {n1, n2};
          buf_elems++;
        } else {
          // WARNING: awful code!
          // If block queue full, enqueue onto global queue
          while (!ensureEnqueue<BLOCK_QUEUE_SIZE>(&count)) {
            int64_t g_q_idx = -1;
            while (g_q_idx != -1) {
              globalQueue->enqueue(&g_q_idx, 1);
            }
            globalQueue->buffer[g_q_idx] = {n1, n2};
            globalQueue->ackEnqueue(1);
          }
          block_queue[atomicAdd(&tail, 1) % BLOCK_QUEUE_SIZE] = {n1, n2};
        }
      }

      next_action++;
    }

    // Perform Free actions
    while (next_action < MAX_ACTIONS && actions[next_action].kind == FREE) {
      if (actions[next_action].action.free) {
        free(left);
      } else {
        free(right);
      }

      next_action++;
    }
  }

  // at the end copy network back to init_network_d
  // final network must be smaller than init network so there will be space
}
