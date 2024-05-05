#include "../include/parallel/actions.hpp"
#include "../include/parallel/inet.hpp"
#include "../include/parallel/kernel.cuh"
#include "../include/parallel/queue.cuh"

#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

namespace cg = cooperative_groups;

const uint8_t NODE_ARITIES_H[NODE_KINDS] = {1, 0, 2, 2, 0, 2, 2, 3, 3, 0,
                                            3, 3, 2, 2, 2, 1, 2, 4, 4, 4,
                                            4, 3, 1, 0, 1, 1, 1, 2, 1, 3};

__constant__ uint8_t NODE_ARITIES[NODE_KINDS];
__constant__ Action actions_map[ACTIONS_MAP_SIZE];

#define BLOCK_QUEUE_SIZE (4 * BLOCK_DIM_X)
#define MAX_NEW_NODES 4
#define THREAD_BUFFER_SIZE 4

__device__ int atomicAggInc(uint32_t *ctr) {
  auto g = cg::coalesced_threads();
  int warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

__host__ void copyConstantData() {
  cudaMemcpyToSymbol(NODE_ARITIES, NODE_ARITIES_H, sizeof(NODE_ARITIES));
  cudaMemcpyToSymbol(actions_map, actions_map_h, sizeof(actions_map));
}

__device__ inline uint32_t actMapIndex(uint8_t left, uint8_t right) {
  return (left * (2 * NODE_KINDS - left + 1) / 2 + right - left) * 2 *
         MAX_ACTIONS;
}

__device__ inline void lock(NodeElement *node) {
  atomicMax(&node[0].header.lock, threadIdx.x + blockDim.x * blockIdx.x);
}
__device__ inline void unlock(NodeElement *node) {
  atomicCAS(&node[0].header.lock, threadIdx.x + blockDim.x * blockIdx.x, 0u);
}
__device__ inline bool is_locked(NodeElement *node) {
  return node[0].header.lock == threadIdx.x + blockDim.x * blockIdx.x;
}

// Could try separating this into a separate kernel
__device__ inline void
copyNetwork(NodeElement *output, NodeElement *dst_network,
            InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[3].port_node[1 + 2 * output[4].port_port].port_node = dst_network;
    memcpy(dst_network, output, 5 * sizeof(NodeElement));

    globalQueue->count = 1;
    globalQueue->head = 0;
    globalQueue->tail = 1;

    globalQueue->buffer[0] = {output[3].port_node, output[3].port_node};
    // Using this to keep track of where to next place a node
    output[0].port_port = 5;
  }
  __shared__ int64_t queue_idx;

  // We assume the output network is a tree structure
  while (globalQueue->count != 0) {
    __syncthreads();
    uint32_t dequed_nodes = min(blockDim.x, globalQueue->count);
    if (threadIdx.x == 0) {
      globalQueue->dequeue(&queue_idx, dequed_nodes);
    }
    __syncthreads();

    if (queue_idx == -1)
      continue;

    // deque all
    NodeElement *node;
    if (threadIdx.x < dequed_nodes) {
      node = globalQueue->buffer[queue_idx + threadIdx.x].n1;
      // printf("%p: %d[%u] - %d\n", node, node[0].header.kind,
      //        node[0].header.value, globalQueue->count);

      if (threadIdx.x == 0) {
        globalQueue->ackDequeue(dequed_nodes);
      }
    }
    __syncthreads();

    if (threadIdx.x < dequed_nodes) {
      int64_t dst_net_idx =
          atomicAdd((unsigned long long *)&output->port_port,
                    1 + 2 * (NODE_ARITIES[node->header.kind] + 1));
      for (int i = 0; i < NODE_ARITIES[node->header.kind] + 1; i++) {
        // Redirect matching ports (probably not necessary)
        node[1 + 2 * i]
            .port_node[1 + 2 * node[1 + 2 * i + 1].port_port]
            .port_node = dst_network + dst_net_idx;

        // Enqueue children
        if (i > 0) {
          int64_t q_idx = -1;
          while (q_idx == -1) {
            globalQueue->enqueue(&q_idx, 1);
          }
          globalQueue->buffer[q_idx] = {node[1 + 2 * i].port_node,
                                        node[1 + 2 * i].port_node};
          globalQueue->ackEnqueue(1);
        }
      }
      memcpy(dst_network + dst_net_idx, node,
             (1 + 2 * (NODE_ARITIES[node->header.kind] + 1)) *
                 sizeof(NodeElement));
      free(node);
    }
  }
}

__global__ void runINet(InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue,
                        size_t inters_size, bool *global_done,
                        NodeElement *output_net, NodeElement *output,
                        NodeElement *network) {

  Interaction interact_buf[THREAD_BUFFER_SIZE];
  uint8_t buf_elems = 0;

  // __shared__ Interaction block_queue[BLOCK_QUEUE_SIZE]; // 16384
  // __shared__ uint32_t head, tail;
  // __shared__ int32_t count;

  __shared__ int64_t global_queue_idx;

  // Copy initial interactions
  if (blockDim.x * blockIdx.x + threadIdx.x < inters_size) {
    interact_buf[0] =
        globalQueue->buffer[blockDim.x * blockIdx.x + threadIdx.x];
    buf_elems++;
  }

  if (threadIdx.x == 0) {
    // count = 0;
    // head = 0;
    // tail = 0;
  }
  __syncthreads();

  while (true) {
    // If all threads in block are done
    if (__syncthreads_and(buf_elems == 0 /*&& count == 0*/)) {
      global_done[blockIdx.x] = true;
      // If all blocks are done
      if (__syncthreads_and(global_done[threadIdx.x % gridDim.x]) &&
          globalQueue->isEmpty())
        // Might need to synchronize across grid
        break;

      if (globalQueue->isEmpty())
        continue;
    }

    // // Attempt to dequeue block_queue if it's full
    // if (count == BLOCK_QUEUE_SIZE) {
    //   if (threadIdx.x == 0)
    //     globalQueue->enqueue(&global_queue_idx, 3 * blockDim.x);
    //   __syncthreads();
    //
    //   // If both the block queue and the global queue are full then spin
    //   if (global_queue_idx == -1)
    //     continue;
    //
    //   // Otherwise copy data from block to global queue
    //   for (int i = 0; i < 3; i++) {
    //     globalQueue->buffer[global_queue_idx + i * blockDim.x + threadIdx.x]
    //     =
    //         block_queue[(head + i * blockDim.x + threadIdx.x) %
    //                     BLOCK_QUEUE_SIZE];
    //   }
    //   __syncthreads();
    //   if (threadIdx.x == 0) {
    //     globalQueue->ackEnqueue(3 * blockDim.x);
    //     head += 3 * blockDim.x;
    //     count -= 3 * blockDim.x;
    //   }
    // }
    //
    // // Attempt to enqueue block_queue from global if nearing empty
    // if (count < blockDim.x) {
    //   if (threadIdx.x == 0)
    //     globalQueue->dequeue(&global_queue_idx, blockDim.x - count);
    //   __syncthreads();
    //
    //   if (global_queue_idx != -1) {
    //     if (threadIdx.x < blockDim.x - count)
    //       block_queue[(tail + threadIdx.x) % BLOCK_QUEUE_SIZE] =
    //           globalQueue->buffer[global_queue_idx + threadIdx.x];
    //     __syncthreads();
    //
    //     if (threadIdx.x == 0) {
    //       globalQueue->ackDequeue(blockDim.x - count);
    //       tail += blockDim.x - count;
    //       count = blockDim.x;
    //     }
    //   }
    // }

    // if (threadIdx.x == 0)
    //   printf("\n");

    bool running = true;
    // if (buf_elems == 0) {
    //   if (ensureDequeue<BLOCK_QUEUE_SIZE>(&count)) {
    //     interact_buf[0] = block_queue[atomicAdd(&head, 1) %
    //     BLOCK_QUEUE_SIZE];
    //   } else
    //     running = false;
    // } else {
    //   buf_elems--;
    // }
    // -----
    // if (head == tail) {
    //   int64_t gq_idx;
    //   globalQueue->dequeue(&gq_idx, 1);
    //   if (gq_idx != -1) {
    //     interact_buf[tail % THREAD_BUFFER_SIZE] =
    //     globalQueue->buffer[gq_idx]; tail++; globalQueue->ackDequeue(1);
    //   } else
    //     running = false;
    // }

    if (buf_elems == 0) {
      int64_t gq_idx;
      globalQueue->dequeue(&gq_idx, 1);
      if (gq_idx != -1) {
        interact_buf[buf_elems] = globalQueue->buffer[gq_idx];
        globalQueue->ackDequeue(1);
      } else
        running = false;
    } else {
      buf_elems--;
    }
    // If there is enough register space, consider loading into register
    NodeElement *left, *right;

    // Load actions
    Action *actions;
    uint8_t next_action = 0;

    NodeElement *active_pair[2];
    NodeElement *new_nodes[MAX_NEW_NODES];
    uint8_t next_new = 0;

    if (running) {
      bool switch_nodes = interact_buf[buf_elems].n1->header.kind >
                          interact_buf[buf_elems].n2->header.kind;
      // If there is enough register space, consider loading into register
      left = switch_nodes ? interact_buf[buf_elems].n2
                          : interact_buf[buf_elems].n1;
      right = switch_nodes ? interact_buf[buf_elems].n1
                           : interact_buf[buf_elems].n2;

      // printf("%p: %d[%u] >-< %p: %d[%u] - %d[%d]\n", left, left->header.kind,
      //        left->header.value, right, right->header.kind,
      //        right->header.value, threadIdx.x + blockDim.x * blockIdx.x,
      //        globalQueue->count);
      printf("%d[%u] >-< %d[%u]\n", left->header.kind, left->header.value,
             right->header.kind, right->header.value);

      // Load actions
      uint32_t index = actMapIndex(left->header.kind, right->header.kind);
      index += MAX_ACTIONS * (left->header.value == right->header.value);
      actions = actions_map + index;

      active_pair[0] = left;
      active_pair[1] = right;

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
        // printf("new: %p\n", new_nodes[next_new]);
        // Should do this in one memory operation
        new_nodes[next_new][0] = {{nna.kind, static_cast<uint16_t>(value), 0}};

        next_action++;
        next_new++;
      }
    }

    // Perform connect actions
    while (true) {
      bool valid_connect = running && next_action < MAX_ACTIONS &&
                           actions[next_action].kind == CONNECT;
      // If all threads in block have done all their connections
      if (__syncthreads_and(!valid_connect))
        break;

      ConnectAction ca;
      NodeElement *n1, *n2;
      uint64_t p1, p2;

      if (valid_connect) {
        ca = actions[next_action].action.connect;
        p1 = connect_p(ca.c1), p2 = connect_p(ca.c2);

        if (connect_g(ca.c1) == ACTIVE_PAIR) {
          n1 = active_pair[connect_n(ca.c1)];
        } else if (connect_g(ca.c1) == VARS) {
          n1 = active_pair[connect_n(ca.c1)][1 + 2 * (connect_p(ca.c1) + 1)]
                   .port_node;
          p1 = active_pair[connect_n(ca.c1)][1 + 2 * (connect_p(ca.c1) + 1) + 1]
                   .port_port;
        } else {
          n1 = new_nodes[connect_n(ca.c1)];
        }

        if (connect_g(ca.c2) == ACTIVE_PAIR) {
          n2 = active_pair[connect_n(ca.c2)];
        } else if (connect_g(ca.c2) == VARS) {
          n2 = active_pair[connect_n(ca.c2)][1 + 2 * (connect_p(ca.c2) + 1)]
                   .port_node;
          p2 = active_pair[connect_n(ca.c2)][1 + 2 * (connect_p(ca.c2) + 1) + 1]
                   .port_port;
        } else {
          n2 = new_nodes[connect_n(ca.c2)];
        }

        lock(active_pair[0]);
        lock(active_pair[1]);
        lock(n1);
        lock(n2);
      }
      __syncthreads();

      if (valid_connect) {
        if (is_locked(active_pair[0]) && is_locked(active_pair[1]) &&
            is_locked(n1) && is_locked(n2)) {
          // printf("%p: %d[%lu] --- %d[%lu]: %p\n", n1, n1->header.kind, p1,
          //        n2->header.kind, p2, n2);
          // printf("%d[%lu] --- %d[%lu]\n", n1->header.kind, p1,
          // n2->header.kind,
          //        p2);

          ((Port *)(n1 + 1))[p1] = {n2, p2};
          ((Port *)(n2 + 1))[p2] = {n1, p1};
          // Add any new interactions
          if (p1 == 0 && p2 == 0) {
            // if (tail - head < THREAD_BUFFER_SIZE) {
            //   interact_buf[tail % THREAD_BUFFER_SIZE] = {n1, n2};
            //   tail++;
            // } else {
            //   // WARNING: awful code!
            //   // If block queue full, enqueue onto global queue
            //   if (!ensureEnqueue<BLOCK_QUEUE_SIZE>(&count)) {
            int64_t g_q_idx = -1;
            while (g_q_idx == -1) {
              globalQueue->enqueue(&g_q_idx, 1);
            }
            globalQueue->buffer[g_q_idx] = {n1, n2};
            globalQueue->ackEnqueue(1);
            //   } else {
            //     uint32_t bq_idx = atomicAdd(&tail, 1);
            //     block_queue[bq_idx % BLOCK_QUEUE_SIZE] = {n1, n2};
            //   }
            // }
          }

          next_action++;
        }

        unlock(active_pair[0]);
        unlock(active_pair[1]);
        unlock(n1);
        unlock(n2);
      }
    }
    __syncthreads();

    // Perform Free actions
    if (running) {
      while (next_action < MAX_ACTIONS && actions[next_action].kind == FREE) {
        if (actions[next_action].action.free) {
          // printf("freeing: %p\n", left);
          free(left);
        } else {
          // printf("freeing: %p\n", right);
          free(right);
        }

        next_action++;
      }
    }
  }

  copyNetwork(output, output_net, globalQueue);
}
