#include "../include/parallel/actions.hpp"
#include "../include/parallel/inet.hpp"
#include "../include/parallel/kernel.cuh"
#include "../include/parallel/queue.cuh"

#include <cstdint>
#include <cstdio>

const uint8_t NODE_ARITIES_H[NODE_KINDS] = {1, 0, 2, 2, 0, 2, 2, 3, 3, 0,
                                            2, 2, 2, 1, 2, 4, 4, 4, 4, 3,
                                            1, 0, 1, 1, 1, 2, 1, 3};

__constant__ uint8_t NODE_ARITIES[NODE_KINDS];
__constant__ Action actions_map[ACTIONS_MAP_SIZE];

#define BLOCK_QUEUE_SIZE (4 * BLOCK_DIM_X)
#define MAX_NEW_NODES 4
#define THREAD_BUFFER_SIZE 4

__host__ void copyConstantData() {
  cudaMemcpyToSymbol(NODE_ARITIES, NODE_ARITIES_H, sizeof(NODE_ARITIES));
  cudaMemcpyToSymbol(actions_map, actions_map_h, sizeof(actions_map));
}

__device__ inline uint32_t actMapIndex(uint8_t left, uint8_t right) {
  return (left * (2 * NODE_KINDS - left + 1) / 2 + right - left) * 2 *
         MAX_ACTIONS;
}

__device__ inline void lock(NodeElement *node) {
  if ((node[0].header.lock >> 12) == blockIdx.x)
    atomicMax(&node[0].header.lock, threadIdx.x + (blockIdx.x << 12));
  else
    atomicCAS(&node[0].header.lock, 0u, threadIdx.x + (blockIdx.x << 12));
}
__device__ inline void unlock(NodeElement *node) {
  atomicCAS(&node[0].header.lock, threadIdx.x + (blockIdx.x << 12), 0u);
}
__device__ inline bool is_locked(NodeElement *node) {
  return node[0].header.lock == threadIdx.x + (blockIdx.x << 12);
}

__global__ void
copyNetwork(NodeElement *output, NodeElement *dst_network,
            InteractionQueue<MAX_INTERACTIONS_SIZE> *global_queue) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[3].port_node[1 + 2 * output[4].port_port].port_node = dst_network;
    memcpy(dst_network, output, 5 * sizeof(NodeElement));

    global_queue->count = 1;
    global_queue->head = 0;
    global_queue->tail = 1;

    global_queue->buffer[0] = {output[3].port_node, output[3].port_node};
    // Using this to keep track of where to next place a node
    output[0].port_port = 5;
  }
  __shared__ int64_t queue_idx;

  // We assume the output network is a tree structure
  while (global_queue->count != 0) {
    __syncthreads();
    uint32_t dequed_nodes = min(blockDim.x, global_queue->count);
    if (threadIdx.x == 0) {
      global_queue->dequeueMany(&queue_idx, dequed_nodes);
    }
    __syncthreads();

    if (queue_idx == -1)
      continue;

    // deque all
    NodeElement *node;
    if (threadIdx.x < dequed_nodes) {
      node = global_queue->buffer[queue_idx + threadIdx.x].n1;
      // printf("%p: %d[%u] - %d\n", node, node[0].header.kind,
      //        node[0].header.value, globalQueue->count);

      if (threadIdx.x == 0) {
        global_queue->ackDequeue(dequed_nodes);
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
          while (!global_queue->enqueue(
              {node[1 + 2 * i].port_node, node[1 + 2 * i].port_node})) {
          }
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
                        bool *global_done, NodeElement *network) {
  if (threadIdx.x == 0)
    global_done[blockIdx.x] = false;
  while (true) {
    // If all threads in block are done
    if (__syncthreads_and(globalQueue->isEmpty())) {
      if (threadIdx.x == 0)
        global_done[blockIdx.x] = true;
      // If all blocks are done
      if (__syncthreads_and(global_done[threadIdx.x % gridDim.x]) &&
          globalQueue->isEmpty())
        // Might need to synchronize across grid
        break;

      continue;
    }

    Interaction interaction;
    bool running = globalQueue->dequeue(&interaction);

    // Load actions
    Action *actions;
    uint8_t next_action = 0;

    NodeElement *active[2];
    NodeElement *new_nodes[MAX_NEW_NODES];

    if (running) {
      bool switch_nodes =
          interaction.n1->header.kind > interaction.n2->header.kind;
      // If there is enough register space, consider loading into register
      active[0] = switch_nodes ? interaction.n2 : interaction.n1;
      active[1] = switch_nodes ? interaction.n1 : interaction.n2;

      printf("%d | %p: %d[%u] >-< %p: %d[%u]\n",
             threadIdx.x + blockDim.x * blockIdx.x, active[0],
             active[0]->header.kind, active[0]->header.value, active[1],
             active[1]->header.kind, active[1]->header.value);

      // printf("%d[%u] >-< %d[%u]\n", left->header.kind, left->header.value,
      //        right->header.kind, right->header.value);

      // Load actions
      uint32_t index =
          actMapIndex(active[0]->header.kind, active[1]->header.kind);
      index +=
          MAX_ACTIONS * (active[0]->header.value == active[1]->header.value);
      actions = actions_map + index;

      // TODO: Test doing it all in a single loop
      while (next_action < MAX_ACTIONS && actions[next_action].kind == NEW) {
        NewNodeAction nna = actions[next_action].action.new_node;
        uint32_t value;
        if (nna.value == -1)
          value = active[0]->header.value;
        else if (nna.value == -2)
          value = active[1]->header.value;
        else if (nna.value == -3)
          value = reinterpret_cast<std::uintptr_t>(active[0]);
        else
          value = nna.value;

        new_nodes[next_action] = (NodeElement *)malloc(
            sizeof(NodeElement) * (1 + 2 * (NODE_ARITIES[nna.kind] + 1)));
        // printf("%d | new: %p\n", threadIdx.x + blockIdx.x * blockDim.x,
        //        new_nodes[next_action]);
        // if (new_nodes[next_action] == (NodeElement *)0x200000000)
        //   printf("HELP\n");

        // Should do this in one memory operation
        new_nodes[next_action][0] = {
            {nna.kind, static_cast<uint16_t>(value), 0}};

        next_action++;
      }
    }

    int32_t new_inters_count = 0;
    Interaction new_inters[5];

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
        n1 = active[connect_n(ca.c1)], n2 = active[connect_n(ca.c2)];

        if (connect_g(ca.c1) == VARS) {
          p1 = n1[1 + 2 * (connect_p(ca.c1) + 1) + 1].port_port;
          n1 = n1[1 + 2 * (connect_p(ca.c1) + 1)].port_node;
        } else if (connect_g(ca.c1) == NEW_NODES) {
          n1 = new_nodes[connect_n(ca.c1)];
        }

        if (connect_g(ca.c2) == VARS) {
          p2 = n2[1 + 2 * (connect_p(ca.c2) + 1) + 1].port_port;
          n2 = n2[1 + 2 * (connect_p(ca.c2) + 1)].port_node;
        } else if (connect_g(ca.c2) == NEW_NODES) {
          n2 = new_nodes[connect_n(ca.c2)];
        }

        lock(active[0]);
        lock(active[1]);
        lock(n1);
        lock(n2);
      }
      __syncthreads();

      if (valid_connect) {
        if (is_locked(active[0]) && is_locked(active[1]) && is_locked(n1) &&
            is_locked(n2)) {
          // printf("%d | %p: %d[%lu] --- %d[%lu]: %p\n",
          //        threadIdx.x + blockIdx.x * blockDim.x, n1,
          //        n1->header.kind, p1, n2->header.kind, p2, n2);

          // Make connection
          ((Port *)(n1 + 1))[p1] = {n2, p2};
          ((Port *)(n2 + 1))[p2] = {n1, p1};

          // Buffer new interactions
          if (p1 == 0 && p2 == 0) {
            new_inters[new_inters_count] = {n1, n2};
            new_inters_count++;
          }

          next_action++;
        }

        unlock(active[0]);
        unlock(active[1]);
        unlock(n1);
        unlock(n2);
      }
    }

    // Add new interactions once the current one is done
    for (int i = 0; i < new_inters_count; i++) {
      while (!globalQueue->enqueue(new_inters[i]))
        ;
    }

    // Perform Free actions
    if (running) {
      while (next_action < MAX_ACTIONS && actions[next_action].kind == FREE) {
        if (actions[next_action].action.free) {
          // printf("%d | freeing: %p\n", threadIdx.x + blockIdx.x * blockDim.x,
          //        active[0]);
          if (active[0] < network)
            free(active[0]);
        } else {
          // printf("%d | freeing: %p\n", threadIdx.x + blockIdx.x * blockDim.x,
          //        active[1]);
          if (active[1] < network)
            free(active[1]);
        }

        next_action++;
      }
    }

    // if (syncthreads_and(globalQueue->isEmpty()))
    //   break;
  }
}
