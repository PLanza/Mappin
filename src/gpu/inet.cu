#include "../../include/gpu/actions.hpp"
#include "../../include/gpu/inet.hpp"
#include "../../include/gpu/kernel.cuh"
#include "../../include/gpu/queue.cuh"

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
  if (!node)
    return;
  atomicCAS(&node[0].header.lock, 0u, threadIdx.x + 1 + (blockIdx.x << 12));
  if ((node[0].header.lock >> 12) == blockIdx.x)
    atomicMax(&node[0].header.lock, threadIdx.x + 1 + (blockIdx.x << 12));
}
__device__ inline void unlock(NodeElement *node) {
  if (!node)
    return;
  atomicCAS(&node[0].header.lock, threadIdx.x + 1 + (blockIdx.x << 12), 0u);
}
__device__ inline bool is_locked(NodeElement *node) {
  if (!node)
    return false;
  return node[0].header.lock == threadIdx.x + 1 + (blockIdx.x << 12);
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
  __shared__ int32_t queue_idx;

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
      Interaction *to_deque = global_queue->buffer + queue_idx + threadIdx.x;

      // Wait until the interactions has been filled in
      while (!to_deque->n1 || !to_deque->n2)
        ;
      node = to_deque->n1;
      *to_deque = {nullptr, nullptr};
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

__global__ void
reduceInteractions(InteractionQueue<MAX_INTERACTIONS_SIZE> *global_queue,
                   NodeElement *network, int32_t inters_count,
                   unsigned long long inters_head) {
  uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  bool running = thread_id < inters_count;
  Interaction interaction;

  if (running) {
    interaction =
        global_queue->buffer[(inters_head + thread_id) % MAX_INTERACTIONS_SIZE];
  }

  Action *actions;
  uint8_t next_action = 0;

  NodeElement *local_nodes[2 + MAX_NEW_NODES];
  if (running) {
    bool switch_nodes =
        interaction.n1->header.kind > interaction.n2->header.kind;
    // If there is enough register space, consider loading into register
    local_nodes[0] = switch_nodes ? interaction.n2 : interaction.n1;
    local_nodes[1] = switch_nodes ? interaction.n1 : interaction.n2;

    printf("%d | %p: %d[%u] >-< %p: %d[%u]\n", thread_id, local_nodes[0],
           local_nodes[0]->header.kind, local_nodes[0]->header.value,
           local_nodes[1], local_nodes[1]->header.kind,
           local_nodes[1]->header.value);

    // printf("%d[%u] >-< %d[%u]\n", active[0]->header.kind,
    //        active[0]->header.value, active[1]->header.kind,
    //        active[1]->header.value);

    // TODO: Coalesce loading actions
    uint32_t index =
        actMapIndex(local_nodes[0]->header.kind, local_nodes[1]->header.kind);
    index += MAX_ACTIONS *
             (local_nodes[0]->header.value == local_nodes[1]->header.value);
    actions = actions_map + index;

    while (next_action < MAX_ACTIONS && actions[next_action].kind == NEW) {
      NewNodeAction nna = actions[next_action].action.new_node;

      local_nodes[2 + next_action] = (NodeElement *)malloc(
          sizeof(NodeElement) * (1 + 2 * (NODE_ARITIES[nna.kind] + 1)));
      printf("%d | new: %p = %d\n", thread_id, local_nodes[2 + next_action],
             nna.kind);

      if (nna.value == -1)
        local_nodes[2 + next_action][0] = {
            {nna.kind, local_nodes[0]->header.value, 0}};
      else if (nna.value == -2)
        local_nodes[2 + next_action][0] = {
            {nna.kind, local_nodes[1]->header.value, 0}};
      else if (nna.value == -3)
        local_nodes[2 + next_action][0] = {
            {nna.kind,
             static_cast<uint16_t>(reinterpret_cast<std::uintptr_t>(
                 local_nodes[2 + next_action])),
             0}};
      else
        local_nodes[2 + next_action][0] = {
            {nna.kind, static_cast<uint16_t>(nna.value), 0}};

      next_action++;
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
      printf("%d | %d --- %d\n", thread_id, ca.c1, ca.c2);

      // Acquire locks in-order to not deadlock
      if (connect_g(ca.c1) == VARS) {
        if (local_nodes[connect_n(ca.c1)] <
            local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 3].port_node) {
          lock(local_nodes[connect_n(ca.c1)]);
          if (is_locked(local_nodes[connect_n(ca.c1)]))
            lock(local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 3]
                     .port_node);
        } else {
          lock(local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 3]
                   .port_node);
          if (is_locked(local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 3]
                            .port_node))
            lock(local_nodes[connect_n(ca.c1)]);
        }
        n1 = local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 3].port_node;
        p1 = local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 4].port_port;
      } else {
        lock(local_nodes[connect_n(ca.c1)]);
        n1 = local_nodes[connect_n(ca.c1) + 2 * connect_g(ca.c1)];
        p1 = connect_p(ca.c1);
      }

      if (connect_g(ca.c2) == VARS) {
        if (local_nodes[connect_n(ca.c2)] <
            local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 3].port_node) {
          lock(local_nodes[connect_n(ca.c2)]);
          if (is_locked(local_nodes[connect_n(ca.c2)]))
            lock(local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 3]
                     .port_node);
        } else {
          lock(local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 3]
                   .port_node);
          if (is_locked(local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 3]
                            .port_node))
            lock(local_nodes[connect_n(ca.c2)]);
        }
        n2 = local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 3].port_node;
        p2 = local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 4].port_port;
      } else {
        lock(local_nodes[connect_n(ca.c2)]);
        n2 = local_nodes[connect_n(ca.c2) + 2 * connect_g(ca.c2)];
        p2 = connect_p(ca.c2);
      }
      printf("%d | %p: %d >-< %p: %d \n\t %p: %d --- %p: %d\n", thread_id,
             local_nodes[connect_n(ca.c1)],
             local_nodes[connect_n(ca.c1)]->header.lock,
             local_nodes[connect_n(ca.c2)],
             local_nodes[connect_n(ca.c2)]->header.lock, n1, n1->header.lock,
             n2, n2->header.lock);
    }
    __syncthreads();

    if (valid_connect) {
      if (is_locked(local_nodes[connect_n(ca.c1)]) &&
          is_locked(local_nodes[connect_n(ca.c2)]) &&
          (connect_g(ca.c1) != VARS || is_locked(n1)) &&
          (connect_g(ca.c2) != VARS || is_locked(n2))) {
        // Make connection
        printf("%d | %p: %d --- %p: %d\n", thread_id, n1, n1->header.lock, n2,
               n2->header.lock);
        ((Port *)(n1 + 1))[p1] = {n2, p2};
        ((Port *)(n2 + 1))[p2] = {n1, p1};

        // Buffer new interactions
        if (p1 == 0 && p2 == 0)
          global_queue->enqueue({n1, n2});

        next_action++;
      }
      unlock(n1);
      unlock(n2);
      unlock(local_nodes[connect_n(ca.c1)]);
      unlock(local_nodes[connect_n(ca.c2)]);
    }
  }
  __syncthreads();

  // Perform Free actions
  if (running) {
    while (next_action < MAX_ACTIONS && actions[next_action].kind == FREE) {
      if (actions[next_action].action.free) {
        printf("%d | freeing: %p\n", thread_id, local_nodes[0]);
        if (local_nodes[0] < network) {
          // Threads in block cannot free the same node so no lock contention
          while (!is_locked(local_nodes[0]))
            lock(local_nodes[0]);
          free(local_nodes[0]);
        }
      } else {
        printf("%d | freeing: %p\n", thread_id, local_nodes[1]);
        if (local_nodes[1] < network) {
          // Threads in block cannot free the same node so no lock contention
          while (!is_locked(local_nodes[1]))
            lock(local_nodes[1]);
          free(local_nodes[1]);
        }
      }

      next_action++;
    }
  }
}
