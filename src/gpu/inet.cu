#include "../../include/gpu/actions.hpp"
#include "../../include/gpu/inet.hpp"
#include "../../include/gpu/kernel.cuh"
#include "../../include/gpu/queue.cuh"

#include <cstdint>

const uint8_t NODE_ARITIES_H[NODE_KINDS] = {1, 0, 2, 2, 0, 2, 2, 3, 3, 0,
                                            2, 2, 2, 1, 2, 4, 4, 4, 4, 3,
                                            1, 0, 1, 1, 1, 2, 1, 3};

__constant__ uint8_t NODE_ARITIES[NODE_KINDS];
__constant__ Action actions_map[ACTIONS_MAP_SIZE];

#define MAX_NEW_NODES 4

__host__ void copyConstantData() {
  cudaMemcpyToSymbol(NODE_ARITIES, NODE_ARITIES_H, sizeof(NODE_ARITIES));
  cudaMemcpyToSymbol(actions_map, actions_map_h, sizeof(actions_map));
}

__device__ inline uint32_t actMapIndex(uint8_t left, uint8_t right) {
  return (left * (2 * NODE_KINDS - left + 1) / 2 + right - left) * 2 *
         MAX_ACTIONS;
}

__device__ inline NodeElement *lock(NodeElement *node) {
  atomicCAS(&node[1].flags, 0u, threadIdx.x + 1 + (blockIdx.x << 12));
  if (node[1].flags >> 12 == blockIdx.x)
    atomicMax(&node[1].flags, threadIdx.x + 1 + (blockIdx.x << 12));
  return node;
}
__device__ inline void unlock(NodeElement *node) {
  atomicCAS(&node[1].flags, threadIdx.x + 1 + (blockIdx.x << 12), 0u);
}
__device__ inline bool is_locked(NodeElement *node) {
  return node[1].flags == threadIdx.x + 1 + (blockIdx.x << 12);
}

__global__ void
copyNetwork(NodeElement *src_network, uint32_t *network_size,
            NodeElement *dst_network,
            ParallelQueue<uint32_t, MAX_INTERACTIONS_SIZE> *copy_queue,
            int32_t copy_count, unsigned long long queue_head) {

  uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  NodeElement *node;

  if (thread_id < copy_count) {
    node = src_network +
           copy_queue->buffer[(queue_head + thread_id) % MAX_INTERACTIONS_SIZE];

    int64_t dst_net_idx =
        atomicAdd(network_size, 4 + 2 * NODE_ARITIES[node->header.kind]);

    node[1].flags = dst_net_idx;
    (dst_network + dst_net_idx)[0] = node[0];
    (dst_network + dst_net_idx)[1].flags = 0;

    for (int i = 0; i < NODE_ARITIES[node->header.kind] + 1; i++) {
      uint32_t neighbor = node[2 + 2 * i].port_node;

      // If neighbor has been copied
      if ((src_network + neighbor)[1].flags != 0) {
        neighbor = (src_network + neighbor)[1].flags;
        (dst_network + neighbor)[2 + 2 * node[3 + 2 * i].port_port].port_node =
            dst_net_idx;

        // Write to destination network
        ((Port *)(dst_network + dst_net_idx + 2))[i] = {
            neighbor, node[3 + 2 * i].port_port};
      } else {
        atomicAggInc(&copy_queue->count);
        copy_queue
            ->buffer[atomicAggInc(&copy_queue->tail) % MAX_INTERACTIONS_SIZE] =
            neighbor;
        ((Port *)(dst_network + dst_net_idx + 2))[i] = ((Port *)node + 2)[i];
      }
    }
  }
}

__global__ void
reduceNetwork(ParallelQueue<Interaction, MAX_INTERACTIONS_SIZE> *global_queue,
              NodeElement *network, uint32_t *network_size,
              int32_t inters_count, unsigned long long inters_head) {
  uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  bool running = thread_id < inters_count;

  Interaction interaction;

  Action *actions;
  uint8_t next_action = 0;

  NodeElement *local_nodes[2 + MAX_NEW_NODES];
  if (running) {
    interaction =
        global_queue->buffer[(inters_head + thread_id) % MAX_INTERACTIONS_SIZE];

    bool switch_nodes = network[interaction.n1].header.kind >
                        network[interaction.n2].header.kind;
    // If there is enough register space, consider loading into register
    local_nodes[0] =
        switch_nodes ? network + interaction.n2 : network + interaction.n1;
    local_nodes[1] =
        switch_nodes ? network + interaction.n1 : network + interaction.n2;

    // printf("%d | %lu: %d[%u] >-< %lu: %d[%u]\n", thread_id,
    //        local_nodes[0] - network, local_nodes[0]->header.kind,
    //        local_nodes[0]->header.value, local_nodes[1] - network,
    //        local_nodes[1]->header.kind, local_nodes[1]->header.value);

    // printf("%d[%u] >-< %d[%u]\n", local_nodes[0]->header.kind,
    //        local_nodes[0]->header.value, local_nodes[1]->header.kind,
    //        local_nodes[1]->header.value);

    // TODO: Coalesce loading actions
    uint32_t index =
        actMapIndex(local_nodes[0]->header.kind, local_nodes[1]->header.kind);
    index += MAX_ACTIONS *
             (local_nodes[0]->header.value == local_nodes[1]->header.value);
    actions = actions_map + index;

    while (next_action < MAX_ACTIONS && actions[next_action].kind == NEW) {
      NewNodeAction nna = actions[next_action].action.new_node;

      local_nodes[2 + next_action] =
          network + atomicAdd(network_size, (4 + 2 * NODE_ARITIES[nna.kind]));

      if (*network_size > MAX_NETWORK_SIZE)
        printf("Ran out of space!\n");

      // printf("%d | new: %p = %d\n", thread_id, local_nodes[2 + next_action],
      //        nna.kind);

      if (nna.value == -1)
        local_nodes[2 + next_action][0] = {
            {nna.kind, local_nodes[0]->header.value}};
      else if (nna.value == -2)
        local_nodes[2 + next_action][0] = {
            {nna.kind, local_nodes[1]->header.value}};
      else if (nna.value == -3)
        local_nodes[2 + next_action][0] = {{
            nna.kind,
            static_cast<uint16_t>(local_nodes[2 + next_action] - network),
        }};
      else
        local_nodes[2 + next_action][0] = {
            {nna.kind, static_cast<uint16_t>(nna.value)}};

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
    uint32_t n1, n2;
    uint32_t p1, p2;

    if (valid_connect) {
      ca = actions[next_action].action.connect;
      // printf("%d | %d --- %d\n", thread_id, ca.c1, ca.c2);
      n1 = 0;
      n2 = 0;

      // Acquire locks in-order to not deadlock
      if (connect_g(ca.c1) == VARS) {
        if (local_nodes[connect_n(ca.c1)] <
            network + local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 4]
                          .port_node) {
          lock(local_nodes[connect_n(ca.c1)]);
          if (is_locked(local_nodes[connect_n(ca.c1)]))
            n1 = lock(network +
                      local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 4]
                          .port_node) -
                 network;
        } else {
          n1 = lock(network +
                    local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 4]
                        .port_node) -
               network;
          if (is_locked(network +
                        local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 4]
                            .port_node))
            lock(local_nodes[connect_n(ca.c1)]);
        }
        p1 = local_nodes[connect_n(ca.c1)][2 * connect_p(ca.c1) + 5].port_port;
      } else {
        lock(local_nodes[connect_n(ca.c1)]);
        n1 = local_nodes[connect_n(ca.c1) + 2 * connect_g(ca.c1)] - network;
        p1 = connect_p(ca.c1);
      }

      if (connect_g(ca.c2) == VARS) {
        if (local_nodes[connect_n(ca.c2)] <
            network + local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 4]
                          .port_node) {
          lock(local_nodes[connect_n(ca.c2)]);
          if (is_locked(local_nodes[connect_n(ca.c2)]))
            n2 = lock(network +
                      local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 4]
                          .port_node) -
                 network;
        } else {
          n2 = lock(network +
                    local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 4]
                        .port_node) -
               network;
          if (is_locked(network +
                        local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 4]
                            .port_node))
            lock(local_nodes[connect_n(ca.c2)]);
        }
        p2 = local_nodes[connect_n(ca.c2)][2 * connect_p(ca.c2) + 5].port_port;
      } else {
        lock(local_nodes[connect_n(ca.c2)]);
        n2 = local_nodes[connect_n(ca.c2) + 2 * connect_g(ca.c2)] - network;
        p2 = connect_p(ca.c2);
      }
      // printf("%d | %lu: %d <-> %lu: %d \n\t %d: %d --- %d: %d\n", thread_id,
      //        local_nodes[connect_n(ca.c1)] - network,
      //        local_nodes[connect_n(ca.c1)][1].flags,
      //        local_nodes[connect_n(ca.c2)] - network,
      //        local_nodes[connect_n(ca.c2)][1].flags, n1,
      //        (network + n1)[1].flags, n2, (network + n2)[1].flags);
    }
    __syncthreads();

    if (valid_connect) {
      if (is_locked(local_nodes[connect_n(ca.c1)]) &&
          is_locked(local_nodes[connect_n(ca.c2)]) &&
          (connect_g(ca.c1) != VARS || is_locked(network + n1)) &&
          (connect_g(ca.c2) != VARS || is_locked(network + n2))) {
        // printf("%d | %d: %d[%d] --- %d: %d[%d]\n", thread_id, n1,
        //        (n1 + network)[0].header.kind, p1, n2,
        //        (n2 + network)[0].header.kind, p2);

        // Make connection
        ((Port *)(network + n1 + 2))[p1] = {n2, p2};
        ((Port *)(network + n2 + 2))[p2] = {n1, p1};

        // Push new interactions to the queue
        if (p1 == 0 && p2 == 0 && (network + n1)->header.kind != DELETE &&
            (network + n2)->header.kind != DELETE) {
          atomicAggInc(&global_queue->count);
          global_queue->buffer[atomicAggInc(&global_queue->tail) %
                               MAX_INTERACTIONS_SIZE] = {n1, n2};
        }

        next_action++;
      }
      unlock(local_nodes[connect_n(ca.c1)]);
      unlock(local_nodes[connect_n(ca.c2)]);
      unlock(network + n1);
      unlock(network + n2);
    }
  }
}
