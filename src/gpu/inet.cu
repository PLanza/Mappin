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
  uint32_t prev =
      atomicCAS(&node[1].flags, 0u, threadIdx.x + 1 + (blockIdx.x << 12));
  if (node[1].flags >> 12 == blockIdx.x) {
    atomicMax(&node[1].flags, threadIdx.x + 1 + (blockIdx.x << 12));
    return node;
  }
  return prev == 0 ? node : 0;
}
__device__ inline void unlock(NodeElement *node) {
  if (!node)
    return;
  atomicCAS(&node[1].flags, threadIdx.x + 1 + (blockIdx.x << 12), 0u);
}
__device__ inline bool is_locked(NodeElement *node) {
  return node != 0 && node[1].flags == threadIdx.x + 1 + (blockIdx.x << 12);
}

__global__ void rewireInteractions(
    NodeElement *old_network,
    ParallelQueue<Interaction, MAX_INTERACTIONS_SIZE> *inters_queue) {
  uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id < inters_queue->count) {
    Interaction interaction =
        inters_queue
            ->buffer[(inters_queue->head + thread_id) % MAX_INTERACTIONS_SIZE];

    uint32_t new_n1 = (old_network + interaction.n1)[1].flags;
    uint32_t new_n2 = (old_network + interaction.n2)[1].flags;

    // printf("%d | %d(%d) --- %d(%d)\n", thread_id, interaction.n1, new_n1,
    //        interaction.n2, new_n2);

    inters_queue->buffer[(inters_queue->head + thread_id) %
                         MAX_INTERACTIONS_SIZE] = {new_n1, new_n2};
  }
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

    // If the node happens to be in the queue twice we return early
    if (atomicCAS(&node[1].flags, 0, 1) != 0)
      return;

    uint32_t dst_net_idx =
        atomicAdd(network_size, 4 + 2 * NODE_ARITIES[node->header.kind]);

    node[1].flags = dst_net_idx;
    (dst_network + dst_net_idx)[0] = node[0];
    (dst_network + dst_net_idx)[1].flags = 0;

    for (int i = 0; i < NODE_ARITIES[node->header.kind] + 1; i++) {
      uint32_t neighbor = node[2 + 2 * i].port_node;

      // If caught between CAS and Add retry
      if ((src_network + neighbor)[1].flags == 1 && dst_net_idx != 7) {
        // printf("%d | fail %ld(%d): %d[%d] --- %d: %d[%d]\n", thread_id,
        //        node - src_network, dst_net_idx, node->header.kind, i,
        //        neighbor, src_network[neighbor].header.kind, node[3 + 2 *
        //        i].port_port);
        i--;
        continue;
      }

      // If neighbor has been copied
      if ((src_network + neighbor)[1].flags != 0) {
        neighbor = (src_network + neighbor)[1].flags;

        // Write to destination network
        ((Port *)(dst_network + dst_net_idx + 2))[i] = {
            neighbor, node[3 + 2 * i].port_port};
        (dst_network + neighbor)[2 + 2 * node[3 + 2 * i].port_port].port_node =
            dst_net_idx;

        // printf("%d | dst %ld(%d): %d[%d] --- %d(%d): %d[%d]\n", thread_id,
        //        node - src_network, dst_net_idx, node->header.kind, i,
        //        node[2 + 2 * i].port_node, neighbor,
        //        dst_network[neighbor].header.kind, node[3 + 2 * i].port_port);
      } else {
        // printf("%d | src %ld(%d): %d[%d] --- %d: %d[%d]\n", thread_id,
        //        node - src_network, dst_net_idx, node->header.kind, i,
        //        neighbor, src_network[neighbor].header.kind, node[3 + 2 *
        //        i].port_port);
        atomicAggInc(&copy_queue->count);
        copy_queue
            ->buffer[atomicAggInc(&copy_queue->tail) % MAX_INTERACTIONS_SIZE] =
            neighbor;
        ((Port *)(dst_network + dst_net_idx + 2))[i] = {
            node[2 + 2 * i].port_node, node[3 + 2 * i].port_port};
      }
    }
  }
}

__global__ void
reduceNetwork(ParallelQueue<Interaction, MAX_INTERACTIONS_SIZE> *inters_queue,
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
        inters_queue->buffer[(inters_head + thread_id) % MAX_INTERACTIONS_SIZE];
    // If interacting nodes have been freed
    running = interaction.n1 != 0 && interaction.n2 != 0;
  }

  if (running) {
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

      if (nna.value == -1)
        local_nodes[2 + next_action][0] = {
            {nna.kind, local_nodes[0]->header.value}};
      else if (nna.value == -2)
        local_nodes[2 + next_action][0] = {
            {nna.kind, local_nodes[1]->header.value}};
      else if (nna.value == -3)
        local_nodes[2 + next_action][0] = {{
            nna.kind,
            static_cast<uint16_t>((local_nodes[2 + next_action] - network) *
                                  (thread_id + 1)),
        }};
      else
        local_nodes[2 + next_action][0] = {
            {nna.kind, static_cast<uint16_t>(nna.value)}};

      local_nodes[2 + next_action][1].flags = 0;

      // printf("%d | new: %ld = %d[%d]\n", thread_id,
      //        local_nodes[2 + next_action] - network, nna.kind,
      //        local_nodes[2 + next_action]->header.value);
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
    NodeElement *ln1, *ln2, *n1, *n2;
    uint32_t p1, p2;

    if (valid_connect) {
      ca = actions[next_action].action.connect;
      // printf("%d | %d --- %d\n", thread_id, ca.c1, ca.c2);
      n1 = 0;
      n2 = 0;

      ln1 = local_nodes[connect_n(ca.c1)];
      ln2 = local_nodes[connect_n(ca.c2)];

      // Acquire locks in-order to not deadlock
      if (connect_g(ca.c1) == VARS) {
        if (ln1 < network + ln1[2 * connect_p(ca.c1) + 4].port_node) {
          lock(ln1);
          if (is_locked(ln1))
            n1 = lock(network + ln1[2 * connect_p(ca.c1) + 4].port_node);
        } else {
          n1 = lock(network + ln1[2 * connect_p(ca.c1) + 4].port_node);
          if (is_locked(n1) &&
              ln1[2 * connect_p(ca.c1) + 4].port_node == n1 - network)
            lock(ln1);
        }
        p1 = ln1[2 * connect_p(ca.c1) + 5].port_port;
      } else {
        lock(ln1);
        n1 = lock(local_nodes[connect_n(ca.c1) + 2 * connect_g(ca.c1)]);
        p1 = connect_p(ca.c1);
      }

      if (connect_g(ca.c2) == VARS) {
        if (ln2 < network + ln2[2 * connect_p(ca.c2) + 4].port_node) {
          lock(ln2);
          if (is_locked(ln2))
            n2 = lock(network + ln2[2 * connect_p(ca.c2) + 4].port_node);
        } else {
          n2 = lock(network + ln2[2 * connect_p(ca.c2) + 4].port_node);
          if (is_locked(n2) &&
              ln2[2 * connect_p(ca.c2) + 4].port_node == n2 - network)
            lock(ln2);
        }
        p2 = ln2[2 * connect_p(ca.c2) + 5].port_port;
      } else {
        lock(ln2);
        // Need to lock new nodes if they've been connected to some other active
        // pair
        n2 = lock(local_nodes[connect_n(ca.c2) + 2 * connect_g(ca.c2)]);
        p2 = connect_p(ca.c2);
      }

      // if (n1 != 0 && n2 != 0)
      //   printf("%d | %lu: %d <-> %lu: %d \n\t %lu: %d --- %lu: %d\n",
      //   thread_id,
      //          ln1 - network, ln1[1].flags, ln2 - network, ln2[1].flags,
      //          n1 - network, n1[1].flags, n2 - network, n2[1].flags);
    }
    __syncthreads();

    if (valid_connect) {
      if (is_locked(ln1) && is_locked(ln2) && is_locked(n1) && is_locked(n2)) {
        // printf("%d | %lu: %d[%d] --- %lu: %d[%d]\n", thread_id, n1 - network,
        //        n1[0].header.kind, p1, n2 - network, n2[0].header.kind, p2);

        // Make connection
        ((Port *)(n1 + 2))[p1] = {(uint32_t)(n2 - network), p2};
        ((Port *)(n2 + 2))[p2] = {(uint32_t)(n1 - network), p1};

        // Push new interactions to the queue
        if (p1 == 0 && p2 == 0 && n1->header.kind != DELETE &&
            n2->header.kind != DELETE) {
          atomicAggInc(&inters_queue->count);
          inters_queue->buffer[atomicAggInc(&inters_queue->tail) %
                               MAX_INTERACTIONS_SIZE] = {
              (uint32_t)(n1 - network), (uint32_t)(n2 - network)};
        }

        next_action++;
      }

      __threadfence();

      unlock(local_nodes[connect_n(ca.c1)]);
      unlock(local_nodes[connect_n(ca.c2)]);
      unlock(n1);
      unlock(n2);
    }
  }
}
