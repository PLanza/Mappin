#include "../include/parallel/inet.cuh"
#include "../include/parallel/queue.cuh"
#include <cstdint>

const uint8_t NODE_ARITIES_H[NODE_KINDS] = {
    1, 0, 2, 2, 0, 2, 2, 3, 3, 0, 3, 3, 2, 2, 2, 1, 1, 1, 0, 1, 1, 2,
};

#define BLOCK_QUEUE_SIZE 4 * 256

__global__ void runINet(NodeElement *network,
                        InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue,
                        size_t inters_size, bool *global_done) {

  Interaction local_stack[4];
  uint8_t next_stack = 1;

  __shared__ uint8_t *block_done;

  __shared__ Interaction block_queue[BLOCK_QUEUE_SIZE];
  __shared__ uint16_t count, head, tail;

  __shared__ int64_t global_queue_idx;

  // Copy initial interactions
  if (blockDim.x * blockIdx.x + threadIdx.x < inters_size)
    local_stack[0] = globalQueue->buffer[blockDim.x * blockIdx.x + threadIdx.x];

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
    block_done[threadIdx.x / 8] |= ((uint8_t)next_stack == 0)
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
        globalQueue->ackEnqueue(3 * blockDim.x);
      }
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
        globalQueue->ackEnqueue(blockDim.x - count);
      }
    }
  }

  // interact(local_q[0])

  // at the end copy network back to init_network_d
  // final network must be smaller than init network so there will be space
}
