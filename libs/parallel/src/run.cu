#include "../include/parallel/actions.cuh"
#include "../include/parallel/inet.cuh"
#include "../include/parallel/network.cuh"
#include "../include/parallel/queue.cuh"
#include "generate/grammar.hpp"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

void checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    const char *errorStr = cudaGetErrorString(err);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}

void checkCudaErrors(cudaError_t err) {
  checkCudaErrors(err, __FILE__, __LINE__);
}

void run(std::unique_ptr<grammar::Grammar> grammar, std::string input_string) {
  cudaDeviceProp *prop;
  checkCudaErrors(cudaGetDeviceProperties(prop, 0));

  dim3 grid_dims(16, 1, 1);
  dim3 block_dims(256, 1, 1);

  initActions();

  checkCudaErrors(cudaMemcpyToSymbol(NODE_ARITIES, NODE_ARITIES_H,
                                     sizeof(uint8_t) * NODE_KINDS));
  checkCudaErrors(cudaMemcpyToSymbol(actions_map, actions_map_h,
                                     sizeof(Action) * ACTIONS_MAP_SIZE));

  std::vector<grammar::Token> tokens = grammar->stringToTokens(input_string);
  HostINetwork host_network(grammar->getStackActions(), tokens);

  size_t interactions_size = host_network.getInteractions();
  size_t network_size = host_network.getNetworkSize();

  Interaction *interactions =
      (Interaction *)malloc(interactions_size * sizeof(Interaction));

  NodeElement *network_d;
  checkCudaErrors(
      cudaMalloc((void **)&network_d, sizeof(NodeElement) * network_size));
  host_network.initNetwork(network_d, interactions);

  // Initialize global queue such that the first set of interactions can be
  // immediately loaded by the threads
  InteractionQueue<MAX_INTERACTIONS_SIZE> globalQueue_h(
      interactions, interactions_size, grid_dims.x * block_dims.x);
  InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue_d;
  checkCudaErrors(cudaMalloc((void **)&globalQueue_d,
                             sizeof(InteractionQueue<MAX_INTERACTIONS_SIZE>)));
  checkCudaErrors(cudaMemcpy(globalQueue_d, &globalQueue_h,
                             sizeof(InteractionQueue<MAX_INTERACTIONS_SIZE>),
                             cudaMemcpyHostToDevice));

  cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                     MAX_INTERACTIONS_SIZE * sizeof(Interaction) +
                         MAX_NETWORK_SIZE * sizeof(NodeElement));

  bool *global_done_d;
  checkCudaErrors(
      cudaMalloc((void **)&global_done_d, sizeof(bool) * grid_dims.x));

  NodeElement *output_network_d;
  checkCudaErrors(cudaMalloc((void **)&output_network_d,
                             sizeof(NodeElement) * network_size));

  // Invoke kernel
  runINet<<<grid_dims, block_dims>>>(globalQueue_d, interactions_size,
                                     global_done_d, output_network_d,
                                     network_d + network_size - 5);

  NodeElement *output_network_h =
      (NodeElement *)malloc(sizeof(NodeElement) * network_size);
  checkCudaErrors(cudaMemcpy(output_network_h, output_network_d,
                             sizeof(NodeElement) * network_size,
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(globalQueue_d));
  checkCudaErrors(cudaFree(network_d));

  std::cout << "Done!" << std::endl;

  // traverse network_h and retrieve parse
}
