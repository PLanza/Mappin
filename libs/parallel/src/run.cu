#include "../include/parallel/actions.hpp"
#include "../include/parallel/inet.hpp"
#include "../include/parallel/kernel.cuh"
#include "../include/parallel/network.hpp"
#include "../include/parallel/queue.cuh"
#include "../include/parallel/run.hpp"
#include "generate/grammar.hpp"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

void _checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    const char *errorStr = cudaGetErrorString(err);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(err) _checkCudaErrors(err, __FILE__, __LINE__)

void parse(std::unique_ptr<grammar::Grammar> grammar,
           std::string &input_string) {
  std::cout << "Parsing: " << input_string << std::endl;

  dim3 grid_dims(GRID_DIM_X, 1, 1);
  dim3 block_dims(BLOCK_DIM_X, 1, 1);

  initActions();

  copyConstantData();

  // Set up starting interaction network
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
  InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue_h =
      new InteractionQueue<MAX_INTERACTIONS_SIZE>(
          interactions, interactions_size, grid_dims.x * block_dims.x);
  InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue_d;
  checkCudaErrors(cudaMalloc((void **)&globalQueue_d,
                             sizeof(InteractionQueue<MAX_INTERACTIONS_SIZE>)));
  checkCudaErrors(cudaMemcpy(globalQueue_d, globalQueue_h,
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
                                     network_d + network_size - 5, network_d);

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
