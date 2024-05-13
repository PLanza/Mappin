#include "../include/parallel/actions.hpp"
#include "../include/parallel/inet.hpp"
#include "../include/parallel/kernel.cuh"
#include "../include/parallel/network.hpp"
#include "../include/parallel/queue.cuh"
#include "../include/parallel/run.hpp"
#include "../include/parallel/timing.cuh"
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
  std::cout << "Network is made of " << network_size << " nodes." << std::endl;
  std::cout << sizeof(actions_map_h) << std::endl;

  Interaction *interactions =
      (Interaction *)malloc(interactions_size * sizeof(Interaction));

  // Allocate network
  NodeElement *network_d;
  checkCudaErrors(
      cudaMalloc((void **)&network_d, sizeof(NodeElement) * network_size));
  host_network.initNetwork(network_d, interactions);

  // Initialize global queue such that the first set of interactions can be
  // immediately loaded by the threads
  InteractionQueue<MAX_INTERACTIONS_SIZE> *globalQueue_h =
      new InteractionQueue<MAX_INTERACTIONS_SIZE>(interactions,
                                                  interactions_size);
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

  struct cudaFuncAttributes funcAttrib;
  checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, runINet));
  printf("%s numRegs=%d\n", "runINet", funcAttrib.numRegs);

  // Timing
  Timing *timing = new Timing();
  timing->StartCounter();

  // while (true) {
  //   int32_t gq_count_h;
  //   checkCudaErrors(cudaMemcpy(&gq_count_h, &globalQueue_d->count,
  //                              sizeof(int32_t), cudaMemcpyDeviceToHost));
  //
  //   if (gq_count_h == 0)
  //     break;
  //
  //   uint32_t gq_head_h;
  //   checkCudaErrors(cudaMemcpy(&gq_head_h, &globalQueue_d->head,
  //                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
  //
  //   // Clear the queue
  //   checkCudaErrors(cudaMemcpy(&globalQueue_d->head, &globalQueue_d->tail,
  //                              sizeof(uint32_t), cudaMemcpyDeviceToDevice));
  //   checkCudaErrors(cudaMemset(&globalQueue_d->count, 0, sizeof(uint32_t)));
  //
  //   // Dynamically launch the kernel
  //   uint32_t grid_dimx = gq_count_h / BLOCK_DIM_X + 1;
  //   resolveINets<<<grid_dimx, block_dims>>>(globalQueue_d, gq_head_h,
  //                                           gq_count_h, network_d);
  //   cudaDeviceSynchronize();
  // }

  // Invoke kernel
  runINet<<<grid_dims, block_dims>>>(globalQueue_d, global_done_d, network_d);
  cudaDeviceSynchronize();

  std::cout << "Parsing took " << timing->GetCounter() << " ms" << std::endl;
  timing->StartCounter();

  copyNetwork<<<grid_dims, block_dims>>>(network_d + network_size - 5,
                                         output_network_d, globalQueue_d);
  std::cout << "Copying the network took " << timing->GetCounter() << " ms"
            << std::endl;

  uint64_t output_net_size;
  checkCudaErrors(cudaMemcpy(&output_net_size, network_d + network_size - 5,
                             sizeof(uint64_t), cudaMemcpyDeviceToHost));
  std::cout << "Output network has " << output_net_size << " NodeElements"
            << std::endl;

  NodeElement *output_network_h =
      (NodeElement *)malloc(sizeof(NodeElement) * network_size);
  checkCudaErrors(cudaMemcpy(output_network_h, output_network_d,
                             sizeof(NodeElement) * output_net_size,
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(globalQueue_d));
  checkCudaErrors(cudaFree(network_d));

  std::cout << "Parsing results: " << std::endl;
  std::vector<grammar::ParseTree *> trees =
      grammar->getParses(output_network_h, output_network_d);
  for (grammar::ParseTree *tree : trees) {
    grammar->printParseTree(tree);
    std::cout << std::endl;
  }
}
