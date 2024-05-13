#include "../../include/generate/grammar.hpp"
#include "../../include/gpu/actions.hpp"
#include "../../include/gpu/inet.hpp"
#include "../../include/gpu/kernel.cuh"
#include "../../include/gpu/network.hpp"
#include "../../include/gpu/queue.cuh"
#include "../../include/gpu/run.hpp"
#include "../../include/gpu/timing.cuh"

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

void runInteractionNet(InteractionQueue<MAX_INTERACTIONS_SIZE> *global_queue_d,
                       NodeElement *network_d) {

  uint32_t total_inters = 0;
  unsigned long long queue_head;
  int32_t queue_count;
  do {
    checkCudaErrors(cudaMemcpy(&queue_head, &global_queue_d->head,
                               sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&queue_count, &global_queue_d->count,
                               sizeof(int32_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&global_queue_d->head, &global_queue_d->tail,
                               sizeof(unsigned long long),
                               cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(&global_queue_d->count, 0, sizeof(int32_t)));

    uint32_t grid_dim_x = queue_count / BLOCK_DIM_X + 1;
    total_inters += queue_count;

    reduceInteractions<<<grid_dim_x, BLOCK_DIM_X>>>(global_queue_d, network_d,
                                                    queue_count, queue_head);
    cudaDeviceSynchronize();

    std::cout << "Total interactions so far: " << total_inters << "\n\n";

  } while (queue_count != 0);
}

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
  checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, reduceInteractions));
  printf("%s numRegs=%d\n", "runINet", funcAttrib.numRegs);

  // Timing
  Timing *timing = new Timing();
  timing->StartCounter();

  runInteractionNet(globalQueue_d, network_d);

  // Invoke kernel
  // runINet<<<grid_dims, block_dims>>>(globalQueue_d, global_done_d,
  // network_d); cudaDeviceSynchronize();

  std::cout << "Parsing took " << timing->GetCounter() << " ms" << std::endl;
  timing->StartCounter();

  uint32_t grid_dim_x = tokens.size() / BLOCK_DIM_X + 1;

  copyNetwork<<<grid_dim_x, block_dims>>>(network_d + network_size - 5,
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
