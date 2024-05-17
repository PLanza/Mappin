#include "../../include/generate/grammar.hpp"
#include "../../include/gpu/actions.hpp"
#include "../../include/gpu/inet.hpp"
#include "../../include/gpu/kernel.cuh"
#include "../../include/gpu/network.hpp"
#include "../../include/gpu/queue.cuh"
#include "../../include/gpu/run.hpp"
#include "../../include/gpu/timing.cuh"

#include <cstdint>
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

NodeElement *
copyInteractionNet(NodeElement *output_node, NodeElement *src_network,
                   NodeElement *dst_network, uint32_t *net_size,
                   ParallelQueue<uint32_t, MAX_INTERACTIONS_SIZE> *copy_queue) {
  unsigned long long queue_head;
  uint32_t value = 1;
  // Copy the output node
  checkCudaErrors(cudaMemcpy(dst_network + 1, output_node,
                             sizeof(NodeElement) * 6,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(&(output_node + 1)->flags, &value,
                             sizeof(uint32_t), cudaMemcpyHostToDevice));

  // Add the first node to the copy queue
  checkCudaErrors(cudaMemcpy(&queue_head, &copy_queue->head,
                             sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(
      cudaMemcpy(&copy_queue->buffer + queue_head % MAX_INTERACTIONS_SIZE,
                 output_node + 4, sizeof(NodeElement), cudaMemcpyDeviceToHost));

  value += 6;
  checkCudaErrors(
      cudaMemcpy(net_size, &value, sizeof(uint32_t), cudaMemcpyHostToDevice));

  int32_t queue_count = 1;
  while (queue_count != 0) {
    checkCudaErrors(cudaMemcpy(&queue_head, &copy_queue->head,
                               sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&copy_queue->head, &copy_queue->tail,
                               sizeof(unsigned long long),
                               cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(&copy_queue->count, 0, sizeof(int32_t)));
    uint32_t grid_dim_x = queue_count / BLOCK_DIM_X + 1;

    copyNetwork<<<grid_dim_x, BLOCK_DIM_X>>>(src_network, net_size, dst_network,
                                             copy_queue, queue_count,
                                             queue_head);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(&queue_count, &copy_queue->count,
                               sizeof(int32_t), cudaMemcpyDeviceToHost));
  }
  return dst_network + 1;
}

NodeElement *runInteractionNet(
    ParallelQueue<Interaction, MAX_INTERACTIONS_SIZE> *inters_queue_d,
    ParallelQueue<uint32_t, MAX_INTERACTIONS_SIZE> *copy_queue_d,
    NodeElement *&network_d, NodeElement *&network_copy_d, uint32_t *net_size_d,
    HostINetwork &starting_net) {

  uint32_t total_inters = 0;
  unsigned long long queue_head = 0;
  int32_t queue_count = starting_net.interactions.size();

  uint32_t net_size_h;
  NodeElement *output_node = network_d + starting_net.network.size() - 6;

  while (queue_count != 0) {
    checkCudaErrors(cudaMemcpy(&queue_head, &inters_queue_d->head,
                               sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&inters_queue_d->head, &inters_queue_d->tail,
                               sizeof(unsigned long long),
                               cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(&inters_queue_d->count, 0, sizeof(int32_t)));

    uint32_t grid_dim_x = queue_count / BLOCK_DIM_X + 1;
    total_inters += queue_count;

    reduceNetwork<<<grid_dim_x, BLOCK_DIM_X>>>(
        inters_queue_d, network_d, net_size_d, queue_count, queue_head);
    cudaDeviceSynchronize();
    std::cout << "Total interactions so far: " << total_inters << "\n\n";

    checkCudaErrors(cudaMemcpy(&queue_count, &inters_queue_d->count,
                               sizeof(int32_t), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(&net_size_h, net_size_d, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));
    // If nearing the limit network size copy collect it
    if (net_size_h + queue_count * 40 > MAX_NETWORK_SIZE) {
      std::cout << "Copy collecting network" << std::endl;
      output_node = copyInteractionNet(output_node, network_d, network_copy_d,
                                       net_size_d, copy_queue_d);
      NodeElement *tmp = network_d;
      network_d = network_copy_d;
      network_copy_d = tmp;
    }
  }
  return output_node;
}

void parse(std::unique_ptr<grammar::Grammar> grammar,
           std::string &input_string) {
  std::cout << "Parsing: " << input_string << std::endl;

  struct cudaFuncAttributes funcAttrib;
  checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, reduceNetwork));
  printf("%s numRegs=%d\n", "runINet", funcAttrib.numRegs);

  initActions();

  copyConstantData();

  // Set up starting interaction network
  std::vector<grammar::Token> tokens = grammar->stringToTokens(input_string);
  HostINetwork host_network(grammar->getStackActions(), tokens);
  uint32_t net_size_h = host_network.network.size();

  std::cout << "Network is made of " << net_size_h << " node elements."
            << std::endl;
  std::cout << sizeof(actions_map_h) << std::endl;

  // Allocate network
  NodeElement *network_d;
  checkCudaErrors(
      cudaMalloc((void **)&network_d, sizeof(NodeElement) * MAX_NETWORK_SIZE));
  checkCudaErrors(cudaMemcpy(network_d, host_network.network.data(),
                             sizeof(NodeElement) * net_size_h,
                             cudaMemcpyHostToDevice));

  NodeElement *network_copy_d;
  checkCudaErrors(cudaMalloc((void **)&network_copy_d,
                             sizeof(NodeElement) * MAX_NETWORK_SIZE));
  uint32_t *net_size_d;
  checkCudaErrors(cudaMalloc((void **)&net_size_d, sizeof(uint32_t)));
  checkCudaErrors(cudaMemcpy(net_size_d, &net_size_h, sizeof(uint32_t),
                             cudaMemcpyHostToDevice));

  // Initialize interaction queue
  ParallelQueue<Interaction, MAX_INTERACTIONS_SIZE> *inters_queue_d =
      newParallelQueue<Interaction, MAX_INTERACTIONS_SIZE>(
          host_network.interactions);

  // Initialize copy queue
  ParallelQueue<uint32_t, MAX_INTERACTIONS_SIZE> *copy_queue_d =
      newParallelQueue<uint32_t, MAX_INTERACTIONS_SIZE>(
          std::vector<uint32_t>());

  Timing *timing = new Timing();
  timing->StartCounter();

  NodeElement *output_node =
      runInteractionNet(inters_queue_d, copy_queue_d, network_d, network_copy_d,
                        net_size_d, host_network);

  std::cout << "Parsing took " << timing->GetCounter() << " ms" << std::endl;
  timing->StartCounter();

  copyInteractionNet(output_node, network_d, network_copy_d, net_size_d,
                     copy_queue_d);
  std::cout << "Copying the network took " << timing->GetCounter() << " ms"
            << std::endl;

  // Get network size
  uint32_t output_net_size;
  checkCudaErrors(cudaMemcpy(&output_net_size, net_size_d, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));
  std::cout << "Output network has " << output_net_size << " NodeElements"
            << std::endl;

  // Copy output network
  NodeElement *output_network_h =
      (NodeElement *)malloc(sizeof(NodeElement) * net_size_h);
  checkCudaErrors(cudaMemcpy(output_network_h, network_copy_d,
                             sizeof(NodeElement) * output_net_size,
                             cudaMemcpyDeviceToHost));

  // Get parsing results
  std::cout << "Parsing results: " << std::endl;
  std::vector<grammar::ParseTree *> trees =
      grammar->getParses(output_network_h);
  for (grammar::ParseTree *tree : trees) {
    grammar->printParseTree(tree);
    std::cout << std::endl;
  }

  checkCudaErrors(cudaFree(inters_queue_d));
  checkCudaErrors(cudaFree(network_d));
  checkCudaErrors(cudaFree(network_copy_d));
  checkCudaErrors(cudaFree(net_size_d));
}
