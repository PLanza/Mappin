#include "../include/parallel/actions.cuh"
#include "../include/parallel/inet.cuh"
#include "../include/parallel/network.cuh"
#include "generate/grammar.hpp"
#include <cstdlib>
#include <memory>
#include <string>

__constant__ uint8_t NODE_ARITIES[NODE_KINDS];
__constant__ Action actions_map[ACTIONS_MAP_SIZE];

void run(std::unique_ptr<grammar::Grammar> grammar, std::string input_string) {

  initActions();

  cudaMemcpyToSymbol(NODE_ARITIES, NODE_ARITIES_H,
                     sizeof(uint8_t) * NODE_KINDS);
  cudaMemcpyToSymbol(actions_map, actions_map_h,
                     sizeof(Action) * ACTIONS_MAP_SIZE);

  std::vector<grammar::Token> tokens = grammar->stringToTokens(input_string);
  HostINetwork host_network(grammar->getStackActions(), tokens);

  size_t interactions_size =
      sizeof(Interaction) * host_network.getInteractions();
  size_t network_size = sizeof(NodeElement) * host_network.getNetworkSize();

  Interaction *interactions_h = (Interaction *)malloc(interactions_size);
  NodeElement *network_h = (NodeElement *)malloc(network_size);

  host_network.initNetwork(network_h, interactions_h);

  Interaction *interactions_d;
  NodeElement *network_d;

  cudaMalloc((void **)&interactions_d, interactions_size);
  cudaMalloc((void **)&network_d, network_size);

  cudaMemcpy(interactions_d, interactions_h, interactions_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(network_d, network_h, network_size, cudaMemcpyHostToDevice);

  // Invoke kernel

  cudaMemcpy(network_h, network_d, network_size, cudaMemcpyDeviceToHost);

  cudaFree(interactions_d);
  cudaFree(network_d);

  // traverse network_h and retrieve parse
}
