#ifndef __MAPPIN_PARALLEL_NETWORK__
#define __MAPPIN_PARALLEL_NETWORK__

#include "inet.cuh"
#include "generate/grammar.hpp"
#include <cstddef>
#include <vector>

class HostINetwork {
  private:
    std::vector<NodeElement *> network;

  public:
    size_t initNetworkArray(NodeElement **);
    void createParserNetwork(std::vector<grammar::StackAction> *, std::vector<grammar::Token> input);
    
};

#endif
