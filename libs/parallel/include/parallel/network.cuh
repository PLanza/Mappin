#ifndef __MAPPIN_PARALLEL_NETWORK__
#define __MAPPIN_PARALLEL_NETWORK__

#include "generate/grammar.hpp"
#include "inet.cuh"
#include <cstddef>
#include <vector>

class HostINetwork {
private:
  std::vector<NodeElement *> network;

  NodeElement *createNode(node_kind, uint32_t);
  NodeElement *stackStateToNode(grammar::StackState state);
  NodeElement *createStackActionNetwork(grammar::StackAction &action);
  NodeElement *createComposeNetwork(NodeElement *xs, NodeElement *ys);
  void createParserNetwork(std::vector<grammar::StackAction> *,
                           std::vector<grammar::Token>);

public:
  HostINetwork(std::vector<grammar::StackAction> *,
               std::vector<grammar::Token>);
  ~HostINetwork();

  size_t initNetworkArray(NodeElement **);
};

#endif
