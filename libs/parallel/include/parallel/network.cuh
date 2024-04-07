#ifndef __MAPPIN_PARALLEL_NETWORK__
#define __MAPPIN_PARALLEL_NETWORK__

#include "generate/grammar.hpp"
#include "inet.cuh"
#include <cstddef>
#include <vector>

class HostINetwork {
private:
  std::vector<NodeElement *> network;
  std::vector<Interaction> interactions;

  uint64_t createNode(node_kind, uint32_t);
  void connect(uint64_t, uint64_t, uint64_t, uint64_t);
  node_kind getNodeKind(uint64_t);

  uint64_t stackStateToNode(grammar::StackState state);
  uint64_t createStackActionNetwork(grammar::StackAction &action);
  uint64_t createComposeNetwork(uint64_t xs, uint64_t ys);
  void createParserNetwork(std::vector<grammar::StackAction> *,
                           std::vector<grammar::Token>);

public:
  HostINetwork(std::vector<grammar::StackAction> *,
               std::vector<grammar::Token>);
  ~HostINetwork();

  size_t getNetworkSize();
  size_t getInteractions();

  void initNetwork(NodeElement *, Interaction *);
};

#endif
