#ifndef __MAPPIN_PARALLEL_NETWORK__
#define __MAPPIN_PARALLEL_NETWORK__

#include "../../include/generate/grammar.hpp"
#include "inet.hpp"
#include <vector>

class HostINetwork {
private:
  uint32_t createNode(node_kind, uint16_t);
  void connect(uint32_t, uint32_t, uint32_t, uint32_t);
  node_kind getNodeKind(uint32_t);

  uint32_t stackStateToNode(grammar::StackState state);
  uint32_t createStackActionNetwork(grammar::StackAction &action);
  uint32_t createComposeNetwork(uint32_t xs, uint32_t ys);
  void createParserNetwork(std::vector<grammar::StackAction> *,
                           std::vector<grammar::Token>);

public:
  std::vector<NodeElement> network;
  std::vector<Interaction> interactions;

  HostINetwork(std::vector<grammar::StackAction> *,
               std::vector<grammar::Token>);
};

#endif
