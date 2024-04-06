#include "../include/parallel/network.cuh"

size_t HostINetwork::initNetworkArray(NodeElement ** network) {

  *network = new NodeElement[this->network.size()];
  for (size_t i = 0; i < this->network.size(); i++)
    (*network)[i] = *this->network[i];
  return this->network.size();
}

void HostINetwork::createParserNetwork(std::vector<grammar::StackAction>*, std::vector<grammar::Token> input) {

}
