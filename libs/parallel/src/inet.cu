#include "../include/parallel/inet.cuh"

const uint8_t NODE_ARITIES[NODE_KINDS] = {
    1, 0, 2, 2, 0, 2, 2, 3, 3, 0, 3, 3, 2, 2, 2, 1, 1, 1, 0, 1, 1, 2,
};

NodeElement *newNode(node_kind kind, uint32_t value) {
  NodeElement *node = new NodeElement[1 + 2*(NODE_ARITIES[kind] + 1)];
  node[0] = {{kind, value}};
  
  return node;
}

void connect(NodeElement *n1, uint64_t p1, NodeElement *n2, uint64_t p2) {
  // We want these assignments to be a single memory write
  ((Port *) (n1 + 1))[p1] = {n2, p2};
  ((Port *) (n2 + 1))[p2] = {n1, p1};

  // if (p1 == 0 && p2 == 0)
  //   interactions.push_back({n1, n2});
}

void freeNode(NodeElement *n) {
  delete[] n;
}
