#include "../include/parallel/network.cuh"

void HostINetwork::connect(uint64_t n1, uint64_t p1, uint64_t n2, uint64_t p2) {
  // We want these assignments to be a single memory write
  ((Port *)(n1 + 1))[p1] = {(NodeElement *)n2, p2};
  ((Port *)(n2 + 1))[p2] = {(NodeElement *)n1, p1};

  if (p1 == 0 && p2 == 0)
    this->interactions.push_back({(NodeElement *)n1, (NodeElement *)n2});
}

uint64_t HostINetwork::createNode(node_kind kind, uint32_t value) {
  uint64_t index = this->network.size();

  NodeElement *node = new NodeElement[1 + 2 * (NODE_ARITIES_H[kind] + 1)];
  node[0] = {{kind, value}};

  this->network.push_back(node);
  return index;
}

node_kind HostINetwork::getNodeKind(uint64_t index) {
  return this->network[index][0].header.kind;
}

uint64_t HostINetwork::stackStateToNode(grammar::StackState state) {
  switch (state.kind) {
  // Converts a token's ID to a value for SYM nodes
  case grammar::SOME:
    return this->createNode(SYM, state.value);
  case grammar::ANY:
    return this->createNode(ANY, 0);
  case grammar::REST: {
    return this->createNode(BAR, 0);
  }
  }
}

uint64_t HostINetwork::createStackActionNetwork(grammar::StackAction &action) {
  uint64_t product = this->createNode(GAMMA, 1);

  // Create reduction rule chains
  uint64_t prev_rule = product;
  for (uint32_t rule : action.reduce_rules) {
    uint64_t temp = this->createNode(SYM, rule);
    connect(prev_rule, 1, temp, 0);
    prev_rule = temp;
  }
  connect(prev_rule, 1, this->createNode(END, 0), 0);

  uint64_t slash = this->createNode(SLASH, 0);
  connect(slash, 0, product, 2);

  uint64_t lhs_start = (action.lhs.size() == 0)
                           ? this->createNode(END, 0)
                           : stackStateToNode(action.lhs[0]);

  uint64_t lhs_end = lhs_start;
  // Create -|X>-|Y>-...-|Z>- chains
  for (size_t i = 1; i < action.lhs.size(); i++) {
    uint64_t temp = stackStateToNode(action.lhs[i]);
    connect(lhs_end, 1, temp, 0);
    lhs_end = temp;
  }
  // Cap chains with | $ >- nodes
  if (this->getNodeKind(lhs_end) == SYM) {
    uint64_t temp = this->createNode(END, 0);
    connect(lhs_end, 1, temp, 0);
    lhs_end = temp;
  }

  uint64_t rhs_start = (action.rhs.size() == 0)
                           ? this->createNode(END, 0)
                           : stackStateToNode(action.rhs[0]);

  uint64_t rhs_end = rhs_start;
  // Create -|X>-|Y>-...-|Z>- chains
  for (size_t i = 1; i < action.rhs.size(); i++) {
    uint64_t temp = stackStateToNode(action.rhs[i]);
    connect(rhs_end, 1, temp, 0);
    rhs_end = temp;
  }
  // Cap chains with | $ >- nodes
  if (this->getNodeKind(rhs_end) == SYM) {
    uint64_t temp = this->createNode(END, 0);
    connect(rhs_end, 1, temp, 0);
    rhs_end = temp;
  }

  assert(this->getNodeKind(lhs_end) == BAR ||
         this->getNodeKind(lhs_end) == END);
  assert(this->getNodeKind(rhs_end) == BAR ||
         this->getNodeKind(rhs_end) == END);

  // Create -< - |-| - >- connections
  if (this->getNodeKind(lhs_end) == BAR && this->getNodeKind(rhs_end) == BAR)
    connect(lhs_end, 1, rhs_end, 1);
  else if (this->getNodeKind(lhs_end) == BAR &&
           this->getNodeKind(rhs_end) == END) {
    uint64_t end = this->createNode(END, 0);
    connect(lhs_end, 1, end, 0);
  } else if (this->getNodeKind(lhs_end) == END &&
             this->getNodeKind(rhs_end) == BAR) {
    uint64_t end = this->createNode(END, 0);
    connect(rhs_end, 1, end, 0);
  }

  connect(slash, 1, lhs_start, 0);
  connect(slash, 2, rhs_start, 0);
  return product;
}

// Return value is fold_xs' port #3
uint64_t HostINetwork::createComposeNetwork(uint64_t xs, uint64_t ys) {
  // If passed in later layers of compositions
  uint64_t fold_xs = this->createNode(FOLD, 0);
  connect(fold_xs, 1, this->createNode(NIL, 0), 0);

  uint64_t fold_ys = this->createNode(FOLD, 0);
  connect(fold_ys, 1, this->createNode(NIL, 0), 0);

  if (this->getNodeKind(xs) == CONS)
    connect(xs, 0, fold_xs, 0);
  else if (this->getNodeKind(xs) == FOLD)
    connect(xs, 3, fold_xs, 0);

  if (this->getNodeKind(ys) == CONS)
    connect(ys, 0, fold_ys, 0);
  else if (this->getNodeKind(ys) == FOLD)
    connect(ys, 3, fold_ys, 0);

  uint64_t gamma_x = this->createNode(GAMMA, 0);
  connect(gamma_x, 0, fold_xs, 2);
  uint64_t gamma_acc_xs = this->createNode(GAMMA, 0);
  connect(gamma_acc_xs, 0, gamma_x, 2);

  uint64_t append = this->createNode(APPEND, 0);
  connect(append, 0, gamma_acc_xs, 1);
  connect(append, 1, fold_ys, 3);
  connect(append, 2, gamma_acc_xs, 2);

  uint64_t gamma_y = this->createNode(GAMMA, 0);
  connect(gamma_y, 0, fold_ys, 2);
  uint64_t gamma_acc_ys = this->createNode(GAMMA, 0);
  connect(gamma_acc_ys, 0, gamma_y, 2);

  uint64_t copy_acc_ys = this->createNode(DELTA, 1);
  connect(copy_acc_ys, 0, gamma_acc_ys, 1);

  uint64_t if_ = this->createNode(IF, 1);
  connect(if_, 2, copy_acc_ys, 2);
  connect(if_, 3, gamma_acc_ys, 2);

  uint64_t action_cons = this->createNode(CONS, 0);
  connect(action_cons, 0, if_, 1);
  connect(action_cons, 2, copy_acc_ys, 1);

  uint64_t cont = this->createNode(CONT, 0);
  connect(cont, 2, if_, 0);

  uint64_t prod_x = this->createNode(GAMMA, 1);
  connect(prod_x, 0, gamma_x, 1);
  connect(prod_x, 2, cont, 0);

  uint64_t prod_y = this->createNode(GAMMA, 1);
  connect(prod_y, 0, gamma_y, 1);
  connect(prod_y, 2, cont, 1);

  uint64_t rule_cons = this->createNode(CONS, 0);
  connect(rule_cons, 1, prod_x, 1);
  connect(rule_cons, 2, prod_y, 1);

  uint64_t prod_result = this->createNode(GAMMA, 1);
  connect(prod_result, 0, action_cons, 1);
  connect(prod_result, 1, rule_cons, 0);
  connect(prod_result, 2, cont, 3);

  return fold_xs;
}

void HostINetwork::createParserNetwork(
    std::vector<grammar::StackAction> *stack_actions,
    std::vector<grammar::Token> input) {
  std::vector<uint64_t> input_action_lists;
  for (grammar::Token token : input) {
    std::vector<grammar::StackAction> &actions = stack_actions[token.id];
    uint64_t tail = this->createNode(NIL, 0);

    for (grammar::StackAction &action : actions) {
      uint64_t action_net = this->createStackActionNetwork(action);
      uint64_t cons = this->createNode(CONS, 0);
      connect(cons, 2, tail, 0);
      connect(cons, 1, action_net, 0);
      tail = cons;
    }
    input_action_lists.push_back(tail);
  }

  std::vector<uint64_t> &prev_layer = input_action_lists;
  while (prev_layer.size() != 1) {
    std::vector<uint64_t> curr_layer;
    for (size_t i = 0; i < prev_layer.size() - 1; i += 2) {
      curr_layer.push_back(
          this->createComposeNetwork(prev_layer[i], prev_layer[i + 1]));
    }
    if (prev_layer.size() % 2 == 1)
      curr_layer.push_back(prev_layer.back());

    prev_layer = curr_layer;
  }

  uint64_t out = this->createNode(OUTPUT, 0);
  connect(out, 1, prev_layer[0], 3);
}

HostINetwork::HostINetwork(std::vector<grammar::StackAction> *stack_actions,
                           std::vector<grammar::Token> input) {
  this->createParserNetwork(stack_actions, input);
}

size_t HostINetwork::getNetworkSize() {
  size_t size = 0;

  for (auto const &element : this->network) {
    size += 1 + 2 * (NODE_ARITIES_H[element[0].header.kind] + 1);
  }

  return size;
}

size_t HostINetwork::getInteractions() { return this->interactions.size(); }

void HostINetwork::initNetwork(NodeElement *network,
                               Interaction *interactions) {
  size_t i = 0;
  for (auto const &element : this->network) {
    size_t element_size = 1 + 2 * (NODE_ARITIES_H[element[0].header.kind] + 1);

    network[i] = element[0];
    for (int j = 1; j < element_size; j++) {
      network[i + j] = element[j];
      if (j % 2 == 1)
        network[i + j].port_node = network + (uint64_t)element[j].port_node;
    }

    i += element_size;
  }

  for (size_t i = 0; i < this->interactions.size(); i++)
    interactions[i] = this->interactions[i];
}

HostINetwork::~HostINetwork() {
  for (int i = 0; i < this->network.size(); i++)
    delete[] this->network[i];
}
