#include "../../include/gpu/network.hpp"

void HostINetwork::connect(uint64_t n1, uint64_t p1, uint64_t n2, uint64_t p2) {
  this->network[n1][1 + 2 * p1] = {.port_node = (NodeElement *)n2};
  this->network[n1][1 + 2 * p1 + 1] = {.port_port = p2};
  this->network[n2][1 + 2 * p2] = {.port_node = (NodeElement *)n1};
  this->network[n2][1 + 2 * p2 + 1] = {.port_port = p1};

  if (p1 == 0 && p2 == 0)
    this->interactions.push_back({(NodeElement *)n1, (NodeElement *)n2});
}

uint64_t HostINetwork::createNode(node_kind kind, uint16_t value) {
  uint64_t index = this->network.size();

  NodeElement *node = new NodeElement[1 + 2 * (NODE_ARITIES_H[kind] + 1)];
  node[0] = {{kind, value, 0}};

  if (this->node_positions.size() == 0)
    this->node_positions.push_back(0);
  else
    this->node_positions.push_back(
        this->node_positions.back() + 1 +
        2 * (NODE_ARITIES_H[this->network.back()[0].header.kind] + 1));

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
  case grammar::REST:
    return this->createNode(BAR, 0);
  case grammar::STAR:
    return this->createNode(STAR, 0);
  case grammar::END_STAR:
    return this->createNode(END_STAR, 0);
  }
  return 0;
}

uint64_t HostINetwork::createStackActionNetwork(grammar::StackAction &action) {
  std::vector<uint64_t> rule_star_nodes;
  uint64_t product = this->createNode(GAMMA, 1);

  // Create reduction rule chains
  uint64_t prev_rule = product;
  uint64_t prev_rule_star;
  for (int32_t rule : action.reduce_rules) {
    if (rule == -1) {
      uint64_t rule_star = this->createNode(RULE_STAR, 0);
      rule_star_nodes.push_back(rule_star);

      this->connect(prev_rule, 1, rule_star, 1);

      prev_rule = rule_star;
      prev_rule_star = rule_star;
    } else if (rule == -2) {
      uint64_t rule_end_star = this->createNode(END_STAR, 0);

      this->connect(prev_rule, 1, rule_end_star, 0);
      this->connect(prev_rule_star, 3, rule_end_star, 2);

      prev_rule = rule_end_star;
    } else {
      uint64_t temp = this->createNode(SYM, rule);

      if (prev_rule == prev_rule_star)
        this->connect(prev_rule, 2, temp, 0);
      else
        this->connect(prev_rule, 1, temp, 0);

      prev_rule = temp;
    }
  }
  this->connect(prev_rule, 1, this->createNode(END, 0), 0);

  uint64_t slash = this->createNode(SLASH, 0);
  this->connect(slash, 0, product, 2);

  uint64_t lhs_start = (action.lhs.size() == 0)
                           ? this->createNode(END, 0)
                           : stackStateToNode(action.lhs[0]);

  uint64_t lhs_end = lhs_start;
  // Create -|X>-|Y>-...-|Z>- chains
  for (size_t i = 1; i < action.lhs.size(); i++) {
    uint64_t temp = stackStateToNode(action.lhs[i]);
    if (this->getNodeKind(temp) == END_STAR) {
      uint64_t rule_star = rule_star_nodes.back();
      rule_star_nodes.pop_back();
      this->connect(rule_star, 0, temp, 2);
    }
    this->connect(lhs_end, 1, temp, 0);
    lhs_end = temp;
  }
  // Cap chains with | $ >- nodes
  if (this->getNodeKind(lhs_end) == SYM) {
    uint64_t temp = this->createNode(END, 0);
    this->connect(lhs_end, 1, temp, 0);
    lhs_end = temp;
  }

  uint64_t rhs_start = (action.rhs.size() == 0)
                           ? this->createNode(END, 0)
                           : stackStateToNode(action.rhs[0]);

  uint64_t rhs_end = rhs_start;
  // Create -|X>-|Y>-...-|Z>- chains
  for (size_t i = 1; i < action.rhs.size(); i++) {
    uint64_t temp = stackStateToNode(action.rhs[i]);
    this->connect(rhs_end, 1, temp, 0);
    rhs_end = temp;
  }
  // Cap chains with | $ >- nodes
  if (this->getNodeKind(rhs_end) == SYM) {
    uint64_t temp = this->createNode(END, 0);
    this->connect(rhs_end, 1, temp, 0);
    rhs_end = temp;
  }

  assert(this->getNodeKind(lhs_end) == BAR ||
         this->getNodeKind(lhs_end) == END);
  assert(this->getNodeKind(rhs_end) == BAR ||
         this->getNodeKind(rhs_end) == END);

  // Create -< - |-| - >- connections
  if (this->getNodeKind(lhs_end) == BAR && this->getNodeKind(rhs_end) == BAR)
    this->connect(lhs_end, 1, rhs_end, 1);
  else if (this->getNodeKind(lhs_end) == BAR &&
           this->getNodeKind(rhs_end) == END) {
    uint64_t end = this->createNode(END, 0);
    this->connect(lhs_end, 1, end, 0);
  } else if (this->getNodeKind(lhs_end) == END &&
             this->getNodeKind(rhs_end) == BAR) {
    uint64_t end = this->createNode(END, 0);
    this->connect(rhs_end, 1, end, 0);
  }

  this->connect(slash, 1, lhs_start, 0);
  this->connect(slash, 2, rhs_start, 0);
  return product;
}

// Return value is fold_xs' port #3
uint64_t HostINetwork::createComposeNetwork(uint64_t xs, uint64_t ys) {
  // If passed in later layers of compositions
  uint64_t fold_xs = this->createNode(FOLD, 0);
  this->connect(fold_xs, 1, this->createNode(NIL, 0), 0);

  uint64_t fold_ys = this->createNode(FOLD, 0);
  this->connect(fold_ys, 1, this->createNode(NIL, 0), 0);

  if (this->getNodeKind(xs) == CONS)
    this->connect(xs, 0, fold_xs, 0);
  else if (this->getNodeKind(xs) == FOLD)
    this->connect(xs, 3, fold_xs, 0);

  if (this->getNodeKind(ys) == CONS)
    this->connect(ys, 0, fold_ys, 0);
  else if (this->getNodeKind(ys) == FOLD)
    this->connect(ys, 3, fold_ys, 0);

  uint64_t gamma_x = this->createNode(GAMMA, 0);
  this->connect(gamma_x, 0, fold_xs, 2);
  uint64_t gamma_acc_xs = this->createNode(GAMMA, 0);
  this->connect(gamma_acc_xs, 0, gamma_x, 2);

  uint64_t append = this->createNode(APPEND, 0);
  this->connect(append, 0, gamma_acc_xs, 1);
  this->connect(append, 1, fold_ys, 3);
  this->connect(append, 2, gamma_acc_xs, 2);

  uint64_t gamma_y = this->createNode(GAMMA, 0);
  this->connect(gamma_y, 0, fold_ys, 2);
  uint64_t gamma_acc_ys = this->createNode(GAMMA, 0);
  this->connect(gamma_acc_ys, 0, gamma_y, 2);

  uint64_t copy_acc_ys = this->createNode(DELTA, 1);
  this->connect(copy_acc_ys, 0, gamma_acc_ys, 1);

  uint64_t if_ = this->createNode(IF, 1);
  this->connect(if_, 2, copy_acc_ys, 2);
  this->connect(if_, 3, gamma_acc_ys, 2);

  uint64_t action_cons = this->createNode(CONS, 0);
  this->connect(action_cons, 0, if_, 1);
  this->connect(action_cons, 2, copy_acc_ys, 1);

  uint64_t cont = this->createNode(IF, 0);
  this->connect(cont, 2, if_, 0);

  uint64_t prod_x = this->createNode(GAMMA, 1);
  this->connect(prod_x, 0, gamma_x, 1);
  this->connect(prod_x, 2, cont, 0);

  uint64_t prod_y = this->createNode(GAMMA, 1);
  this->connect(prod_y, 0, gamma_y, 1);
  this->connect(prod_y, 2, cont, 1);

  uint64_t rule_cons = this->createNode(CONS, 0);
  this->connect(rule_cons, 1, prod_x, 1);
  this->connect(rule_cons, 2, prod_y, 1);

  uint64_t prod_result = this->createNode(GAMMA, 1);
  this->connect(prod_result, 0, action_cons, 1);
  this->connect(prod_result, 1, rule_cons, 0);
  this->connect(prod_result, 2, cont, 3);

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
      this->connect(cons, 2, tail, 0);
      this->connect(cons, 1, action_net, 0);
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
  this->connect(out, 1, prev_layer[0], 3);
}

HostINetwork::HostINetwork(std::vector<grammar::StackAction> *stack_actions,
                           std::vector<grammar::Token> input) {
  this->createParserNetwork(stack_actions, input);
}

size_t HostINetwork::getNetworkSize() {
  // The position of the final node + 5 (the size of an OUTPUT node)
  return this->node_positions.back() + 5;
}

size_t HostINetwork::getInteractions() { return this->interactions.size(); }

// Return the output node
void HostINetwork::initNetwork(NodeElement *network,
                               Interaction *interactions) {
  for (size_t i = 0; i < this->network.size(); i++) {
    NodeElement *element = this->network[i];
    size_t element_size = 1 + 2 * (NODE_ARITIES_H[element[0].header.kind] + 1);

    NodeElement node[element_size];
    node[0] = element[0];

    for (int j = 1; j < element_size; j++) {
      if (j % 2 == 1)
        node[j].port_node =
            network + this->node_positions[(uint64_t)element[j].port_node];
      else
        node[j] = element[j];
    }
    cudaMemcpy(network + this->node_positions[i], node,
               sizeof(NodeElement) * element_size, cudaMemcpyHostToDevice);
  }

  for (size_t i = 0; i < this->interactions.size(); i++)
    interactions[i] = {
        network + this->node_positions[(uint64_t)this->interactions[i].n1],
        network + this->node_positions[(uint64_t)this->interactions[i].n2]};
}

HostINetwork::~HostINetwork() {
  for (int i = 0; i < this->network.size(); i++)
    delete[] this->network[i];
}
