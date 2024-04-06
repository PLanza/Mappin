#include "../include/parallel/network.cuh"

// TODO: merge with inet.cuh
void connect(NodeElement *n1, uint64_t p1, NodeElement *n2, uint64_t p2) {
  // We want these assignments to be a single memory write
  ((Port *)(n1 + 1))[p1] = {n2, p2};
  ((Port *)(n2 + 1))[p2] = {n1, p1};

  // if (p1 == 0 && p2 == 0)
  //   interactions.push_back({n1, n2});
}

NodeElement *HostINetwork::createNode(node_kind kind, uint32_t value) {
  NodeElement *node = new NodeElement[1 + 2 * (NODE_ARITIES[kind] + 1)];
  node[0] = {{kind, value}};

  this->network.push_back(node);
  return node;
}

NodeElement *HostINetwork::stackStateToNode(grammar::StackState state) {
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

NodeElement *
HostINetwork::createStackActionNetwork(grammar::StackAction &action) {
  NodeElement *product = this->createNode(GAMMA, 1);

  // Create reduction rule chains
  NodeElement *prev_rule = product;
  for (uint32_t rule : action.reduce_rules) {
    NodeElement *temp = this->createNode(SYM, rule);
    connect(prev_rule, 1, temp, 0);
    prev_rule = temp;
  }
  connect(prev_rule, 1, this->createNode(END, 0), 0);

  NodeElement *slash = this->createNode(SLASH, 0);
  connect(slash, 0, product, 2);

  NodeElement *lhs_start = (action.lhs.size() == 0)
                               ? this->createNode(END, 0)
                               : stackStateToNode(action.lhs[0]);

  NodeElement *lhs_end = lhs_start;
  // Create -|X>-|Y>-...-|Z>- chains
  for (size_t i = 1; i < action.lhs.size(); i++) {
    NodeElement *temp = stackStateToNode(action.lhs[i]);
    connect(lhs_end, 1, temp, 0);
    lhs_end = temp;
  }
  // Cap chains with | $ >- nodes
  if (lhs_end[0].header.kind == SYM) {
    NodeElement *temp = this->createNode(END, 0);
    connect(lhs_end, 1, temp, 0);
    lhs_end = temp;
  }

  NodeElement *rhs_start = (action.rhs.size() == 0)
                               ? this->createNode(END, 0)
                               : stackStateToNode(action.rhs[0]);

  NodeElement *rhs_end = rhs_start;
  // Create -|X>-|Y>-...-|Z>- chains
  for (size_t i = 1; i < action.rhs.size(); i++) {
    NodeElement *temp = stackStateToNode(action.rhs[i]);
    connect(rhs_end, 1, temp, 0);
    rhs_end = temp;
  }
  // Cap chains with | $ >- nodes
  if (rhs_end[0].header.kind == SYM) {
    NodeElement *temp = this->createNode(END, 0);
    connect(rhs_end, 1, temp, 0);
    rhs_end = temp;
  }

  assert(lhs_end[0].header.kind == BAR || lhs_end[0].header.kind == END);
  assert(rhs_end[0].header.kind == BAR || rhs_end[0].header.kind == END);

  // Create -< - |-| - >- connections
  if (lhs_end[0].header.kind == BAR && rhs_end[0].header.kind == BAR)
    connect(lhs_end, 1, rhs_end, 1);
  else if (lhs_end[0].header.kind == BAR && rhs_end[0].header.kind == END) {
    NodeElement *end = this->createNode(END, 0);
    connect(lhs_end, 1, end, 0);
  } else if (lhs_end[0].header.kind == END && rhs_end[0].header.kind == BAR) {
    NodeElement *end = this->createNode(END, 0);
    connect(rhs_end, 1, end, 0);
  }

  connect(slash, 1, lhs_start, 0);
  connect(slash, 2, rhs_start, 0);
  return product;
}

// Return value is fold_xs' port #3
NodeElement *HostINetwork::createComposeNetwork(NodeElement *xs,
                                                NodeElement *ys) {
  // If passed in later layers of compositions
  NodeElement *fold_xs = this->createNode(FOLD, 0);
  connect(fold_xs, 1, this->createNode(NIL, 0), 0);

  NodeElement *fold_ys = this->createNode(FOLD, 0);
  connect(fold_ys, 1, this->createNode(NIL, 0), 0);

  if (xs[0].header.kind == CONS)
    connect(xs, 0, fold_xs, 0);
  else if (xs[0].header.kind == FOLD)
    connect(xs, 3, fold_xs, 0);

  if (ys[0].header.kind == CONS)
    connect(ys, 0, fold_ys, 0);
  else if (ys[0].header.kind == FOLD)
    connect(ys, 3, fold_ys, 0);

  NodeElement *gamma_x = this->createNode(GAMMA, 0);
  connect(gamma_x, 0, fold_xs, 2);
  NodeElement *gamma_acc_xs = this->createNode(GAMMA, 0);
  connect(gamma_acc_xs, 0, gamma_x, 2);

  NodeElement *append = this->createNode(APPEND, 0);
  connect(append, 0, gamma_acc_xs, 1);
  connect(append, 1, fold_ys, 3);
  connect(append, 2, gamma_acc_xs, 2);

  NodeElement *gamma_y = this->createNode(GAMMA, 0);
  connect(gamma_y, 0, fold_ys, 2);
  NodeElement *gamma_acc_ys = this->createNode(GAMMA, 0);
  connect(gamma_acc_ys, 0, gamma_y, 2);

  NodeElement *copy_acc_ys = this->createNode(DELTA, 1);
  connect(copy_acc_ys, 0, gamma_acc_ys, 1);

  NodeElement *if_ = this->createNode(IF, 1);
  connect(if_, 2, copy_acc_ys, 2);
  connect(if_, 3, gamma_acc_ys, 2);

  NodeElement *action_cons = this->createNode(CONS, 0);
  connect(action_cons, 0, if_, 1);
  connect(action_cons, 2, copy_acc_ys, 1);

  NodeElement *cont = this->createNode(CONT, 0);
  connect(cont, 2, if_, 0);

  NodeElement *prod_x = this->createNode(GAMMA, 1);
  connect(prod_x, 0, gamma_x, 1);
  connect(prod_x, 2, cont, 0);

  NodeElement *prod_y = this->createNode(GAMMA, 1);
  connect(prod_y, 0, gamma_y, 1);
  connect(prod_y, 2, cont, 1);

  NodeElement *rule_cons = this->createNode(CONS, 0);
  connect(rule_cons, 1, prod_x, 1);
  connect(rule_cons, 2, prod_y, 1);

  NodeElement *prod_result = this->createNode(GAMMA, 1);
  connect(prod_result, 0, action_cons, 1);
  connect(prod_result, 1, rule_cons, 0);
  connect(prod_result, 2, cont, 3);

  return fold_xs;
}

void HostINetwork::createParserNetwork(
    std::vector<grammar::StackAction> *stack_actions,
    std::vector<grammar::Token> input) {
  std::vector<NodeElement *> input_action_lists;
  for (grammar::Token token : input) {
    std::vector<grammar::StackAction> &actions = stack_actions[token.id];
    NodeElement *tail = this->createNode(NIL, 0);

    for (grammar::StackAction &action : actions) {
      NodeElement *action_net = this->createStackActionNetwork(action);
      NodeElement *cons = this->createNode(CONS, 0);
      connect(cons, 2, tail, 0);
      connect(cons, 1, action_net, 0);
      tail = cons;
    }
    input_action_lists.push_back(tail);
  }

  std::vector<NodeElement *> &prev_layer = input_action_lists;
  while (prev_layer.size() != 1) {
    std::vector<NodeElement *> curr_layer;
    for (size_t i = 0; i < prev_layer.size() - 1; i += 2) {
      curr_layer.push_back(
          this->createComposeNetwork(prev_layer[i], prev_layer[i + 1]));
    }
    if (prev_layer.size() % 2 == 1)
      curr_layer.push_back(prev_layer.back());

    prev_layer = curr_layer;
  }

  NodeElement *out = this->createNode(OUTPUT, 0);
  connect(out, 1, prev_layer[0], 3);
}

HostINetwork::HostINetwork(std::vector<grammar::StackAction> *stack_actions,
                           std::vector<grammar::Token> input) {
  this->createParserNetwork(stack_actions, input);
}
size_t HostINetwork::initNetworkArray(NodeElement **network) {
  *network = new NodeElement[this->network.size()];

  for (size_t i = 0; i < this->network.size(); i++)
    (*network)[i] = *this->network[i];

  return this->network.size();
}

HostINetwork::~HostINetwork() {
  for (int i = 0; i < this->network.size(); i++)
    delete[] this->network[i];
}
