#include <cassert>
#include <deque>
#include <iostream>
#include <memory>

#include "inet.hpp"
#include "nodes.hpp"
#include "parser.hpp"

namespace inet {

Node *tokenToNode(grammar::Token token) {
  switch (token.kind) {
  // Converts a token's ID to a value for SYM nodes
  case grammar::TERM:
    return newNode(SYM, token.id * 2);
  case grammar::NON_TERM:
    return newNode(SYM, token.id * 2 + 1);
  case grammar::ANY:
    std::cout << "TODO" << std::endl;
  case grammar::REST: {
    return newNode(BAR, 0);
  }
  }
}

Node *getStackActionNetwork(grammar::StackAction &action) {
  Node *product = newNode(GAMMA, 1);

  // Create reduction rule chains
  Node *prev_rule = product;
  for (uint32_t rule : action.reduce_rules) {
    Node *temp = newNode(RULE, rule);
    connect(prev_rule, 1, temp, 0);
    prev_rule = temp;
  }
  connect(prev_rule, 1, newNode(END, 0), 0);

  Node *slash = newNode(SLASH, 0);
  connect(slash, 0, product, 2);

  Node *lhs_start =
      (action.lhs.size() == 0) ? newNode(END, 0) : tokenToNode(action.lhs[0]);

  Node *lhs_end = lhs_start;
  // Create -|X>-|Y>-...-|Z>- chains
  for (size_t i = 1; i < action.lhs.size(); i++) {
    Node *temp = tokenToNode(action.lhs[i]);
    connect(lhs_end, 1, temp, 0);
    lhs_end = temp;
  }
  // Cap chains with | $ >- nodes
  if (lhs_end->kind == SYM) {
    Node *temp = newNode(END, 0);
    connect(lhs_end, 1, temp, 0);
    lhs_end = temp;
  }

  Node *rhs_start =
      (action.rhs.size() == 0) ? newNode(END, 0) : tokenToNode(action.rhs[0]);

  Node *rhs_end = rhs_start;
  // Create -|X>-|Y>-...-|Z>- chains
  for (size_t i = 1; i < action.rhs.size(); i++) {
    Node *temp = tokenToNode(action.rhs[i]);
    connect(rhs_end, 1, temp, 0);
    rhs_end = temp;
  }
  // Cap chains with | $ >- nodes
  if (rhs_end->kind == SYM) {
    Node *temp = newNode(END, 0);
    connect(rhs_end, 1, temp, 0);
    rhs_end = temp;
  }

  assert(lhs_end->kind == BAR || lhs_end->kind == END);
  assert(rhs_end->kind == BAR || rhs_end->kind == END);
  // Note that this violates the end token rule which we ignore
  assert(lhs_end->kind == rhs_end->kind);

  // Create -< - |-| - >- connections
  if (lhs_end->kind == BAR && rhs_end->kind == BAR)
    connect(lhs_end, 1, rhs_end, 1);

  connect(slash, 1, lhs_start, 0);
  connect(slash, 2, rhs_start, 0);
  return product;
}

// Return value is fold_xs' port #3
Node *createComposeNetwork(Node *xs, Node *ys) {
  // If passed in later layers of compositions
  Node *fold_xs = newNode(FOLD, 0);
  connect(fold_xs, 1, newNode(NIL, 0), 0);

  Node *fold_ys = newNode(FOLD, 0);
  connect(fold_ys, 1, newNode(NIL, 0), 0);

  if (xs->kind == CONS)
    connect(xs, 0, fold_xs, 0);
  else if (xs->kind == FOLD)
    connect(xs, 3, fold_xs, 0);

  if (ys->kind == CONS)
    connect(ys, 0, fold_ys, 0);
  else if (ys->kind == FOLD)
    connect(ys, 3, fold_ys, 0);

  Node *gamma_x = newNode(GAMMA, 0);
  connect(gamma_x, 0, fold_xs, 2);
  Node *gamma_acc_xs = newNode(GAMMA, 0);
  connect(gamma_acc_xs, 0, gamma_x, 2);

  Node *append = newNode(APPEND, 0);
  connect(append, 0, gamma_acc_xs, 1);
  connect(append, 1, fold_ys, 3);
  connect(append, 2, gamma_acc_xs, 2);

  Node *gamma_y = newNode(GAMMA, 0);
  connect(gamma_y, 0, fold_ys, 2);
  Node *gamma_acc_ys = newNode(GAMMA, 0);
  connect(gamma_acc_ys, 0, gamma_y, 2);

  Node *copy_acc_ys = newNode(DELTA, 1);
  connect(copy_acc_ys, 0, gamma_acc_ys, 1);

  Node *if_ = newNode(IF, 1);
  connect(if_, 2, copy_acc_ys, 2);
  connect(if_, 3, gamma_acc_ys, 2);

  Node *action_cons = newNode(CONS, 0);
  connect(action_cons, 0, if_, 1);
  connect(action_cons, 2, copy_acc_ys, 1);

  Node *cont = newNode(CONT, 0);
  connect(cont, 2, if_, 0);

  Node *prod_x = newNode(GAMMA, 1);
  connect(prod_x, 0, gamma_x, 1);
  connect(prod_x, 2, cont, 0);

  Node *prod_y = newNode(GAMMA, 1);
  connect(prod_y, 0, gamma_y, 1);
  connect(prod_y, 2, cont, 1);

  Node *rule_cons = newNode(CONS, 0);
  connect(rule_cons, 1, prod_x, 1);
  connect(rule_cons, 2, prod_y, 1);

  Node *prod_result = newNode(GAMMA, 1);
  connect(prod_result, 0, action_cons, 1);
  connect(prod_result, 1, rule_cons, 0);
  connect(prod_result, 2, cont, 3);

  return fold_xs;
}

// Returns the output node
Node *createParserNetwork(std::vector<grammar::StackAction> *stack_actions,
                          std::vector<grammar::Token> input) {

  std::vector<Node *> input_action_lists;
  for (grammar::Token token : input) {
    std::vector<grammar::StackAction> &actions = stack_actions[token.id];
    Node *tail = newNode(NIL, 0);
    for (grammar::StackAction &action : actions) {
      Node *action_net = getStackActionNetwork(action);
      Node *cons = newNode(CONS, 0);
      connect(cons, 2, tail, 0);
      connect(cons, 1, action_net, 0);
      tail = cons;
    }
    input_action_lists.push_back(tail);
  }

  std::vector<Node *> &prev_layer = input_action_lists;
  while (prev_layer.size() != 1) {
    std::vector<Node *> curr_layer;
    for (size_t i = 0; i < prev_layer.size() - 1; i += 2) {
      curr_layer.push_back(
          createComposeNetwork(prev_layer[i], prev_layer[i + 1]));
    }
    if (prev_layer.size() % 2 == 1)
      curr_layer.push_back(prev_layer.back());

    prev_layer = curr_layer;
  }

  Node *out = newNode(OUTPUT, 0);
  connect(out, 1, prev_layer[0], 3);

  return out;
}

void traverseRules(Node *cons, std::deque<uint32_t> &stack,
                   std::unique_ptr<grammar::Grammar> &grammar) {
  if (cons->kind == CONS) {
    traverseRules(cons->ports[1].node, stack, grammar);
    traverseRules(cons->ports[2].node, stack, grammar);
    return;
  }

  while (cons->kind == RULE) {
    auto const rule = grammar->getRule(cons->value);
    grammar::Token const &head = std::get<0>(rule);
    std::cout << grammar->getNonTerminalString(head.id);
    std::vector<grammar::Token> const &rhs = std::get<1>(rule);

    unsigned int nonterms = 0;
    for (auto const &token : rhs) {
      if (token.kind == grammar::NON_TERM)
        nonterms++;
    }
    if (nonterms == 0) {
      stack.front()--;
      while (stack.front() == 0) {
        std::cout << " ]";
        stack.pop_front();
        stack.front()--;
      }
    } else {
      std::cout << "[ ";
      stack.push_front(nonterms);
    }

    cons = cons->ports[1].node;
  }
}

void getParse(Node *product, std::unique_ptr<grammar::Grammar> &grammar) {
  // For each parse, check the stack action for incomplete parses
  Node *stack_action = product->ports[2].node;
  if (stack_action->ports[1].node->kind != END &&
      stack_action->ports[2].node->kind != END)
    return;

  // If valid then traverse the reduction rules and print parse
  Node *cons = product->ports[1].node;
  std::deque<uint32_t> stack;
  traverseRules(cons, stack, grammar);
  std::cout << std::endl;
}

void getParses(Node *output, std::unique_ptr<grammar::Grammar> &grammar) {
  // Traverse the list, each element a different parse
  Node *cons = output->ports[1].node;
  while (cons->kind == CONS) {
    getParse(cons->ports[1].node, grammar);
    cons = cons->ports[2].node;
  }
}

} // namespace inet
