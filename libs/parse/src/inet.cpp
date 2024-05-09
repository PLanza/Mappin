#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "inet.hpp"
#include "nodes.hpp"

namespace inet {

const uint32_t MAX_NEW_NODES = 4;

std::unordered_set<Node *> nodes;

Node *newNode(node_kind kind, uint32_t value) {
  Node *node = new Node;

  node->kind = kind;
  node->value = value;

  Port *ports = new Port[node_arities[kind] + 1];
  node->ports = ports;

  nodes.insert(node);

  return node;
}

void connect(Node *n1, std::size_t p1, Node *n2, std::size_t p2) {
  n1->ports[p1] = {n2, p2};
  n2->ports[p2] = {n1, p1};

  // Check for interaction
  if (p1 == 0 && p2 == 0)
    interactions.push_back({n1, n2}); // Add to interaction queue
}

Interaction connect2(Node *n1, std::size_t p1, Node *n2, std::size_t p2) {
  n1->ports[p1] = {n2, p2};
  n2->ports[p2] = {n1, p1};

  // Check for interaction
  if (p1 == 0 && p2 == 0)
    return {n1, n2}; // Add to interaction queue
  else
    return {nullptr, nullptr};
}

void freeNode(Node *n) {
  delete[] n->ports;

  nodes.erase(n);
  delete n;
}

Action::Action(node_kind kind, int32_t value) : kind(NEW) {
  this->action.new_node.kind = kind;
  this->action.new_node.value = value;
}
Action::Action(Connect c1, Connect c2) : kind(CONNECT) {
  this->action.connect.c1 = c1;
  this->action.connect.c2 = c2;
}
Action::Action(bool node) : kind(FREE) { this->action.free = node; }

size_t total_interactions = 0;
#define BLOCK_SIZE 32
void interact() {
  // std::cout << interactions.size() << " active interactions" << std::endl;

  Interaction inters[BLOCK_SIZE];
  Interaction to_add[15 * BLOCK_SIZE];
  Port connections[15 * BLOCK_SIZE][2];

  uint8_t active = 0;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    if (interactions.empty())
      break;

    inters[i] = interactions.front();
    interactions.pop_front();
    active++;
  }

  for (int i = 0; i < BLOCK_SIZE; i++) {
    if (i >= active)
      continue;

    Node *left =
        inters[i].n1->kind < inters[i].n2->kind ? inters[i].n1 : inters[i].n2;
    Node *right =
        inters[i].n1->kind < inters[i].n2->kind ? inters[i].n2 : inters[i].n1;

    total_interactions++;

    std::cout << left->kind << "[" << left->value << "]"
              << " >-< " << right->kind << "[" << right->value << "]"
              << std::endl;

    std::vector<Action> &actions =
        getActions(left->kind, right->kind, left->value == right->value);

    if (actions.empty()) {
      std::cout << "Reached unimplemented interaction " << left->kind << " >< "
                << right->kind << std::endl;
      return;
    }

    size_t next_action = 0;

    Node *active_pair[] = {left, right};

    Node *new_nodes[MAX_NEW_NODES];
    size_t next_new = 0;

    // Make new nodes
    while (next_action < actions.size() && actions[next_action].kind == NEW) {
      NewNodeAction nna = actions[next_action].action.new_node;

      if (nna.value == -1)
        new_nodes[next_new] = newNode(nna.kind, left->value);
      else if (nna.value == -2)
        new_nodes[next_new] = newNode(nna.kind, right->value);
      else if (nna.value == -3)
        new_nodes[next_new] =
            newNode(nna.kind, reinterpret_cast<std::uintptr_t>(left));
      else
        new_nodes[next_new] = newNode(nna.kind, nna.value);

      next_action++;
      next_new++;
    }

    uint32_t connects = 0;
    // Make connections
    while (next_action < actions.size() &&
           actions[next_action].kind == CONNECT) {
      Connect c1 = actions[next_action].action.connect.c1;
      Connect c2 = actions[next_action].action.connect.c2;

      Node *n1, *n2;
      std::size_t p1 = c1.port, p2 = c2.port;

      switch (c1.group) {
      case ACTIVE_PAIR: {
        n1 = active_pair[c1.node];
        break;
      }
      case VARS: {
        n1 = active_pair[c1.node]->ports[c1.port + 1].node;
        p1 = active_pair[c1.node]->ports[c1.port + 1].port;
        break;
      }
      case NEW_NODES: {
        n1 = new_nodes[c1.node];
        break;
      }
      }

      switch (c2.group) {
      case ACTIVE_PAIR: {
        n2 = active_pair[c2.node];
        break;
      }
      case VARS: {
        n2 = active_pair[c2.node]->ports[c2.port + 1].node;
        p2 = active_pair[c2.node]->ports[c2.port + 1].port;
        break;
      }
      case NEW_NODES: {
        n2 = new_nodes[c2.node];
        break;
      }
      }

      to_add[connects * BLOCK_SIZE + i] = connect2(n1, p1, n2, p2);
      connections[connects * BLOCK_SIZE + i][0] = {n1, p1};
      connections[connects * BLOCK_SIZE + i][1] = {n2, p2};

      next_action++;
      connects++;
    }

    // Free nodes
    while (next_action < actions.size() && actions[next_action].kind == FREE) {
      if (actions[next_action].action.free)
        freeNode(left);
      else
        freeNode(right);

      next_action++;
    }
  }
  for (int i = 0; i < 15 * BLOCK_SIZE; i++) {
    if (connections[i][0].node == nullptr)
      continue;
    else {
      std::cout << connections[i][0].node->kind << "[" << connections[i][0].port
                << "] --- " << connections[i][1].node->kind << "["
                << connections[i][1].port << "]" << std::endl;
    }
    if (to_add[i].n1 == nullptr)
      continue;
    else
      interactions.push_back(to_add[i]);
  }
}

} // namespace inet
