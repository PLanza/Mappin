#include <cstddef>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "inet.hpp"
#include "parse_net.hpp"

namespace inet {

const uint32_t MAX_NEW_NODES = 5;

Node *newNode(node_kind kind) {
  Node *node = new Node;
  node->kind = kind;

  Port *ports = new Port[node_arities[kind]+ 1];
  node->ports = ports;
  return node;
}

void connect(Node *n1, std::size_t p1, Node *n2, std::size_t p2) {
  n1->ports[p1] = {n2, p2};
  n2->ports[p2] = {n1, p1};

  //Check for interaction
  if (p1 == 0 && p2 == 0) 
    interactions.push({n1, n2}); // Add to interaction queue
}

void freeNode(Node* n) {
  free(n->ports);
  free(n);
}

Action::Action(node_kind kind): kind(NEW) {
  this->action.new_node = kind;
}
Action::Action(Connect c1, Connect c2): kind(CONNECT) {
  this->action.connect.c1 = c1;
  this->action.connect.c2 = c2;
}
Action::Action(bool node): kind(FREE) {
  this->action.free = node;
}

void interact() {
  Interaction interaction = interactions.front();
  interactions.pop();
  Node *left = interaction.n1;
  Node *right = interaction.n2;

  std::vector<Action> actions = actions_map[left->kind*NODE_KINDS + right->kind];
  if (actions.empty())
    return;
  size_t next_action = 0;

  Node *active_pair[] = {left, right};

  Node *new_nodes[MAX_NEW_NODES];
  size_t next_new = 0;

  // Make new nodes
  while (actions[next_action].kind == NEW)  {
    std::cout << "Making new node: " << actions[next_action].action.new_node << std::endl;
    new_nodes[next_new] = newNode(actions[next_action].action.new_node); 

    next_action++;
    next_new++;
  }

  // Make connections
  while (actions[next_action].kind == CONNECT) {
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
        n1 = active_pair[c1.node]->ports[c1.port+1].node;
        p1 = active_pair[c1.node]->ports[c1.port+1].port;
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
        n2 = active_pair[c2.node]->ports[c2.port+1].node;
        p2 = active_pair[c2.node]->ports[c2.port+1].port;
        break;
      }
      case NEW_NODES: {
        n2 = new_nodes[c2.node];
        break;
      }
    }

    std::cout << "Connecting Nodes: (" << n1->kind << ", " << p1 << ") - (" 
      << n2->kind << ", " << p2 << ")"
      << std::endl;
    connect(n1, p1, n2, p2);

    next_action++;
  }

  //Free nodes
  while(actions[next_action].kind == FREE) {
    if (actions[next_action].action.free) {
      std::cout << "Freeing Node: " << left->kind << std::endl;
      freeNode(left);
    } else {
      std::cout << "Freeing Node: " << right->kind << std::endl;
      freeNode(right);
    }

    next_action++;
  }
}

}