#include <cstddef>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "inet.hpp"
#include "parse_net.hpp"

namespace inet {

const uint32_t MAX_NEW_NODES = 5;

Node *newNode(node_kind kind) {
  Port *ports = (Port *) calloc(node_arities[kind] + 1,  sizeof(Port));
  Node *node = (Node *) malloc(sizeof(Node));

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
  Node *n1 = interaction.n1;
  Node *n2 = interaction.n2;

  std::vector<Action> actions = actions_map[n1->kind][n2->kind];
  size_t next_action = 0;

  // Initialize all ports potentially needed
  Port *active_pair[] = {n1->ports, n2->ports};

  Port lvars[node_arities[n1->kind]];
  Port rvars[node_arities[n2->kind]];
  Port *vars[] = {lvars, rvars};

  Port *new_nodes[MAX_NEW_NODES];
  size_t next_new = 0;

  Port **all_ports[] = {active_pair, vars, new_nodes}; 

  // Get all variable ports
  for (int i = 0; i < node_arities[n1->kind]; i++) {
    lvars[i] = n1->ports[i+1];
  }
  for (int i = 0; i < node_arities[n2->kind]; i++) {
    rvars[i] = n2->ports[i+1];
  }

  // Make new nodes
  while (actions[next_action].kind == NEW)  {
    std::cout << "Making new node: " << actions[next_action].action.new_node << std::endl;
    Node *new_node = newNode(actions[next_action].action.new_node);
    new_nodes[next_new] = new_node->ports;

    next_action++;
    next_new++;
  }

  // Make connections
  while (actions[next_action].kind == CONNECT) {
    Connect c1 = actions[next_action].action.connect.c1;
    Connect c2 = actions[next_action].action.connect.c2;

    Port p1 = all_ports[c1.group][c1.index][c1.port];
    Port p2 = all_ports[c2.group][c2.index][c2.port];

    std::cout << "Connecting Nodes: (" << p1.node << ", " << p1.port << ") - (" 
      << p2.node << ", " << p2.port 
      << std::endl;
    connect(p1.node, p1.port, p2.node, p2.port);

    next_new++;
  }

  //Free nodes
  while(actions[next_action].kind == FREE) {
    if (actions[next_action].action.free) {
      std::cout << "Freeing Node: " << n1->kind << std::endl;
      freeNode(n1);
    } else {
      std::cout << "Freeing Node: " << n2->kind << std::endl;
      freeNode(n2);
    }

    next_action++;
  }
}

}
