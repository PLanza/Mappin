#include <cstdlib>
#include <queue>
#include <vector>

#include "parse_net.hpp"
#include "inet.hpp"

namespace inet {

const int NODE_KINDS = 3;
enum NodeKinds {ZERO, SUCC, ADD};

int *node_arities = new int[NODE_KINDS];
std::queue<Interaction> interactions;
std::vector<Action> *actions_map = new std::vector<Action> [NODE_KINDS * NODE_KINDS];

void init()  {
  // Initialize ports
  node_arities[ZERO] = 0;
  node_arities[SUCC] = 1;
  node_arities[ADD] = 2;
  
  // Initialize actions map
  actions_map[ZERO*NODE_KINDS + ADD] = {
    Action({VARS, 1, 0}, {VARS, 1, 1}),
    Action(true),  // Frees ZERO
    Action(false), // Frees ADD
  };
  actions_map[ADD*NODE_KINDS + ZERO] = { 
    Action({VARS, 0, 0}, {VARS, 0, 1}),
    Action(false), // Frees ZERO
    Action(true),  // Frees ADD
  };

  actions_map[SUCC*NODE_KINDS + ADD] = {
    Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
    Action({VARS, 1, 1}, {ACTIVE_PAIR, 0, 0}),
    Action({ACTIVE_PAIR, 0, 1}, {ACTIVE_PAIR, 1, 2}),
  };

  actions_map[ADD*NODE_KINDS + SUCC] = {
    Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
    Action({VARS, 0, 1}, {ACTIVE_PAIR, 1, 0}),
    Action({ACTIVE_PAIR, 1, 1}, {ACTIVE_PAIR, 0, 2}),
  };

  actions_map[ZERO*NODE_KINDS + SUCC] = {};
  actions_map[SUCC*NODE_KINDS + ZERO] = {};

  // Create network (toy example)
  Node *z1 = newNode(ZERO);
  Node *z2 = newNode(ZERO);
  Node *z3 = newNode(ZERO);
  Node *s1 = newNode(SUCC);
  Node *s2 = newNode(SUCC);
  Node *add = newNode(ADD);

  connect(z1, 0, s1, 1);
  connect(s1, 0, add, 0);
  connect(s2, 0, add, 1);
  connect(z2, 0, s1, 1);
  connect(z3, 0, add, 2);
}

}
