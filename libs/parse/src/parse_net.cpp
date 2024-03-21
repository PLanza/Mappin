#include <queue>
#include <vector>

#include "parse_net.hpp"
#include "inet.hpp"

namespace inet {

int node_arities[20] = {0, 2, 2, 0, 2, 2, 3, 3, 0, 0, 3, 3, 2, 2, 2, 1, 1, 1, 0, 1};
std::queue<Interaction> interactions;
// (Node × Node × bool) -> Action[] 
// bool represents if the two nodes' values match
std::vector<Action> *actions_map = new std::vector<Action> [NODE_KINDS * NODE_KINDS * 2];

// Alternatively, only fill out half the table and index with ordered node kinds
void get_inverse_actions(std::vector<Action> &actions) {
  for (auto &action : actions) {
    switch (action.kind){
    case NEW: {
      if (action.action.new_node.value == -1) 
        action.action.new_node.value = -2;
      else if (action.action.new_node.value == -2) 
        action.action.new_node.value = -1;
      break;
    } 
    case CONNECT: {
      if (action.action.connect.c1.group != NEW_NODES)
        action.action.connect.c1.node = action.action.connect.c1.node ^ 1;
      if (action.action.connect.c2.group != NEW_NODES)
        action.action.connect.c2.node = action.action.connect.c2.node ^ 1;
      break;
    } 
    case FREE: {
      action.action.free ^= true;
      break;
    }
  }
  }
}

void add_actions(NodeKind n1, NodeKind n2, std::vector<Action> actions) {
  actions_map[2*(n1*NODE_KINDS + n2)] = actions;
  actions_map[2*(n1*NODE_KINDS + n2) + 1] = actions;

  get_inverse_actions(actions);
  actions_map[2*(n2*NODE_KINDS + n1)] = actions;
  actions_map[2*(n2*NODE_KINDS + n1) + 1] = actions;
}

void add_actions(NodeKind n1, NodeKind n2, std::vector<Action> actions, bool matches) {
  if (matches) {
    actions_map[2*(n1*NODE_KINDS + n2)] = actions;
    get_inverse_actions(actions);
    actions_map[2*(n2*NODE_KINDS + n1)] = actions;
  } else  { 
    actions_map[2*(n1*NODE_KINDS + n2) + 1] = actions;
    get_inverse_actions(actions);
    actions_map[2*(n2*NODE_KINDS + n1) + 1] = actions;
  }
}

void add_delete_actions() {
  // Assume DELETE on left side
  for (unsigned int kind = DELETE; kind <= SYM; kind++) {
    std::vector<Action> actions;

    for (int i = 1; i < node_arities[kind]; i++)
      actions.push_back(Action(DELETE, 0));

    if (node_arities[kind] > 0) {
      actions.push_back(Action({ACTIVE_PAIR, 0, 0}, {VARS, 1, 0}));
    }

    for (size_t i = 1; i < node_arities[kind]; i++)
      actions.push_back(Action({NEW_NODES, i-1, 0}, {VARS, 1, i}));

    actions.push_back(Action(false));
    add_actions(DELETE, SYM, actions);
  }
}

void add_delta_actions() {
  // Assume DELTA on left side
  for (unsigned int kind = DELTA; kind <= SYM; kind++) {
    std::vector<Action> actions;
    actions.push_back(Action((node_kind) kind, -2));

    actions.push_back(Action({ACTIVE_PAIR, 1, 0}, {VARS, 0, 0}));
    actions.push_back(Action({NEW_NODES, 0, 0}, {VARS, 0, 1}));

    if (node_arities[kind] == 0) {
      actions.push_back(Action(true));
      add_actions(DELTA, SYM, actions);
      continue;
    }

    for (int i = 1; i < node_arities[kind]; i++)
      actions.push_back(Action(DELTA, 0));

    actions.push_back(Action({ACTIVE_PAIR, 0, 0}, {VARS, 1, 0}));
    actions.push_back(Action({ACTIVE_PAIR, 0, 1}, {ACTIVE_PAIR, 1, 1}));
    actions.push_back(Action({ACTIVE_PAIR, 0, 2}, {NEW_NODES, 0, 1}));

    for (size_t i = 1; i < node_arities[kind]; i++) {
      actions.push_back(Action({NEW_NODES, i, 0}, {VARS, 1, i}));

      actions.push_back(Action({NEW_NODES, i, 1}, {ACTIVE_PAIR, 1, i+1}));
      actions.push_back(Action({NEW_NODES, i, 2}, {NEW_NODES, 0, i+1}));
    }

    add_actions(DELTA, SYM, actions);
  }
}

void init()  {
  add_delete_actions();
  add_delta_actions();
}

}
