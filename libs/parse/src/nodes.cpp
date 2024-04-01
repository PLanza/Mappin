#include <vector>

#include "inet.hpp"
#include "nodes.hpp"

namespace inet {

int node_arities[NODE_KINDS] = {
    1, 0, 2, 2, 0, 2, 2, 3, 3, 0, 3, 3, 2, 2, 2, 1, 1, 1, 0, 1, 1, 2, 1,
};
std::string node_strings[NODE_KINDS] = {
    "out", "DEL",  "δ", "γ",   "[]", "::",  "@",   "fold",
    "if",  "bool", "k", "k\'", "/",  "○",   "○_X", "○_$",
    "-",   "-\'",  "$", "X",   "_",  "○__", "R",
};

std::deque<Interaction> interactions;
// (Node × Node × bool) -> Action[]
// bool represents if the two nodes' values match
std::vector<Action> *actions_map =
    new std::vector<Action>[NODE_KINDS * NODE_KINDS * 2];

std::vector<Action> &getActions(node_kind k1, node_kind k2, bool match) {
  if (match)
    return actions_map[2 * (k1 * NODE_KINDS + k2)];
  else
    return actions_map[2 * (k1 * NODE_KINDS + k2) + 1];
}

// Alternatively, only fill out half the table and index with ordered node kinds
void getInverseActions(std::vector<Action> &actions) {
  for (auto &action : actions) {
    switch (action.kind) {
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

void addActions(NodeKind n1, NodeKind n2, std::vector<Action> actions) {
  // TODO: check actions
  actions_map[2 * (n1 * NODE_KINDS + n2)] = actions;
  actions_map[2 * (n1 * NODE_KINDS + n2) + 1] = actions;

  getInverseActions(actions);
  actions_map[2 * (n2 * NODE_KINDS + n1)] = actions;
  actions_map[2 * (n2 * NODE_KINDS + n1) + 1] = actions;
}

void addActions(NodeKind n1, NodeKind n2, bool matches,
                std::vector<Action> actions) {
  if (matches) {
    actions_map[2 * (n1 * NODE_KINDS + n2)] = actions;
    getInverseActions(actions);
    actions_map[2 * (n2 * NODE_KINDS + n1)] = actions;
  } else {
    actions_map[2 * (n1 * NODE_KINDS + n2) + 1] = actions;
    getInverseActions(actions);
    actions_map[2 * (n2 * NODE_KINDS + n1) + 1] = actions;
  }
}

void addDeleteActions() {
  // Assume DELETE on left side
  for (unsigned int kind = DELETE; kind < NODE_KINDS; kind++) {
    std::vector<Action> actions;

    for (int i = 1; i < node_arities[kind]; i++)
      actions.push_back(Action(DELETE, 0));

    if (node_arities[kind] > 0) {
      actions.push_back(Action({ACTIVE_PAIR, 0, 0}, {VARS, 1, 0}));
    }

    for (size_t i = 1; i < node_arities[kind]; i++)
      actions.push_back(Action({NEW_NODES, i - 1, 0}, {VARS, 1, i}));

    actions.push_back(Action(false));
    if (node_arities[kind] == 0)
      actions.push_back(Action(true));

    addActions(DELETE, (NodeKind)kind, actions);
  }
}

void addDeltaActions() {
  addActions(DELTA, DELTA, true,
             {Action({VARS, 0, 0}, {VARS, 1, 0}),
              Action({VARS, 0, 1}, {VARS, 1, 1}), Action(true), Action(false)});

  addActions(DELTA, DELTA, false,
             {
                 Action(DELTA, -1),
                 Action(DELTA, -2),
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action({VARS, 0, 1}, {NEW_NODES, 1, 0}),
                 Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
                 Action({VARS, 1, 1}, {NEW_NODES, 0, 0}),
                 Action({ACTIVE_PAIR, 1, 1}, {ACTIVE_PAIR, 0, 1}),
                 Action({ACTIVE_PAIR, 0, 2}, {NEW_NODES, 1, 1}),
                 Action({ACTIVE_PAIR, 1, 2}, {NEW_NODES, 0, 1}),
                 Action({NEW_NODES, 0, 2}, {NEW_NODES, 1, 2}),
             });

  // Assume DELTA on left side
  for (unsigned int kind = GAMMA; kind < NODE_KINDS; kind++) {
    std::vector<Action> actions;
    actions.push_back(Action((node_kind)kind, -2));

    for (int i = 1; i < node_arities[kind]; i++)
      actions.push_back(Action(DELTA, -1));

    actions.push_back(Action({ACTIVE_PAIR, 1, 0}, {VARS, 0, 0}));
    actions.push_back(Action({NEW_NODES, 0, 0}, {VARS, 0, 1}));

    if (node_arities[kind] == 0) {
      actions.push_back(Action(true));
      addActions(DELTA, (NodeKind)kind, actions);
      continue;
    }

    actions.push_back(Action({ACTIVE_PAIR, 0, 0}, {VARS, 1, 0}));
    actions.push_back(Action({ACTIVE_PAIR, 0, 1}, {ACTIVE_PAIR, 1, 1}));
    actions.push_back(Action({ACTIVE_PAIR, 0, 2}, {NEW_NODES, 0, 1}));

    for (size_t i = 1; i < node_arities[kind]; i++) {
      actions.push_back(Action({NEW_NODES, i, 0}, {VARS, 1, i}));

      actions.push_back(Action({NEW_NODES, i, 1}, {ACTIVE_PAIR, 1, i + 1}));
      actions.push_back(Action({NEW_NODES, i, 2}, {NEW_NODES, 0, i + 1}));
    }

    addActions(DELTA, (NodeKind)kind, actions);
  }
}

void init() {
  addDeleteActions();
  addDeltaActions();

  addActions(GAMMA, GAMMA, true,
             {
                 Action({VARS, 0, 0}, {VARS, 1, 0}),
                 Action({VARS, 0, 1}, {VARS, 1, 1}),
                 Action(true),
                 Action(false),
             });

  addActions(APPEND, NIL,
             {
                 Action({VARS, 0, 0}, {VARS, 0, 1}),
                 Action(true),
                 Action(false),
             });
  addActions(APPEND, CONS,
             {Action({VARS, 0, 1}, {ACTIVE_PAIR, 1, 0}),
              Action({VARS, 1, 1}, {ACTIVE_PAIR, 0, 0}),
              Action({ACTIVE_PAIR, 0, 2}, {ACTIVE_PAIR, 1, 2})});

  addActions(FOLD, NIL,
             {
                 Action(DELETE, 0),
                 Action({VARS, 0, 0}, {VARS, 0, 2}),
                 Action({NEW_NODES, 0, 0}, {VARS, 0, 1}),
                 Action(true),
                 Action(false),
             });
  addActions(FOLD, CONS,
             {
                 Action(DELTA, -3),
                 Action(GAMMA, 0),
                 Action(GAMMA, 0),
                 Action({VARS, 0, 1}, {NEW_NODES, 0, 0}),
                 Action({VARS, 0, 2}, {NEW_NODES, 2, 2}),
                 Action({VARS, 1, 0}, {NEW_NODES, 1, 1}),
                 Action({VARS, 1, 1}, {ACTIVE_PAIR, 0, 0}),
                 Action({NEW_NODES, 0, 1}, {NEW_NODES, 1, 0}),
                 Action({NEW_NODES, 0, 2}, {ACTIVE_PAIR, 0, 2}),
                 Action({NEW_NODES, 1, 2}, {NEW_NODES, 2, 0}),
                 Action({NEW_NODES, 2, 1}, {ACTIVE_PAIR, 0, 3}),
                 Action(false),
             });

  addActions(IF, BOOL, true,
             {
                 Action(DELETE, 0), // Not sure about this delete
                 Action({VARS, 0, 0}, {VARS, 0, 2}),
                 Action({NEW_NODES, 0, 0}, {VARS, 0, 1}),
                 Action(true),
                 Action(false),
             });
  addActions(IF, BOOL, false,
             {
                 Action(DELETE, 0), // Not sure about this delete
                 Action({VARS, 0, 1}, {VARS, 0, 2}),
                 Action({NEW_NODES, 0, 0}, {VARS, 0, 0}),
                 Action(true),
                 Action(false),
             });

  addActions(SLASH, CONT,
             {Action(CONT_AUX, 0), Action(BAR_AUX, 0),
              Action({VARS, 0, 0}, {NEW_NODES, 1, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 1}),
              Action({VARS, 1, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 1, 1}, {NEW_NODES, 0, 2}),
              Action({VARS, 1, 2}, {ACTIVE_PAIR, 0, 0}),
              Action({ACTIVE_PAIR, 0, 1}, {NEW_NODES, 1, 1}),
              Action({ACTIVE_PAIR, 0, 2}, {NEW_NODES, 0, 3}), Action(false)});
  addActions(CONT_AUX, SLASH,
             {Action(COMP, 0), Action(BAR_AUX, 0),
              Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 2}),
              Action({VARS, 0, 2}, {NEW_NODES, 1, 1}),
              Action({VARS, 1, 0}, {NEW_NODES, 0, 1}),
              Action({VARS, 1, 1}, {NEW_NODES, 1, 0}), Action(true),
              Action(false)});

  addActions(SYM, COMP,
             {Action(COMP_SYM, -1), Action({VARS, 0, 0}, {NEW_NODES, 0, 1}),
              Action({VARS, 1, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 1, 1}, {NEW_NODES, 0, 2}), Action(true),
              Action(false)});
  addActions(ANY, COMP,
             {Action(COMP_ANY, -1), Action({VARS, 0, 0}, {NEW_NODES, 0, 1}),
              Action({VARS, 1, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 1, 1}, {NEW_NODES, 0, 2}), Action(true),
              Action(false)});
  addActions(BAR, COMP,
             {Action(BOOL, 1), Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
              Action({VARS, 1, 1}, {NEW_NODES, 0, 0}), Action(false)});
  addActions(END, COMP,
             {Action(COMP_END, 0), Action({VARS, 1, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 1, 1}, {NEW_NODES, 0, 1}), Action(true),
              Action(false)});

  addActions(COMP_SYM, SYM, true,
             {Action(COMP, 0), Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 2}),
              Action({VARS, 1, 0}, {NEW_NODES, 0, 1}), Action(true),
              Action(false)});
  addActions(COMP_SYM, SYM, false,
             {Action(BOOL, 0), Action(DELETE, 0), Action(DELETE, 0),
              Action({VARS, 0, 0}, {NEW_NODES, 1, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 0}),
              Action({VARS, 1, 0}, {NEW_NODES, 2, 0}), Action(true),
              Action(false)});
  addActions(COMP_SYM, ANY,
             {Action(COMP, 0), Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 2}),
              Action({VARS, 1, 0}, {NEW_NODES, 0, 1}), Action(true),
              Action(false)});
  addActions(COMP_SYM, END,
             {Action(BOOL, 0), Action(DELETE, 0),
              Action({VARS, 0, 0}, {NEW_NODES, 1, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 0}), Action(true),
              Action(false)});
  addActions(COMP_SYM, BAR,
             {
                 Action(BOOL, 1),
                 Action(SYM, -1),
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action({VARS, 0, 1}, {NEW_NODES, 0, 0}),
                 Action({VARS, 1, 0}, {NEW_NODES, 1, 0}),
                 Action({ACTIVE_PAIR, 1, 1}, {NEW_NODES, 1, 1}),
                 Action(true),
             });

  addActions(COMP_END, SYM,
             {Action(BOOL, 0), Action(DELETE, 0),
              Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 1, 0}, {NEW_NODES, 1, 0}), Action(true),
              Action(false)});
  addActions(COMP_END, ANY,
             {Action(BOOL, 0), Action(DELETE, 0),
              Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 1, 0}, {NEW_NODES, 1, 0}), Action(true),
              Action(false)});
  addActions(
      COMP_END, BAR,
      {Action(BOOL, 1), Action(END, 0), Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
       Action({VARS, 1, 0}, {NEW_NODES, 1, 0}), Action(true), Action(false)});
  addActions(COMP_END, END,
             {Action(BOOL, 1), Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
              Action(true), Action(false)});

  addActions(BAR, BAR,
             {
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
                 Action({ACTIVE_PAIR, 0, 1}, {ACTIVE_PAIR, 1, 1}),
             });
  addActions(BAR, SYM,
             {
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
                 Action({ACTIVE_PAIR, 0, 1}, {ACTIVE_PAIR, 1, 1}),
             });
  addActions(BAR, ANY,
             {
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
                 Action({ACTIVE_PAIR, 0, 1}, {ACTIVE_PAIR, 1, 1}),
             });
  addActions(BAR, END,
             {
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action(true),
             });
  addActions(BAR, BAR_AUX,
             {
                 Action({VARS, 0, 0}, {VARS, 1, 0}),
                 Action(true),
                 Action(false),
             });
  addActions(BAR_AUX, SYM,
             {
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
                 Action({ACTIVE_PAIR, 0, 1}, {ACTIVE_PAIR, 1, 1}),
             });
  addActions(BAR_AUX, ANY,
             {
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
                 Action({ACTIVE_PAIR, 0, 1}, {ACTIVE_PAIR, 1, 1}),
             });
  addActions(BAR_AUX, END,
             {
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action(true),
             });

  addActions(END, END,
             {
                 Action(true),
                 Action(false),
             });
  addActions(END, SYM,
             {
                 Action({VARS, 1, 0}, {ACTIVE_PAIR, 0, 0}),
                 Action(false),
             });

  addActions(COMP_ANY, SYM,
             {Action(COMP, 0), Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 2}),
              Action({VARS, 1, 0}, {NEW_NODES, 0, 1}), Action(true),
              Action(false)});
  addActions(COMP_ANY, ANY,
             {Action(COMP, 0), Action({VARS, 0, 0}, {NEW_NODES, 0, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 2}),
              Action({VARS, 1, 0}, {NEW_NODES, 0, 1}), Action(true),
              Action(false)});
  addActions(COMP_ANY, END,
             {Action(BOOL, 0), Action(DELETE, 0),
              Action({VARS, 0, 0}, {NEW_NODES, 1, 0}),
              Action({VARS, 0, 1}, {NEW_NODES, 0, 0}), Action(true),
              Action(false)});
  addActions(COMP_ANY, BAR,
             {
                 Action(BOOL, 1),
                 Action(SYM, -1),
                 Action({VARS, 0, 0}, {ACTIVE_PAIR, 1, 0}),
                 Action({VARS, 0, 1}, {NEW_NODES, 0, 0}),
                 Action({VARS, 1, 0}, {NEW_NODES, 1, 0}),
                 Action({ACTIVE_PAIR, 1, 1}, {NEW_NODES, 1, 1}),
                 Action(true),
             });
}

} // namespace inet
