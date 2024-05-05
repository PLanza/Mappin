#include <cassert>
#include <iostream>
#include <vector>

#include "../include/parallel/actions.hpp"
#include "../include/parallel/inet.hpp"

// DO NOT REMOVE!: idk why but without this line we get segfaults
__constant__ Action actions_m[ACTIONS_MAP_SIZE];

Action createAction(node_kind kind, int8_t value) {
  return Action{NEW, {.new_node = NewNodeAction{kind, value}}};
}

uint8_t createConnect(Group g, uint8_t n, uint8_t p) {
  return p | (n << 3) | (g << 6);
}

Action createAction(Group g1, uint8_t n1, uint8_t p1, Group g2, uint8_t n2,
                    uint8_t p2) {
  return Action{
      CONNECT,
      {.connect = {createConnect(g1, n1, p1), createConnect(g2, n2, p2)}}};
}

Action createAction(bool node) { return Action{FREE, {.free = node}}; }

// Triangular matrix
Action actions_map_h[ACTIONS_MAP_SIZE]; // ~32Kb

// Get index into actions_map
size_t actMapIndex(NodeKind n1, NodeKind n2) {
  size_t i = n1 <= n2 ? n1 : n2;
  size_t j = n1 <= n2 ? n2 : n1;

  return (i * (2 * NODE_KINDS - i + 1) / 2 + j - i) * 2 * MAX_ACTIONS;
}

bool checkActions(NodeKind n1, NodeKind n2, std::vector<Action> actions) {
  bool seen_connect = false;
  bool seen_free = false;
  bool seen_none = false;

  NodeKind active_pair[2] = {n1, n2};

  std::vector<uint8_t> new_arities;
  for (Action const &action : actions) {
    switch (action.kind) {
    case NEW: {
      if (seen_connect || seen_free || seen_none)
        return false;

      new_arities.push_back(NODE_ARITIES_H[action.action.new_node.kind]);
      break;
    }
    case CONNECT: {
      if (seen_free || seen_none)
        return false;
      seen_connect = true;
      auto &[c1, c2] = action.action.connect;

      switch (connect_g(c1)) {
      case ACTIVE_PAIR:
      case VARS: {
        if (connect_n(c1) > 1 ||
            connect_p(c1) > NODE_ARITIES_H[active_pair[connect_n(c1)]])
          return false;
        break;
      }
      case NEW_NODES: {
        if (connect_n(c1) > new_arities.size() ||
            connect_p(c1) > new_arities[connect_n(c1)])
          return false;
        break;
      }
      }

      switch (connect_g(c2)) {
      case ACTIVE_PAIR:
      case VARS: {
        if (connect_n(c2) > 1 ||
            connect_p(c2) > NODE_ARITIES_H[active_pair[connect_n(c2)]])
          return false;
        break;
      }
      case NEW_NODES: {
        if (connect_n(c2) > new_arities.size() ||
            connect_p(c2) > new_arities[connect_n(c2)])
          return false;
        break;
      }
      }
      break;
    }
    case FREE: {
      seen_free = true;
      if (seen_none)
        return false;
      break;
    }
    case NONE: {
      seen_none = true;
      break;
    }
    }
  }
  return true;
}

void addActions(NodeKind n1, NodeKind n2, std::vector<Action> actions) {
  if (!checkActions(n1, n2, actions)) {
    std::cout << "Error adding action between node types " << n1 << " and "
              << n2 << std::endl;
    return;
  }
  for (int i = 0; i < actions.size(); i++) {
    actions_map_h[actMapIndex(n1, n2) + i] = actions[i];
    actions_map_h[actMapIndex(n1, n2) + MAX_ACTIONS + i] = actions[i];
  }
}

void addActions(NodeKind n1, NodeKind n2, bool matches,
                std::vector<Action> actions) {

  assert(n1 <= n2);

  if (!checkActions(n1, n2, actions)) {
    std::cout << "Error adding action between node types " << n1 << " and "
              << n2 << std::endl;
    return;
  }
  for (int i = 0; i < actions.size(); i++) {
    if (matches)
      actions_map_h[actMapIndex(n1, n2) + i + MAX_ACTIONS] = actions[i];
    else
      actions_map_h[actMapIndex(n1, n2) + i] = actions[i];
  }
}

void addDeleteActions() {
  for (uint8_t kind = DELETE; kind < NODE_KINDS; kind++) {
    std::vector<Action> actions;

    for (int i = 1; i < NODE_ARITIES_H[kind]; i++)
      actions.push_back(createAction(DELETE, 0));

    if (NODE_ARITIES_H[kind] > 0) {
      actions.push_back(createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0));
    }

    for (size_t i = 1; i < NODE_ARITIES_H[kind]; i++)
      actions.push_back(createAction(VARS, 1, i, NEW_NODES, i - 1, 0));

    actions.push_back(createAction(false));
    if (NODE_ARITIES_H[kind] == 0)
      actions.push_back(createAction(true));

    addActions(DELETE, (NodeKind)kind, actions);
  }
}

void addDeltaActions() {
  addActions(DELTA, DELTA, true,
             {createAction(VARS, 0, 0, VARS, 1, 0),
              createAction(VARS, 0, 1, VARS, 1, 1), createAction(true),
              createAction(false)});

  addActions(DELTA, DELTA, false,
             {
                 createAction(DELTA, -1),
                 createAction(DELTA, -2),
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 1, 0),
                 createAction(VARS, 1, 1, NEW_NODES, 0, 0),
                 createAction(ACTIVE_PAIR, 1, 1, ACTIVE_PAIR, 0, 1),
                 createAction(ACTIVE_PAIR, 0, 2, NEW_NODES, 1, 1),
                 createAction(ACTIVE_PAIR, 1, 2, NEW_NODES, 0, 1),
                 createAction(NEW_NODES, 0, 2, NEW_NODES, 1, 2),
             });

  for (unsigned int kind = GAMMA; kind < NODE_KINDS; kind++) {
    std::vector<Action> actions;
    actions.push_back(createAction((node_kind)kind, -2));

    for (int i = 1; i < NODE_ARITIES_H[kind]; i++)
      actions.push_back(createAction(DELTA, -1));

    actions.push_back(createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0));
    actions.push_back(createAction(VARS, 0, 1, NEW_NODES, 0, 0));

    if (NODE_ARITIES_H[kind] == 0) {
      actions.push_back(createAction(true));
      addActions(DELTA, (NodeKind)kind, actions);
      continue;
    }

    actions.push_back(createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0));
    for (size_t i = 1; i < NODE_ARITIES_H[kind]; i++)
      actions.push_back(createAction(VARS, 1, i, NEW_NODES, i, 0));

    actions.push_back(createAction(ACTIVE_PAIR, 0, 1, ACTIVE_PAIR, 1, 1));
    actions.push_back(createAction(ACTIVE_PAIR, 0, 2, NEW_NODES, 0, 1));
    for (size_t i = 1; i < NODE_ARITIES_H[kind]; i++)
      actions.push_back(createAction(ACTIVE_PAIR, 1, i + 1, NEW_NODES, i, 1));

    for (size_t i = 1; i < NODE_ARITIES_H[kind]; i++)
      actions.push_back(createAction(NEW_NODES, i, 2, NEW_NODES, 0, i + 1));

    addActions(DELTA, (NodeKind)kind, actions);
  }
}

void initActions() {
  for (int i = 0; i < NODE_KINDS * NODE_KINDS * 2 * MAX_ACTIONS; i++)
    actions_map_h[i] = {NONE, {false}};
  addDeltaActions();
  addDeleteActions();
  addActions(GAMMA, GAMMA, true,
             {
                 createAction(VARS, 0, 0, VARS, 1, 0),
                 createAction(VARS, 0, 1, VARS, 1, 1),
                 createAction(true),
                 createAction(false),
             });

  addActions(NIL, APPEND,
             {
                 createAction(VARS, 1, 0, VARS, 1, 1),
                 createAction(true),
                 createAction(false),
             });
  addActions(CONS, APPEND,
             {createAction(VARS, 0, 1, ACTIVE_PAIR, 1, 0),
              createAction(VARS, 1, 1, ACTIVE_PAIR, 0, 0),
              createAction(ACTIVE_PAIR, 0, 2, ACTIVE_PAIR, 1, 2)});

  addActions(NIL, FOLD,
             {
                 createAction(DELETE, 0),
                 createAction(VARS, 1, 0, VARS, 1, 2),
                 createAction(VARS, 1, 1, NEW_NODES, 0, 0),
                 createAction(true),
                 createAction(false),
             });
  addActions(CONS, FOLD,
             {
                 createAction(DELTA, -3),
                 createAction(GAMMA, 0),
                 createAction(GAMMA, 0),
                 createAction(VARS, 0, 1, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 1, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 2, NEW_NODES, 2, 2),
                 createAction(VARS, 0, 0, NEW_NODES, 1, 1),
                 createAction(ACTIVE_PAIR, 1, 2, NEW_NODES, 0, 2),
                 createAction(ACTIVE_PAIR, 1, 3, NEW_NODES, 2, 1),
                 createAction(NEW_NODES, 0, 1, NEW_NODES, 1, 0),
                 createAction(NEW_NODES, 1, 2, NEW_NODES, 2, 0),
                 createAction(true),
             });

  addActions(IF, BOOL, true,
             {
                 createAction(DELETE, 0), // Not sure about this delete
                 createAction(VARS, 0, 0, VARS, 0, 2),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 0),
                 createAction(true),
                 createAction(false),
             });
  addActions(IF, BOOL, false,
             {
                 createAction(DELETE, 0), // Not sure about this delete
                 createAction(VARS, 0, 1, VARS, 0, 2),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(true),
                 createAction(false),
             });

  addActions(CONT, SLASH,
             {createAction(CONT_AUX, 0), createAction(BAR, 1),
              createAction(VARS, 0, 2, ACTIVE_PAIR, 1, 0),
              createAction(VARS, 1, 0, NEW_NODES, 1, 0),
              createAction(VARS, 1, 1, NEW_NODES, 0, 1),
              createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2),
              createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 1, 1),
              createAction(ACTIVE_PAIR, 1, 2, NEW_NODES, 0, 3),
              createAction(true)});
  addActions(CONT_AUX, SLASH,
             {createAction(COMP, 0), createAction(BAR, 1),
              createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2),
              createAction(VARS, 0, 2, NEW_NODES, 1, 1),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1),
              createAction(VARS, 1, 1, NEW_NODES, 1, 0), createAction(true),
              createAction(false)});

  addActions(COMP, SYM,
             {createAction(COMP_SYM, -2),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1),
              createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2), createAction(true),
              createAction(false)});
  addActions(COMP, ANY,
             {createAction(COMP_ANY, 0),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1),
              createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2), createAction(true),
              createAction(false)});
  addActions(COMP, BAR,
             {createAction(BOOL, 1),
              createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 0), createAction(true)});
  addActions(COMP, END,
             {createAction(COMP_END, 0),
              createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 1), createAction(true),
              createAction(false)});

  addActions(COMP_SYM, SYM, true,
             {createAction(COMP, 0), createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1), createAction(true),
              createAction(false)});
  addActions(COMP_SYM, SYM, false,
             {createAction(BOOL, 0), createAction(DELETE, 0),
              createAction(DELETE, 0),
              createAction(VARS, 0, 0, NEW_NODES, 1, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 0),
              createAction(VARS, 1, 0, NEW_NODES, 2, 0), createAction(true),
              createAction(false)});
  addActions(COMP_SYM, ANY,
             {createAction(COMP, 0), createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1), createAction(true),
              createAction(false)});
  addActions(COMP_SYM, END,
             {createAction(BOOL, 0), createAction(DELETE, 0),
              createAction(VARS, 0, 0, NEW_NODES, 1, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 0), createAction(true),
              createAction(false)});
  addActions(COMP_SYM, BAR,
             {
                 createAction(BOOL, 1),
                 createAction(SYM, -1),
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 0, NEW_NODES, 1, 0),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 1, 1),
                 createAction(true),
             });

  addActions(COMP_END, SYM,
             {createAction(BOOL, 0), createAction(DELETE, 0),
              createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 1, 0, NEW_NODES, 1, 0), createAction(true),
              createAction(false)});
  addActions(COMP_END, ANY,
             {createAction(BOOL, 0), createAction(DELETE, 0),
              createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 1, 0, NEW_NODES, 1, 0), createAction(true),
              createAction(false)});
  addActions(COMP_END, BAR,
             {createAction(BOOL, 1), createAction(END, 0),
              createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 1, 0, NEW_NODES, 1, 0), createAction(true),
              createAction(false)});
  addActions(COMP_END, END,
             {createAction(BOOL, 1), createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(true), createAction(false)});

  addActions(COMP_ANY, SYM,
             {createAction(COMP, 0), createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1), createAction(true),
              createAction(false)});
  addActions(COMP_ANY, ANY,
             {createAction(COMP, 0), createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1), createAction(true),
              createAction(false)});
  addActions(COMP_ANY, END,
             {createAction(BOOL, 0), createAction(DELETE, 0),
              createAction(VARS, 0, 0, NEW_NODES, 1, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 0), createAction(true),
              createAction(false)});
  addActions(COMP_ANY, BAR,
             {
                 createAction(BOOL, 1),
                 createAction(SYM, -1),
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 0, NEW_NODES, 1, 0),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 1, 1),
                 createAction(true),
             });

  addActions(BAR, BAR, true,
             {
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 1, ACTIVE_PAIR, 1, 1),
             });
  addActions(BAR, SYM,
             {
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 1, ACTIVE_PAIR, 1, 1),
             });
  addActions(BAR, ANY,
             {
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 1, ACTIVE_PAIR, 1, 1),
             });
  addActions(BAR, END,
             {
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(true),
             });
  addActions(BAR, BAR, false,
             {
                 createAction(VARS, 0, 0, VARS, 1, 0),
                 createAction(true),
                 createAction(false),
             });

  addActions(BAR, STAR,
             {
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 1, ACTIVE_PAIR, 1, 1),
             });
  addActions(BAR, END_STAR,
             {
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 1, ACTIVE_PAIR, 1, 1),
             });

  addActions(END, END,
             {
                 createAction(true),
                 createAction(false),
             });
  addActions(END, SYM,
             {
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(false),
             });

  addActions(COMP_ANY, SYM,
             {createAction(COMP, 0), createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1), createAction(true),
              createAction(false)});
  addActions(COMP_ANY, ANY,
             {createAction(COMP, 0), createAction(VARS, 0, 0, NEW_NODES, 0, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 2),
              createAction(VARS, 1, 0, NEW_NODES, 0, 1), createAction(true),
              createAction(false)});
  addActions(COMP_ANY, END,
             {createAction(BOOL, 0), createAction(DELETE, 0),
              createAction(VARS, 0, 0, NEW_NODES, 1, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 0), createAction(true),
              createAction(false)});
  addActions(COMP_ANY, BAR,
             {
                 createAction(BOOL, 1),
                 createAction(SYM, -1),
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 0, NEW_NODES, 1, 0),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 1, 1),
                 createAction(true),
             });

  addActions(COMP_SYM, STAR,
             {
                 createAction(SLASH, 0),
                 createAction(SLASH, 0),
                 createAction(SYM, -1),
                 createAction(COMP_STAR_SYM, -1),
                 createAction(VARS, 0, 0, NEW_NODES, 3, 1),
                 createAction(VARS, 0, 1, NEW_NODES, 3, 2),
                 createAction(VARS, 1, 0, NEW_NODES, 3, 0),
                 createAction(ACTIVE_PAIR, 1, 0, NEW_NODES, 0, 1),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 0, 2),
                 createAction(NEW_NODES, 0, 0, NEW_NODES, 3, 4),
                 createAction(NEW_NODES, 1, 0, NEW_NODES, 3, 3),
                 createAction(NEW_NODES, 2, 0, NEW_NODES, 1, 1),
                 createAction(NEW_NODES, 2, 1, NEW_NODES, 1, 2),
                 createAction(true),
             });
  addActions(COMP_STAR_SYM, SYM, true,
             {
                 createAction(COMP_STAR, 0),
                 createAction(VARS, 0, 3, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 2),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 3),
                 createAction(VARS, 1, 0, NEW_NODES, 0, 1),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 0, 4),
                 createAction(true),
             });
  addActions(COMP_STAR_SYM, SYM, false,
             {
                 createAction(COMP_STAR_AUX, 0),
                 createAction(STAR_DEL, 0),
                 createAction(DELETE, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 1),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 3),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 3, NEW_NODES, 2, 0),
                 createAction(VARS, 1, 0, NEW_NODES, 1, 0),
                 createAction(NEW_NODES, 1, 1, NEW_NODES, 0, 2),
                 createAction(true),
                 createAction(false),
             });
  addActions(COMP_STAR_SYM, ANY,
             {
                 createAction(COMP_STAR, 0),
                 createAction(VARS, 0, 3, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 2),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 3),
                 createAction(VARS, 1, 0, NEW_NODES, 0, 1),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 0, 4),
                 createAction(true),
             });
  addActions(COMP_STAR_SYM, END_STAR,
             {
                 createAction(COMP_STAR_AUX, 1),
                 createAction(SYM, -1),
                 createAction(ANY, 1),
                 createAction(DELETE, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 1, 1),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 3),
                 createAction(VARS, 0, 2, NEW_NODES, 3, 0),
                 createAction(VARS, 0, 3, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 1, NEW_NODES, 2, 0),
                 createAction(ACTIVE_PAIR, 1, 0, NEW_NODES, 0, 2),
                 createAction(ACTIVE_PAIR, 1, 2, NEW_NODES, 2, 1),
                 createAction(NEW_NODES, 1, 0, NEW_NODES, 0, 1),
                 createAction(true),
             });
  addActions(COMP_STAR_ANY, SYM,
             {
                 createAction(COMP_STAR, 0),
                 createAction(VARS, 0, 3, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 2),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 3),
                 createAction(VARS, 1, 0, NEW_NODES, 0, 1),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 0, 4),
                 createAction(true),
             });
  addActions(COMP_STAR_ANY, ANY,
             {
                 createAction(COMP_STAR, 0),
                 createAction(VARS, 0, 3, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 2),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 3),
                 createAction(VARS, 1, 0, NEW_NODES, 0, 1),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 0, 4),
                 createAction(true),
             });
  addActions(COMP_STAR_ANY, END_STAR,
             {
                 createAction(COMP_STAR_AUX, 1),
                 createAction(ANY, 0),
                 createAction(ANY, 1),
                 createAction(DELETE, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 1, 1),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 3),
                 createAction(VARS, 0, 2, NEW_NODES, 3, 0),
                 createAction(VARS, 0, 3, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 1, NEW_NODES, 2, 0),
                 createAction(ACTIVE_PAIR, 1, 0, NEW_NODES, 0, 2),
                 createAction(ACTIVE_PAIR, 1, 2, NEW_NODES, 2, 1),
                 createAction(NEW_NODES, 1, 0, NEW_NODES, 0, 1),
                 createAction(true),
             });
  addActions(COMP_STAR_END, SYM,
             {
                 createAction(COMP_STAR_AUX, 0),
                 createAction(STAR_DEL, 0),
                 createAction(DELETE, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 1),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 3),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 3, NEW_NODES, 2, 0),
                 createAction(VARS, 1, 0, NEW_NODES, 1, 0),
                 createAction(NEW_NODES, 1, 1, NEW_NODES, 0, 2),
                 createAction(true),
                 createAction(false),
             });
  addActions(COMP_STAR_END, ANY,
             {
                 createAction(COMP_STAR_AUX, 0),
                 createAction(STAR_DEL, 0),
                 createAction(DELETE, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 1),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 3),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 3, NEW_NODES, 2, 0),
                 createAction(VARS, 1, 0, NEW_NODES, 1, 0),
                 createAction(NEW_NODES, 1, 1, NEW_NODES, 0, 2),
                 createAction(true),
                 createAction(false),
             });
  addActions(COMP_STAR_END, END_STAR,
             {
                 createAction(COMP_STAR_AUX, 1),
                 createAction(ANY, 1),
                 createAction(DELETE, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 1),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 3),
                 createAction(VARS, 0, 2, NEW_NODES, 2, 0),
                 createAction(VARS, 0, 3, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 1, NEW_NODES, 1, 0),
                 createAction(ACTIVE_PAIR, 1, 0, NEW_NODES, 0, 2),
                 createAction(ACTIVE_PAIR, 1, 2, NEW_NODES, 1, 1),
                 createAction(true),
             });

  addActions(COMP_STAR, SYM,
             {
                 createAction(COMP_STAR_SYM, -2),
                 createAction(VARS, 0, 2, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 2),
                 createAction(VARS, 0, 3, NEW_NODES, 0, 4),
                 createAction(VARS, 1, 0, NEW_NODES, 0, 1),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 0, 3),
                 createAction(true),
             });
  addActions(COMP_STAR, ANY,
             {
                 createAction(COMP_STAR_SYM, -2),
                 createAction(VARS, 0, 2, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 2),
                 createAction(VARS, 0, 3, NEW_NODES, 0, 4),
                 createAction(VARS, 1, 0, NEW_NODES, 0, 1),
                 createAction(ACTIVE_PAIR, 1, 1, NEW_NODES, 0, 3),
                 createAction(true),
             });
  addActions(COMP_STAR, BAR,
             {
                 createAction(COMP_STAR_END, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 2),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 3),
                 createAction(VARS, 0, 3, NEW_NODES, 0, 4),
                 createAction(ACTIVE_PAIR, 1, 0, NEW_NODES, 0, 1),
                 createAction(true),
             });
  addActions(COMP_STAR, END,
             {
                 createAction(COMP_STAR_END, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 0, 1, NEW_NODES, 0, 2),
                 createAction(VARS, 0, 2, NEW_NODES, 0, 3),
                 createAction(VARS, 0, 3, NEW_NODES, 0, 4),
                 createAction(ACTIVE_PAIR, 1, 0, NEW_NODES, 0, 1),
                 createAction(true),
             });

  addActions(SLASH, COMP_STAR_AUX, true,
             {
                 createAction(COMP, 0),
                 createAction(VARS, 0, 1, VARS, 1, 0),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 1, NEW_NODES, 0, 1),
                 createAction(VARS, 1, 2, NEW_NODES, 0, 2),
                 createAction(true),
                 createAction(false),
             });
  addActions(SLASH, COMP_STAR_AUX, false,
             {
                 createAction(COMP, 0),
                 createAction(VARS, 0, 1, VARS, 1, 1),
                 createAction(VARS, 0, 0, NEW_NODES, 0, 1),
                 createAction(VARS, 1, 0, NEW_NODES, 0, 0),
                 createAction(VARS, 1, 2, NEW_NODES, 0, 2),
                 createAction(true),
                 createAction(false),
             });
  addActions(SLASH, SYM,
             {
                 createAction(VARS, 0, 1, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 2, ACTIVE_PAIR, 1, 1),
             });
  addActions(SLASH, ANY,
             {
                 createAction(VARS, 0, 1, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 2, ACTIVE_PAIR, 1, 1),
             });
  addActions(SLASH, STAR,
             {
                 createAction(VARS, 0, 1, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 2, ACTIVE_PAIR, 1, 1),
             });

  addActions(SYM, STAR_DEL,
             {createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0), createAction(true)});
  addActions(ANY, STAR_DEL,
             {createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0), createAction(true)});
  addActions(END_STAR, STAR_DEL,
             {createAction(NIL, 1), createAction(VARS, 0, 0, VARS, 1, 0),
              createAction(VARS, 0, 1, NEW_NODES, 0, 0), createAction(true),
              createAction(false)});
  addActions(NIL, RULE_STAR,
             {createAction(VARS, 1, 0, VARS, 1, 2),
              createAction(VARS, 1, 1, ACTIVE_PAIR, 0, 0),
              createAction(false)});
  addActions(NIL, END_STAR,
             {createAction(VARS, 1, 0, VARS, 1, 1), createAction(true),
              createAction(false)});
  addActions(
      NIL, SYM,
      {createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0), createAction(false)});

  // LOOP
  addActions(ANY, RULE_STAR, false,
             {createAction(VARS, 1, 1, ACTIVE_PAIR, 1, 0),
              createAction(VARS, 1, 2, ACTIVE_PAIR, 1, 2),
              createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 3), createAction(true)});
  addActions(SYM, RULE_STAR,
             {
                 createAction(SYM, -1),
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(VARS, 1, 1, NEW_NODES, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 1, ACTIVE_PAIR, 1, 1),
                 createAction(NEW_NODES, 0, 1, ACTIVE_PAIR, 1, 2),
             });
  addActions(ANY, RULE_STAR, true,
             {
                 createAction(SYM, -1),
                 createAction(VARS, 0, 0, ACTIVE_PAIR, 1, 0),
                 createAction(VARS, 1, 0, ACTIVE_PAIR, 0, 0),
                 createAction(VARS, 1, 1, NEW_NODES, 0, 0),
                 createAction(ACTIVE_PAIR, 0, 1, ACTIVE_PAIR, 1, 1),
                 createAction(NEW_NODES, 0, 1, ACTIVE_PAIR, 1, 2),
             });
  addActions(END_STAR, RULE_STAR,
             {
                 createAction(VARS, 1, 1, ACTIVE_PAIR, 0, 0),
                 createAction(VARS, 0, 1, ACTIVE_PAIR, 1, 2),
                 createAction(VARS, 1, 2, ACTIVE_PAIR, 1, 0),
                 createAction(ACTIVE_PAIR, 1, 3, ACTIVE_PAIR, 0, 2),
             });
}
