#ifndef __MAPPIN_PARSE_NODES__
#define __MAPPIN_PARSE_NODES__

#include <deque>
#include <string>
#include <vector>

#include "inet.hpp"

namespace inet {

const int NODE_KINDS = 29;

enum NodeKind {
  OUTPUT,
  DELETE,
  DELTA,
  GAMMA, // γ_0 is regular γ, γ_1 is ×
  NIL,   // Also for deleting RULE_STAR
  CONS,
  APPEND,
  FOLD,
  IF,
  BOOL,
  SLASH,
  COMP,
  COMP_SYM,
  COMP_END,
  COMP_ANY,
  COMP_STAR,
  COMP_STAR_SYM,
  COMP_STAR_ANY,
  COMP_STAR_END,
  COMP_STAR_AUX,
  BAR,
  END,
  SYM, // Also acts as RULE
  ANY, // Also acts as LOOP
  STAR,
  END_STAR,
  STAR_DEL,
  RULE_STAR, // Also CONT_AUX
};

extern int node_arities[NODE_KINDS];
extern std::string node_strings[NODE_KINDS];

struct Interaction {
  Node *n1 = nullptr;
  Node *n2 = nullptr;
};

extern std::deque<Interaction> interactions;
extern std::vector<Action> *actions_map;

void init();

std::vector<Action> &getActions(node_kind, node_kind, bool);
} // namespace inet

#endif
