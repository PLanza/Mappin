#ifndef __MAPPIN_PARSE_NODES__
#define __MAPPIN_PARSE_NODES__

#include <deque>
#include <string>
#include <vector>

#include "inet.hpp"

namespace inet {

const int NODE_KINDS = 23;

enum NodeKind {
  OUTPUT,
  DELETE,
  DELTA,
  GAMMA, // γ_0 is regular γ, γ_1 is ×
  NIL,
  CONS,
  APPEND,
  FOLD,
  IF,
  BOOL,
  CONT,
  CONT_AUX,
  SLASH,
  COMP,
  COMP_SYM,
  COMP_END,
  COMP_ANY,
  BAR,
  BAR_AUX,
  END,
  SYM,
  ANY,
  RULE, // Could use SYM for RULE
};

extern int node_arities[NODE_KINDS];
extern std::string node_strings[NODE_KINDS];

typedef struct {
  Node *n1;
  Node *n2;
} Interaction;

extern std::deque<Interaction> interactions;
extern std::vector<Action> *actions_map;

void init();

std::vector<Action> &getActions(node_kind, node_kind, bool);
} // namespace inet

#endif
