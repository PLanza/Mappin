#ifndef __MAPPIN_PARSE_NODES__
#define __MAPPIN_PARSE_NODES__

#include <queue>
#include <string>
#include <vector>

#include "inet.hpp"

namespace inet {

const int NODE_KINDS = 19;

enum NodeKind {
  DELETE, DELTA, GAMMA, 
  NIL, CONS, APPEND, FOLD, 
  IF, BOOL,
  CONT, CONT_AUX, 
  SLASH, COMP, COMP_SYM, COMP_END, 
  BAR, BAR_AUX, END, SYM 
};

extern int node_arities[NODE_KINDS];
extern std::string node_strings[NODE_KINDS];

typedef struct {
  Node *n1;
  Node *n2;
} Interaction;

extern std::queue<Interaction> interactions;
extern std::vector<Action> *actions_map;

void init();
}

#endif
