#ifndef __MAPPIN_PARSE_PARSE_NET__
#define __MAPPIN_PARSE_PARSE_NET__

#include <queue>
#include <vector>

#include "inet.hpp"

namespace inet {

extern const int NODE_KINDS;

extern int *node_arities;

typedef struct {
  Node *n1;
  Node *n2;
} Interaction;

extern std::queue<Interaction> interactions;
extern std::vector<Action> **actions_map;

void init();
}

#endif
