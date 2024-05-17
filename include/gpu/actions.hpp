#ifndef __MAPPIN_PARALLEL_ACTIONS__
#define __MAPPIN_PARALLEL_ACTIONS__

#include <cstddef>
#include <cstdint>

#include "inet.hpp"

enum ActionKind { NEW, CONNECT, NONE };
enum Group { ACTIVE_PAIR, NEW_NODES, VARS };

typedef struct {
  node_kind kind;
  int8_t value;
} NewNodeAction;

#define connect_g(connect) (connect >> 6 & 0b11)
#define connect_n(connect) (connect >> 3 & 0b111)
#define connect_p(connect) (connect & 0b111)

typedef struct {
  uint8_t c1;
  uint8_t c2;
} ConnectAction;

struct Action {
  uint16_t kind;
  union {
    NewNodeAction new_node;
    ConnectAction connect;
    bool free;
  } action;
};

inline const uint8_t MAX_ACTIONS = 18;
inline const size_t ACTIONS_MAP_SIZE =
    ((NODE_KINDS * NODE_KINDS + NODE_KINDS) / 2) * 2 * MAX_ACTIONS;

void initActions();

extern Action actions_map_h[ACTIONS_MAP_SIZE];

#endif
