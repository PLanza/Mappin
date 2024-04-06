#ifndef __MAPPIN_PARALLEL_ACTIONS__
#define __MAPPIN_PARALLEL_ACTIONS__

#include <cstddef>
#include <cstdint>

#include "./inet.cuh"

enum ActionKind { NEW, CONNECT, FREE, NONE };
enum Group { ACTIVE_PAIR, VARS, NEW_NODES };

typedef struct {
  node_kind kind;
  int16_t value;
} NewNodeAction;

typedef struct {
  uint8_t c1;
  uint8_t c2;
} ConnectAction;

struct __align__(4) Action {
  uint8_t kind;
  union {
    NewNodeAction new_node;
    ConnectAction connect;
    bool free;
  } action;
};

inline const uint8_t MAX_ACTIONS = 16;
inline const size_t ACTIONS_MAP_SIZE = 
    ((NODE_KINDS * NODE_KINDS + NODE_KINDS) / 2) * 2 * MAX_ACTIONS;

extern Action actions_map[ACTIONS_MAP_SIZE];

size_t actMapIndex(NodeKind, NodeKind);
void initActions();

#endif
