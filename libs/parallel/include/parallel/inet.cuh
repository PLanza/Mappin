#ifndef __MAPPIN_PARALLEL_INET__
#define __MAPPIN_PARALLEL_INET__

#include <cstddef>
#include <cstdint>

typedef uint64_t node_elem;
typedef uint8_t node_kind;

inline const size_t NODE_KINDS = 22;

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
  BAR,
  BAR_AUX,
  END,
  SYM, // also for RULE
  ANY,
  COMP_ANY,
};

extern const uint8_t NODE_ARITIES_H[NODE_KINDS];

struct __align__(8) NodeElement {
  union {
    struct {
      uint8_t kind;
      uint32_t value;
    } header;
    NodeElement *port_node;
    uint64_t port_port;
  };
};

struct __align__(16) Port {
  NodeElement *node;
  uint64_t port;
};

struct __align__(16) Interaction {
  NodeElement *n1;
  NodeElement *n2;
};
#endif
