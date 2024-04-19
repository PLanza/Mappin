#ifndef __MAPPIN_PARALLEL_INET__
#define __MAPPIN_PARALLEL_INET__

#include <cstdint>

typedef uint64_t node_elem;
typedef uint8_t node_kind;

inline const uint8_t NODE_KINDS = 22;

enum NodeKind {
  OUTPUT, // Not really necessary
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
  SYM, // also for RULE
  ANY,
};

extern const uint8_t NODE_ARITIES_H[NODE_KINDS];

void copyConstantData();

// TODO: Manage the network directly such that it is located in a fixed address
// space Then, nodes are instead 32-bit indexes and we can condense nodes and
// ports into one 64-bit value. Similaraly for Interactions.
struct NodeElement {
  union {
    struct {
      uint8_t kind;
      uint16_t value;
      uint32_t lock;
    } header;
    NodeElement *port_node;
    uint64_t port_port;
  };
};

struct Port {
  NodeElement *node;
  uint64_t port;
};

struct Interaction {
  NodeElement *n1;
  NodeElement *n2;
};

#endif
