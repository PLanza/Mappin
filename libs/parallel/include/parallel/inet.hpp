#ifndef __MAPPIN_PARALLEL_INET__
#define __MAPPIN_PARALLEL_INET__

#include <cstdint>

typedef uint64_t node_elem;
typedef uint8_t node_kind;

inline const uint8_t NODE_KINDS = 28;

enum NodeKind {
  OUTPUT,
  DELETE,
  DELTA,
  GAMMA, // γ_0 is regular γ, γ_1 is ×
  NIL,   // Also for deleting RULE_STAR
  CONS,
  APPEND,
  FOLD,
  IF, // Also CONT
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

extern const uint8_t NODE_ARITIES_H[NODE_KINDS];

void copyConstantData();

// TODO: Manage the network directly such that it is located in a fixed address
// space Then, nodes are instead 32-bit indexes and we can condense nodes and
// ports into one 64-bit value. Similaraly for Interactions.
struct NodeElement {
  union {
    struct {
      node_kind kind;
      uint16_t value;
      uint32_t lock;
    } header;
    NodeElement *port_node = nullptr;
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
