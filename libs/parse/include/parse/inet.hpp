#ifndef __MAPPIN_PARSE_INET__
#define __MAPPIN_PARSE_INET__

#include <cstddef>
#include <cstdint>

namespace inet {
  
typedef uint32_t node_kind;

struct Node;

// A node's port that connecting to some node `node` at its port `port`
struct Port {
  Node *node;
  size_t port;
};

// Will need to extend to include a value field
struct Node {
  node_kind kind;
  Port *ports;
  uint32_t value;
};


Node *newNode(node_kind kind, uint32_t value);
void connect(Node *n1, size_t p1, Node *n2, size_t p2);
void freeNode(Node *n);

// Need a node allocator (want to keep node data contiguous)
// Probably want a garbage collector then
// Copy-collect might be the best strategy

enum ActionKind { NEW, CONNECT, FREE };
enum Group { ACTIVE_PAIR, VARS, NEW_NODES };

struct Connect {
  Group group;
  size_t node;
  size_t port;
};

typedef struct {
  Connect c1;
  Connect c2;
} ConnectAction;

typedef struct {
  node_kind kind;
  int32_t value; // -1 means copy the left node's value, -2 means copy the right node's value
} NewNodeAction;

struct Action {
  ActionKind kind;
  union {
    NewNodeAction new_node;
    ConnectAction connect;
    bool free; // only free the interacting nodes so a choice between 2
  } action;

  Action(node_kind, int32_t);
  Action(Connect, Connect);
  Action(bool);
};

void interact();
}

#endif
