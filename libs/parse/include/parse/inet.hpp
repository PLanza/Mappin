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
};


Node *newNode(node_kind kind);
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
} Connection;

struct Action {
  ActionKind kind;
  union {
    node_kind new_node;
    Connection connect;
    bool free; // only free the interacting nodes so a choice between 2
  } action;

  Action(node_kind);
  Action(Connect, Connect);
  Action(bool);
};

void interact();
}

#endif
