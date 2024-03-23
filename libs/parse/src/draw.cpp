#include "draw.hpp"
#include "inet.hpp"
#include "nodes.hpp"

#include <cstddef>
#include <drag/drag.hpp>
#include <drag/drawing/draw.hpp>
#include <drag/types.hpp>
#include <iostream>

namespace inet {

void drawNetwork(std::string *terminals) {
  drag::graph g;
  std::unordered_map<Node *, drag::vertex_t> node_map;

  drag::drawing_options opts;

  // Add nodes
  for (Node *node : nodes) {
    drag::vertex_t graph_node = g.add_node();
    node_map[node] = graph_node;

    if (node->kind == BOOL) {
      opts.labels[graph_node] = node->value == 1 ? "T" : "F";
    } else if (node -> kind == SYM) {
      opts.labels[graph_node] = terminals[node->value];
    } else if (node -> kind == COMP_SYM) {
      opts.labels[graph_node] = "○_" + terminals[node->value];
    } else {
      opts.labels[graph_node] = node_strings[node->kind];
    }
  }

  std::cout << "drew nodes" << std::endl;

  // Add edges
  for (Node *n : nodes) {
    for (size_t i = 0; i < node_arities[n->kind] + 1; i++) {
      g.add_edge(node_map[n], node_map[n->ports[i].node]);
    }
  }

  drag::sugiyama_layout layout(g);

  auto image = drag::draw_svg_image(layout, opts);
  image.save("network.svg");
}

} // namespace inet
