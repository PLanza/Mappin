#include "nodes.hpp"
#include "inet.hpp"
#include "draw.hpp"

#include <cstddef>
#include <drag/drag.hpp>
#include <drag/drawing/draw.hpp>
#include <drag/types.hpp>

namespace inet {

void drawNetwork() {
  drag::graph g;
  std::unordered_map<Node *, drag::vertex_t> node_map;

  // Add nodes
  for (Node *node : nodes) {
    node_map[node] = g.add_node();
  }

  // Add edges
  for (Node *n : nodes) {
    for (size_t i = 0; i < node_arities[n->kind]; i++) {
      g.add_edge(node_map[n], node_map[n->ports[i].node]);
    }
  }

  drag::sugiyama_layout layout(g);

  auto image = drag::draw_svg_image(layout, drag::drawing_options{});
  image.save("network.svg");
}

} // namespace inet
