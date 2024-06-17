#include "../../include/gpu/draw.hpp"

#include <cstdio>
#include <drag/drag.hpp>
#include <drag/drawing/draw.hpp>
#include <drag/types.hpp>
#include <queue>

void drawNetwork(NodeElement *network) {
  static int counter = 0;
  std::queue<uint32_t> to_visit;
  to_visit.push(1);
  drag::graph g;
  std::unordered_map<uint32_t, drag::vertex_t> node_map;

  drag::drawing_options opts;

  node_map[1] = g.add_node();
  opts.labels[node_map[1]] = "0";

  // Add nodes
  while (!to_visit.empty()) {
    uint32_t node = to_visit.front();
    to_visit.pop();

    for (uint32_t i = 0; i < NODE_ARITIES_H[network[node].header.kind] + 1;
         i++) {
      if (i == 0 && network[node].header.kind == OUTPUT)
        continue;

      uint32_t neighbor = (network + node)[2 + 2 * i].port_node;
      // printf("%d --- %d\n", node, neighbor);
      // printf("%d: %d[%d] --- %d: %d[%d]\n", node, network[node].header.kind,
      // i,
      //        neighbor, network[neighbor].header.kind,
      //        (network + node)[3 + 2 * i].port_port);
      if (node_map.find(neighbor) == node_map.end()) {
        node_map[neighbor] = g.add_node();
        opts.labels[node_map[neighbor]] =
            std::to_string(network[neighbor].header.kind);
        to_visit.push(neighbor);
      }

      if (i == 0) {
        g.add_edge(node_map[node], node_map[neighbor]);
        if ((network + node)[3 + 2 * i].port_port == 0)
          opts.edge_colors[{node_map[node], node_map[neighbor]}] = "red";
        else
          opts.edge_colors[{node_map[node], node_map[neighbor]}] = "blue";
      } else {
        g.add_edge(node_map[node], node_map[neighbor]);
      }
    }
  }

  drag::sugiyama_layout layout(g);

  auto image = drag::draw_svg_image(layout, opts);
  std::string file_name = "images/" + std::to_string(counter);
  file_name += ".svg";
  image.save(file_name);
  counter++;
}
