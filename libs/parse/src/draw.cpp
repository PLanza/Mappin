#include "draw.hpp"
#include "inet.hpp"
#include "nodes.hpp"
#include <generate/grammar.hpp>

#include <cstddef>
#include <drag/drag.hpp>
#include <drag/drawing/draw.hpp>
#include <drag/types.hpp>

namespace inet {

grammar::Token nodeToToken(uint32_t value) {
  if (value % 2 == 0)
    return {grammar::TERM, value / 2};
  else
    return {grammar::NON_TERM, (value - 1) / 2};
}

void drawNetwork(std::unique_ptr<grammar::Grammar> &grammar, bool as_tokens) {
  static int counter = 0;
  drag::graph g;
  std::unordered_map<Node *, drag::vertex_t> node_map;

  drag::drawing_options opts;

  // Add nodes
  for (Node *node : nodes) {
    drag::vertex_t graph_node = g.add_node();
    node_map[node] = graph_node;

    if (node->kind == BOOL) {
      opts.labels[graph_node] = node->value == 1 ? "T" : "F";
    } else if (node->kind == SYM) {
      if (as_tokens) {
        grammar::Token token = nodeToToken(node->value);
        opts.labels[graph_node] = token.kind == grammar::TERM
                                      ? grammar->getTerminalString(token.id)
                                      : grammar->getNonTerminalString(token.id);
      } else
        opts.labels[graph_node] = std::to_string(node->value);
    } else if (node->kind == COMP_SYM) {
      grammar::Token token = nodeToToken(node->value);
      opts.labels[graph_node] = "○_";
      opts.labels[graph_node] += token.kind == grammar::TERM
                                     ? grammar->getTerminalString(token.id)
                                     : grammar->getNonTerminalString(token.id);
    } else if (node->kind == DELTA) {
      opts.labels[graph_node] = node_strings[node->kind];
      opts.labels[graph_node] += std::to_string(node->value % 100);
    } else if (node->kind == RULE) {
      opts.labels[graph_node] = std::to_string(node->value);
    } else {
      opts.labels[graph_node] = node_strings[node->kind];
    }
  }

  // Add edges
  for (Node *n : nodes) {
    for (size_t i = 0; i < node_arities[n->kind] + 1; i++) {
      if (i == 0 && n->kind != OUTPUT) {
        g.add_edge(node_map[n], node_map[n->ports[i].node]);
        if (n->ports[i].port == 0)
          opts.edge_colors[{node_map[n], node_map[n->ports[i].node]}] = "red";
        else
          opts.edge_colors[{node_map[n], node_map[n->ports[i].node]}] = "blue";
      } else {
        if (n->ports[i].port != 0)
          g.add_edge(node_map[n], node_map[n->ports[i].node]);
      }
    }
  }

  opts.edge_colors[{node_map[interactions.front().n1],
                    node_map[interactions.front().n2]}] = "green";
  opts.edge_colors[{node_map[interactions.front().n2],
                    node_map[interactions.front().n1]}] = "green";

  drag::sugiyama_layout layout(g);

  auto image = drag::draw_svg_image(layout, opts);
  std::string file_name = "images/" + std::to_string(counter);
  file_name += ".svg";
  image.save(file_name);
  counter++;
}

} // namespace inet
