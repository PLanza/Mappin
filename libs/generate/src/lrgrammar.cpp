#include "grammar/lrgrammar.hpp"
#include "grammar.hpp"
#include "parse/nodes.hpp"
#include <boost/container_hash/extensions.hpp>
#include <cstddef>
#include <cstring>
#include <deque>
#include <iostream>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace grammar {

LRGrammar::LRGrammar(const char *file_name) : Grammar(file_name) {}

LRGrammar::~LRGrammar() {
  delete this->parse_table;
  delete[] this->stack_actions;
}

void LRGrammar::finalize() {
  // Add S' -> S$ rule
  Token start_token = std::get<0>(rules[this->start_rule]);

  std::string start_name;
  for (auto const &[nonterm, id] : this->nonterm_id_map) {
    if (id == start_token.id) {
      start_name = nonterm;
      break;
    }
  }
  start_name += "\'";

  this->addRule(start_name, {start_token, Token{TERM, this->term_id_map[">"]}},
                true, std::get<2>(rules[this->start_rule]));

  this->fillStringArrays();
}

void LRGrammar::makeParseTable() { this->parse_table = new LRParseTable(this); }

LRParseTable *LRGrammar::getParseTable() { return this->parse_table; }

void LRGrammar::printParseTable() {
  if (this->parse_table == nullptr) {
    std::cerr << "ERROR: Attempted to print parse table before it "
                 "was made."
              << std::endl;
    return;
  }

  uint32_t action_width = 7 * (this->terms_size - 1);
  uint32_t goto_width = 7 * (this->nonterms_size - 1);

  std::cout << "state | ";
  for (int i = 0; i < action_width / 2 - 3; i++)
    std::cout << " ";
  std::cout << "action";
  for (int i = 0; i < action_width / 2 - 3; i++)
    std::cout << " ";
  std::cout << "  | ";
  for (int i = 0; i < goto_width / 2 - 2; i++)
    std::cout << " ";
  std::cout << "goto";
  for (int i = 0; i < goto_width / 2 - 2; i++)
    std::cout << " ";
  std::cout << std::endl;

  printf("%5s | ", "");
  for (int t_id = 1; t_id < this->terms_size; t_id++)
    printf("%-6s ", this->terminals[t_id].c_str());
  std::cout << " | ";
  for (int t_id = 0; t_id < this->nonterms_size - 1; t_id++)
    printf("%-6s ", this->nonterminals[t_id].c_str());

  std::cout << std::endl;
  for (int i = 0; i < 11 + action_width + goto_width; i++)
    std::cout << "-";
  std::cout << std::endl;

  for (uint32_t state = 0; state < this->parse_table->state_count; state++) {
    printf("%5d | ", state);
    for (uint32_t t_id = 1; t_id < this->terms_size; t_id++) {
      ParseAction action = this->parse_table->getAction(state, t_id);
      switch (action.kind) {
      case grammar::EMPTY: {
        printf("%6s ", "");
        break;
      }
      case grammar::SHIFT: {
        printf("S%-5d ", action.reduce_rule);
        break;
      }
      case grammar::REDUCE: {
        printf("R%-5d ", action.reduce_rule);
        break;
      }
      case grammar::ACCEPT: {
        printf("%-6s ", "acc");
        break;
      }
      }
    }
    std::cout << " | ";
    for (uint32_t t_id = 0; t_id < this->nonterms_size - 1; t_id++) {
      int goto_state = this->parse_table->getGoto(state, t_id);
      if (goto_state == -1)
        printf("%6s ", "");
      else
        printf("%-6d ", goto_state);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void LRGrammar::printStackActions() { Grammar::printStackActions(false); }

uint32_t mergeStar(size_t &i_l, size_t &i_r, StackAction left_action,
                   StackAction right_action) {
  size_t start_i_l = i_l;
  size_t start_i_r = i_r;

  uint32_t loops = 0;

  i_l++;
  while (true) {
    // If reached the end of right_action or reach `-` -> done
    if (i_r >= right_action.lhs.size() || right_action.lhs[i_r].kind == REST) {
      // advance i_l until reached END_STAR
      while (left_action.rhs[i_l].kind != END_STAR)
        i_l++;
      i_r = start_i_r - 1;
      return loops;
    }

    StackState left_state = left_action.rhs[i_l],
               right_state = right_action.lhs[i_r];

    if (left_state.kind == SOME && right_state.kind == SOME &&
        left_state.value != right_state.value) {
      // advance i_l until reached END_STAR
      while (left_action.rhs[i_l].kind != END_STAR)
        i_l++;
      i_r = start_i_r - 1;
      return loops;
    }

    if (left_state.kind == END_STAR) {
      i_l = start_i_l;
      start_i_r = i_r;
      i_r--;
      loops++;
    }

    i_l++, i_r++;
  }
}

std::optional<StackAction> mergeStackActions(StackAction left_action,
                                             StackAction right_action) {
  // A-/ should not be able to merge with anything
  if (left_action.rhs.size() == 0)
    return std::nullopt;

  size_t i_l = 0, i_r = 0;

  // Traverse until reach rest
  while (i_l < left_action.rhs.size() && i_r < right_action.lhs.size()) {
    StackState left_state = left_action.rhs[i_l],
               right_state = right_action.lhs[i_r];
    if (left_state.kind == REST) {
      // Remove `-`
      left_action.lhs.pop_back();
      // Append suffix
      left_action.lhs.insert(left_action.lhs.end(),
                             right_action.lhs.begin() + i_r,
                             right_action.lhs.end());
      break;
    }
    if (right_state.kind == REST) {
      if (right_action.rhs.size() == 0)
        break;

      // Remove `-`
      right_action.rhs.pop_back();
      // Append suffix
      right_action.rhs.insert(right_action.rhs.end(),
                              left_action.rhs.begin() + i_l,
                              left_action.rhs.end());
      break;
    }

    if (left_state.kind == SOME && right_state.kind == SOME &&
        left_state.value != right_state.value)
      return std::nullopt;

    if (left_state.kind == STAR)
      mergeStar(i_l, i_r, left_action, right_action);

    i_l++, i_r++;
  }

  left_action.rhs = right_action.rhs;
  // Not sure this works for reduce chains longer than 3
  left_action.reduce_rules.insert(left_action.reduce_rules.begin(),
                                  right_action.reduce_rules.begin(),
                                  right_action.reduce_rules.end());

  left_action.shifted = right_action.shifted;

  return left_action;
}

struct Markers {
  int32_t state;
  uint32_t rule;
};
struct UnfinishedStackAction {
  StackAction action;
  std::unordered_map<uint32_t, Markers> visited;
};

void LRGrammar::getStackActionClosure(uint32_t term) {
  // Those that have shifted once
  std::vector<StackAction> final_actions;
  // Those in the process of being reduced
  std::vector<UnfinishedStackAction> current_actions;

  // Initialize current actions
  for (int i = 0; i < this->stack_actions[term].size(); i++) {
    StackAction &action = this->stack_actions[term][i];
    if (action.shifted)
      final_actions.push_back(action);
    else
      current_actions.push_back({action,
                                 {{i,
                                   {(int32_t)action.lhs.size() - 1,
                                    (uint32_t)action.reduce_rules.size()}}}});
  }

  while (!current_actions.empty()) {
    UnfinishedStackAction left_action = current_actions.back();
    current_actions.pop_back();

    for (int i = 0; i < this->stack_actions[term].size(); i++) {

      if (auto merged_action = mergeStackActions(
              left_action.action, this->stack_actions[term][i])) {
        // If looping then encapsulate loop with STARs
        if (left_action.visited.find(i) != left_action.visited.end()) {
          if (left_action.visited[i].state != -1) {
            // Encapsulate with STARs
            merged_action->lhs.insert(merged_action->lhs.begin() +
                                          left_action.visited[i].state,
                                      STAR_STATE);
            merged_action->lhs.insert(merged_action->lhs.end() - 1,
                                      END_STAR_STATE);

            merged_action->reduce_rules.insert(
                merged_action->reduce_rules.end() - left_action.visited[i].rule,
                -2);
            merged_action->reduce_rules.insert(
                merged_action->reduce_rules.begin(), -1);

            // Do not loop again
            current_actions.push_back({*merged_action, left_action.visited});
            current_actions.back().visited[i].state = -1;
          }
        } else {
          if (merged_action->shifted)
            final_actions.push_back(*merged_action);
          else {
            current_actions.push_back({*merged_action, left_action.visited});
            current_actions.back().visited[i].state =
                merged_action->lhs.size() - 1;
            current_actions.back().visited[i].rule =
                merged_action->reduce_rules.size();
          }
        }
      }
    }
  }

  this->stack_actions[term] = final_actions;
  this->clearRepeatedActions(term);
}

int32_t containsLoop(StackAction &action) {
  for (int32_t i = 0; i < action.lhs.size(); i++)
    if (action.lhs[i].kind == STAR)
      return i;

  return false;
}

bool areEqual(StackAction &left, StackAction &right, int32_t star) {
  if (right.lhs.size() < star)
    return false;

  uint32_t il = 0, ir = 0;
  while (il < left.lhs.size() && ir < right.lhs.size()) {
    if (left.lhs[il].kind == STAR) {
      while (left.lhs[il].kind != END_STAR)
        il++;
      il++;
    }
    if (!(left.lhs[il] == right.lhs[ir]))
      return false;

    il++;
    ir++;
  }
  return true;
}

void LRGrammar::clearRepeatedActions(uint32_t term) {
  std::vector<StackAction> &actions = this->stack_actions[term];

  for (auto left = actions.begin(); left < actions.end(); left++) {
    int32_t star = containsLoop(*left);
    if (star == -1)
      continue;

    for (auto right = actions.begin(); right < actions.end(); right++) {
      if (left != right && areEqual(*left, *right, star))
        actions.erase(right);
    }
  }
}

void LRGrammar::removeUselessActions() {

  for (uint32_t r_term = 1; r_term < this->terms_size; r_term++) {
    // std::cout << "Merging " << terminals[r_term] << std::endl;

    std::vector<StackAction> &r_actions = this->stack_actions[r_term];
    for (auto ra_iter = r_actions.begin(); ra_iter < r_actions.end();
         ra_iter++) {
      StackAction r_action = *ra_iter;
      bool has_merged = false;

      for (uint32_t l_term = 0; l_term < this->terms_size; l_term++) {
        auto l_follow = this->parse_table->getFollowSet({TERM, l_term});
        if (l_follow.find({TERM, r_term}) != l_follow.end()) {
          std::vector<StackAction> &l_actions = this->stack_actions[l_term];
          // std::cout << "\t and " << terminals[l_term] << std::endl;

          for (StackAction l_action : l_actions) {
            // this->printStackAction(l_action, false);
            // std::cout << " o ";
            // this->printStackAction(r_action, false);
            // std::cout << std::endl;

            if (mergeStackActions(l_action, r_action).has_value()) {
              // std::cout << "Success" << std::endl;
              has_merged = true;
              break;
            }
          }
        }
        if (has_merged)
          break;
      }
      if (!has_merged) {
        r_actions.erase(ra_iter);
        ra_iter--;
      }
      // std::cout << std::endl;
    }
  }
}

void LRGrammar::generateStackActions() {
  if (this->parse_table == nullptr) {
    std::cerr
        << "ERROR: Attempted to generate stack actions before the parse table"
        << std::endl;
    return;
  }
  std::cout << "Generating stack actions for LR(0) Grammar" << std::endl;

  this->stack_actions = new std::vector<StackAction>[this->terms_size];
  this->stack_actions[0] = {StackAction{{}, {{SOME, 0}}, {}}};

  for (uint32_t term = 1; term < this->terms_size; term++) {
    for (uint32_t state = 0; state < this->parse_table->state_count; state++) {
      ParseAction action = this->parse_table->getAction(state, term);
      switch (action.kind) {
      case SHIFT: {
        this->stack_actions[term].push_back(
            StackAction{{{SOME, state}, REST_STATE},
                        {{SOME, action.reduce_rule}, {SOME, state}, REST_STATE},
                        {},
                        true});
        break;
      }
      case REDUCE: {
        std::deque<StackState> lhs_prefix = {{SOME, state}};
        auto &[rule_lhs, rule_rhs, _] = this->rules[action.reduce_rule];
        for (int i = 1; i < rule_rhs.size(); i++)
          lhs_prefix.push_back(ANY_STATE);

        for (uint32_t state = 0; state < this->parse_table->state_count;
             state++) {
          std::deque<StackState> lhs = lhs_prefix;
          int goto_state = this->parse_table->getGoto(state, rule_lhs.id);
          if (goto_state < 0)
            continue;

          lhs.push_back({SOME, state});
          lhs.push_back(REST_STATE);
          this->stack_actions[term].push_back(
              StackAction{lhs,
                          {{SOME, static_cast<uint32_t>(goto_state)},
                           {SOME, state},
                           REST_STATE},
                          {(int32_t)action.reduce_rule},
                          false});
        }
        break;
      }
      case ACCEPT: {
        this->stack_actions[term].push_back(
            StackAction{{{SOME, state}, REST_STATE}, {}, {}, true});
        break;
      }
      case EMPTY:
        break;
      }
    }
    this->getStackActionClosure(term);
  }
  this->removeUselessActions();
}

void LRGrammar::traverseRules(inet::Node *cons,
                              std::deque<ParseTree *> &stack) {
  if (cons->kind == inet::CONS) {
    this->traverseRules(cons->ports[1].node, stack);
    this->traverseRules(cons->ports[2].node, stack);
  } else if (cons->kind == inet::SYM) {
    this->traverseRules(cons->ports[1].node, stack);
    auto const &[head, rhs, _] = this->getRule(cons->value);
    ParseTree *tree = new ParseTree(NON_TERM, cons->value, rhs.size());

    for (size_t i = 0; i < rhs.size(); i++) {
      tree->children[i] = stack.back();
      stack.pop_back();
    }
    stack.push_back(tree);
  } else if (cons->kind == inet::END) {
    stack.push_back(new ParseTree(TERM, 0, 0));
  }
}

ParseTree *LRGrammar::getParse(inet::Node *product) {
  // For each parse, check the stack action for incomplete parses
  inet::Node *stack_action = product->ports[2].node;
  if (stack_action->ports[1].node->kind != inet::END &&
      stack_action->ports[2].node->kind != inet::END)
    return nullptr;

  // If valid then traverse the reduction rules and print parse
  inet::Node *cons = product->ports[1].node;
  std::deque<ParseTree *> stack;
  this->traverseRules(cons, stack);

  auto const &[head, rhs, _] = this->getRule(this->start_rule);
  ParseTree *tree = new ParseTree(NON_TERM, this->start_rule, rhs.size());

  for (size_t i = 0; i < rhs.size(); i++) {
    tree->children[i] = stack.back();
    stack.pop_back();
  }
  delete stack.back();

  return tree;
}

#define translate(ptr) (host + (ptr - device))
void LRGrammar::traverseRules(NodeElement *cons, std::deque<ParseTree *> &stack,
                              NodeElement *host, NodeElement *device) {
  if (cons[0].header.kind == inet::CONS) {
    this->traverseRules(translate(cons[3].port_node), stack, host, device);
    this->traverseRules(translate(cons[5].port_node), stack, host, device);
  } else if (cons[0].header.kind == inet::SYM) {
    this->traverseRules(translate(cons[3].port_node), stack, host, device);
    auto const &[head, rhs, _] = this->getRule(cons[0].header.value);
    ParseTree *tree = new ParseTree(NON_TERM, cons[0].header.value, rhs.size());

    for (size_t i = 0; i < rhs.size(); i++) {
      tree->children[i] = stack.back();
      stack.pop_back();
    }
    stack.push_back(tree);
  } else if (cons[0].header.kind == inet::END) {
    stack.push_back(new ParseTree(TERM, 0, 0));
  }
}

ParseTree *LRGrammar::getParse(NodeElement *product, NodeElement *host,
                               NodeElement *device) {
  // For each parse, check the stack action for incomplete parses
  NodeElement *stack_action = translate(product[5].port_node);
  if (translate(stack_action[3].port_node)[0].header.kind != inet::END &&
      translate(stack_action[5].port_node)[0].header.kind != inet::END)
    return nullptr;

  // If valid then traverse the reduction rules and print parse
  NodeElement *cons = translate(product[3].port_node);
  std::deque<ParseTree *> stack;
  this->traverseRules(cons, stack, host, device);

  auto const &[head, rhs, _] = this->getRule(this->start_rule);
  ParseTree *tree = new ParseTree(NON_TERM, this->start_rule, rhs.size());

  for (size_t i = 0; i < rhs.size(); i++) {
    tree->children[i] = stack.back();
    stack.pop_back();
  }
  delete stack.back();

  return tree;
}

void LRGrammar::printParseTree(ParseTree *tree) {
  if (tree->kind == TERM) {
    std::cout << "_";
  } else {
    auto const &[head, rhs, _] = this->getRule(tree->value);
    std::cout << this->getNonTerminalString(head.id) << "[ ";
    for (size_t i = 1; i <= rhs.size(); i++) {
      this->printParseTree(tree->children[rhs.size() - i]);
      std::cout << " ";
    }
    std::cout << "]";
  }
}

} // namespace grammar
