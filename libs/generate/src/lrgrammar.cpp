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

namespace boost {
template <> struct hash<grammar::Config> {
  size_t operator()(const grammar::Config &i) const {
    size_t hash = 0;
    boost::hash_combine(hash, i.rule_idx);
    boost::hash_combine(hash, i.bullet);
    return hash;
  }
};
} // namespace boost

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
      current_actions.push_back(
          {action,
           {{i,
             {(int32_t)action.lhs.size() - 1,
              (uint32_t)action.reduce_rules.size() - 1}}}});
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
                merged_action->reduce_rules.begin() +
                    left_action.visited[i].rule,
                -1);
            merged_action->reduce_rules.insert(
                merged_action->reduce_rules.end() - 1, -2);

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
                left_action.action.lhs.size();
            current_actions.back().visited[i].rule =
                left_action.action.reduce_rules.size();
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

void LRGrammar::generateStackActions() {
  if (this->parse_table == nullptr) {
    std::cerr
        << "ERROR: Attempted to generate stack actions before the parse table"
        << std::endl;
    return;
  }
  std::cout << "Generating stack actions for LR(0) Grammar" << std::endl;

  this->stack_actions =
      new std::vector<StackAction>[this->parse_table->state_count];
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

typedef std::unordered_set<Config, boost::hash<Config>> unfinished_lrstate;

LRState getStateClosure(grammar_rules const &rules,
                        unfinished_lrstate &unfinished) {
  uint32_t prev_size = 0;
  while (unfinished.size() != prev_size) {
    prev_size = unfinished.size();

    for (size_t i = 0; i < rules.size(); i++) {
      for (Config const &config : unfinished) {
        std::vector<Token> const &rhs = std::get<1>(rules[config.rule_idx]);
        if (config.bullet >= rhs.size())
          continue;

        if (rhs[config.bullet].kind == NON_TERM) {
          uint32_t nonterm_id = rhs[config.bullet].id;
          if (std::get<0>(rules[i]).id == nonterm_id)
            unfinished.insert({i, 0});
        }
      }
    }
  }

  LRState state;
  for (Config c : unfinished)
    state.push_back(c);
  return state;
}

bool equalStates(LRState const &left, LRState const &right) {
  for (Config const &l_config : left) {
    bool contains = false;
    for (Config const &r_config : right) {
      contains |= l_config == r_config;
    }
    if (!contains)
      return false;
  }
  return true;
}

void LRParseTable::generateStates() {
  std::size_t curr_state = 0;
  std::size_t prev_size = this->states.size();

  while (curr_state < this->states.size()) {
    // Maps transition tokens to the new state generated
    std::unordered_map<Token, unfinished_lrstate, boost::hash<Token>>
        new_states;
    std::vector<size_t> new_reductions;
    // Create sets that curr_state transitions into
    for (Config const &config : this->states[curr_state]) {
      std::vector<Token> const &rhs =
          std::get<1>(this->grammar->rules[config.rule_idx]);

      // A -> α ... β •
      if (config.bullet >= rhs.size()) {
        new_reductions.push_back(config.rule_idx);
        continue;
      }
      // A -> α ... β • eoi
      if (rhs[config.bullet] == Token{TERM, 1}) {
        this->end_states.push_back(curr_state);
        continue;
      }

      size_t set_idx;
      // A -> α ... • β  ...
      if (new_states.find(rhs[config.bullet]) != new_states.end())
        new_states[rhs[config.bullet]].insert(
            Config(config.rule_idx, config.bullet + 1));
      else {
        new_states[rhs[config.bullet]] = {
            Config(config.rule_idx, config.bullet + 1)};
      }
    }

    this->reductions.push_back(new_reductions);

    uint32_t terms = this->grammar->terms_size;
    uint32_t nonterms = this->grammar->nonterms_size;
    this->trans_table.push_back(new int[terms + nonterms]);
    for (int i = 0; i < terms + nonterms; i++)
      this->trans_table.back()[i] = -1;

    // Get the closure of each new state
    for (auto &[token, state] : new_states) {
      LRState finished_state = getStateClosure(this->grammar->rules, state);

      int repeat = -1;
      // Check if this new state is the same to any previous ones
      for (size_t other = 0; other < prev_size; other++) {
        if (equalStates(finished_state, this->states[other]))
          repeat = other;
      }

      size_t table_idx = token.id + (token.kind == TERM ? 0 : terms);
      if (repeat == -1) {
        // If the new state is original then transition to it and add it to
        // the set of states
        this->trans_table.back()[table_idx] = this->states.size();
        this->states.push_back(finished_state);
      } else
        // If new state has already been seen then transition into the
        // original
        this->trans_table.back()[table_idx] = repeat;
    }

    curr_state++;
    prev_size = this->states.size();
  }

  assert(this->trans_table.size() == this->states.size());
  this->state_count = this->states.size();
}

void LRParseTable::printConfig(Config const &config) {
  const auto &[head, rhs, _] = this->grammar->rules[config.rule_idx];

  std::cout << this->grammar->nonterminals[head.id] << " := ";
  for (uint32_t i = 0; i < rhs.size(); i++) {
    if (i == config.bullet)
      std::cout << "• ";

    Token token = rhs[i];
    if (token.kind == TERM)
      std::cout << this->grammar->terminals[token.id] << " ";
    else
      std::cout << this->grammar->nonterminals[token.id] << " ";
  }
  if (config.bullet == rhs.size())
    std::cout << "•";
}

void LRParseTable::printStates() {
  for (LRState const &state : states) {
    std::cout << "\n ---------- \n" << std::endl;
    for (Config const &config : state) {
      this->printConfig(config);
      std::cout << std::endl;
    }
  }
}

std::unordered_set<Token, boost::hash<Token>>
LRParseTable::getTokenHeads(uint32_t non_term) {
  if (this->token_heads.find(non_term) != this->token_heads.end())
    return this->token_heads[non_term];

  std::unordered_set<Token, boost::hash<Token>> theads;
  std::vector<uint32_t> head_stack = {non_term};

  while (!head_stack.empty()) {
    uint32_t curr_head = head_stack.back();
    head_stack.pop_back();

    for (auto const &[head, rhs, _] : this->grammar->rules) {
      if (head.id == curr_head) {
        if (rhs[0].kind == TERM)
          theads.insert(rhs[0]);
        else {
          bool in_stack = false;
          for (uint32_t const &stack_h : head_stack) {
            if (stack_h == rhs[0].id)
              in_stack = true;
          }
          if (!in_stack)
            head_stack.push_back(rhs[0].id);
        }
      }
    }
  }

  this->token_heads[non_term] = theads;
  return this->token_heads[non_term];
}

std::vector<LTStackToken> LRParseTable::getOriginators(size_t state_idx,
                                                       Config config) {
  std::vector<LTStackToken> originators;
  LRState &state = this->states[state_idx];

  // A -> • α ... => B -> ... • A ...
  if (config.bullet == 0) {
    Token head = std::get<0>(this->grammar->rules[config.rule_idx]);
    for (uint32_t c = 0; c < state.size(); c++) {
      std::vector<Token> rhs =
          std::get<1>(this->grammar->rules[state[c].rule_idx]);
      if (rhs[state[c].bullet] == head)
        originators.push_back({state_idx, c, false, false});
    }
  } else {
    // A -> ... α • => A -> ... • α => ... => A -> • ... α => B -> ... • A

    Token trans_tok =
        std::get<1>(this->grammar->rules[config.rule_idx])[config.bullet - 1];
    size_t table_col =
        trans_tok.id + (trans_tok.kind == TERM ? 0 : this->grammar->terms_size);
    for (size_t src_idx = 0; src_idx < this->state_count; src_idx++) {
      // If a state transitions into `state` with token `trans_tok`
      if (this->trans_table[src_idx][table_col] == state_idx) {
        LRState &src_state = this->states[src_idx];
        // Grab the config that originated this transition
        for (uint32_t c = 0; c < src_state.size(); c++) {
          if (src_state[c].rule_idx == config.rule_idx &&
              src_state[c].bullet == config.bullet - 1) {
            std::vector<LTStackToken> new_orig =
                this->getOriginators(src_idx, src_state[c]);
            // Recursively find its originator
            originators.insert(originators.end(), new_orig.begin(),
                               new_orig.end());
          }
        }
      }
    }
    // add the config with the same rule as config
  }
  return originators;
}

void LRParseTable::addContexts(
    std::unordered_set<Token, boost::hash<Token>> &gened_contexts,
    LTStackToken c_0_ltst, std::vector<LTStackToken> const &lane) {

  int32_t c_0_idx;
  for (int32_t i = lane.size() - 1; i >= 0; i--) {
    if (lane[i] == c_0_ltst)
      c_0_idx = i;
  }

  for (int32_t i = c_0_idx; i >= 0; i--) {
    if (lane[i].marker || lane[i].zero)
      continue;

    if (gened_contexts.empty())
      break;

    Config &config_i = this->states[lane[i].state][lane[i].config];
    for (Token token : config_i.context) {
      gened_contexts.erase(token);
    }
    for (Token token : gened_contexts) {
      config_i.context.insert(token);
    }
  }
}

void moveMarkers(std::vector<LTStackToken> &lane, LTStackToken c_0,
                 LTStackToken c_1, bool tests_failed) {
  int32_t c_0_idx, c_1_idx;
  for (int32_t i = lane.size() - 1; i >= 0; i--) {
    if (lane[i] == c_0)
      c_0_idx = i;
    if (lane[i] == c_1)
      c_1_idx = i;
  }

  uint32_t markers = 0;
  for (uint32_t i = c_0_idx + 1; i < c_1_idx; i++) {
    if (lane[i].marker) {
      lane[i] = {0, 0, false, true};
      markers++;
    }
  }

  LTStackToken lane_top = lane.back();
  if (tests_failed)
    lane.pop_back();
  for (uint32_t i = 0; i < markers; i++)
    lane.push_back({0, 0, true, false});
  if (tests_failed)
    lane.push_back(lane_top);
}

void LRParseTable::laneTrace(size_t state, uint32_t config) {
  std::vector<LTStackToken> lane;
  std::vector<LTStackToken> stack;

  LaneTraceFlags global_flags = NONE;
  std::unordered_set<Token, boost::hash<Token>> gened_contexts;

  Config &config_0 = this->states[state][config];

  if (config_0.flags & COMPLETE)
    return;

  config_0.flags = (LaneTraceFlags)(config_0.flags | IN_LANE);
  lane.push_back({state, config, false});

  do {
    LTStackToken c_0_ltst = lane.back();
    config_0 = this->states[c_0_ltst.state][c_0_ltst.config];

    uint32_t failed_origins = 0;

    // DO LOOP
    for (LTStackToken c : this->getOriginators(c_0_ltst.state, config_0)) {
      Config &config_1 = this->states[c.state][c.config];
      std::vector<Token> cf_1_rhs =
          std::get<1>(this->grammar->rules[config_1.rule_idx]);

      // Test A
      if (config_1.bullet < cf_1_rhs.size() - 1) {
        Token head = cf_1_rhs[config_1.bullet + 1];
        if (head.kind == TERM)
          gened_contexts = {head};
        else
          gened_contexts = this->getTokenHeads(head.id);

        // We don't allow ε reductions so immediately add the context
        this->addContexts(gened_contexts, c_0_ltst, lane);
      }
      // Test B
      else if (config_1.flags & COMPLETE) {
        gened_contexts = config_1.context;
        this->addContexts(gened_contexts, c_0_ltst, lane);
      }
      // Test C
      else if (config_1.flags & IN_LANE) {
        moveMarkers(lane, c_0_ltst, c, failed_origins > 0);
        gened_contexts = config_1.context;
        this->addContexts(gened_contexts, c_0_ltst, lane);
      }
      // All tests failed
      else {
        if (failed_origins == 0) {
          lane.push_back(c);
          config_1.flags = (LaneTraceFlags)(config_1.flags | IN_LANE);
        } else if (failed_origins == 1) {
          LTStackToken c_prev = lane.back();
          lane.pop_back();
          lane.push_back({0, 0, true});
          lane.push_back(c_prev);

          stack.push_back({0, 0, true});
          stack.push_back(c);
        } else {
          stack.push_back(c);
        }
        failed_origins++;
      }
    }

    if (failed_origins > 0)
      continue;

    while (true) {
      if (lane.back().marker) {
        LTStackToken stack_top = stack.back();
        LaneTraceFlags &stk_top_flags =
            this->states[stack_top.state][stack_top.config].flags;
        while (!stack_top.marker && (stk_top_flags & COMPLETE)) {
          stack.pop_back();
        }
        if (stack_top.marker) {
          stack.pop_back();
          lane.pop_back();
        } else {
          stack.pop_back();
          stk_top_flags = (LaneTraceFlags)(stk_top_flags | IN_LANE);
          lane.push_back(stack_top);
          break;
        }
      } else if (lane.back().zero) {
        lane.pop_back();
      } else {
        Config &lane_top = this->states[lane.back().state][lane.back().config];
        lane_top.flags = (LaneTraceFlags)(lane_top.flags & ~IN_LANE);
        lane_top.flags = (LaneTraceFlags)(lane_top.flags | COMPLETE);

        // Propagate contexts
        if (lane_top.bullet == 0) {
          for (Config &c : this->states[lane.back().state]) {
            if (c.bullet == 0) {
              c.context = lane_top.context;
              c.flags = (LaneTraceFlags)(c.flags | COMPLETE);
            }
          }
        }
        if (lane.size() > 1)
          lane.pop_back();
        else
          break;
      }
    }

    // End DO LOOP
  } while (lane.size() > 1);
}

// Returns the config index holding the right context
uint32_t LRParseTable::laneTracing(size_t state, size_t reduce_rule) {
  auto const &[head, rhs, _] = this->grammar->rules[reduce_rule];
  for (uint32_t i = 0; i < this->states[state].size(); i++) {
    Config config = this->states[state][i];
    if (config.rule_idx == reduce_rule && config.bullet == rhs.size()) {
      this->laneTrace(state, i);
      return i;
    }
  }
  // Shouldn't reach here
  std::cerr << "Lane tracing rule " << reduce_rule
            << " not corresponded with reduction in state " << state
            << std::endl;
  return 0;
}

LRParseTable::LRParseTable(Grammar *grammar) : grammar(grammar) {
  std::cout << "Making LR(0) Grammar parse table" << std::endl;

  unfinished_lrstate start_set = {Config{this->grammar->start_rule, 0}};
  LRState start_state = getStateClosure(this->grammar->rules, start_set);
  this->states = {start_state};

  this->generateStates();
  this->printStates();

  uint32_t terms = this->grammar->terms_size;
  uint32_t nonterms = this->grammar->nonterms_size;

  this->goto_table = new int[this->state_count * nonterms];
  this->action_table = new ParseAction[this->state_count * terms];
  memset(this->action_table, 0,
         sizeof(ParseAction) * this->state_count * terms);

  std::unordered_map<size_t,
                     std::vector<std::unordered_set<Token, boost::hash<Token>>>>
      conflicting_states;

  for (size_t state = 0; state < this->state_count; state++) {
    std::vector<std::unordered_set<Token, boost::hash<Token>>> contexts;
    // Add the transition rules
    for (uint32_t token = 0; token < terms; token++) {
      if (this->trans_table[state][token] != -1) {
        contexts.push_back({Token{TERM, token}});
        this->action_table[state * terms + token] = {
            SHIFT, static_cast<uint32_t>(this->trans_table[state][token])};
      }
    }

    // Fill the GOTO table
    for (size_t token = 0; token < nonterms; token++)
      this->goto_table[state * nonterms + token] =
          this->trans_table[state][terms + token];

    // Add reduction rule contexts
    bool conflicting = false;
    for (size_t reduction : this->reductions[state]) {
      uint32_t config = this->laneTracing(state, reduction);
      auto &config_context = this->states[state][config].context;
      contexts.push_back(config_context);

      for (Token const &token : config_context) {
        if (action_table[state * terms + token.id].kind != EMPTY) {
          conflicting = true;
          std::cout << "Ambiguous Grammar: "
                    << action_table[state * terms + token.id].reduce_rule << " "
                    << reduction << std::endl;
        }
        this->action_table[state * terms + token.id] = {REDUCE,
                                                        (uint32_t)reduction};
      }
    }
    if (conflicting)
      conflicting_states[state] = contexts;
  }

  // Add Accept actions
  for (int const &state : this->end_states)
    this->action_table[state * terms + 1] = {ACCEPT, 0};

  // At this point we have a LALR(0) parser
}

LRParseTable::~LRParseTable() {
  delete[] this->action_table;
  delete[] this->goto_table;
  for (int *&row : this->trans_table)
    delete[] row;
}

ParseAction LRParseTable::getAction(uint32_t state, uint32_t token) const {
  return this->action_table[state * this->grammar->terms_size + token];
}

int LRParseTable::getGoto(uint32_t state, uint32_t nonterm) const {
  return this->goto_table[state * this->grammar->nonterms_size + nonterm];
}

} // namespace grammar
