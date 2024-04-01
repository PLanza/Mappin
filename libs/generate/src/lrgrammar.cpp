#include "grammar/lrgrammar.hpp"
#include "grammar.hpp"
#include <algorithm>
#include <boost/container_hash/extensions.hpp>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <unordered_set>

namespace boost {
template <> struct hash<grammar::Item> {
  size_t operator()(const grammar::Item &i) const {
    size_t hash = 0;
    boost::hash_combine(hash, i.rule_idx);
    boost::hash_combine(hash, i.bullet_pos);
    return hash;
  }
};
} // namespace boost

namespace grammar {

LR0Grammar::LR0Grammar(const char *file_name) : Grammar(file_name) {}

LR0Grammar::~LR0Grammar() {
  delete this->parse_table;
  delete[] this->stack_actions;
}

void LR0Grammar::finalize() {
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

void LR0Grammar::makeParseTable() {
  this->parse_table =
      new LRParseTable(this->terms_size, this->nonterms_size, this->rules,
                       this->start_rule, this->file_name);
}

LRParseTable *LR0Grammar::getParseTable() { return this->parse_table; }

void LR0Grammar::printParseTable() {
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

  for (uint32_t state = 0; state < this->parse_table->states; state++) {
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

void LR0Grammar::printStackActions() { Grammar::printStackActions(false); }

bool mergeStackActions(StackAction &left_action, StackAction right_action) {
  // A-/ should not be able to merge with anything
  if (left_action.rhs.size() == 0)
    return false;

  size_t max_i = std::min(left_action.rhs.size(), right_action.lhs.size());
  size_t i = 0;
  // Traverse until reach rest
  for (; i < max_i; i++) {
    StackState left_state = left_action.rhs[i],
               right_state = right_action.lhs[i];
    if (left_state.kind == REST || right_state.kind == REST)
      break;
    if (left_state.kind == SOME && right_state.kind == SOME) {
      if (left_state.value != right_state.value)
        return false;
    }
  }

  if (left_action.rhs[i].kind == REST) {
    left_action.lhs.erase(left_action.lhs.end() - 1);
    left_action.lhs.insert(left_action.lhs.end(), right_action.lhs.begin() + i,
                           right_action.lhs.end());
  } else if (right_action.lhs[i].kind == REST && right_action.rhs.size() > 0) {
    right_action.rhs.erase(right_action.rhs.end() - 1);
    right_action.rhs.insert(right_action.rhs.end(), left_action.rhs.begin() + i,
                            left_action.rhs.end());
  }
  left_action.rhs = right_action.rhs;
  left_action.reduce_rules.insert(left_action.reduce_rules.end(),
                                  right_action.reduce_rules.begin(),
                                  right_action.reduce_rules.end());
  return true;
}

void LR0Grammar::getStackActionClosure(uint32_t term) {
  std::vector<StackAction> curr_actions;
  while (this->stack_actions[term] != curr_actions) {
    curr_actions = this->stack_actions[term];
    for (StackAction &left_action : this->stack_actions[term]) {
      for (StackAction &right_action : this->stack_actions[term]) {
        if (mergeStackActions(left_action, right_action))
          break;
      }
    }
  }
}

void LR0Grammar::generateStackActions() {
  if (this->parse_table == nullptr) {
    std::cerr
        << "ERROR: Attempted to generate stack actions before the parse table"
        << std::endl;
    return;
  }
  std::cout << "Generating stack actions for LR(0) Grammar" << std::endl;

  this->stack_actions = new std::vector<StackAction>[this->parse_table->states];
  this->stack_actions[0] = {StackAction{{}, {{SOME, 0}}, {}}};

  for (uint32_t term = 1; term < this->terms_size; term++) {
    for (uint32_t state = 0; state < this->parse_table->states; state++) {
      ParseAction action = this->parse_table->getAction(state, term);
      switch (action.kind) {
      case SHIFT: {
        this->stack_actions[term].push_back(
            StackAction{{{SOME, state}, REST_STATE},
                        {{SOME, action.reduce_rule}, {SOME, state}, REST_STATE},
                        {}});
        break;
      }
      case REDUCE: {
        std::deque<StackState> lhs_prefix = {{SOME, state}};
        auto &[rule_lhs, rule_rhs, _] = this->rules[action.reduce_rule];
        for (int i = 1; i < rule_rhs.size(); i++)
          lhs_prefix.push_back(ANY_STATE);

        for (uint32_t state = 0; state < this->parse_table->states; state++) {
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
                          {action.reduce_rule}});
        }
        break;
      }
      case ACCEPT: {
        this->stack_actions[term].push_back(
            StackAction{{{SOME, state}, REST_STATE}, {}, {}});
        break;
      }
      case EMPTY:
        break;
      }
    }
    this->getStackActionClosure(term);
  }
}

void LR0Grammar::getParses(inet::Node *) {}

void getSetClosure(grammar_rules const &rules,
                   std::unordered_set<Item, boost::hash<Item>> &set) {
  std::unordered_set<Item, boost::hash<Item>> prev_set;
  while (set.size() != prev_set.size()) {
    prev_set = set;

    for (size_t i = 0; i < rules.size(); i++) {
      for (Item const &item : prev_set) {
        std::vector<Token> const &rhs = std::get<1>(rules[item.rule_idx]);
        if (item.bullet_pos >= rhs.size())
          continue;

        if (rhs[item.bullet_pos].kind == NON_TERM) {
          uint32_t nonterm_id = rhs[item.bullet_pos].id;
          if (std::get<0>(rules[i]).id == nonterm_id)
            set.insert({i, 0});
        }
      }
    }
  }
}

LRParseTable::LRParseTable(uint32_t terminals, uint32_t nonterminals,
                           grammar_rules const &rules, uint32_t start_rule,
                           const char *file_name)
    : terms(terminals), nonterms(nonterminals) {
  std::cout << "Making LR(0) Grammar parse table" << std::endl;

  // The set of items that correspond to the parser's states: state -> {item}
  std::unordered_set<Item, boost::hash<Item>> start_set = {{start_rule, 0}};
  getSetClosure(rules, start_set);
  std::vector<std::unordered_set<Item, boost::hash<Item>>> item_sets = {
      start_set};

  // (state × token) -> state
  std::vector<int *> trans_table;
  std::vector<int> end_states;
  std::vector<std::tuple<size_t, ParseAction>> reduce_states;

  std::size_t curr_state = 0;
  std::size_t prev_size = item_sets.size();

  while (curr_state < item_sets.size()) {
    std::unordered_map<Token, std::unordered_set<Item, boost::hash<Item>>,
                       boost::hash<Token>>
        new_sets;
    for (Item const &item : item_sets[curr_state]) {
      std::vector<Token> const &rhs = std::get<1>(rules[item.rule_idx]);

      // A -> α ... β •
      if (item.bullet_pos >= rhs.size()) {
        reduce_states.push_back(
            {curr_state,
             ParseAction{REDUCE, static_cast<uint32_t>(item.rule_idx)}});
        continue;
      }
      // A -> α ... β • eoi
      if (rhs[item.bullet_pos] == Token{TERM, 1}) {
        end_states.push_back(curr_state);
        continue;
      }

      size_t set_idx;
      if (new_sets.find(rhs[item.bullet_pos]) != new_sets.end())
        new_sets[rhs[item.bullet_pos]].insert(
            {item.rule_idx, item.bullet_pos + 1});
      else {
        new_sets[rhs[item.bullet_pos]] = {{item.rule_idx, item.bullet_pos + 1}};
      }
    }

    trans_table.push_back(new int[terminals + nonterminals]);
    for (int i = 0; i < terminals + nonterminals; i++)
      trans_table.back()[i] = -1;

    for (auto &[token, set] : new_sets) {
      getSetClosure(rules, set);

      int repeat = -1;
      for (size_t state = 0; state < prev_size; state++) {
        if (set == item_sets[state])
          repeat = state;
      }

      size_t table_idx = token.id + (token.kind == TERM ? 0 : terminals);
      if (repeat == -1) {
        trans_table.back()[table_idx] = item_sets.size();
        item_sets.push_back(set);
      } else
        trans_table.back()[table_idx] = repeat;
    }

    curr_state++;
    prev_size = item_sets.size();
  }

  assert(trans_table.size() == item_sets.size());
  this->states = item_sets.size();

  this->goto_table = new int[this->states * this->nonterms];
  this->action_table = new ParseAction[this->states * this->terms];
  for (int i = 0; i < this->states; i++) {
    for (int j = 0; j < this->terms; j++) {
      this->action_table[i * this->terms + j] = ParseAction{EMPTY, 0};
    }
  }

  for (size_t state = 0; state < trans_table.size(); state++) {
    // Add the Shift actions
    for (size_t term = 0; term < terminals; term++) {
      if (trans_table[state][term] != -1)
        this->action_table[state * terminals + term] = {
            SHIFT, static_cast<uint32_t>(trans_table[state][term])};
    }
    // Fill the GOTO table
    for (size_t nonterm = 0; nonterm < nonterminals; nonterm++)
      this->goto_table[state * nonterminals + nonterm] =
          trans_table[state][terminals + nonterm];
  }
  // Add Reduce actions
  for (auto const &[state, action] : reduce_states) {
    for (size_t token = 0; token < terminals; token++) {
      if (action_table[state * terminals + token].kind != EMPTY)
        throw exceptionOnLine(AMBIGUOUS_GRAMMAR, file_name,
                              std::get<2>(rules[action.reduce_rule]));
      this->action_table[state * terminals + token] = action;
    }
  }
  // Add Accept actions
  for (int const &state : end_states)
    this->action_table[state * terminals + 1] = {ACCEPT, 0};

  for (int *&row : trans_table)
    delete[] row;
}

LRParseTable::~LRParseTable() {
  delete[] this->action_table;
  delete[] this->goto_table;
}

ParseAction LRParseTable::getAction(uint32_t state, uint32_t token) const {
  return this->action_table[state * this->terms + token];
}

int LRParseTable::getGoto(uint32_t state, uint32_t nonterm) const {
  return this->goto_table[state * this->nonterms + nonterm];
}

} // namespace grammar
