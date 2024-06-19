#include "../../include/generate/grammar/lrtable.hpp"
#include <algorithm>
#include <iostream>

namespace boost {
template <> struct hash<grammar::Config> {
  size_t operator()(const grammar::Config &c) const {
    size_t hash = 0;
    boost::hash_combine(hash, c.rule_idx);
    boost::hash_combine(hash, c.bullet);
    boost::hash_combine(hash, c.lookahead);
    return hash;
  }
};

} // namespace boost
namespace grammar {

std::unordered_set<uint32_t> LRParseTable::getFirstSet(Token token) {
  if (token.kind == TERM)
    return {token.id};

  uint32_t non_term = token.id;
  if (this->first_sets.find(non_term) != this->first_sets.end())
    return this->first_sets[non_term];

  this->first_sets[non_term] = {};
  std::unordered_set<uint32_t> &first = this->first_sets[non_term];

  std::vector<uint32_t> to_visit = {non_term};
  uint32_t visiting = 0;

  while (visiting < to_visit.size()) {
    for (auto const &[head, rhs, _] : this->grammar->rules) {
      if (head.id == to_visit[visiting]) {
        if (rhs[0].kind == TERM) {
          first.insert(rhs[0].id);
        } else {
          bool in_stack = false;
          for (uint32_t const &non_term : to_visit) {
            if (non_term == rhs[0].id)
              in_stack = true;
          }
          if (!in_stack)
            to_visit.push_back(rhs[0].id);
        }
      }
    }
    visiting++;
  }

  return first;
}

void insertConfig(LRState &s, Config c) {
  if (std::find(s.begin(), s.end(), c) == s.end())
    s.push_back(c);
}

void LRParseTable::getStateClosure(LRState &unfinished) {
  grammar_rules &rules = this->grammar->rules;

  std::size_t curr_config = 0;
  while (curr_config < unfinished.size()) {
    Config config = unfinished[curr_config];
    curr_config++;

    std::vector<Token> config_rule_rhs = std::get<1>(rules[config.rule_idx]);
    if (config.bullet == config_rule_rhs.size())
      continue;

    Token next = config_rule_rhs[config.bullet];
    if (next.kind == TERM)
      continue;

    for (size_t rule = 0; rule < rules.size(); rule++) {
      if (std::get<0>(rules[rule]).id != next.id)
        continue;

      if (config.bullet + 1 == config_rule_rhs.size() ||
          config.rule_idx == this->grammar->start_rule) {
        insertConfig(unfinished, Config(rule, 0, config.lookahead));
        continue;
      } else {
        std::unordered_set<uint32_t> first_set =
            this->getFirstSet(config_rule_rhs[config.bullet + 1]);
        for (uint32_t lookahead : first_set)
          insertConfig(unfinished, Config(rule, 0, lookahead));
      }
    }
  }
}

LRState LRParseTable::generateNextState(uint32_t state, Token token) {
  LRState unfinished;
  for (const Config &config : this->states[state]) {
    if (std::get<1>(this->grammar->rules[config.rule_idx]).size() ==
        config.bullet)
      continue;
    else if (std::get<1>(
                 this->grammar->rules[config.rule_idx])[config.bullet] == token)
      unfinished.push_back(
          Config(config.rule_idx, config.bullet + 1, config.lookahead));
  }

  this->getStateClosure(unfinished);
  return unfinished;
}

// Checks if two states contain the same configurations
bool equalStates(const LRState &left, const LRState &right) {
  for (Config const &l_config : left) {
    bool contains = false;
    for (Config const &r_config : right)
      contains |= l_config == r_config;

    if (!contains)
      return false;
  }
  return true;
}

int LRParseTable::findState(const LRState &state) {
  for (int i = 0; i < this->states.size(); i++) {
    if (equalStates(state, this->states[i])) {
      return i;
    }
  }

  return -1;
}

void LRParseTable::generateStates() {
  LRState start_state = {Config{this->grammar->start_rule, 0, 1}};
  this->getStateClosure(start_state);
  this->states = {start_state};

  std::size_t curr_state = 0;
  while (curr_state < this->states.size()) {

    this->trans_table.push_back({});
    for (uint32_t term = 2; term < this->grammar->terms_size; term++) {
      LRState next_state =
          this->generateNextState(curr_state, Token{TERM, term});

      if (!next_state.empty()) {
        if (int same = this->findState(next_state); same == -1) {
          this->trans_table.back()[Token{TERM, term}] = this->states.size();
          this->states.push_back(next_state);
        } else {
          this->trans_table.back()[Token{TERM, term}] = same;
        }
      }
    }

    for (uint32_t nonterm = 0; nonterm < this->grammar->nonterms_size;
         nonterm++) {
      LRState next_state =
          this->generateNextState(curr_state, Token{NON_TERM, nonterm});

      if (!next_state.empty()) {
        if (int same = this->findState(next_state); same == -1) {
          this->trans_table.back()[Token{NON_TERM, nonterm}] =
              this->states.size();
          this->states.push_back(next_state);
        } else {
          this->trans_table.back()[Token{NON_TERM, nonterm}] = same;
        }
      }
    }

    curr_state++;
  }

  this->state_count = this->states.size();
}

void LRParseTable::fillTables() {
  uint32_t terms = this->grammar->terms_size;
  uint32_t nonterms = this->grammar->nonterms_size;
  const grammar_rules &rules = this->grammar->rules;

  this->goto_table = new int[this->state_count * nonterms];
  this->action_table = new ParseAction[this->state_count * terms];
  memset(this->action_table, 0,
         sizeof(ParseAction) * this->state_count * terms);

  // Fill in action table
  for (uint32_t state = 0; state < this->state_count; state++) {
    for (const auto &config : this->states[state]) {
      const auto &[head, rhs, _] = rules[config.rule_idx];
      ParseAction *action_table_row = this->action_table + state * terms;

      // Accept action
      if (config.bullet == rhs.size() - 1 &&
          config.rule_idx == this->grammar->start_rule) {
        action_table_row[1] = ParseAction{ACCEPT, 0};
      } // Shift actions
      else if (config.bullet < rhs.size() && rhs[config.bullet].kind == TERM) {
        action_table_row[rhs[config.bullet].id] =
            ParseAction{SHIFT, this->trans_table[state][rhs[config.bullet]]};
      } // Reduce action
      else if (config.bullet == rhs.size() &&
               config.rule_idx != this->grammar->start_rule) {
        action_table_row[config.lookahead] =
            ParseAction{REDUCE, static_cast<uint32_t>(config.rule_idx)};
      }
    }
  }

  // Fill in GOTO table
  for (uint32_t state = 0; state < this->state_count; state++) {
    for (int nt = 0; nt < nonterms; nt++)
      this->goto_table[state * nonterms + nt] = -1;

    for (const auto &[token, goto_state] : this->trans_table[state]) {
      if (token.kind == NON_TERM) {
        this->goto_table[state * nonterms + token.id] = goto_state;
      }
    }
  }
}

LRParseTable::LRParseTable(Grammar *grammar) : grammar(grammar) {
  std::cout << "Making LR(1) Grammar parse table" << std::endl;

  this->generateStates();

  this->printStates();

  this->fillTables();
}

LRParseTable::~LRParseTable() {
  delete[] this->action_table;
  delete[] this->goto_table;
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

  std::cout << ", " << this->grammar->terminals[config.lookahead];
}

void LRParseTable::printStates() {
  for (LRState const &state : states) {
    for (Config const &config : state) {
      this->printConfig(config);
      std::cout << std::endl;
    }
    std::cout << "\n ---------- \n" << std::endl;
  }
}

std::unordered_set<uint32_t> LRParseTable::getFollowSet(Token token) {
  if (this->follow_sets.find(token) != this->follow_sets.end())
    return this->follow_sets[token];

  this->follow_sets[token] = {};
  std::unordered_set<uint32_t> &follow = this->follow_sets[token];

  if (token.kind == TERM && token.id == 0) {
    follow = this->getFirstSet(
        std::get<0>(this->grammar->rules[this->grammar->start_rule]));
    return this->follow_sets[token];
  }

  for (auto const &[head, rhs, _] : this->grammar->rules) {
    for (uint32_t i = 0; i < rhs.size(); i++) {
      if (rhs[i] == token) {
        if (i == rhs.size() - 1) {
          auto to_append = this->getFollowSet(head);
          follow.merge(to_append);
        } else if (rhs[i + 1].kind == TERM) {
          follow.insert(rhs[i + 1].id);
        } else {
          auto to_append = this->getFirstSet(rhs[i + 1]);
          follow.merge(to_append);
        }
      }
    }
  }

  return this->follow_sets[token];
}

ParseAction LRParseTable::getAction(uint32_t state, uint32_t token) const {
  return this->action_table[state * this->grammar->terms_size + token];
}

int LRParseTable::getGoto(uint32_t state, uint32_t nonterm) const {
  return this->goto_table[state * this->grammar->nonterms_size + nonterm];
}
} // namespace grammar
