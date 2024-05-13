#include "../../include/generate/grammar/lrtable.hpp"
#include <iostream>

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

std::unordered_set<Token, boost::hash<Token>>
LRParseTable::getFollowSet(Token token) {
  if (this->follow_sets.find(token) != this->follow_sets.end())
    return this->follow_sets[token];

  this->follow_sets[token] = {};
  std::unordered_set<Token, boost::hash<Token>> &follow =
      this->follow_sets[token];

  if (token.kind == TERM && token.id == 0) {
    follow = this->getTokenHeads(
        std::get<0>(this->grammar->rules[this->grammar->start_rule]).id);
    return this->follow_sets[token];
  }

  for (auto const &[head, rhs, _] : this->grammar->rules) {
    for (uint32_t i = 0; i < rhs.size(); i++) {
      if (rhs[i] == token) {
        if (i == rhs.size() - 1) {
          auto to_append = this->getFollowSet(head);
          follow.merge(to_append);
        } else if (rhs[i + 1].kind == TERM) {
          follow.insert(rhs[i + 1]);
        } else {
          auto to_append = this->getTokenHeads(rhs[i + 1].id);
          follow.merge(to_append);
        }
      }
    }
  }

  return this->follow_sets[token];
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
    LTStackToken lane_item = lane[i];
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

  if (this->states[state][config].flags == COMPLETE)
    return;

  this->states[state][config].flags = IN_LANE;
  lane.push_back({state, config, false});

  do {
    LTStackToken c_0_ltst = lane.back();
    Config config_0 = this->states[c_0_ltst.state][c_0_ltst.config];

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
      else if (config_1.flags == COMPLETE) {
        gened_contexts = config_1.context;
        this->addContexts(gened_contexts, c_0_ltst, lane);
      }
      // Test C
      else if (config_1.flags == IN_LANE) {
        moveMarkers(lane, c_0_ltst, c, failed_origins > 0);
        gened_contexts = config_1.context;
        this->addContexts(gened_contexts, c_0_ltst, lane);
      }
      // All tests failed
      else {
        if (failed_origins == 0) {
          lane.push_back(c);
          config_1.flags = IN_LANE;
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
        while (!stack_top.marker && stk_top_flags == COMPLETE) {
          stack.pop_back();
        }
        if (stack_top.marker) {
          stack.pop_back();
          lane.pop_back();
        } else {
          stack.pop_back();
          stk_top_flags = IN_LANE;
          lane.push_back(stack_top);
          break;
        }
      } else if (lane.back().zero) {
        lane.pop_back();
      } else {
        Config &lane_top = this->states[lane.back().state][lane.back().config];
        lane_top.flags = COMPLETE;

        // Propagate contexts
        if (lane_top.bullet == 0) {
          for (Config &c : this->states[lane.back().state]) {
            if (c.bullet == 0) {
              c.context = lane_top.context;
              c.flags = COMPLETE;
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
  // this->printStates();

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
