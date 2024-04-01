#include "grammar/llgrammar.hpp"

#include <iostream>
#include <unordered_set>
#include <utility>

#include <parse/nodes.hpp>

namespace grammar {
LLGrammar::LLGrammar(const char *file_name) : Grammar(file_name) {}

LLGrammar::~LLGrammar() {
  delete this->parse_table;
  delete[] this->stack_actions;
}

void LLGrammar::finalize() { this->fillStringArrays(); }

void LLGrammar::makeParseTable() {
  this->parse_table = new LLParseTable(this->terms_size, this->nonterms_size,
                                       this->rules, this->file_name);
}

LLParseTable *LLGrammar::getParseTable() { return this->parse_table; }

void LLGrammar::generateStackActions() {
  if (this->parse_table == nullptr) {
    std::cerr
        << "ERROR: Attempted to generate stack actions before the parse table"
        << std::endl;
    return;
  }
  std::cout << "Generating stack actions for LL Grammar" << std::endl;

  this->stack_actions = new std::vector<StackAction>[this->terms_size];

  // '<' => /A
  this->stack_actions[0].push_back(StackAction{
      {}, {std::get<0>(this->rules[start_rule]).toStackState()}, {}});

  // '>' => -/
  this->stack_actions[1].push_back(StackAction{{REST_STATE}, {}, {}});

  // SHIFT rules
  for (uint32_t t_id = 2; t_id < this->terms_size; t_id++) {
    this->stack_actions[t_id].push_back(StackAction{
        {Token{TERM, t_id}.toStackState(), REST_STATE}, {REST_STATE}, {}});
  }

  // REDUCE rules
  for (uint32_t t_id = 2; t_id < this->terms_size; t_id++) {
    for (uint32_t nt_id = 0; nt_id < this->nonterms_size; nt_id++) {
      ParseAction action = this->parse_table->getAction(nt_id, t_id);
      switch (action.kind) {
      case EMPTY:
      case SHIFT:
      case ACCEPT:
        // Shouldn't reach here for LL Grammar
        break;
      case REDUCE: {
        auto [right_stack, reduce_rules] =
            this->findTerminal(t_id, action.reduce_rule);

        right_stack.push_back(REST_STATE);
        this->stack_actions[t_id].push_back(
            StackAction{{Token{NON_TERM, nt_id}.toStackState(), REST_STATE},
                        right_stack,
                        reduce_rules});
        break;
      }
      }
    }
  }
}

std::pair<std::deque<StackState>, std::deque<uint32_t>>
LLGrammar::findTerminal(uint32_t term_id, uint32_t rule) {
  std::vector<Token> rhs = std::get<1>(this->rules[rule]);
  Token first = rhs[0];
  std::deque<StackState> stack_suffix;
  for (size_t i = 1; i < rhs.size(); i++) {
    stack_suffix.push_back(rhs[i].toStackState());
  }

  switch (first.kind) {
  case (TERM): {
    if (term_id == first.id)
      return {stack_suffix, {rule}};
    else {
      std::cerr << "error: got wrong terminal \'" << this->terminals[term_id]
                << "\' when generating stack actions\n"
                << "Probably a problem with the parse table." << std::endl;
      return {{}, {}};
    }
  }
  case (NON_TERM): {
    uint32_t next_rule =
        this->parse_table->getAction(first.id, term_id).reduce_rule;

    auto [stack_prefix, reduce_rules] = this->findTerminal(term_id, next_rule);
    stack_prefix.insert(stack_prefix.end(), stack_suffix.begin(),
                        stack_suffix.end());
    reduce_rules.insert(reduce_rules.begin(), rule);

    return {stack_prefix, reduce_rules};
  }
  default:
    // Should never arrive here
    std::cerr << "error finding terminal \'" << this->terminals[term_id]
              << "\' when generating stack actions" << std::endl;
    return {};
  }
}

LLParseTable::LLParseTable(uint32_t terminals, uint32_t nonterminals,
                           grammar_rules &rules, const char *file_name)
    : rows(nonterminals), cols(terminals) {
  std::cout << "Making LL Grammar parse table" << std::endl;

  // Initialize parse table as a contiguous array
  this->table = new ParseAction[this->rows * this->cols];
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      this->table[i * this->cols + j] = ParseAction{EMPTY, 0};
    }
  }

  // Initialize first set
  // first 32 bytes are the terminal id, last 32 bytes are the rule
  // index
  std::unordered_set<uint64_t> first_set[this->rows];

  bool not_done;
  do {
    not_done = false;
    for (uint32_t i = 0; i < rules.size(); i++) {
      const auto &[head, rhs, _] = rules[i];
      Token first = rhs[0];

      if (first.kind == TERM)
        not_done |= first_set[head.id]
                        .insert(static_cast<uint64_t>(first.id) << 32 |
                                static_cast<uint64_t>(i))
                        .second;
      else {
        for (const uint64_t &element : first_set[first.id]) {
          not_done |= first_set[head.id]
                          .insert(element & (uint64_t(0xFFFFFFFF) << 32) |
                                  static_cast<uint64_t>(i))
                          .second;
        }
      }
    }
  } while (not_done);

  // Set REDUCE rules from first set
  for (uint32_t non_term = 0; non_term < this->rows; non_term++) {
    for (const uint64_t &element : first_set[non_term]) {
      uint32_t term = static_cast<uint32_t>(element >> 32);
      uint32_t rule = static_cast<uint32_t>(element);

      if (this->table[non_term * this->cols + term].kind != EMPTY) {
        throw exceptionOnLine(AMBIGUOUS_GRAMMAR, file_name,
                              std::get<2>(rules[rule]));
      }
      this->table[non_term * this->cols + term] = {REDUCE, rule};
    }
  }
}

void LLGrammar::printParseTable() {
  if (this->parse_table == nullptr) {
    std::cerr << "ERROR: Attempted to print parse table before it "
                 "was made."
              << std::endl;
    return;
  }
  // Top row
  printf("%10s | ", "");
  for (int t_id = 2; t_id < this->terms_size; t_id++)
    printf("%-8s ", this->terminals[t_id].c_str());

  std::cout << std::endl;
  for (int i = 0; i < 13 + 9 * (this->terms_size - 2); i++)
    std::cout << "-";
  std::cout << std::endl;

  for (int nt_id = 0; nt_id < this->nonterms_size; nt_id++) {
    printf("%10s | ", this->nonterminals[nt_id].c_str());
    for (int t_id = 2; t_id < this->terms_size; t_id++) {
      ParseAction action = this->parse_table->getAction(nt_id, t_id);
      switch (action.kind) {
      case grammar::EMPTY: {
        std::cout << "         ";
        break;
      }
      case grammar::SHIFT: {
        std::cout << "S        ";
        break;
      }
      case grammar::REDUCE: {
        std::cout << "R" << action.reduce_rule << "       ";
        break;
      }
      case grammar::ACCEPT: {
        std::cout << "acc      ";
        break;
      }
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void LLGrammar::printStackActions() { Grammar::printStackActions(true); }

ParseAction LLParseTable::getAction(uint32_t nonterm, uint32_t term) const {
  return this->table[nonterm * this->cols + term];
}

LLParseTable::~LLParseTable() { delete[] this->table; }

void LLGrammar::getParse(inet::Node *product) {
  // For each parse, check the stack action for incomplete parses
  inet::Node *stack_action = product->ports[2].node;
  if (stack_action->ports[1].node->kind != inet::END &&
      stack_action->ports[2].node->kind != inet::END)
    return;

  // If valid then traverse the reduction rules and print parse
  inet::Node *cons = product->ports[1].node;
  std::deque<uint32_t> stack;
  this->traverseRules(cons, stack);
  std::cout << std::endl;
}

void LLGrammar::traverseRules(inet::Node *cons, std::deque<uint32_t> &stack) {
  if (cons->kind == inet::CONS) {
    this->traverseRules(cons->ports[1].node, stack);
    this->traverseRules(cons->ports[2].node, stack);
    return;
  }

  while (cons->kind == inet::RULE) {
    auto const &rule = this->getRule(cons->value);
    grammar::Token const &head = std::get<0>(rule);
    std::cout << this->getNonTerminalString(head.id);
    std::vector<grammar::Token> const &rhs = std::get<1>(rule);

    unsigned int nonterms = 0;
    for (auto const &token : rhs) {
      if (token.kind == grammar::NON_TERM)
        nonterms++;
    }
    if (nonterms == 0) {
      stack.front()--;
      while (stack.front() == 0) {
        std::cout << " ]";
        stack.pop_front();
        stack.front()--;
      }
    } else {
      std::cout << "[ ";
      stack.push_front(nonterms);
    }

    cons = cons->ports[1].node;
  }
}

} // namespace grammar
