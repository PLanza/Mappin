#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "grammar.hpp"
#include "util/util.hpp"

namespace grammar {

class AmbiguousGrammarException : public MappinException {
  const char *message() const override {
    return "The following rule makes the grammar ambiguous: ";
  }

  AmbiguousGrammarException(const char *file, Span span, std::string line)
      : MappinException(file, span, line) {}

  friend class MappinException;
};

Grammar::Grammar(const char *file_name)
    : file_name(file_name), rules({}), parse_table(nullptr),
      stack_actions(nullptr), lock(false), nonterms_size(0), terminals(nullptr),
      nonterminals(nullptr), nonterm_id_map({}) {
  this->terms_size = 0;
  this->term_id_map = {};

  this->newTerminal("_START_");
  this->newTerminal("_END_");
}

Token Grammar::newToken(TokenKind kind, std::string name) {
  if (kind == TERM)
    return this->newTerminal(name);
  else
    return this->newNonTerminal(name);
}

Token Grammar::newTerminal(std::string name) {
  if (auto id_index = this->term_id_map.find(name);
      id_index == this->term_id_map.end()) {
    term_id_map[name] = this->terms_size++;
  }
  return {TERM, term_id_map[name], name};
}

Token Grammar::newNonTerminal(std::string name) {
  if (auto id_index = this->nonterm_id_map.find(name);
      id_index == this->nonterm_id_map.end()) {
    nonterm_id_map[name] = this->nonterms_size++;
  }
  return {NON_TERM, nonterm_id_map[name], name};
}

void Grammar::addRule(std::string name, std::vector<Token> rhs, bool start,
                      std::size_t line) {
  if (this->lock) {
    std::cerr << "ERROR: Attempted to add rule to grammar after it was locked."
              << std::endl;
    return;
  }

  Token head = this->newNonTerminal(name);

  if (start)
    this->start_rule = head.id;

  this->rules.push_back({head, rhs, line});
}

void Grammar::printGrammar() {
  for (const auto &[head, rhs, _] : this->rules) {
    if (head.id == this->start_rule)
      std::cout << "$ ";
    std::cout << head.name << " := ";

    for (auto &token : rhs) {
      std::cout << "(" << token.id << ": " << token.name << ") ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

void Grammar::printToken(Token token) {
  switch (token.kind) {
  case TERM: {
    std::cout << this->terminals[token.id];
    break;
  }
  case NON_TERM: {
    std::cout << this->nonterminals[token.id];
    break;
  }
  case ANY: {
    std::cout << "_";
    break;
  }
  case REST: {
    std::cout << "-";
    break;
  }
  }
}

void Grammar::printStackActions() {
  if (!this->stack_actions) {
    std::cerr
        << "ERROR: Attempted to print stack actions before they were generated"
        << std::endl;
    return;
  }

  for (uint32_t i = 0; i < this->terms_size; i++) {
    printf("%8s : ", this->terminals[i].c_str());

    for (const auto &stack_action : stack_actions[i]) {
      for (const grammar::Token &token : stack_action.lhs) {
        this->printToken(token);
      }
      std::cout << "/";
      for (const grammar::Token &token : stack_action.rhs) {
        this->printToken(token);
      }
      std::cout << "  ";
    }
    std::cout << std::endl;
  }
}

void Grammar::lockGrammar() {
  this->lock = true;

  this->terminals = new std::string[this->terms_size];
  for (const auto &[name, t_id] : this->term_id_map)
    this->terminals[t_id] = name;

  this->nonterminals = new std::string[this->nonterms_size];
  for (const auto &[name, nt_id] : this->nonterm_id_map)
    this->nonterminals[nt_id] = name;
}

MappinException *exceptionOnLine(GrammarExceptionKind kind,
                                 const char *file_name, std::size_t line) {
  std::ifstream file_stream;
  file_stream.open(file_name);

  std::string grammar_line;

  for (int i = 1; i <= line; i++)
    std::getline(file_stream, grammar_line);

  Span span = Span{{line, 1}, {line, grammar_line.length() + 1}};

  switch (kind) {
  case AMBIGUOUS_GRAMMAR:
    return MappinException::newMappinException<AmbiguousGrammarException>(
               file_name, span, std::optional<std::string>(grammar_line))
        .value();
  default:
    // Should never reach here but makes compiler happy
    return nullptr;
  }
}

LLGrammar::LLGrammar(const char *file_name) : Grammar(file_name) {}

void LLGrammar::makeParseTable() {
  this->lockGrammar();

  this->parse_table = new LLParseTable(this->terms_size, this->nonterms_size,
                                       this->rules, this->file_name);
}

void LLGrammar::generateStackActions() {
  this->stack_actions = new std::vector<StackAction>[this->terms_size];

  // _START_ => /A
  this->printToken(std::get<0>(this->rules[start_rule]));
  this->stack_actions[0].push_back(
      StackAction{{}, {std::get<0>(this->rules[start_rule])}});

  // _END_ => -/
  this->stack_actions[1].push_back(StackAction{{REST_TOKEN}, {}});

  // SHIFT rules
  for (uint32_t t_id = 2; t_id < this->terms_size; t_id++) {
    this->stack_actions[t_id].push_back(StackAction{
        {Token{TERM, t_id, this->terminals[t_id]}, REST_TOKEN}, {REST_TOKEN}});
  }

  // REDUCE rules
  for (uint32_t t_id = 2; t_id < this->terms_size; t_id++) {
    for (uint32_t nt_id = 0; nt_id < this->nonterms_size; nt_id++) {
      ParseAction action = this->parse_table->getAction(nt_id, t_id);
      switch (action.kind) {
      case EMPTY:
      case SHIFT:
        // Shouldn't reach here for LL Grammar
        break;
      case REDUCE: {
        std::vector<Token> right_stack =
            this->findTerminal(t_id, action.reduce_rule);
        right_stack.push_back(REST_TOKEN);
        this->stack_actions[t_id].push_back(StackAction{
            {Token{NON_TERM, nt_id, this->nonterminals[nt_id]}, REST_TOKEN},
            right_stack});
        break;
      }
      }
    }
  }
}

std::vector<Token> LLGrammar::findTerminal(uint32_t term_id, uint32_t rule) {
  std::vector<Token> rhs = std::get<1>(this->rules[rule]);
  Token first = rhs[0];

  switch (first.kind) {
  case (TERM): {
    if (term_id == first.id)
      return {rhs.begin() + 1, rhs.end()};
    else {
      std::cerr << "error: got wrong terminal \'" << this->terminals[term_id]
                << "\' when generating stack actions\n"
                << "Probably a problem with the parse table." << std::endl;
      return {};
    }
  }
  case (NON_TERM): {
    uint32_t next_rule =
        this->parse_table->getAction(first.id, term_id).reduce_rule;
    std::vector<Token> prefix = this->findTerminal(term_id, next_rule);
    prefix.insert(prefix.begin(), rhs.begin() + 1, rhs.end());
    return prefix;
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
  std::cout << "Making a " << this->rows << " x " << this->cols
            << " LL parse table" << std::endl;

  // Initialize parse table as a contiguous array
  ParseAction *parse_table = new ParseAction[this->rows * this->cols];
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      parse_table[i * this->cols + j] = ParseAction{EMPTY, 0};
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
        for (auto &element : first_set[first.id]) {
          not_done |= first_set[head.id]
                          .insert(element & (0xFFFFFFFFull << 32) |
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

      if (parse_table[non_term * this->cols + term].kind != EMPTY) {
        throw exceptionOnLine(AMBIGUOUS_GRAMMAR, file_name,
                              std::get<2>(rules[rule]));
      }
      parse_table[non_term * this->cols + term] = {REDUCE, rule};
    }
  }

  this->table = parse_table;
}

void LLGrammar::printParseTable() {
  if (!this->lock) {
    std::cerr << "ERROR: Attempted to print parse table before it "
                 "was made."
              << std::endl;
    return;
  }
  // Top row
  printf("%10s |", "");
  for (int t_id = 0; t_id < this->terms_size; t_id++)
    printf("%-8s ", this->terminals[t_id].c_str());

  std::cout << std::endl;
  for (int i = 0; i < 12 + 9 * this->terms_size; i++)
    std::cout << "-";
  std::cout << std::endl;

  for (int nt_id = 0; nt_id < this->nonterms_size; nt_id++) {
    printf("%10s |", this->nonterminals[nt_id].c_str());
    for (int t_id = 0; t_id < this->terms_size; t_id++) {
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
      }
    }
    std::cout << std::endl;
  }
}

ParseAction LLParseTable::getAction(uint32_t nonterm, uint32_t term) {
  return this->table[nonterm * this->cols + term];
}

LLParseTable::~LLParseTable() { delete[] this->table; }

} // namespace grammar
