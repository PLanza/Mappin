#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "grammar.hpp"

namespace grammar {

Grammar::Grammar()
    : next_nonterm_id(0), next_term_id(2), rules({}),
      term_id_map({{"_START_", 0}, {"_END_", 1}}){};

Token Grammar::newToken(TokenKind kind, std::string name) {
  if (kind == TERM)
    return this->newTerminal(name);
  else
    return this->newNonTerminal(name);
}

Token Grammar::newTerminal(std::string name) {
  if (auto id_index = this->term_id_map.find(name);
      id_index == this->term_id_map.end()) {
    term_id_map[name] = this->next_term_id++;
  }
  return {TERM, term_id_map[name], name};
}

Token Grammar::newNonTerminal(std::string name) {
  if (auto id_index = this->nonterm_id_map.find(name);
      id_index == this->nonterm_id_map.end()) {
    nonterm_id_map[name] = this->next_nonterm_id++;
  }
  return {NON_TERM, nonterm_id_map[name], name};
}

// Inserts a new rule to the grammar
// If the non-terminal has already been defined, then
// append the new right hand sides to the existing rule
void Grammar::addRule(std::string name, std::vector<Token> rhs, bool start) {
  Token head = this->newNonTerminal(name);

  if (start) {
    this->start_rule = head.id;
  }

  this->rules.push_back({head, rhs});
}

ParseAction *Grammar::makeParseTable() { throw new std::bad_function_call(); }

void Grammar::print() {
  for (const auto &[head, rhs] : this->rules) {
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

LLGrammar::LLGrammar() : Grammar() {}

ParseAction *LLGrammar::makeParseTable() {
  unsigned int rows = Grammar::next_nonterm_id;
  unsigned int cols = Grammar::next_term_id;

  std::cout << "Making a " << rows << " x " << cols << " LL parse table"
            << std::endl;

  // Initialize parse table as a contiguous array
  ParseAction *parse_table = new ParseAction[rows * cols];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      parse_table[i * cols + j] = ParseAction{EMPTY, 0};
    }
  }

  // Initialize first set
  // first 32 bytes are the terminal id, last 32 bytes are the rule index
  std::unordered_set<unsigned long long> first_set[Grammar::next_nonterm_id];

  bool not_done;
  do {
    not_done = false;
    for (unsigned int i = 0; i < Grammar::rules.size(); i++) {
      const auto &[head, rhs] = Grammar::rules[i];
      Token first = rhs[0];

      if (first.kind == TERM)
        not_done |=
            first_set[head.id]
                .insert(static_cast<unsigned long long>(first.id) << 32 |
                        static_cast<unsigned long long>(i))
                .second;
      else {
        for (auto &element : first_set[first.id]) {
          not_done |= first_set[head.id]
                          .insert(element & (0xFFFFFFFFull << 32) |
                                  static_cast<unsigned long long>(i))
                          .second;
        }
      }
    }
  } while (not_done);

  // Set REDUCE rules from first set
  for (unsigned int non_term = 0; non_term < Grammar::next_nonterm_id;
       non_term++) {
    for (const unsigned long long &element : first_set[non_term]) {
      unsigned int term = static_cast<unsigned int>(element >> 32);
      if (parse_table[non_term * cols + term].kind != EMPTY) {
        // throw ambiguous grammar error
      }
      parse_table[non_term * cols + term] = {
          REDUCE, static_cast<unsigned int>(element)};
    }
  }

  return parse_table;
}
} // namespace grammar
