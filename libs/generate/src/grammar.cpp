#include <iostream>
#include <string>
#include <unordered_map>
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
    rhs.insert(rhs.begin(), newTerminal("_START_"));
    rhs.push_back(newTerminal("_END_"));
  }

  this->rules.push_back({head, rhs});
}

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

LLGrammar::LLGrammar(Grammar grammar) {
  // Generate parsing table
}

} // namespace grammar
