#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Grammar.hpp"

namespace grammar {

TermOrNonTerm newTerminal(const char *name) {
  return {.kind = TERM, .term = {name}};
}

TermOrNonTerm newNonTerminal(const char *name) {
  return {.kind = NON_TERM, .non_term = {name}};
}

Grammar::Grammar() : rules({}), start_rule("") {}

// Inserts a new rule to the grammar
// If the non-terminal has already been defined, then
// append the new right hand sides to the existing rule
void Grammar::addRule(std::string name,
                      std::vector<std::vector<TermOrNonTerm>> rhs, bool start) {
  if (!this->rules.insert({name, rhs}).second)
    this->rules[name].insert(this->rules[name].end(), rhs.begin(), rhs.end());

  if (start)
    this->start_rule = name;
}

void Grammar::print() {
  for (const auto &[name, rules] : this->rules) {
    if (name == this->start_rule)
      std::cout << "$ ";
    std::cout << name << " := \n";

    for (auto rhs : rules) {
      std::cout << "  | ";
      for (auto term : rhs) {
        if (term.kind == TERM)
          std::cout << term.term.name << " ";
        else
          std::cout << term.non_term.name << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << std::endl;
}

} // namespace grammar
