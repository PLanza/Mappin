#include <unordered_map>

#include "Grammar.hpp"

namespace grammar {

TermOrNonTerm newTerminal(const char *name) {
  return {.kind = TERM, .term = {name}};
}

TermOrNonTerm newNonTerminal(const char *name) {
  return {.kind = NON_TERM, .non_term = {name}};
}

} // namespace grammar

grammar::Grammar::Grammar() : rules({}), start_rule({""}) {}
