#ifndef __MAPPIN_GEN_GRAMMAR__
#define __MAPPIN_GEN_GRAMMAR__

#include <string>
#include <unordered_map>
#include <vector>

namespace grammar {

struct Terminal {
  const char *name;
};

struct NonTerminal {
  const char *name;
};

enum TermKind { TERM, NON_TERM };

struct TermOrNonTerm {
  TermKind kind;
  union {
    Terminal term;
    NonTerminal non_term;
  };
};

TermOrNonTerm newTerminal(const char *name);
TermOrNonTerm newNonTerminal(const char *name);

class Grammar {
public:
  Grammar();
  void addRule(std::string, std::vector<std::vector<TermOrNonTerm>>, bool);
  void print();

private:
  std::unordered_map<std::string, std::vector<std::vector<TermOrNonTerm>>>
      rules;
  std::string start_rule;
};

} // namespace grammar

#endif
