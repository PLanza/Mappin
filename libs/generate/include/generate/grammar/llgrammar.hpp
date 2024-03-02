#ifndef __MAPPIN_GEN_LLGRAMMAR__
#define __MAPPIN_GEN_LLGRAMMAR__

#include "../grammar.hpp"
#include <utility>

namespace grammar {

class LLParseTable : public ParseTable {
public:
  LLParseTable(uint32_t, uint32_t, grammar_rules &, const char *);
  ~LLParseTable();

  ParseAction getAction(uint32_t, uint32_t) override;

private:
  ParseAction *table;
  uint32_t rows;
  uint32_t cols;
};

class LLGrammar : public Grammar {
public:
  LLGrammar(const char *);
  void makeParseTable() override;
  void generateStackActions() override;
  void printParseTable() override;

private:
  std::pair<std::vector<Token>, std::vector<uint32_t>> findTerminal(uint32_t,
                                                                    uint32_t);
};
} // namespace grammar
#endif
