#ifndef __MAPPIN_GEN_LRGRAMMAR__
#define __MAPPIN_GEN_LRGRAMMAR__

#include "../grammar.hpp"
#include <utility>

namespace grammar {

struct Item {
  std::size_t rule_idx;
  std::size_t bullet_pos;

  bool operator==(Item const &other) const {
    return this->rule_idx == other.rule_idx &&
           this->bullet_pos == other.bullet_pos;
  }
};

class LRParseTable : public ParseTable {
public:
  LRParseTable(uint32_t, uint32_t, grammar_rules const &, uint32_t,
               const char *);
  ~LRParseTable();

  uint32_t states;

  ParseAction getAction(uint32_t, uint32_t) const override;
  int getGoto(uint32_t, uint32_t) const;

private:
  ParseAction *action_table;
  int *goto_table;
  const uint32_t terms;
  const uint32_t nonterms;
};

class LR0Grammar : public Grammar {
public:
  LR0Grammar(const char *);
  ~LR0Grammar();

  void finalize() override;
  void makeParseTable() override;
  void generateStackActions() override;
  void printParseTable() override;

protected:
  LRParseTable *getParseTable() override;

private:
  std::pair<std::vector<Token>, std::vector<uint32_t>> findTerminal(uint32_t,
                                                                    uint32_t);
  LRParseTable *parse_table;
};
} // namespace grammar

#endif
