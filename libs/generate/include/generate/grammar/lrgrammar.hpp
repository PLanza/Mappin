#ifndef __MAPPIN_GEN_LRGRAMMAR__
#define __MAPPIN_GEN_LRGRAMMAR__

#include "../grammar.hpp"
#include "parse/inet.hpp"

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
  ParseAction *action_table = nullptr;
  int *goto_table = nullptr;
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
  void printStackActions() override;
  void printParseTree(ParseTree *) override;

protected:
  LRParseTable *getParseTable() override;
  ParseTree *getParse(inet::Node *) override;
  ParseTree *getParse(NodeElement *, NodeElement *, NodeElement *) override;

private:
  void getStackActionClosure(uint32_t);
  LRParseTable *parse_table = nullptr;
  void traverseRules(inet::Node *, std::deque<ParseTree *> &);
  void traverseRules(NodeElement *, std::deque<ParseTree *> &, NodeElement *,
                     NodeElement *);
};
} // namespace grammar

#endif
