#ifndef __MAPPIN_GEN_LRGRAMMAR__
#define __MAPPIN_GEN_LRGRAMMAR__

#include "../grammar.hpp"
#include "lrtable.hpp"
#include "parse/inet.hpp"

namespace grammar {

class LRGrammar : public Grammar {
public:
  LRGrammar(const char *);
  ~LRGrammar();

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
  void clearRepeatedActions(uint32_t);
  LRParseTable *parse_table = nullptr;
  void traverseRules(inet::Node *, std::deque<ParseTree *> &);
  void traverseRules(NodeElement *, std::deque<ParseTree *> &, NodeElement *,
                     NodeElement *);
};
} // namespace grammar

#endif
