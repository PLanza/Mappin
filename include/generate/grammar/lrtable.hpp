#ifndef __MAPPIN_GEN_LRTABLE__
#define __MAPPIN_GEN_LRTABLE__

#include "../grammar.hpp"

namespace grammar {

struct Config {
  std::size_t rule_idx;
  std::size_t bullet;
  uint32_t lookahead;

  bool operator==(Config const &other) const {
    return this->rule_idx == other.rule_idx && this->bullet == other.bullet &&
           this->lookahead == other.lookahead;
  }
  bool operator<(Config const &other) const {
    return this->rule_idx < other.rule_idx ||
           (this->rule_idx == other.rule_idx && this->bullet < other.bullet);
  }
  Config(std::size_t rule_idx, std::size_t bullet, uint32_t lookahead)
      : rule_idx(rule_idx), bullet(bullet), lookahead(lookahead) {}
};

typedef std::vector<Config> LRState;

class LRParseTable : public ParseTable {
public:
  LRParseTable(Grammar *);
  ~LRParseTable();

  uint32_t state_count;

  ParseAction getAction(uint32_t, uint32_t) const override;
  int getGoto(uint32_t, uint32_t) const;
  std::unordered_set<uint32_t> getFollowSet(Token);

  void printConfig(Config const &);
  void printStates();

private:
  Grammar *grammar;
  ParseAction *action_table = nullptr;
  int *goto_table = nullptr;
  std::vector<LRState> states;

  std::vector<std::unordered_map<Token, uint32_t, boost::hash<Token>>>
      trans_table;

  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> first_sets;
  std::unordered_map<Token, std::unordered_set<uint32_t>, boost::hash<Token>>
      follow_sets;

  void getStateClosure(LRState &unfinished);
  LRState generateNextState(uint32_t, Token);
  void generateStates();
  void fillTables();
  int findState(const LRState &state);
  std::unordered_set<uint32_t> getFollowSet(uint32_t, uint32_t);
  std::unordered_set<uint32_t> getFirstSet(Token);
};

} // namespace grammar
#endif
