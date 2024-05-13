#ifndef __MAPPIN_GEN_LRTABLE__
#define __MAPPIN_GEN_LRTABLE__

#include "../grammar.hpp"

namespace grammar {

enum LaneTraceFlags {
  NONE = 0,
  COMPLETE = 1,
  IN_LANE = 1 << 1,
};

struct Config {
  std::size_t rule_idx;
  std::size_t bullet;

  LaneTraceFlags flags = NONE;
  std::unordered_set<Token, boost::hash<Token>> context;

  bool operator==(Config const &other) const {
    return this->rule_idx == other.rule_idx && this->bullet == other.bullet;
  }
  bool operator<(Config const &other) const {
    return this->rule_idx < other.rule_idx ||
           (this->rule_idx == other.rule_idx && this->bullet < other.bullet);
  }
  Config(std::size_t rule_idx, std::size_t bullet)
      : rule_idx(rule_idx), bullet(bullet), flags(NONE), context({}) {}
};

typedef std::vector<Config> LRState;

struct LTStackToken {
  size_t state;
  uint32_t config;
  bool marker;
  bool zero = false;
  bool operator==(LTStackToken const &other) const {
    return (this->marker == other.marker) && (this->zero == other.zero) &&
           this->state == other.state && this->config == other.config;
  }
};

class LRParseTable : public ParseTable {
public:
  LRParseTable(Grammar *);
  ~LRParseTable();

  uint32_t state_count;

  ParseAction getAction(uint32_t, uint32_t) const override;
  int getGoto(uint32_t, uint32_t) const;

  std::unordered_set<Token, boost::hash<Token>> getFollowSet(Token);

  void printConfig(Config const &);
  void printStates();

private:
  Grammar *grammar;
  ParseAction *action_table = nullptr;
  int *goto_table = nullptr;
  std::vector<LRState> states;

  std::vector<int *> trans_table;
  std::vector<int> end_states;
  std::vector<std::vector<size_t>> reductions;

  // NT id -> {T id}
  std::unordered_map<uint32_t, std::unordered_set<Token, boost::hash<Token>>>
      token_heads;

  std::unordered_map<Token, std::unordered_set<Token, boost::hash<Token>>,
                     boost::hash<Token>>
      follow_sets;

  void generateStates();
  uint32_t laneTracing(size_t, size_t);
  void laneTrace(size_t, uint32_t);
  std::vector<LTStackToken> getOriginators(size_t, Config);
  std::unordered_set<Token, boost::hash<Token>> getTokenHeads(uint32_t);
  void addContexts(std::unordered_set<Token, boost::hash<Token>> &,
                   LTStackToken, std::vector<LTStackToken> const &);
};

} // namespace grammar
#endif
