#ifndef __MAPPIN_PARSE_PARSER__
#define __MAPPIN_PARSE_PARSER__

#include "inet.hpp"
#include <generate/grammar.hpp>
#include <memory>

namespace inet {

Node *createParserNetwork(std::vector<grammar::StackAction> *,
                          std::vector<grammar::Token>);

void getParses(Node *, std::unique_ptr<grammar::Grammar> &);
} // namespace inet
#endif
