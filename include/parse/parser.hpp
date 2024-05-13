#ifndef __MAPPIN_PARSE_PARSER__
#define __MAPPIN_PARSE_PARSER__

#include "../generate/grammar.hpp"
#include "inet.hpp"

namespace inet {

Node *createParserNetwork(std::vector<grammar::StackAction> *,
                          std::vector<grammar::Token>);

} // namespace inet
#endif
