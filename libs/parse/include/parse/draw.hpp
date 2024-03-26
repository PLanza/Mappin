#ifndef __MAPPIN_PARSE_DRAW__
#define __MAPPIN_PARSE_DRAW__

#include <string>
#include <generate/grammar.hpp>
#include <memory>

namespace inet {

void drawNetwork(std::unique_ptr<grammar::Grammar> &);

}

#endif