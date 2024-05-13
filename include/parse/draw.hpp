#ifndef __MAPPIN_PARSE_DRAW__
#define __MAPPIN_PARSE_DRAW__

#include "../generate/grammar.hpp"
#include <memory>

namespace inet {

void drawNetwork(std::unique_ptr<grammar::Grammar> &, bool);

}

#endif
