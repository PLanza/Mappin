cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME
  generate
)

set(LIBRARY_NAME
  generate
)

set(SOURCE_DIR
    src
)
set(SOURCE
    ${SOURCE_DIR}/grammar_parser.cpp
    ${SOURCE_DIR}/grammar.cpp
    ${SOURCE_DIR}/llgrammar.cpp
    ${SOURCE_DIR}/lrgrammar.cpp
)

set(HEADERS_DIR
  include/${LIBRARY_NAME}
)
set(HEADERS
  ${HEADERS_DIR}/grammar_parser.hpp
  ${HEADERS_DIR}/grammar.hpp
  ${HEADERS_DIR}/grammar/llgrammar.hpp
  ${HEADERS_DIR}/grammar/lrgrammar.hpp
)

find_package(Boost)
include_directories(${Boost_INCLUDE_DIR})

set(LIBRARIES
  util
  parse
  ${Boost_LIBRARIES}
)

project(${PROJECT_NAME}
        DESCRIPTION "Mappin's parser generator"
        LANGUAGES CXX
)

# foreach(LIBRARY ${LIBRARIES})
#   add_subdirectory("../${LIBRARY}" "../../build/libs/${LIBRARY}")
# endforeach(LIBRARY)

add_library(${LIBRARY_NAME} STATIC ${HEADERS} ${SOURCE})

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})

# Imports grammar_parser.hpp as `#include "grammar_parser.hpp` when used internally
# and as `#include <generator/grammar_parser.hpp>` when used externally
target_include_directories(${LIBRARY_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
  $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

