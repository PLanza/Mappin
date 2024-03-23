cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME
  parse
)

set(LIBRARY_NAME
  parse
)

set(SOURCE_DIR
    src
)
set(SOURCE
    ${SOURCE_DIR}/inet.cpp
    ${SOURCE_DIR}/nodes.cpp
    ${SOURCE_DIR}/parser.cpp
    ${SOURCE_DIR}/draw.cpp
)

set(HEADERS_DIR
  include/${LIBRARY_NAME}
)
set(HEADERS
  ${HEADERS_DIR}/inet.hpp
  ${HEADERS_DIR}/nodes.hpp
  ${HEADERS_DIR}/parser.hpp
  ${HEADERS_DIR}/draw.hpp
)

set(LIBRARIES
  generate
  drag
)

project(${PROJECT_NAME}
        DESCRIPTION "Mappin's parser"
        LANGUAGES CXX
)

add_library(${LIBRARY_NAME} STATIC ${HEADERS} ${SOURCE})

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})

target_include_directories(${LIBRARY_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
  $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
