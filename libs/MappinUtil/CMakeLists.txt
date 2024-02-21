cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME
  MappinUtil
)

set(LIBRARY_NAME
  MappinUtil
)

set(LIBRARY_SOURCE_DIR
    src
)
set(LIBRARY_SOURCE
    ${LIBRARY_SOURCE_DIR}/MappinUtil.cpp
    ${LIBRARY_SOURCE_DIR}/error.cpp
)

set(LIBRARY_HEADERS_DIR
  include/${LIBRARY_NAME}
)
set(LIBRARY_HEADERS
  ${LIBRARY_HEADERS_DIR}/MappinUtil.hpp
)

project(${PROJECT_NAME}
        DESCRIPTION "Mappin's utilities"
        LANGUAGES CXX
)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_HEADERS} ${LIBRARY_SOURCE})

target_include_directories(${LIBRARY_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
  $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
