cmake_minimum_required(VERSION 3.18)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

#Create compile_commands.json
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PROJECT_NAME
  Mappin
)

set(SOURCE_DIR
    src
)
set(SOURCE
    ${SOURCE_DIR}/main.cpp
)

set(HEADERS_DIR
  include
)
set(_HEADERS
)

set(LIBRARIES_DIR
  libs
)
set(LIBRARIES
  MappinUtil
  MappinGen
  MappinParse
)

project(${PROJECT_NAME} 
        DESCRIPTION "Massively Parallel Parsing with Interaction Nets"
        LANGUAGES CXX CUDA
)

# Create an executable using source files specified as SOURCE and with header
# files in specified in HEADERS
add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCE})

foreach(LIBRARY ${LIBRARIES})
  add_subdirectory("${LIBRARIES_DIR}/${LIBRARY}")
endforeach(LIBRARY)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
