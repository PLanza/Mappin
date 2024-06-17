cmake_minimum_required(VERSION 3.18)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

#Create compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")

# Required for coding remotely on a computer without a GPU
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

set(PROJECT_NAME
  mappin
)

include(FetchContent)
FetchContent_Declare(
  drag
  GIT_REPOSITORY https://github.com/bigno78/drag.git
  GIT_TAG master
)
FetchContent_MakeAvailable(drag)

set(SOURCE
    src/main.cpp
    src/generate/grammar_parser.cpp
    src/generate/grammar.cpp
    src/generate/llgrammar.cpp
    src/generate/lrgrammar.cpp
    src/generate/lrtable.cpp
    src/generate/error.cpp
    src/gpu/actions.cu
    src/gpu/network.cpp
    src/gpu/run.cu
    src/gpu/inet.cu
    src/gpu/timing.cu
    src/gpu/draw.cpp
)

set(HEADERS
  include/generate/util.hpp
  include/generate/grammar_parser.hpp
  include/generate/grammar.hpp
  include/generate/grammar/llgrammar.hpp
  include/generate/grammar/lrgrammar.hpp
  include/generate/grammar/lrtable.hpp
  include/gpu/draw.hpp
  include/gpu/actions.hpp
  include/gpu/inet.hpp
  include/gpu/run.hpp
  include/gpu/kernel.cuh
  include/gpu/network.hpp
  include/gpu/queue.cuh
  include/gpu/timing.cuh
)

set(BOOST_ROOT /usr/lib/x86_64-linux-gnu/cmake)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

set(LIBRARIES
  drag
  ${Boost_LIBRARIES}
)

project(${PROJECT_NAME} 
        DESCRIPTION "Massively Parallel Parsing with Interaction Nets"
        LANGUAGES CXX CUDA
)

# Create an executable using source files specified as SOURCE and with header
# files in specified in HEADERS
add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCE})

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
