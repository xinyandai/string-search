cmake_minimum_required(VERSION 3.10)
project(string_search)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( PkgConfig )
find_package(OpenMP)
find_package(Boost COMPONENTS)
include_directories(${Boost_INCLUDE_DIRS})

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

link_libraries(-lpthread)
link_libraries(-lcnpy)
link_libraries(-lz)

add_executable(string_search main.cpp)
target_link_libraries(string_search ${Boost_LIBRARIES})