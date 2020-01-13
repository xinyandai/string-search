cmake_minimum_required(VERSION 3.10)
project(string_search)

SET( CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Ofast -lrt -lm -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -mtune=generic -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
find_package( PkgConfig )
find_package(OpenMP)
find_package(Boost COMPONENTS)
include_directories(${Boost_INCLUDE_DIRS})

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES} )


link_libraries(-lpthread)
link_libraries(-lcnpy)
link_libraries(-lfaiss)
link_libraries(-lz)

add_executable(string_search main.cpp)
add_executable(gt gt.cc)
add_executable(join join.cc)
add_executable(bf_threshold bf_threshold.cc)
add_executable(bf_topk bf_topk.cc)
add_executable(embed_threshold embed_threshold.cc)
add_executable(embed_topk embed_topk.cc)
add_executable(allpair_distance allpair_distance.cc)
target_link_libraries(join ${BLAS_LIBRARIES})
target_link_libraries(string_search ${Boost_LIBRARIES})