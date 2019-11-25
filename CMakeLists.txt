cmake_minimum_required(VERSION 3.9.5)
project(admm_end)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_FLAGS -fopenmp)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories (include)

aux_source_directory (src SRC)
aux_source_directory (./src/sub_problem SRC_SUB_PROBLEM)
aux_source_directory (./src/utils SRC_UTILS)

add_executable(admm_end ${SRC} ${SRC_SUB_PROBLEM} ${SRC_UTILS} src/admm_comm.cpp include/admm_comm.h include/admm.cpp include/admm.h src/sub_problem/l2r_lr_fun.cpp include/sub_problem/l2r_lr_fun.h include/sub_problem/optimizer.h src/sub_problem/gd.cpp include/sub_problem/gd.h)