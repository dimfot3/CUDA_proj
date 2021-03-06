cmake_minimum_required(VERSION 3.12)
set(CMAKE_C_COMPILER nvcc)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
project(cuda_proj LANGUAGES CUDA C CXX)

add_library(utils src/utils.cu)
target_include_directories(utils PUBLIC include)

add_library(seq src/sequential.cu)
target_link_libraries(seq utils)
target_include_directories(seq PUBLIC include)

add_library(cuda_11 src/cuda_prog_11.cu)
target_link_libraries(cuda_11 utils)
target_include_directories(cuda_11 PUBLIC include)

add_library(cuda_m1 src/cuda_prog_m1.cu)
target_link_libraries(cuda_m1 utils)
target_include_directories(cuda_m1 PUBLIC include)

add_library(cuda_share src/cuda_prog_m1_share.cu)
target_link_libraries(cuda_share utils)
target_include_directories(cuda_share PUBLIC include)

add_executable(main_program src/cprogram.cu)
target_link_libraries(main_program utils seq cuda_11 cuda_m1 cuda_share)  
target_include_directories(main_program PUBLIC include)

file(GLOB SCRIPTS
  "scripts/*.sh"
)
file(COPY ${SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR})

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif(BUILD_TESTING)
