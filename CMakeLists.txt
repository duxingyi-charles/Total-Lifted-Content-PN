cmake_minimum_required(VERSION 3.15.0)

project(Lifted VERSION 0.1.0 LANGUAGES CXX)

add_executable(lifted_exe lifted_eigen_test.cpp)

set(CMAKE_CXX_STANDARD 11)

set(SUITESPARSE_INCLUDE_DIR "/usr/local/include" CACHE STRING "Suitesparse include directory")
set(SUITESPARSE_LIB_DIR "/usr/local/lib" CACHE STRING "Suitesparse library directory")

target_include_directories(lifted_exe PUBLIC /usr/local/include)
target_include_directories(lifted_exe PUBLIC "${SUITESPARSE_INCLUDE_DIR}")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lifted_exe PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(lifted_exe PUBLIC -L/usr/local/lib -L"${SUITESPARSE_LIB_DIR}" cholmod)
