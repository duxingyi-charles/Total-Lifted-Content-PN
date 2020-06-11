cmake_minimum_required(VERSION 3.15.0)
project(TLC_PN VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

add_executable(findInjective_PN TLC_PN.cpp)

set(SUITESPARSE_INCLUDE_DIR /usr/local/include CACHE PATH "suitesparse include directory")
set(SUITESPARSE_LIB_DIR /usr/local/lib CACHE PATH "suitesparse library directory")

target_include_directories(findInjective_PN PUBLIC /usr/local/include)
target_include_directories(findInjective_PN PUBLIC "${SUITESPARSE_INCLUDE_DIR}")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(findInjective_PN PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(findInjective_PN PUBLIC -L/usr/local/lib -L"${SUITESPARSE_LIB_DIR}" cholmod)
