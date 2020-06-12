cmake_minimum_required(VERSION 3.15.0)
project(TLC_PN VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

add_executable(findInjective_PN TLC_PN.cpp)

# Eigen

set(EIGEN_INCLUDE_DIR /usr/local/include CACHE PATH "Eigen3 include directory")
target_include_directories(findInjective_PN PUBLIC "${EIGEN_INCLUDE_DIR}")

# suitesparse

set(SUITESPARSE_INCLUDE_DIR /usr/local/include CACHE PATH "suitesparse include directory")
target_include_directories(findInjective_PN PUBLIC "${SUITESPARSE_INCLUDE_DIR}")

if (WIN32)
# ------------------------------------------------------------------
# Detect SuiteSparse libraries:
# If not found automatically, set SuiteSparse_DIR in CMake to the
# directory where SuiteSparse-config.cmake was installed.
# ------------------------------------------------------------------
	find_package(SuiteSparse CONFIG REQUIRED)
	target_link_libraries(findInjective_PN PRIVATE SuiteSparse::cholmod)
else()
	set(SUITESPARSE_LIB_DIR /usr/local/lib CACHE PATH "suitesparse library directory")
	target_link_libraries(findInjective_PN PUBLIC -L"${SUITESPARSE_LIB_DIR}" cholmod)
endif()


# OpenMP

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(findInjective_PN PUBLIC OpenMP::OpenMP_CXX)
endif()