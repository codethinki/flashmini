# Copyright (c) 2026 Lukas Thomann
# Licensed under the MIT License

cmake_minimum_required(VERSION 4.0.0)

message(STATUS "---- fm_cmake toolchain ----")

#append cmake dir to module path
set(FM_CMAKE_LIBRARY_DIR ${CMAKE_CURRENT_LIST_DIR})

list(APPEND CMAKE_MODULE_PATH "${FM_CMAKE_LIBRARY_DIR}")
message(STATUS "appended cmake/utils/ to cmake module path")

list(APPEND CMAKE_MODULE_PATH "${FM_CMAKE_LIBRARY_DIR}/../")
message(STATUS "appended (${FM_CMAKE_LIBRARY_DIR}/../) cmake/ to cmake module path")



include(fm_assertions)
include(fm_tool_utilities)

#delegate to vcpkg

fm_assert_program(vcpkg REASON "fm needs vcpkg" HINTS "$ENV{VCPKG_ROOT}")

set(VCPKG_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

fm_assert_file(${VCPKG_TOOLCHAIN_FILE} REASON "fm requires a vcpkg toolchain but it wasn't found at ${VCPKG_TOOLCHAIN_FILE}")

message(STATUS "using vcpkg toolchain file: ${VCPKG_TOOLCHAIN_FILE}")    

message(STATUS "---- fl toolchain handing off to vcpkg ----")
include("${VCPKG_TOOLCHAIN_FILE}")
