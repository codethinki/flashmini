# Copyright (c) 2026 Lukas Thomann
# Licensed under the MIT License

include(fm_assertions)

#[[.rst:
.. command:: fm_find_program

   .. code-block:: cmake

      fm_find_program(<out_var> <prog> [args...])

   Locates an external program and exports its path to the parent scope.

   :param OUT_VAR variable to export program path to
   :param prog: Name of the program to find
   :type prog: string
   :param args: Additional arguments to pass to find_program (e.g., PATHS, HINTS)
   :type args: optional arguments

   :post: <OUT_VAR> variable is set in PARENT_SCOPE with the full path to the program, or configuration terminates with FATAL_ERROR if not found
#]]
function(fm_find_program OUT_VAR prog)
    
    find_program(${OUT_VAR} "${prog}" ${ARGN})
    
    fm_assert_true(${OUT_VAR} REASON "Program '${prog}' not found")
    
    set(${OUT_VAR} "${${OUT_VAR}}" PARENT_SCOPE)
endfunction()

#[[.rst:
.. command:: fm_enable_build_cache

   .. code-block:: cmake

      fm_enable_build_cache()

   Enables BuildCache globally for all targets by setting compiler launcher variables.

   :pre: buildcache program is found in PATH
   :post: CMAKE_C_COMPILER_LAUNCHER and CMAKE_CXX_COMPILER_LAUNCHER are set to buildcache in PARENT_SCOPE

   .. note::
      This affects ALL targets in the current scope and below.
      For per-target control, use ``fm_target_enable_build_cache()`` instead.

   .. warning::
      BuildCache must be installed and available in PATH.
      The function will fail with FATAL_ERROR if buildcache is not found.

   .. seealso::
      Use ``fm_target_enable_build_cache()`` from fm_target_utilities for per-target control.

#]]
function(fm_enable_build_cache)
    fm_find_program(BUILDCACHE_EXECUTABLE buildcache)

    message(STATUS "Enabling buildcache globally: ${BUILDCACHE_EXECUTABLE}")

    set(CMAKE_C_COMPILER_LAUNCHER "${BUILDCACHE_EXECUTABLE}" PARENT_SCOPE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${BUILDCACHE_EXECUTABLE}" PARENT_SCOPE)
endfunction()

#[[.rst:
.. command:: fm_find_clang_format

   .. code-block:: cmake

      fm_find_clang_format()

   Locates the clang-format executable and exports its path to the parent scope.

   :post: CLANG_FORMAT_EXECUTABLE is set in PARENT_SCOPE with the full path to clang-format, or configuration terminates with FATAL_ERROR if not found

   .. note::
      The clang-format executable must be available in PATH.

   .. warning::
      This function will fail with FATAL_ERROR if clang-format is not found.
      Ensure clang-format is installed and available in your system PATH.

   .. seealso::
      Use ``fm_add_clang_format_target()`` from fm_target_utilities to create a format target.

#]]
function(fm_find_clang_format)
   fm_find_program(CLANG_FORMAT_EXECUTABLE clang-format)
   
   message(STATUS "Found external clang-format: ${CLANG_FORMAT_EXECUTABLE}")

   set(CLANG_FORMAT_EXECUTABLE ${CLANG_FORMAT_EXECUTABLE} PARENT_SCOPE)
endfunction()