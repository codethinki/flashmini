# Copyright (c) 2026 Lukas Thomann
# Licensed under the MIT License

include(fm_assertions)

#[[.rst:
.. command:: fm_find_program

   .. code-block:: cmake

      fm_find_program(<out_var> <prog> [OPTIONAL] [args...])

   Locates an external program and exports its path to the parent scope.
   If OPTIONAL is specified, does not error if program is not found.

   :param OUT_VAR variable to export program path to
   :param prog: Name of the program to find
   :param OPTIONAL: If specified, do not raise FATAL_ERROR if program is not found
   :param args: Additional arguments to pass to find_program

   :post: <OUT_VAR> variable is set in PARENT_SCOPE with the full path to the program, or configuration terminates with FATAL_ERROR if not found and not OPTIONAL

#]]
function(fm_find_program OUT_VAR prog)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "OPTIONAL" "" "")

    find_program(${OUT_VAR} "${prog}" ${ARG_UNPARSED_ARGUMENTS})
    
    if(${OUT_VAR})
        message(STATUS "${prog} found: ${${OUT_VAR}}")
        message(VERBOSE "${prog} location: ${${OUT_VAR}}")
        set(${OUT_VAR} "${${OUT_VAR}}" PARENT_SCOPE)
        return()
    endif()

    if(ARG_OPTIONAL)
        message(STATUS "${prog} not found")
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    fm_assert_true(${OUT_VAR} REASON "Program '${prog}' not found")
endfunction()

#[[.rst:
.. command:: fm_enable_build_cache

   .. code-block:: cmake

      fm_enable_build_cache([OPTIONAL])

   Enables BuildCache globally for all targets by setting compiler launcher variables.

   :param OPTIONAL: If specified, do not raise FATAL_ERROR if buildcache is not found
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
    cmake_parse_arguments(PARSE_ARGV 0 ARG "OPTIONAL" "" "")

    if (ARG_OPTIONAL)
        fm_find_program(BUILDCACHE_EXECUTABLE buildcache OPTIONAL)
    else()
        fm_find_program(BUILDCACHE_EXECUTABLE buildcache)
    endif()
    
    if(BUILDCACHE_EXECUTABLE)
        message(STATUS "BuildCache globally enabled")
    else()
        message(STATUS "Couldn't enable BuildCache globally")
        return()
    endif()

    set(CMAKE_C_COMPILER_LAUNCHER "${BUILDCACHE_EXECUTABLE}" PARENT_SCOPE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${BUILDCACHE_EXECUTABLE}" PARENT_SCOPE)
endfunction()

#[[.rst:
.. command:: fm_find_clang_format

   .. code-block:: cmake

      fm_find_clang_format([OPTIONAL])

   Locates a required clang-format executable and exports its path to the parent scope.
   If OPTIONAL is specified, does not error if clang-format is not found.

   :post: CLANG_FORMAT_EXECUTABLE is set in PARENT_SCOPE with the full path to clang-format, or configuration terminates with FATAL_ERROR if not found and not OPTIONAL

   .. seealso::
      Use ``fm_add_clang_format_target()`` from fm_target_utilities to create a format target.

#]]
function(fm_find_clang_format)
   cmake_parse_arguments(PARSE_ARGV 0 ARG "OPTIONAL" "" "")

   if(ARG_OPTIONAL)
      fm_find_program(CLANG_FORMAT_EXECUTABLE clang-format OPTIONAL)
   else()
      fm_find_program(CLANG_FORMAT_EXECUTABLE clang-format)
   endif()
   
   set(CLANG_FORMAT_EXECUTABLE ${CLANG_FORMAT_EXECUTABLE} PARENT_SCOPE)
endfunction()

#[[.rst:
.. command:: fm_find_uncrustify

   .. code-block:: cmake

      fm_find_uncrustify([OPTIONAL])

   Locates a required uncrustify executable and exports its path to the parent scope.
   If OPTIONAL is specified, does not error if uncrustify is not found.

   :post: UNCRUSTIFY_EXECUTABLE is set in PARENT_SCOPE with the full path to uncrustify, or configuration terminates with FATAL_ERROR if not found and not OPTIONAL

   .. seealso::
      Use ``fm_add_uncrustify_target()`` from fm_target_utilities to create a format target.

#]]
function(fm_find_uncrustify)
   cmake_parse_arguments(PARSE_ARGV 0 ARG "OPTIONAL" "" "")

   if(ARG_OPTIONAL)
      fm_find_program(UNCRUSTIFY_EXECUTABLE uncrustify OPTIONAL)
   else()
      fm_find_program(UNCRUSTIFY_EXECUTABLE uncrustify)
   endif()
   
   set(UNCRUSTIFY_EXECUTABLE ${UNCRUSTIFY_EXECUTABLE} PARENT_SCOPE)
endfunction()