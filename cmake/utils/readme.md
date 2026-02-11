# overview

## requirements
- vcpkg is installed


## fm_assertions
Simple assertions that every language should have:
  - `fm_assert_true` / `fm_assert_false` — check boolean conditions and fail configuration when the condition is (not) met.
  - `fm_assert_[_not]_cmd` — verify a CMake command/function is (not) present and fail on mismatch.
  - `fm_assert_[_not]_target` — assert a CMake target does (not) exists in the current scope.
  - `fm_assert_[_not]_empty` — assert a string value is (not) empty.
  - `fm_assert_program` — locate an external program (supports `find_program` args) and export `<PROG>_PROGRAM` to the parent scope (fails if not found).
  - `fm_assert_file` — assert that a particular file exists and is not a directory.


## fm_target_utilities
To help you set up targets and dependencies quicker:

  - `fm_glob` — generic recursive glob for specified file patterns/masks and append results to a variable. **Supports multiple paths.**
  - `fm_glob_cpp` — recursive glob for common C++ source/header/file-set extensions and append results to a variable. **Supports multiple paths.**
  - `fm_glob_cppm` — recursive glob for C++ module interface files (.cppm). **Supports multiple paths.**
  - `fm_add_resources` — add a POST_BUILD step to copy resource directories next to a target's binary. **Supports multiple paths.**
  - `fm_target_add_modules` — add C++ module files to a target with PUBLIC/PRIVATE visibility.
  - `fm_target_enable_sanitizers` — enable Address/Undefined sanitizers for specified targets/configurations.
  - `fm_target_enable_build_cache` — enable per-target build-cache integration ([installation](#optional))
  - `fm_add_clang_format_target` — create a "format" target that runs clang-format on specified source files

## fm_install_utilities
**Ever wanted to create a cmake installable package?**  
Now made easy, just build the `<main-component>_package` target and you are good to go:

  - `fm_pkg_target_add_modules` — add C++ module file-sets to a target (via `fm_target_add_modules`) and register it for installation.
  - `fm_pkg_target_find_package` — wrap `find_package` and record the dependency for generated package config files.
  - `fm_pkg_target_include_directories` — configure target include directories with appropriate install interfaces.
  - `fm_create_package` — finalize export sets, generate config/version files, and create the package target.

**This has naming implications**, subcomponents should be named `<main-component>_<subcomponent>` to be installable via `<main-component>::<subcomponent>`.

This will also create additional cmake targets but dont worry about it.

## fm_setup_utilities
this is more or less for me, very handy but no backwards compatibility guaranteed

  - `fm_set_compiler_specifics` — apply compiler-specific common flags (MSVC vs others).
  - `fm_set_newest_c_cpp_standard` (macro) — prefer the newest supported C/C++ standard and set related policy/flags.


## fm_tool_utilities
  - `fm_enable_build_cache` — enable BuildCache globally by setting C/C++ compiler launcher variables. ([installation](#optional))
  - `fm_find_clang_format` — locate clang-format executable and export path to parent scope

## toolchain.cmake
  - (toolchain configuration) — contains the project's recommended toolchain preset for CMake.
