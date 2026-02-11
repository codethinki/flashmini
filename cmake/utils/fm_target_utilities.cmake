# Copyright (c) 2026 Lukas Thomann
# Licensed under the MIT License

include(fm_assertions)

#[[.rst:
.. command:: fm_glob

   .. code-block:: cmake

      fm_glob(<out_var> <sub_paths...> [PATTERNS <patterns...>])

   Recursively globs for files matching specified patterns in one or more subdirectories.

   :param out_var: Variable to append found files to
   :type out_var: string
   :param sub_paths: One or more subdirectory paths to search within
   :type sub_paths: string or list of strings
   :param PATTERNS: List of file patterns to match (e.g., "*.cpp", "*.hpp")
   :type PATTERNS: list of strings

   :post: out_var contains the list of found files appended to existing content

   .. note::
      Uses GLOB_RECURSE with CONFIGURE_DEPENDS to ensure CMake re-runs if files change.
      Results are appended to out_var, preserving any existing content.
      When multiple paths are provided, all paths are searched and results are combined.

#]]
function(fm_glob OUT_VAR)
    set(multiValueArgs PATTERNS)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "${multiValueArgs}")

    set(SUB_PATHS ${ARG_UNPARSED_ARGUMENTS})

    set(GLOB_PATTERNS "")
    foreach(SUB_PATH IN LISTS SUB_PATHS)
        foreach(PATTERN IN LISTS ARG_PATTERNS)
            list(APPEND GLOB_PATTERNS "${SUB_PATH}/${PATTERN}")
        endforeach()
    endforeach()

    if(GLOB_PATTERNS)
        file(GLOB_RECURSE FOUND_FILES
            CONFIGURE_DEPENDS
            ${GLOB_PATTERNS}
        )
        set(${OUT_VAR} ${${OUT_VAR}} ${FOUND_FILES} PARENT_SCOPE)
    endif()
endfunction()

#[[.rst:
.. command:: fm_glob_cpp

   .. code-block:: cmake

      fm_glob_cpp(<out_var> <sub_paths...>)

   Recursively globs for common C++ source and header files in one or more subdirectories.

   :param out_var: Variable to append found files to
   :type out_var: string
   :param sub_paths: One or more subdirectory paths to search within
   :type sub_paths: string or list of strings

   :post: out_var contains the list of found files appended to existing content

   .. note::
      Searches for files with extensions: .cpp, .hpp, .inl
      When multiple paths are provided, all paths are searched and results are combined.

#]]
function(fm_glob_cpp OUT_VAR)
    fm_glob(${OUT_VAR} ${ARGN} PATTERNS "*.cpp" "*.hpp" "*.inl")
    set(${OUT_VAR} ${${OUT_VAR}} PARENT_SCOPE)
endfunction()


#[[.rst:
.. command:: fm_glob_cppm

   .. code-block:: cmake

      fm_glob_cppm(<out_var> <sub_paths...>)

   Recursively globs for C++ module interface files in one or more subdirectories.

   :param out_var: Variable to append found files to
   :type out_var: string
   :param sub_paths: One or more subdirectory paths to search within
   :type sub_paths: string or list of strings

   :post: out_var contains the list of found files appended to existing content

   .. note::
      Searches for files with extension: .cppm (C++ module interface files)
      When multiple paths are provided, all paths are searched and results are combined.

#]]
function(fm_glob_cppm OUT_VAR)
    fm_glob(${OUT_VAR} ${ARGN} PATTERNS "*.cppm")
    set(${OUT_VAR} ${${OUT_VAR}} PARENT_SCOPE)
endfunction()



#[[.rst:
.. command:: fm_add_resources

   .. code-block:: cmake

      fm_add_resources(<target_name> <resource_paths...>)

   Adds post-build commands to copy one or more resource directories to the target's output directory.

   :param target_name: Name of the target to add resources to
   :type target_name: string
   :param resource_paths: One or more paths to resource directories to copy
   :type resource_paths: string or list of strings

   :pre: target_name exists
   :post: Resources are copied to the target directory after build, preserving directory structure

   .. note::
      Each resource directory will be copied to ``$<TARGET_FILE_DIR:target>/<resource_path>``.
      Directory structure is preserved relative to the original resource_path.
      When multiple paths are provided, each directory is copied separately.

#]]
function(fm_add_resources TARGET_NAME)
    fm_assert_target("${TARGET_NAME}")

    set(RESOURCE_PATHS ${ARGN})

    foreach(RESOURCE_PATH IN LISTS RESOURCE_PATHS)
        # Get the absolute path to the source resources
        get_filename_component(ABS_RESOURCE_PATH ${RESOURCE_PATH} ABSOLUTE)

        add_custom_command(
            TARGET ${TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${ABS_RESOURCE_PATH}"
                    # Use the original RESOURCE_PATH to preserve the structure
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>/${RESOURCE_PATH}"
            COMMENT "Copying resource directory: ${RESOURCE_PATH}"
            VERBATIM
        )
    endforeach()
endfunction()



#[[.rst:
.. command:: fm_target_enable_sanitizers

   .. code-block:: cmake

      fm_target_enable_sanitizers(<target> [CONFIGS <configs...>] [SANITIZERS <sanitizers...>])

   Enables sanitizers for the specified target and build configurations.

   :param target: Name of the target to enable sanitizers for
   :type target: string
   :param CONFIGS: List of build configurations to enable sanitizers for (e.g., Debug, Release)
   :type CONFIGS: list of strings
   :param SANITIZERS: List of sanitizers to enable
   :type SANITIZERS: list of strings

   :pre: target exists
   :pre: SANITIZERS list is not empty
   :pre: CONFIGS list is not empty
   :post: Sanitizers are enabled for the target in specified configurations

   .. note::
      **Supported sanitizers:**

      - ``address`` - AddressSanitizer (ASan): detects memory errors
      - ``undefined`` - UndefinedBehaviorSanitizer (UBSan): detects undefined behavior

   .. warning::
      **MSVC limitations:**

      - UndefinedBehaviorSanitizer (``undefined``) is NOT supported on MSVC and will be skipped with a warning
      - Only AddressSanitizer (``address``) is available on MSVC

   .. warning::
      Unsupported sanitizer names will cause a FATAL_ERROR.

#]]
function(fm_target_enable_sanitizers target)
    # 1. Validate Target
    fm_assert_target("${target}")

    # 2. Parse Arguments
    set(multiValueArgs CONFIGS SANITIZERS)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "${multiValueArgs}")

    # 3. Validate Parsing & Requirements via Assertions
    fm_assert_empty("${ARG_UNPARSED_ARGUMENTS}")

    fm_assert_not_empty("${ARG_SANITIZERS}")
    fm_assert_not_empty("${ARG_CONFIGS}")

    # 4. Prepare Generator Expressions
    string(REPLACE ";" "," CONFIG_CSV "${ARG_CONFIGS}")
    set(GENEX_CONDITION "$<CONFIG:${CONFIG_CSV}>")

    # 5. Iterate and Apply
    foreach(sanitizer IN LISTS ARG_SANITIZERS)
        if(sanitizer STREQUAL "address")
            target_compile_options(${target} PRIVATE
                "$<${GENEX_CONDITION}:$<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fsanitize=address>>"
            )
            target_link_options(${target} PRIVATE
                "$<${GENEX_CONDITION}:$<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fsanitize=address>>"
            )

        elseif(sanitizer STREQUAL "undefined")
            if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                message(WARNING "fm_target_enable_sanitizers: MSVC does not support UBSan. Skipping for target '${target}'.")
            else()
                target_compile_options(${target} PRIVATE "$<${GENEX_CONDITION}:-fsanitize=undefined>")
                target_link_options(${target} PRIVATE "$<${GENEX_CONDITION}:-fsanitize=undefined>")
            endif()

        else()
            message(FATAL_ERROR "ERROR fm_target_enable_sanitizers: Unsupported sanitizer '${sanitizer}' [args: ${ARGV}]")
        endif()
    endforeach()
endfunction()



#[[.rst:
.. command:: fm_target_enable_build_cache

   .. code-block:: cmake

      fm_target_enable_build_cache(<target>)

   Enables build caching for the specified target using BuildCache.

   :param target: Name of the target to enable build caching for
   :type target: string

   :pre: target exists
   :pre: buildcache program is found in PATH
   :post: C and C++ compiler launchers are set to buildcache for the target

   .. note::
      BuildCache must be installed and available in PATH.
      See the installation guide for setup instructions.

   .. seealso::
      Use ``fm_enable_build_cache()`` from fm_tool_utilities to enable globally.

#]]
function(fm_target_enable_build_cache target)
    fm_assert_target("${target}")

    fm_assert_program(buildcache)

    set_target_properties(${target} PROPERTIES
        C_COMPILER_LAUNCHER "${BUILDCACHE_PROGRAM}"
        CXX_COMPILER_LAUNCHER "${BUILDCACHE_PROGRAM}"
    )
endfunction()



#[[.rst:
.. command:: fm_target_add_modules

   .. code-block:: cmake

      fm_target_add_modules(<target_name> [PUBLIC <files...>] [PRIVATE <files...>])

   Adds C++ module files to a target with specified visibility.

   :param target_name: Name of the target to add modules to
   :type target_name: string
   :param PUBLIC: List of public module files (.cppm)
   :type PUBLIC: list of file paths
   :param PRIVATE: List of private module files (.cppm)
   :type PRIVATE: list of file paths

   :pre: target_name exists
   :pre: target_name is NOT an INTERFACE library (C++ modules not supported)
   :pre: At least one of PUBLIC or PRIVATE arguments is provided
   :post: Module files are added to target with appropriate file sets and CXX_SCAN_FOR_MODULES is enabled

   .. note::
      **Public modules:**

      - Added to ``CXX_MODULES`` file set
      - Visible to consumers of the target

      **Private modules:**

      - Added to ``<target_name>_private_modules`` file set
      - Only visible within the target itself

   .. warning::
      INTERFACE libraries do NOT support C++ modules and will cause a FATAL_ERROR.

#]]
function(fm_target_add_modules TARGET_NAME)
    # 1. Basic existence check
    fm_assert_target("${TARGET_NAME}")

    # 2. Interface check (C++ Modules cannot be added to INTERFACE libraries)
    get_target_property(TGT_TYPE ${TARGET_NAME} TYPE)
    fm_assert_false(
        "${TGT_TYPE}" STREQUAL "INTERFACE_LIBRARY"
        REASON "'${TARGET_NAME}' is an INTERFACE library which do NOT support modules"
    )

    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs PUBLIC PRIVATE)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    fm_assert_empty(
        "${ARGS_UNPARSED_ARGUMENTS}"
        REASON "Unparsed arguments found: '${ARGS_UNPARSED_ARGUMENTS}'. All modules must be specified with PUBLIC or PRIVATE keywords."
    )

    # 3. Enable modules & scanning
    set_target_properties(
        ${TARGET_NAME} PROPERTIES
        CXX_SCAN_FOR_MODULES ON
    )

    # 4. Private Modules
    if(ARGS_PRIVATE)
        target_sources(${TARGET_NAME} PRIVATE 
            FILE_SET "${TARGET_NAME}_private_modules" TYPE CXX_MODULES FILES ${ARGS_PRIVATE}
        )
    endif()

    # 5. Public Modules
    if(ARGS_PUBLIC)
        target_sources(${TARGET_NAME} PUBLIC 
            FILE_SET CXX_MODULES TYPE CXX_MODULES FILES ${ARGS_PUBLIC}
        )
    endif()
endfunction()


#[[.rst:
.. command:: fm_add_clang_format_target

   .. code-block:: cmake

      fm_add_clang_format_target(<files...>)

   Creates a custom target named "format" that runs clang-format on specified files.

   :param files: List of source files to format
   :type files: list of file paths

   :pre: clang-format executable is available in PATH
   :post: A custom target named "format" is created that formats the specified files in-place

   .. note::
      - The format target uses ``-i`` flag to format files in-place
      - The ``-style=file`` flag means clang-format will look for a .clang-format configuration file
      - Files are formatted relative to CMAKE_SOURCE_DIR

   .. warning::
      This function will fail if clang-format is not found in PATH.

   **Example usage:**

   .. code-block:: cmake

      # Format specific files
      fm_add_clang_format_target(
          src/main.cpp
          src/utils.cpp
          include/header.hpp
      )

      # Then run: cmake --build . --target format

   .. seealso::
      - ``fm_find_clang_format()`` from fm_tool_utilities to locate clang-format
      - Create a .clang-format file in your project root to define formatting style

#]]
function(fm_add_clang_format_target TARGET_NAME)
    fm_assert_not_empty(${TARGET_NAME} REASON "add_clang_format_target requires a target name")

    include(fm_tool_utilities)
    fm_find_clang_format()

    set(FILES_TO_FORMAT ${ARGN})

    


    add_custom_target(
        ${TARGET_NAME}
        COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FILES_TO_FORMAT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Formatting all source files with clang-format..."
        VERBATIM
    )
endfunction()