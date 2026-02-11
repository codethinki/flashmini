# Copyright (c) 2026 Lukas Thomann
# Licensed under the MIT License

include(fm_target_utilities)

#[[.rst:
.. command:: fm_pkg_target_add_modules

   .. code-block:: cmake

      fm_pkg_target_add_modules(<target_name> [PUBLIC <files...>] [PRIVATE <files...>])

   Adds C++ module files to a target and registers it for installation.

   :param target_name: Name of the target to add modules to
   :type target_name: string
   :param PUBLIC: List of public module files (.cppm)
   :type PUBLIC: list of file paths
   :param PRIVATE: List of private module files (.cppm)
   :type PRIVATE: list of file paths

   :pre: target_name exists
   :pre: target_name is NOT an INTERFACE library (C++ modules not supported)
   :pre: At least one of PUBLIC or PRIVATE arguments is provided
   :post: Module files are added to target and target is registered for installation

   .. note::
      This function delegates to ``fm_target_add_modules()`` for core module handling,
      then registers the target for installation.

   .. seealso::
      Use ``fm_target_add_modules()`` from fm_target_utilities if installation is not needed.

#]]
function(fm_pkg_target_add_modules TARGET_NAME)
    # 1. Add modules to target
    fm_target_add_modules(${TARGET_NAME} ${ARGN})
    
    # 2. Register Target for installation logic
    get_property(INSTALLABLE_TARGETS GLOBAL PROPERTY _fm_INSTALLABLE_TARGETS)
    if(NOT "${TARGET_NAME}" IN_LIST INSTALLABLE_TARGETS)
        list(APPEND INSTALLABLE_TARGETS ${TARGET_NAME})
        set_property(GLOBAL PROPERTY _fm_INSTALLABLE_TARGETS "${INSTALLABLE_TARGETS}")
    endif()
endfunction()

#[[.rst:
.. command:: fm_pkg_target_find_package

   .. code-block:: cmake

      fm_pkg_target_find_package(<target_name> <find_package_args>...)

   Wraps find_package to ensure dependencies are found during build AND recorded for package config files.

   :param target_name: Name of the target that depends on the package
   :type target_name: string
   :param find_package_args: Arguments to pass to find_package (package name, version, components, etc.)
   :type find_package_args: variable arguments

   :post: Package is found via find_package and recorded for generated Config.cmake file using find_dependency

   .. note::
      The first argument in find_package_args should be the package name.
      All arguments are recorded and will be passed to find_dependency() in the generated package config.

   .. note::
      If the package is not found and REQUIRED is specified, a clear error message is generated
      indicating which component depends on the missing package.

#]]
function(fm_pkg_target_find_package TARGET_NAME)
    # 1. Standard find_package for the current build
    find_package(${ARGN})

    # 2. Record for installation
    list(GET ARGN 0 PKG_NAME)
    
    # Create a safe argument list for checking existence (remove REQUIRED)
    # This ensures find_package(... QUIET) doesn't fatal-error if the package is missing,
    # allowing us to print our custom message.
    set(ARGS_CHECK_LIST ${ARGN})
    list(REMOVE_ITEM ARGS_CHECK_LIST "REQUIRED")
    
    list(JOIN ARGS_CHECK_LIST " " ARGS_CHECK_STR)

    # The full arguments for the actual dependency enforcement (includes REQUIRED)
    list(JOIN ARGN " " ARGS_STR)
    
    # We create a check block that runs find_package QUIETly first (without REQUIRED).
    # block(SCOPE_FOR VARIABLES) ensures CMAKE_MESSAGE_LOG_LEVEL changes don't leak out.
    set(CHECK_BLOCK "
block(SCOPE_FOR VARIABLES)
    set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
    find_package(${ARGS_CHECK_STR} QUIET)
    if(NOT ${PKG_NAME}_FOUND)
        set(MSG \"${CMAKE_FIND_PACKAGE_NAME} component '${TARGET_NAME}' dependency missing: find_package(${ARGS_STR}) failed\")
        message(FATAL_ERROR \"\${MSG}\")
    endif()
endblock()
find_dependency(${ARGS_STR})
")
    set_property(GLOBAL APPEND_STRING PROPERTY _fm_PKG_DEPENDENCIES "${CHECK_BLOCK}\n")
endfunction()

#[[.rst:
.. command:: fm_pkg_target_include_directories

   .. code-block:: cmake

      fm_pkg_target_include_directories(<target_name>
                                         [PUBLIC <dirs...>]
                                         [PRIVATE <dirs...>]
                                         [INTERFACE <dirs...>])

   Configures target include directories with appropriate build and install interfaces.

   :param target_name: Name of the target to configure
   :type target_name: string
   :param PUBLIC: List of public include directories
   :type PUBLIC: list of directory paths
   :param PRIVATE: List of private include directories
   :type PRIVATE: list of directory paths
   :param INTERFACE: List of interface include directories
   :type INTERFACE: list of directory paths

   :pre: target_name exists
   :post: Include directories are configured with BUILD_INTERFACE and INSTALL_INTERFACE generator expressions
   :post: Public/Interface headers are installed to their respective directories
   :post: Target EXPORT_NAME is set (strips project name prefix if present)

   .. note::
      **Directory handling:**

      - BUILD_INTERFACE: Points to source directory during build
      - INSTALL_INTERFACE: Points to install directory for consumers
      - PRIVATE directories are NOT exported to install interface

   .. note::
      **Export name stripping:**

      If target name starts with ``${PROJECT_NAME}_``, the prefix is removed for the export name.
      Example: ``myproject_core`` → export name ``core`` → imported as ``myproject::core``

#]]
function(fm_pkg_target_include_directories TARGET_NAME)
    fm_assert_target("${TARGET_NAME}")
    set(oneValueArgs "")
    set(multiValueArgs PUBLIC PRIVATE INTERFACE)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    include(GNUInstallDirs)

    # --- strip project name prefix for EXPORT_NAME ---
    set(PREFIX_TO_STRIP "${PROJECT_NAME}_")
    string(FIND "${TARGET_NAME}" "${PREFIX_TO_STRIP}" PREFIX_POS)
    if(PREFIX_POS EQUAL 0)
        string(LENGTH "${PREFIX_TO_STRIP}" PREFIX_LENGTH)
        string(SUBSTRING "${TARGET_NAME}" ${PREFIX_LENGTH} -1 CLEAN_EXPORT_NAME)
        set_property(TARGET ${TARGET_NAME} PROPERTY EXPORT_NAME ${CLEAN_EXPORT_NAME})
    endif()

    # --- configure include directories ---
    foreach (SCOPE PUBLIC PRIVATE INTERFACE)
        if (DEFINED ARGS_${SCOPE})
            set(PROCESSED_DIRS "")
            foreach (DIR ${ARGS_${SCOPE}})
                list(APPEND PROCESSED_DIRS "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${DIR}>")
                if (NOT "${SCOPE}" STREQUAL "PRIVATE")
                    list(APPEND PROCESSED_DIRS "$<INSTALL_INTERFACE:${DIR}>")
                endif ()
            endforeach ()
            target_include_directories(${TARGET_NAME} ${SCOPE} ${PROCESSED_DIRS})
        endif ()
    endforeach ()

    # --- install public headers (File-based install is safe to keep here) ---
    set(PUBLIC_HEADER_DIRS ${ARGS_PUBLIC} ${ARGS_INTERFACE})
    if(PUBLIC_HEADER_DIRS)
        list(REMOVE_DUPLICATES PUBLIC_HEADER_DIRS)
        foreach(DIR ${PUBLIC_HEADER_DIRS})
            install(DIRECTORY ${DIR}/ DESTINATION ${DIR})
        endforeach()
    endif()

    # Register Target
    get_property(INSTALLABLE_TARGETS GLOBAL PROPERTY _fm_INSTALLABLE_TARGETS)
    if(NOT "${TARGET_NAME}" IN_LIST INSTALLABLE_TARGETS)
        list(APPEND INSTALLABLE_TARGETS ${TARGET_NAME})
        set_property(GLOBAL PROPERTY _fm_INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS})
    endif()
endfunction()

# _fm_finalize_pkg_targets()
# Internal function that performs the actual install(TARGETS) call.
#]]
function(_fm_finalize_pkg_targets)
    get_property(INSTALLABLE_TARGETS GLOBAL PROPERTY _fm_INSTALLABLE_TARGETS)
    include(GNUInstallDirs)
    
    # We use a consistent export set name based on the project name
    set(EXPORT_SET_NAME "${PROJECT_NAME}-targets")

    foreach(TGT ${INSTALLABLE_TARGETS})
        if(NOT TARGET ${TGT})
            continue()
        endif()

        get_target_property(TGT_TYPE ${TGT} TYPE)

        set(INSTALL_COMPONENTS "")

        # 1. Standard Binaries
        if(NOT "${TGT_TYPE}" STREQUAL "INTERFACE_LIBRARY")
            list(APPEND INSTALL_COMPONENTS
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
        endif()

        # 2. C++ Modules
        get_target_property(HAS_MODS ${TGT} CXX_MODULE_SETS)
        if(HAS_MODS)
            list(APPEND INSTALL_COMPONENTS
                FILE_SET CXX_MODULES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/modules/${TGT}"
            )
        endif()

        install(TARGETS ${TGT} 
                EXPORT "${EXPORT_SET_NAME}"
                ${INSTALL_COMPONENTS}
        )
    endforeach()
endfunction()





# _fm_setup_package()
# Updated to support C++ Module metadata export.
#]]
function(_fm_setup_package)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    set(EXPORT_SET_NAME "${PROJECT_NAME}-targets")
    set(NAMESPACE "${PROJECT_NAME}::")
    set(INSTALL_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
    set(TARGETS_FILENAME "${EXPORT_SET_NAME}.cmake")

    # --- Part 1: Install the Export Set ---
    install(EXPORT ${EXPORT_SET_NAME}
            FILE ${TARGETS_FILENAME}
            NAMESPACE ${NAMESPACE}
            DESTINATION ${INSTALL_CONFIG_DIR}
            # Module BMI folder
            CXX_MODULES_DIRECTORY "cmake/${PROJECT_NAME}-modules" 
    )

    # --- Part 2: Auto-generate and Install Config/Version files ---
    get_property(PKG_DEPENDENCIES GLOBAL PROPERTY _fm_PKG_DEPENDENCIES)

    set(TEMP_CONFIG_IN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in")
    file(WRITE ${TEMP_CONFIG_IN_PATH}
            "@PACKAGE_INIT@\n\n"
            "include(CMakeFindDependencyMacro)\n"
            "${PKG_DEPENDENCIES}\n"
            "include(\"\${CMAKE_CURRENT_LIST_DIR}/${TARGETS_FILENAME}\")\n"
    )

    configure_package_config_file(${TEMP_CONFIG_IN_PATH}
            "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
            INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
    )

    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion
    )

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION ${INSTALL_CONFIG_DIR}
    )
endfunction()

# _fm_add_pkg_target()
# builds and installs all registered package targets
# creates a custom target named "${PROJECT_NAME}_install"
# pre: _fm_INSTALLABLE_TARGETS global property is not empty
#]]
function(_fm_add_pkg_target)
    get_property(INSTALLABLE_TARGETS GLOBAL PROPERTY _fm_INSTALLABLE_TARGETS)
    fm_assert_not_empty("${INSTALLABLE_TARGETS}" "No installable targets were registered — use fm_pkg_target_include_directories or add to _fm_INSTALLABLE_TARGETS manually")

    set(INSTALL_TARGET_NAME "${PROJECT_NAME}_package")
    set(INSTALL_COMMENT "Packaging ${PROJECT_NAME} project...")

    # --- FIX START: Filter out INTERFACE libraries from build dependencies ---
    set(BUILDABLE_TARGETS "")
    foreach(TGT ${INSTALLABLE_TARGETS})
        get_target_property(TGT_TYPE ${TGT} TYPE)
        # We only add to DEPENDS if it creates a real file (Static/Shared Lib or Executable)
        # INTERFACE_LIBRARY does not create a file, so we skip it here.
        if(NOT "${TGT_TYPE}" STREQUAL "INTERFACE_LIBRARY")
            list(APPEND BUILDABLE_TARGETS ${TGT})
        endif()
    endforeach()
    # --- FIX END ---

    set(PKG_DUMMY_SOURCE "${CMAKE_BINARY_DIR}/_pkg_dummy_source.cpp")
    if(WIN32)
        file(WRITE ${PKG_DUMMY_SOURCE} 
            "#define WIN32_LEAN_AND_MEAN\n"
            "#include <Windows.h>\n"
            "int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) { return 0; }\n"
        )
        add_executable(${INSTALL_TARGET_NAME} WIN32 ${PKG_DUMMY_SOURCE})
    else()
        file(WRITE ${PKG_DUMMY_SOURCE}
            "#include<print>\n int main() { std::println(\"installed :)\"); return 0; }"
        )
        add_executable(${INSTALL_TARGET_NAME} ${PKG_DUMMY_SOURCE})
    endif()

    add_custom_target(_do_${INSTALL_TARGET_NAME}_install
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_INSTALL_PREFIX}"
            COMMAND ${CMAKE_COMMAND} --install . --prefix "${CMAKE_INSTALL_PREFIX}"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "${INSTALL_COMMENT}"
            DEPENDS ${BUILDABLE_TARGETS}  # <--- Use the filtered list here
    )

    add_dependencies(${INSTALL_TARGET_NAME} _do_${INSTALL_TARGET_NAME}_install)
endfunction()


#[[.rst:
.. command:: fm_create_package

   .. code-block:: cmake

      fm_create_package()

   Finalizes and creates the installable package for the project.

   :post: All registered targets are finalized for installation
   :post: Package config and version files are generated
   :post: Package target named ``${PROJECT_NAME}_package`` is created

   .. note::
      This function orchestrates the complete package creation process:

      1. Finalizes all registered installable targets
      2. Generates CMake package configuration files
      3. Creates a custom build target for packaging

   .. note::
      After calling this, build the ``${PROJECT_NAME}_package`` target to create the package.

#]]
function(fm_create_package)
    _fm_finalize_pkg_targets()
    _fm_setup_package()
    _fm_add_pkg_target()
endfunction()