# Copyright (c) 2026 Lukas Thomann
# Licensed under the MIT License

#[[.rst:
.. command:: _fm_assertion_failure (internal)

   .. code-block:: cmake

      _fm_assertion_failure(<reason> <args...>)

   Internal macro to terminate configuration with a formatted error message.

   :param reason: Error message describing the failure
   :type reason: string
   :param args: Additional context to append to error message
   :type args: optional arguments

   :post: Configuration terminates with FATAL_ERROR

   .. warning::
      This is an internal function. Use the public assertion functions instead.
#]]
macro(_fm_assertion_failure reason)
    

    if("${reason}" STREQUAL "")
        set(reason "unknown reason")
    endif()
    if("${ARGN}" STREQUAL "")
        set(ARG_STR "")  
    else()
        set(ARG_STR "[args: ${ARGN}]")
    endif()



    message(FATAL_ERROR "ERROR: ${reason}${ARG_STR}")
endmacro()

#[[.rst:
.. command:: fm_assert_true

   .. code-block:: cmake

      fm_assert_true(<condition...> REASON <reason>)

   Asserts that a boolean condition evaluates to TRUE, terminating configuration otherwise.

   :param condition: CMake boolean expression to evaluate
   :type condition: boolean expression
   :param REASON: Error message to display if the condition is FALSE
   :type REASON: string

   :pre: condition is a valid CMake boolean expression
   :post: condition evaluates to TRUE, or configuration terminates with FATAL_ERROR

#]]
function(fm_assert_true)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${oneValueArgs}" "")
    
    if(NOT ${ARG_UNPARSED_ARGUMENTS})
        _fm_assertion_failure("${ARG_REASON}")
    endif()
endfunction()

#[[.rst:
.. command:: fm_assert_false

   .. code-block:: cmake

      fm_assert_false(<condition...> REASON <reason>)

   Asserts that a boolean condition evaluates to FALSE, terminating configuration otherwise.

   :param condition: CMake boolean expression to evaluate
   :type condition: boolean expression
   :param REASON: Error message to display if the condition is TRUE
   :type REASON: string

   :pre: condition is a valid CMake boolean expression
   :post: condition evaluates to FALSE, or configuration terminates with FATAL_ERROR

#]]
function(fm_assert_false)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${oneValueArgs}" "")
    
    if(${ARG_UNPARSED_ARGUMENTS})
        _fm_assertion_failure("${ARG_REASON}")
    endif()
endfunction()

#[[.rst:
.. command:: fm_assert_not_cmd

   .. code-block:: cmake

      fm_assert_not_cmd(<cmd> [REASON <reason>])

   Asserts that a CMake command, function, or macro is NOT defined.

   :param cmd: Name of the command to check
   :type cmd: string
   :param REASON: Optional error message to display if the assertion fails
   :type REASON: string

   :post: cmd is NOT a defined command/function/macro, or configuration terminates with FATAL_ERROR

#]]
function(fm_assert_not_cmd cmd)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")
    
    if(NOT ARG_REASON)
        set(ARG_REASON "Command '${cmd}' already defined")
    endif()
    
    fm_assert_false(COMMAND ${cmd} REASON "${ARG_REASON}")
endfunction()

#[[.rst:
.. command:: fm_assert_cmd

   .. code-block:: cmake

      fm_assert_cmd(<cmd> [REASON <reason>])

   Asserts that a CMake command, function, or macro is defined.

   :param cmd: Name of the command to check
   :type cmd: string
   :param REASON: Optional error message to display if the assertion fails
   :type REASON: string

   :post: cmd is a defined command/function/macro, or configuration terminates with FATAL_ERROR

#]]
function(fm_assert_cmd cmd)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")
    
    if(NOT ARG_REASON)
        set(ARG_REASON "Command '${cmd}' not defined")
    endif()
    
    fm_assert_true(COMMAND ${cmd} REASON "${ARG_REASON}")
endfunction()

#[[.rst:
.. command:: fm_assert_target

   .. code-block:: cmake

      fm_assert_target(<target> [REASON <reason>])

   Asserts that a CMake target exists in the current scope.

   :param target: Name of the target to check
   :type target: string
   :param REASON: Optional error message to display if the assertion fails
   :type REASON: string

   :post: target exists, or configuration terminates with FATAL_ERROR

#]]
function(fm_assert_target target)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")
    
    if(NOT ARG_REASON)
        set(ARG_REASON "Target '${target}' does not exist")
    endif()
    
    fm_assert_true(TARGET ${target} REASON "${ARG_REASON}")
endfunction()

#[[.rst:
.. command:: fm_assert_not_target

   .. code-block:: cmake

      fm_assert_not_target(<target> [REASON <reason>])

   Asserts that a CMake target does NOT exist in the current scope.

   :param target: Name of the target to check
   :type target: string
   :param REASON: Optional error message to display if the assertion fails
   :type REASON: string

   :post: target does NOT exist, or configuration terminates with FATAL_ERROR

#]]
function(fm_assert_not_target target)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")
    
    if(NOT ARG_REASON)
        set(ARG_REASON "Target '${target}' already exists")
    endif()
    
    fm_assert_false(TARGET ${target} REASON "${ARG_REASON}")
endfunction()

#[[.rst:
.. command:: fm_assert_empty

   .. code-block:: cmake

      fm_assert_empty(<value> [REASON <reason>])

   Asserts that a value is an empty string.

   :param value: Value to check for emptiness
   :type value: string
   :param REASON: Optional error message to display if the assertion fails
   :type REASON: string

   :post: value is an empty string, or configuration terminates with FATAL_ERROR

#]]
function(fm_assert_empty value)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")
    
    if(NOT ("${value}" STREQUAL ""))
        if(NOT ARG_REASON)
            set(ARG_REASON "Value not empty: '${value}'")
        endif()
        _fm_assertion_failure("${ARG_REASON}")
    endif()
endfunction()

#[[.rst:
.. command:: fm_assert_not_empty

   .. code-block:: cmake

      fm_assert_not_empty(<value> [REASON <reason>])

   Asserts that a value is NOT an empty string.

   :param value: Value to check for non-emptiness
   :type value: string
   :param REASON: Optional error message to display if the assertion fails
   :type REASON: string

   :post: value is NOT an empty string, or configuration terminates with FATAL_ERROR

#]]
function(fm_assert_not_empty value)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")
    
    if("${value}" STREQUAL "")
        if(NOT ARG_REASON)
            set(ARG_REASON "Value is empty")
        endif()
        _fm_assertion_failure("${ARG_REASON}")
    endif()
endfunction()

#[[.rst:
.. command:: fm_assert_program

   .. code-block:: cmake

      fm_assert_program(<prog> [REASON <reason>] [args...])

   Asserts an external program exists.

   :param prog: Name of the program to find
   :type prog: string
   :param REASON: Optional error message to display if the assertion fails
   :type REASON: string
   :param args: Additional arguments to pass to find_program (e.g., PATHS, HINTS)
   :type args: optional arguments

   :post: program found or configuration terminates with FATAL_ERROR
#]]
function(fm_assert_program prog)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")
    
    find_program(TEMP "${prog}" ${ARG_UNPARSED_ARGUMENTS})
    
    if(NOT ARG_REASON)
        set(ARG_REASON "Program '${prog}' not found")
    endif()
    
    fm_assert_true(${VAR_NAME} REASON "${ARG_REASON}")
endfunction()

#[[.rst:
.. command:: fm_assert_file

   .. code-block:: cmake

      fm_assert_file(<file> [REASON <reason>])

   Asserts that a file exists and is not a directory.

   :param file: Path to the file to check
   :type file: string
   :param REASON: Optional error message to display if the assertion fails
   :type REASON: string

   :post: file exists and is not a directory or configuration terminates with FATAL_ERROR
#]]
function(fm_assert_file file)
    set(oneValueArgs REASON)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")

    if(NOT EXISTS "${file}" OR IS_DIRECTORY "${file}")
        if(NOT ARG_REASON)
            set(ARG_REASON "File '${file}' does not exist or is a directory")
        endif()
        _fm_assertion_failure("${ARG_REASON}")
    endif()
endfunction()