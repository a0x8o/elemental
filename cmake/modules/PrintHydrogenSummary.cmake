function (print_full_hydrogen_summary)
  set(_OPTIONS ONLY_PRINT_TRUE_BOOLS PRINT_EMPTY_VARIABLES)
  set(_ONE_VALUE_PARAMS TIMESTAMP)
  set(_MULTI_VALUE_PARAMS
    VERSION_VARIABLES PATH_VARIABLES STRING_VARIABLES BOOLEAN_VARIABLES)

  cmake_parse_arguments(__ARGS
    "${_OPTIONS}" "${_ONE_VALUE_PARAMS}" "${_MULTI_VALUE_PARAMS}"
    ${ARGN})


  set(__max_var_name_length 0)
  foreach (__var ${__ARGS_VERSION_VARIABLES} ${__ARGS_PATH_VARIABLES}
      ${__ARGS_STRING_VARIABLES} ${__ARGS_BOOLEAN_VARIABLES})
    string(LENGTH "${__var}" __var_name_length)

    if (__var_name_length GREATER __max_var_name_length)
      set(__max_var_name_length ${__var_name_length})
    endif ()
  endforeach ()
  math(EXPR __max_var_name_length "${__max_var_name_length} + 2")

  set(__STAR_STR
    "********************************************************************************")
  set(__DOTS_STR
    "......................................................................")
  set(__SPACE_STR
    "                                                                      ")
  set(__SUMMARY_STR "\n${__STAR_STR}\n")
  string(APPEND __SUMMARY_STR "* ${PROJECT_NAME} Build\n")
  if (__ARGS_TIMESTAMP)
    string(APPEND __SUMMARY_STR "* ${__ARGS_TIMESTAMP}\n")
  endif ()
  string(APPEND __SUMMARY_STR "*\n* Version information:\n")

  # Print the version variables
  foreach (__var ${__ARGS_VERSION_VARIABLES})
    string(LENGTH "${__var}" __var_name_length)
    math(EXPR __num_spaces "${__max_var_name_length} - ${__var_name_length}")
    string(SUBSTRING "${__SPACE_STR}" 0 ${__num_spaces} __spaces)

    string(APPEND __SUMMARY_STR "*   ${__var}${__spaces}${${__var}}\n")
  endforeach ()

  string(APPEND __SUMMARY_STR "*\n* Important paths:\n")

  foreach (__var ${__ARGS_PATH_VARIABLES})
    if (NOT ${__var} AND NOT __ARGS_PRINT_EMPTY_VARIABLES)
      continue()
    endif ()

    string(LENGTH "${__var}" __var_name_length)
    math(EXPR __num_spaces "${__max_var_name_length} - ${__var_name_length}")
    string(SUBSTRING "${__SPACE_STR}" 0 ${__num_spaces} __spaces)

    if (__ARGS_PRINT_EMPTY_VARIABLES AND NOT ${__var})
      string(APPEND __SUMMARY_STR "*   ${__var}${__spaces}<unset>\n")
    else ()
      string(APPEND __SUMMARY_STR "*   ${__var}${__spaces}${${__var}}\n")
    endif ()
  endforeach ()

  string(APPEND __SUMMARY_STR "*\n* Important strings:\n")

  foreach (__var ${__ARGS_STRING_VARIABLES})
    if (NOT ${__var} AND NOT __ARGS_PRINT_EMPTY_VARIABLES)
      continue()
    endif ()

    string(LENGTH "${__var}" __var_name_length)
    math(EXPR __num_spaces "${__max_var_name_length} - ${__var_name_length}")
    string(SUBSTRING "${__SPACE_STR}" 0 ${__num_spaces} __spaces)

    if (__ARGS_PRINT_EMPTY_VARIABLES AND NOT ${__var})
      string(APPEND __SUMMARY_STR "*   ${__var}${__spaces}\"\"\n")
    else ()
      string(APPEND __SUMMARY_STR "*   ${__var}${__spaces}\"${${__var}}\"\n")
    endif ()
  endforeach ()

  if (NOT __ARGS_ONLY_PRINT_TRUE_BOOLS)
    string(APPEND __SUMMARY_STR "*\n* The following switches are set:\n")

    foreach (__var ${__ARGS_BOOLEAN_VARIABLES})

      string(LENGTH "${__var}" __var_name_length)
      math(EXPR __num_spaces "${__max_var_name_length} - ${__var_name_length}")
      string(SUBSTRING "${__SPACE_STR}" 0 ${__num_spaces} __spaces)

      if (NOT ${__var})
        string(APPEND __SUMMARY_STR "*   ${__var}${__spaces}OFF\n")
      else ()
        string(APPEND __SUMMARY_STR "*   ${__var}${__spaces}ON\n")
      endif ()
    endforeach ()

  else ()

    string(APPEND __SUMMARY_STR "*\n* The following switches are ON:\n")

    foreach (__var ${__ARGS_BOOLEAN_VARIABLES})
      if (${__var})
        string(APPEND __SUMMARY_STR "*   ${__var}\n")
      endif ()
    endforeach ()
  endif ()

  string(APPEND __SUMMARY_STR "*\n${__STAR_STR}\n")

  # Get the message to STDOUT
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${__SUMMARY_STR}")
endfunction ()
