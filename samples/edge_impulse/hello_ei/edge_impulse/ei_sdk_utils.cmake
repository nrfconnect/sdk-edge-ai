#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Process Edge Impulse URI list from configuration

# Function to parse and process Edge Impulse URI(s)
# Supports multiple URIs separated by space, newline, or semicolon
# Converts relative paths to absolute paths
#
# Arguments:
#   uri_input - Input URI string from CONFIG_EDGE_IMPULSE_URI
#   output_var - Variable name to store the processed URI list
#
function(process_edge_impulse_uri uri_input output_var)
  # Match first and any URI in the middle of input string
  # URI can be separated by space, new line or semicolon
  string(REGEX MATCHALL ".+[ \r\n;]" uri_prepare_list "${uri_input}")

  # Match the last URI in input string
  string(REGEX MATCH "[^ \n\r;].+$" uri_list_end "${uri_input}")

  list(APPEND uri_prepare_list ${uri_list_end})

  set(uri_list "")
  foreach(uri IN LISTS uri_prepare_list)
    # Remove trailing spaces
    string(STRIP ${uri} uri_string)

    # If URI is NOT a URL (http://, https://, file://), treat it as a local path
    if(NOT ${uri_string} MATCHES "^[a-z]+://")
      # Expand any ${VARIABLES} in the path
      string(CONFIGURE ${uri_string} uri_string)

      # Convert relative paths to absolute paths
      if(NOT IS_ABSOLUTE ${uri_string})
        # Using application source directory as base directory for relative path
        set(uri_string ${APPLICATION_SOURCE_DIR}/${uri_string})
      endif()
    endif()

    list(APPEND uri_list ${uri_string})
  endforeach()

  # Remove duplicated URIs from list
  list(REMOVE_DUPLICATES uri_list)

  # Return the processed list
  set(${output_var} ${uri_list} PARENT_SCOPE)

endfunction()

# Get Edge Impulse API key header from sysbuild configuration
#
# Retrieves authentication header for downloading private Edge Impulse models
# Returns empty string if not configured or sysbuild not available
#
# Arguments:
#   OUTPUT_VAR - Variable name to store API key header
function(get_edge_impulse_api_key OUTPUT_VAR)
  if(COMMAND zephyr_get)
    zephyr_get(EI_API_KEY_HEADER SYSBUILD GLOBAL)
  else()
    set(EI_API_KEY_HEADER "")
  endif()

  set(${OUTPUT_VAR} "${EI_API_KEY_HEADER}" PARENT_SCOPE)
endfunction()
