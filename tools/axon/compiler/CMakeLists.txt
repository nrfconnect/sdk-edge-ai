#
# Copyright (c) 2023, Nordic Semiconductor ASA. All Rights Reserved.
#
# The information contained herein is confidential property of Nordic Semiconductor ASA.
# The use, copying, transfer or disclosure of such information is prohibited except by
# express written agreement with Nordic Semiconductor ASA.
#

if (${NOT_A_ZEPHYR_BUILD})
  target_include_directories(nrf-axon-compiler-interface
    INTERFACE
      include
  )

else()

  zephyr_include_directories(
    include
  )

endif()
