cmake_minimum_required(VERSION 3.13)

project(NRF_EdgeAI_User_Model
   VERSION 1.0
   LANGUAGES
   C
)

add_library(${PROJECT_NAME}
   nrf_edgeai_user_model.c
)

add_library(NRF::EdgeAI::UserModel ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
   PRIVATE
       NRF::EdgeAI
)
