cmake_minimum_required(VERSION 3.8)
project(go2_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Declare the list of messages you want to generate
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LidarDecoded.msg"
  DEPENDENCIES std_msgs
)

# Export message runtime dependency
ament_export_dependencies(rosidl_default_runtime std_msgs)


ament_package()
