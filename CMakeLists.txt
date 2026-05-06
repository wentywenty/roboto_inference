cmake_minimum_required(VERSION 3.16)
project(roboto-inference)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Boost COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

find_package(roboto_motors REQUIRED)
find_package(roboto_imu REQUIRED)
find_package(roboto_bms REQUIRED)

add_subdirectory(thirdparty)

set(PUBLIC_DEPENDENCIES
    ${Boost_LIBRARIES} pthread Eigen3::Eigen fmt::fmt spdlog::spdlog ${ONNXRUNTIME_LIB_FILE} cnpy yaml-cpp)

add_subdirectory(src)

add_library(robot STATIC
  src/robot_interface.cpp
)

target_include_directories(robot
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(robot PUBLIC ${PUBLIC_DEPENDENCIES} utils)

# Link hardware packages
foreach(_pkg roboto_motors roboto_imu roboto_bms)
  if(TARGET ${_pkg}::${_pkg})
    target_link_libraries(robot PUBLIC ${_pkg}::${_pkg})
  else()
    ament_target_dependencies(robot PUBLIC ${_pkg})
  endif()
endforeach()

pybind11_add_module(robot_py src/pybind_module.cpp)
target_link_libraries(robot_py PUBLIC robot)

add_executable(inference_node src/inference_node.cpp src/ros_interface.cpp src/obs_manager.cpp)
target_include_directories(inference_node
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CNPY_INCLUDE_DIR}
)
target_link_libraries(inference_node PUBLIC ${PUBLIC_DEPENDENCIES} utils robot)
ament_target_dependencies(inference_node PUBLIC rclcpp sensor_msgs geometry_msgs std_srvs)

set_target_properties(inference_node PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")

install(TARGETS inference_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(TARGETS robot utils
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(FILES
  src/utils/close_chain_mapping.hpp
  src/utils/thread_pool.hpp
  DESTINATION include/utils)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
install(FILES ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1 ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.21.0
        DESTINATION lib OPTIONAL)

set(PYTHON_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

install(TARGETS robot_py
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
  ARCHIVE DESTINATION ${PYTHON_INSTALL_DIR}
  RUNTIME DESTINATION ${PYTHON_INSTALL_DIR})

ament_export_dependencies(rclcpp sensor_msgs geometry_msgs)
ament_export_include_directories(include)
ament_export_libraries(robot)

ament_package()
