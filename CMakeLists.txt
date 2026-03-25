cmake_minimum_required(VERSION 3.16)
project(inference)

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

# --- roboto_motors ---
find_package(roboto_motors QUIET)
if (NOT roboto_motors_FOUND)
  message(WARNING "roboto_motors ROS config not found, falling back to manual library search...")
  find_path(ROBOTO_MOTORS_INCLUDE_DIR NAMES motor_driver.hpp PATHS /opt/roboparty/include NO_DEFAULT_PATH)
  
  find_library(LIB_MOTORS          NAMES motors          PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  find_library(LIB_MOTORS_PROTOCOL NAMES motors_protocol PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  find_library(LIB_DM_MOTORS       NAMES dm_motors       PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  find_library(LIB_EVO_MOTORS      NAMES evo_motors      PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  
  if (ROBOTO_MOTORS_INCLUDE_DIR AND LIB_MOTORS AND LIB_MOTORS_PROTOCOL AND LIB_DM_MOTORS AND LIB_EVO_MOTORS)
    message(STATUS "Found roboto_motors manually.")
    add_library(roboto_motors::roboto_motors INTERFACE IMPORTED)
    target_include_directories(roboto_motors::roboto_motors INTERFACE ${ROBOTO_MOTORS_INCLUDE_DIR})
    target_link_libraries(roboto_motors::roboto_motors INTERFACE 
        ${LIB_MOTORS} ${LIB_MOTORS_PROTOCOL} ${LIB_DM_MOTORS} ${LIB_EVO_MOTORS})
    set(roboto_motors_FOUND TRUE)
  else()
    message(FATAL_ERROR "Could not find roboto_motors (checked ROS config and manual paths)")
  endif()
endif()

# --- roboto_imu ---
find_package(roboto_imu QUIET)
if (NOT roboto_imu_FOUND)
  message(WARNING "roboto_imu ROS config not found, falling back to manual library search...")
  find_path(ROBOTO_IMU_INCLUDE_DIR NAMES imu_driver.hpp PATHS /opt/roboparty/include NO_DEFAULT_PATH)
  
  find_library(LIB_IMU             NAMES imu             PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  find_library(LIB_IMU_PROTOCOL    NAMES imu_protocol    PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  find_library(LIB_HIPNUC_IMU      NAMES hipnuc_imu      PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  
  if (ROBOTO_IMU_INCLUDE_DIR AND LIB_IMU AND LIB_IMU_PROTOCOL AND LIB_HIPNUC_IMU)
    message(STATUS "Found roboto_imu manually.")
    add_library(roboto_imu::roboto_imu INTERFACE IMPORTED)
    target_include_directories(roboto_imu::roboto_imu INTERFACE ${ROBOTO_IMU_INCLUDE_DIR})
    target_link_libraries(roboto_imu::roboto_imu INTERFACE 
        ${LIB_IMU} ${LIB_IMU_PROTOCOL} ${LIB_HIPNUC_IMU})
    set(roboto_imu_FOUND TRUE)
  else()
    message(FATAL_ERROR "Could not find roboto_imu (checked ROS config and manual paths)")
  endif()
endif()

# --- roboto_bms ---
find_package(roboto_bms QUIET)
if (NOT roboto_bms_FOUND)
  message(WARNING "roboto_bms ROS config not found, falling back to manual library search...")
  find_path(ROBOTO_BMS_INCLUDE_DIR NAMES bms/bms_driver.hpp PATHS /opt/roboparty/include NO_DEFAULT_PATH)
  find_library(ROBOTO_BMS_LIB NAMES bms PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  find_library(ROBOTO_TWS_BMS_LIB NAMES tws_bms PATHS /opt/roboparty/lib NO_DEFAULT_PATH)
  
  if (ROBOTO_BMS_INCLUDE_DIR AND ROBOTO_BMS_LIB AND ROBOTO_TWS_BMS_LIB)
    message(STATUS "Found roboto_bms manually: ${ROBOTO_BMS_LIB}")
    add_library(roboto_bms::roboto_bms INTERFACE IMPORTED)
    target_include_directories(roboto_bms::roboto_bms INTERFACE ${ROBOTO_BMS_INCLUDE_DIR})
    target_link_libraries(roboto_bms::roboto_bms INTERFACE ${ROBOTO_BMS_LIB} ${ROBOTO_TWS_BMS_LIB})
    set(roboto_bms_FOUND TRUE)
  else()
    message(FATAL_ERROR "Could not find roboto_bms (checked ROS config and manual paths)")
  endif()
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
set(ONNXRUNTIME_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/onnxruntime-linux-x64-1.21.0)
set(CNPY_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/cnpy-linux-x64)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
set(ONNXRUNTIME_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/onnxruntime-linux-aarch64-1.21.0)
set(CNPY_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/cnpy-linux-aarch64)
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT_DIR}/lib)

set(CNPY_INCLUDE_DIR ${CNPY_ROOT_DIR}/include)
set(CNPY_LIB_DIR ${CNPY_ROOT_DIR}/lib)

option(YAML_BUILD_SHARED_LIBS "Build shared library" OFF)
option(YAML_CPP_BUILD_TESTS "Build tests" OFF)
option(YAML_CPP_BUILD_TOOLS "Build tools" OFF)
add_subdirectory(thirdparty/yaml-cpp-0.8.0)

set(PUBLIC_DEPENDENCIES
    ${Boost_LIBRARIES} pthread Eigen3::Eigen fmt::fmt spdlog::spdlog ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so ${CNPY_LIB_DIR}/libcnpy.so yaml-cpp)

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

# Link Motors
if(TARGET roboto_motors::roboto_motors)
  target_link_libraries(robot PUBLIC roboto_motors::roboto_motors)
else()
  ament_target_dependencies(robot PUBLIC roboto_motors)
endif()

# Link IMU
if(TARGET roboto_imu::roboto_imu)
  target_link_libraries(robot PUBLIC roboto_imu::roboto_imu)
else()
  ament_target_dependencies(robot PUBLIC roboto_imu)
endif()

# Link BMS
if(TARGET roboto_bms::roboto_bms)
  target_link_libraries(robot PUBLIC roboto_bms::roboto_bms)
else()
  ament_target_dependencies(robot PUBLIC roboto_bms)
endif()

pybind11_add_module(robot_py src/pybind_module.cpp)
target_link_libraries(robot_py PUBLIC robot)

add_executable(${PROJECT_NAME}_node src/inference_node.cpp src/ros_interface.cpp)
target_include_directories(${PROJECT_NAME}_node
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CNPY_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME}_node PUBLIC ${PUBLIC_DEPENDENCIES} utils robot)
ament_target_dependencies(${PROJECT_NAME}_node PUBLIC rclcpp sensor_msgs geometry_msgs std_srvs)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(TARGETS robot utils
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
install(FILES ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1 ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.21.0 ${CNPY_LIB_DIR}/libcnpy.so
        DESTINATION lib)

set(PYTHON_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

install(TARGETS robot_py
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
  ARCHIVE DESTINATION ${PYTHON_INSTALL_DIR}
  RUNTIME DESTINATION ${PYTHON_INSTALL_DIR})

ament_export_dependencies(rclcpp sensor_msgs geometry_msgs)
ament_export_libraries(robot)

ament_package()
