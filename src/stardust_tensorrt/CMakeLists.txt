cmake_minimum_required(VERSION 3.8)

option(USE_TENSORRT "Compile tensorrt framework" ON)
message(STATUS "USE_TENSORRT: ${USE_TENSORRT}\n")

if(USE_TENSORRT)
  set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  project(stardust_tensorrt LANGUAGES CXX CUDA)
  option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
else()
  project(stardust_tensorrt)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O0")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

configure_file(include/config.h.in include/config.h)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# OnnxRuntime
set(ONNXRUNTIME_LIBS /usr/lib/libonnxruntime.so)

# CUDA
if(USE_TENSORRT)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
  get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
  message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

  # TensorRT
  set(TensorRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
  set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)

  message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
  message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

  list(APPEND INCLUDE_DIRS
          ${CUDA_INCLUDE_DIRS}
          ${OpenCV_INCLUDE_DIRS}
          ${TensorRT_INCLUDE_DIRS}
          include
          )
  
else()
  list(APPEND INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
  include
  )
endif()

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
if(USE_TENSORRT)
  message(STATUS "Source files: \n${SRCS}\n")
  add_executable(yolo ${SRCS})
  target_include_directories(yolo PUBLIC ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/include)
  target_link_libraries(yolo nvinfer nvinfer_plugin ${ONNXRUNTIME_LIBS})
  target_link_directories(yolo PUBLIC ${TensorRT_LIBRARIES})
  ament_target_dependencies(yolo OpenCV CUDA rclcpp image_transport sensor_msgs cv_bridge tf2_ros pcl_ros)
else()
  list(REMOVE_ITEM SRCS ${PROJECT_SOURCE_DIR}/src/tensorrt.cpp)
  message(STATUS "Source files: \n${SRCS}\n")
  add_executable(yolo ${SRCS})
  target_include_directories(yolo PUBLIC ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/include)
  target_link_libraries(yolo ${ONNXRUNTIME_LIBS})
  ament_target_dependencies(yolo OpenCV rclcpp image_transport sensor_msgs cv_bridge tf2_ros pcl_ros)
endif()

install(TARGETS
  yolo
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY 
  ${PROJECT_SOURCE_DIR}/include
  DESTINATION include
)

install(DIRECTORY 
  ${PROJECT_BINARY_DIR}/include
  DESTINATION include
)

install(DIRECTORY 
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY 
  config
  DESTINATION share/${PROJECT_NAME}
)
      

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
