# set the minimum required CMake version
cmake_minimum_required(VERSION 3.16)

# specify the project name and language
# project(opencl-benchmark LANGUAGES CXX)
project(opencl-benchmark)

# enable C++14 standard
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set( CMAKE_INCLUDE_DIRECTORIES_BEFORE ON )

# find and include the ITK and Elastix libraries
find_package(Elastix REQUIRED)
include(${ELASTIX_USE_FILE})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

message("${OpenCL_INCLUDE_DIR}")
message(${PROJECT_SOURCE_DIR}/src)

# add the main.cpp file as a target
add_executable(transformix-benchmark src/transformix_benchmark.cpp)
add_executable(resampler-benchmark src/resampler_benchmark.cpp)
add_executable(denis-resampler-benchmark src/denis_resampler_benchmark.cpp src/CommandLineArgumentHelper.cxx)

# link the MyProject target against ITK and Elastix libraries
target_link_libraries(transformix-benchmark ${ITK_LIBRARIES} ${ELASTIX_LIBRARIES} elxOpenCL)
target_link_libraries(resampler-benchmark ${ITK_LIBRARIES} ${ELASTIX_LIBRARIES} elxOpenCL)
target_link_libraries(denis-resampler-benchmark ${ITK_LIBRARIES} ${ELASTIX_LIBRARIES} elxOpenCL)

target_include_directories(resampler-benchmark PUBLIC ${OpenCL_INCLUDE_DIR})
target_include_directories(denis-resampler-benchmark PUBLIC ${PROJECT_SOURCE_DIR}/src ${OpenCL_INCLUDE_DIR})
