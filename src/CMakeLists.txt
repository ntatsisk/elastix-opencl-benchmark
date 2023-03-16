#=========================================================================
# CMakeLists.txt for ITK4OpenCL project

cmake_minimum_required( VERSION 2.8 )
cmake_policy( SET CMP0012 NEW )

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH} )

set( PROJECT_NAME_COMBINED "ITK4OpenCL-${OPENCL_SDK_STRING}" CACHE INTERNAL "Combine project name and OpenCL platform" )
string( REPLACE " " "-" PROJECT_NAME_COMBINED ${PROJECT_NAME_COMBINED} )

set(ITK4OpenCL_PROJECT_NAME ${PROJECT_NAME_COMBINED} CACHE string "Project Name" FORCE)
project(${ITK4OpenCL_PROJECT_NAME})

# Use grouping in Visual Studio
set_property( GLOBAL PROPERTY USE_FOLDERS ON )

# Define ITK4OpenCL common path
set( ITK4OpenCL_COMMON_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  CACHE INTERNAL "ITK4OpenCL common path" )

# set the ITK4OpenCL include dirs
set( ITK4OpenCL_INCLUDE_DIRS
# ${CMAKE_CURRENT_BINARY_DIR}/itkgpu
  ${ITK4OpenCL_COMMON_PATH}
  CACHE INTERNAL "ITK4OpenCL include directories" )

# Define build variable for ITK4OpenCL
set( ITK4OpenCL_BUILD ON CACHE BOOL
  "Build ITK4OpenCL project. On build. Off don't build." )

# Define build variable for ITK4OpenCL test
set( ITK4OpenCL_TEST_BUILD ON CACHE BOOL
  "Build ITK4OpenCL test project. On build. Off don't build." )

# Define skip cpu, skip gpu flags
set( ITK4OpenCL_SKIP_CPU CACHE BOOL
  "With this option CPU filter will not be called." )
set( ITK4OpenCL_SKIP_GPU CACHE BOOL
  "With this option GPU filter will not be called." )

# Define ITK4OpenCL platform x86, x64
if( CMAKE_CL_64 )
  set( ITK4OpenCL_PLATFORM "x64" CACHE INTERNAL
    "ITK4OpenCL platform x86 or x64" )
else()
  set( ITK4OpenCL_PLATFORM "x86" CACHE INTERNAL
    "ITK4OpenCL platform x86 or x64" )
endif()

# macro: MODULE_TARGET_NAME(target_name)
# Does library and executable output name mangling, using the following rules.
# BaseTargetName: ITK4OpenCL
# Shared lib/app release: ITK4OpenCL
# Shared lib/app debug:   ITK4OpenCL_d
# Static lib release:     ITK4OpenCLStatic
# Static lib debug:       ITK4OpenCLStatic_d
# Static app release:     ITK4OpenCL
# Static app debug:       ITK4OpenCL_d
macro( MODULE_TARGET_NAME target_name )
  # Get the target property (EXECUTABLE, STATIC_LIBRARY, SHARED_LIBRARY, MODULE_LIBRARY, UTILITY, INSTALL_FILES or INSTALL_PROGRAMS)
  get_target_property( TARGET_TYPE ${target_name} TYPE )

  # set the base name of the target
  if( ${ARGC} GREATER 1 )
    set( target_base_name ${ARGV1} )
  else()
    set( target_base_name ${target_name} )
  endif()

  # add "Static" to STATIC_LIBRARY targets
  if( TARGET_TYPE STREQUAL "STATIC_LIBRARY" )
    set( NAME_APPENDIX "Static" )
  endif()

  # Define the "normal" target name
  set_target_properties( ${target_name}
    PROPERTIES OUTPUT_NAME ${target_base_name}${NAME_APPENDIX} )
  # Define the "debug" target name
  set_target_properties( ${target_name}
    PROPERTIES DEBUG_OUTPUT_NAME ${target_base_name}${NAME_APPENDIX}_d )
endmacro( MODULE_TARGET_NAME target_name )

# macro: ADD_FILES_TO_GROUP(group_name [files])
# Add the listed `files` to the `group_name` group using the SOURCE_GROUP command
# and stores them in the ${SOURCES} variable.
# The SOURCE_GROUP is used to create folders in the Visual Studio projects
# Note:
# Use 4 backslashes to indicate subfolder relations
# Use "&amp\;" instead of "&" in group names
# Example:
# ADD_FILES_TO_GROUP("Filters" Foo1.cpp Foo2.cpp)
# ADD_FILES_TO_GROUP("Filters\\\\SubFilters" Bar1.cpp Bar2.cpp)
# ADD_LIBRARY(FooBar ${SOURCES}
macro(ADD_FILES_TO_GROUP group_name)
  if(NOT ${group_name} STREQUAL " " )
    source_group(${group_name} FILES ${ARGN})
  endif()
  set(SOURCES ${ARGN} "${SOURCES}")
endmacro(ADD_FILES_TO_GROUP)

# Add directories here
# add_subdirectory( itkgpu )

# Build ITK4OpenCL
if( ITK4OpenCL_BUILD )
  add_subdirectory( common )
endif()

# Build ITK4OpenCL test
if( ITK4OpenCL_TEST_BUILD )
  enable_testing()
  add_subdirectory( test )
  if(ITK4OpenCL_SKIP_CPU)
    #set(DART_TESTING_TIMEOUT 30)
    set(DART_TESTING_TIMEOUT 15)
  else()
    set(DART_TESTING_TIMEOUT 120)
  endif()
  include( CTest )
endif()

#---------------------------------------------------------------------
# Set default build type to Release, if none was specified
# Taken from ITK CMake list
if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
  message( STATUS "Setting build type to 'Release' as none was specified." )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
  # Set the possible values of build type for cmake-gui
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )
endif()
