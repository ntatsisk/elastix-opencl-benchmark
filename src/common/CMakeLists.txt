project( ITK4OpenCL )

# Find Elastix
find_package( ELASTIX REQUIRED )

# Find ITK.
find_package( ITK REQUIRED )
include( ${ITK_USE_FILE} )

find_package( OpenCL REQUIRED )
if( NOT ${OPENCL_FOUND} )
  message( FATAL_ERROR "Could not find OpenCL" )
else()
  include( ${CMAKE_MODULE_PATH}/elastixOpenCL.cmake )
endif()

# Enable ordered includes
set( CMAKE_INCLUDE_DIRECTORIES_BEFORE ON )

# Get the files in the main directory
file( GLOB OpenCL_CommonFiles itk*.* )
list( APPEND OpenCL_CommonFiles CommandLineArgumentHelper.cxx CommandLineArgumentHelper.h )
source_group( "Common" FILES ${OpenCL_CommonFiles} )

# Set the files, so that they are shown in the editor
set( OpenCL_Files
  ${OpenCL_CommonFiles}
  )

# Microsoft specific items
if( MSVC )
  # Kill the anoying MS VS warning about non-safe functions.
  # They hide real warnings.
  add_definitions(
  /D_SCL_SECURE_NO_DEPRECATE
  /D_CRT_SECURE_NO_DEPRECATE
  /D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
  )
endif()

# Create the openCL library
add_library( ITK4OpenCL ${OpenCL_Files} )

# Add "Static" and "_d" to output names
MODULE_TARGET_NAME( ITK4OpenCL )

# Define include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OPENCL_INCLUDE_DIRS}
  ${ELASTIX_INCLUDE_DIRS}
)

include_directories( AFTER
  ${ITK_INCLUDE_DIRS}
)

# Define the link libraries
target_link_libraries(ITK4OpenCL
  # OpenCL libraries first
  ${OPENCL_LIBRARIES}

  # ITK libraries
  ${ITK_LIBRARIES}
)
