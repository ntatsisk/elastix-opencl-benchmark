project( ITK4OpenCLTest )

# Set some directories
set( ITK4OpenCL_TestDataDir   ${CMAKE_SOURCE_DIR}/data )
set( ITK4OpenCL_TestOutputDir ${CMAKE_BINARY_DIR}/Testing )

# Find Elastix
find_package( ELASTIX REQUIRED )

# Find ITK.
find_package( ITK REQUIRED )
include( ${ITK_USE_FILE} )

# Find OpenCL.
find_package( OpenCL REQUIRED )
if( NOT ${OPENCL_FOUND} )
  message(FATAL_ERROR "Could not find OpenCL")
else()
  # Force OPENCL_OPTIMIZATION_MAD_ENABLE to on
  if( DEFINED OPENCL_OPTIMIZATION_MAD_ENABLE )
    set( OPENCL_OPTIMIZATION_MAD_ENABLE ON CACHE BOOL
      "Allow a * b + c to be replaced by a mad. The mad computes a * b + c with reduced accuracy." FORCE )
  endif()
endif()

# Define CMake switches for tests
set( ITK4OpenCL_TEST_1D ON CACHE BOOL "ITK4OpenCL add 1D data tests (image-512-1D.mha)" )
set( ITK4OpenCL_TEST_2D ON CACHE BOOL "ITK4OpenCL add 2D data tests (image-256x256-2D.mha)" )
set( ITK4OpenCL_TEST_3D_TINY OFF CACHE BOOL "ITK4OpenCL add 3D data tests (image-64x64x64-3D.mha)" )
set( ITK4OpenCL_TEST_3D_SMALL OFF CACHE BOOL "ITK4OpenCL add 3D data tests (image-100x100x100-3D.mha)" )
set( ITK4OpenCL_TEST_3D_UNEVEN ON CACHE BOOL "ITK4OpenCL add 3D data tests (image-115x157x129-3D.mha)" )
set( ITK4OpenCL_TEST_3D_MEDIUM OFF CACHE BOOL "ITK4OpenCL add 3D data tests (image-256x256x256-3D.mha)" )
set( ITK4OpenCL_TEST_3D_BIG OFF CACHE BOOL "ITK4OpenCL add 3D data tests (image-512x512x256-3D.mha)" )

set( ITK4OpenCL_TEST_GENERATE_OUTPUT OFF CACHE BOOL "ITK4OpenCL generate output files." )

# Enable ordered includes
set( CMAKE_INCLUDE_DIRECTORIES_BEFORE ON )


# #################################################
# Some global parameters
# #################################################

# Skip_parameters variable
set( skip_parameters "" )
if( ITK4OpenCL_SKIP_CPU)
  set( skip_parameters ${skip_parameters} "-skipcpu" )
endif()
if( ITK4OpenCL_SKIP_GPU)
  set( skip_parameters ${skip_parameters} "-skipgpu" )
endif()

# Select output or not
if( ITK4OpenCL_TEST_GENERATE_OUTPUT )
  set( write_output "" )
else()
  set( write_output "-nooutput" )
endif()

# Input data files
set( PARAMETERS_IN ${ITK4OpenCL_TestDataDir}/BSplineDisplacements.txt )
set( DATA1D_IN ${ITK4OpenCL_TestDataDir}/image-512-1D.mha )
set( DATA2D_IN ${ITK4OpenCL_TestDataDir}/image-256x256-2D.mha )
set( DATA3D_TINY_IN ${ITK4OpenCL_TestDataDir}/image-64x64x64-3D.mha )
set( DATA3D_SMALL_IN ${ITK4OpenCL_TestDataDir}/image-100x100x100-3D.mha )
set( DATA3D_UNEVEN_IN ${ITK4OpenCL_TestDataDir}/image-115x157x129-3D.mha )
set( DATA3D_MEDIUM_IN ${ITK4OpenCL_TestDataDir}/image-256x256x256-3D.mha )
set( DATA3D_BIG_IN ${ITK4OpenCL_TestDataDir}/image-512x512x256-3D.mha )


# #################################################
# Macro that simplifies the addition of tests
# Usage:
# ADD_GPU_TEST_EXECUTABLE( <name_of_test> )
# We assume here that the name of the source which includes the main is:
# "itk"name_of_test".cxx".
# #################################################
macro( ADD_GPU_TEST_EXECUTABLE name )
  # Microsoft specific items
  if( MSVC )
    # Kill the annoying MS VS warning about non-safe functions.
    # They hide real warnings.
    add_definitions(
      /D_SCL_SECURE_NO_DEPRECATE
      /D_CRT_SECURE_NO_DEPRECATE
      /D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
    )

    # Increases address capacity
    if( WIN32 )
      set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj" )
      set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj" )
      set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /bigobj" )
    endif()
  endif()

  # Define some helper variables.
  set( GPUTEST_EXECUTABLE_NAME itk${name} )
  set( GPUTEST_SOURCE_NAME itk${name}.cxx
    itkTestHelper.h
    itkTestOutputWindow.cxx
    itkTestOutputWindow.h )

  # Create the test executable.
  add_executable( ${GPUTEST_EXECUTABLE_NAME} ${GPUTEST_SOURCE_NAME} )

  include_directories(
    # ${ITK4GPUCommon_SOURCE_DIR}
    ${ITK4OpenCL_INCLUDE_DIRS}

    # 3rd party libraries
    ${OPENCL_INCLUDE_DIRS}

    ${ELASTIX_INCLUDE_DIRS}
  )

  include_directories( AFTER
    ${ITK_INCLUDE_DIRS}
  )

  # Link against other libraries.
  target_link_libraries( ${GPUTEST_EXECUTABLE_NAME}
    # OpenCL libraries first
    ${OPENCL_LIBRARIES}

    # Other libraries
    ITK4OpenCL
    ${ITK_LIBRARIES}
    ${ELASTIX_LIBRARIES}
  )

  # Group in IDE's like Visual Studio
  set_property( TARGET ${GPUTEST_EXECUTABLE_NAME} PROPERTY FOLDER "tests OpenCL filters" )
endmacro()


# #################################################
# Macro that defines output file names
# #################################################
macro( set_test_output data_in data_out_cpu data_out_gpu test_name test_postfix )
  # Get the filename without the path
  get_filename_component( DATA_IN_NO_PATH ${data_in} NAME )

  # Construct CPU and GPU output names
  if( ${test_postfix} STREQUAL "none" OR ${test_postfix} STREQUAL "NONE" )
    string( REPLACE ".mha" "-out-cpu.mha" DATA_FILE_OUT_CPU ${DATA_IN_NO_PATH} )
    string( REPLACE ".mha" "-out-gpu.mha" DATA_FILE_OUT_GPU ${DATA_IN_NO_PATH} )
  else()
    string( REPLACE ".mha" "-out-${test_postfix}-cpu.mha" DATA_FILE_OUT_CPU ${DATA_IN_NO_PATH} )
    string( REPLACE ".mha" "-out-${test_postfix}-gpu.mha" DATA_FILE_OUT_GPU ${DATA_IN_NO_PATH} )
  endif()

  # Append with the output path
  set( ${data_out_cpu} "${ITK4OpenCL_TestOutputDir}/${test_name}/${DATA_FILE_OUT_CPU}" )
  set( ${data_out_gpu} "${ITK4OpenCL_TestOutputDir}/${test_name}/${DATA_FILE_OUT_GPU}" )

  # Create output directory
  if( ITK4OpenCL_TEST_GENERATE_OUTPUT )
    file( MAKE_DIRECTORY ${ITK4OpenCL_TestOutputDir}/${test_name} )
  endif()
endmacro()


# #################################################
# Macro that defines resample affine transform test
# #################################################
macro( ADD_GPU_RESAMPLE_TEST interpolator transform combo RMSEs )

  set( TEST_EXE itkGPUResampleImageFilterTest )

  # Extra parameters variable
  set( extra_parameters ${skip_parameters} )

  # Check if transform has word BSpline anywhere
  foreach( elem ${transform} )
    if( ${elem} STREQUAL "BSpline" )
      set( extra_parameters "-p" ${PARAMETERS_IN} )
      break()
    endif()
  endforeach()

  # Check if transform has word Euler or Similarity anywhere
  set( transformContainsES FALSE )
  foreach( elem ${transform} )
    if( ${elem} STREQUAL "Euler" OR ${elem} STREQUAL "Similarity" )
      set( transformContainsES TRUE )
      break()
    endif()
  endforeach()

  # if combo transform
  if( ${combo} )
    set( extra_parameters ${extra_parameters} "-c" )
    set( combo_test_name Combo )
  endif()

  # Construct the test base name
  set( test_base_name1 "${interpolator}${combo_test_name}" )
  foreach( elem ${transform} )
    set( test_base_name1 ${test_base_name1}${elem} )
  endforeach()
  set( test_base_name GPUResampleImageFilterTest-${test_base_name1} )

  # Add all requested tests
  set( listOfTests 1D 2D 3D_SMALL 3D_UNEVEN 3D_MEDIUM 3D_BIG )
  set( rmses ${RMSEs} ) # temporary copy
  foreach( test ${listOfTests} )
    # Get input data
    set( data_in ${DATA${test}_IN} )

    # Get output names, create output directory
    set( output_dir_name ResamplerTest-${test_base_name1} )
    set_test_output( ${data_in} data_OUT_CPU data_OUT_GPU ${output_dir_name} none )

    # Get RMSE for this test and pop the list
    list( GET rmses 0 rmse )
    list( REMOVE_AT rmses 0 )

    # Check For 1D transform
    set( only2d3dtransform FALSE )
    if( ${test} EQUAL "1D" AND ${transformContainsES} )
      set( only2d3dtransform TRUE )
    endif()

    # Add the test
    if( ITK4OpenCL_TEST_${test} AND NOT ${only2d3dtransform} )
      add_test( ${test_base_name}-${test} ${TEST_EXE} -in ${data_in} -out ${data_OUT_CPU} ${data_OUT_GPU} ${write_output}
        -i ${interpolator} -t ${transform} ${extra_parameters} -rmse ${rmse} )
      set_tests_properties( ${test_base_name}-${test} PROPERTIES RUN_SERIAL true )
    endif()
  endforeach()
endmacro()

# #################################################
# GPUFactoriesTest
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPUFactoriesTest )
add_test( GPUFactoriesTest itkGPUFactoriesTest )

# #################################################
# GPUGenericMultiResolutionPyramidImageFilter test
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPUGenericMultiResolutionPyramidImageFilterTest )

set( TEST_EXE itkGPUGenericMultiResolutionPyramidImageFilterTest )

# Add all requested tests
set( listOfTests 2D 3D_SMALL 3D_UNEVEN 3D_MEDIUM 3D_BIG )
set( listOfPermutations "-mrss;-mrrs;-mrss -mrrs;-mrss -sif;-mrrs -sif;-mrss -mrrs -sif" )
foreach( test ${listOfTests} )
  # Get input data
  set( data_in ${DATA${test}_IN} )

  foreach( perm ${listOfPermutations} )
    # Create test name
    string( REPLACE " " "" permNoSpace ${perm} )
    string( SUBSTRING ${permNoSpace} 1 -1 permNoSpace )
    set( test_name GPUGenericMultiResolutionPyramidImageFilterTest-${permNoSpace}-${test} )

    # Get output names
    set_test_output( ${data_in} data_OUT_CPU data_OUT_GPU test1 ${permNoSpace} )

    # Process the permutation list
    string( REPLACE " " ";" permaslist ${perm} )

    # Add the test
    if( ITK4OpenCL_TEST_${test} )
      add_test( ${test_name} ${TEST_EXE} -in ${data_in} -out ${data_OUT_CPU} ${data_OUT_GPU} ${write_output}
        ${skip_parameters} ${permaslist} -rmse 0.3 )
      set_tests_properties( ${test_name} PROPERTIES RUN_SERIAL true )
    endif()
  endforeach()
endforeach()


# #################################################
# RecursiveGaussianImageFilter test
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPURecursiveGaussianImageFilterTest )

set( TEST_EXE itkGPURecursiveGaussianImageFilterTest )

# Add all requested tests
set( listOfTests 1D 2D 3D_SMALL 3D_UNEVEN 3D_MEDIUM 3D_BIG )
foreach( test ${listOfTests} )
  # Get input data
  set( data_in ${DATA${test}_IN} )

  # Create test name
  set( test_name GPURecursiveGaussianImageFilterTest-${test} )

  # Get output names
  set_test_output( ${data_in} data_OUT_CPU data_OUT_GPU test2 none )

  # Add the test
  if( ITK4OpenCL_TEST_${test} )
    add_test( ${test_name} ${TEST_EXE} -in ${data_in} -out ${data_OUT_CPU} ${data_OUT_GPU} ${write_output}
      ${skip_parameters} -rmse 0.01 )
    set_tests_properties( ${test_name} PROPERTIES RUN_SERIAL true )
  endif()
endforeach()


# #################################################
# GPUSmoothingRecursiveGaussianImageFilter test
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPUSmoothingRecursiveGaussianImageFilterTest )

set( TEST_EXE itkGPUSmoothingRecursiveGaussianImageFilterTest )

# Add all requested tests
set( listOfTests 2D 3D_SMALL 3D_UNEVEN 3D_MEDIUM 3D_BIG )
foreach( test ${listOfTests} )
  # Get input data
  set( data_in ${DATA${test}_IN} )

  # Create test name
  set( test_name GPUSmoothingRecursiveGaussianImageFilterTest-${test} )

  # Get output names
  set_test_output( ${data_in} data_OUT_CPU data_OUT_GPU test3 none )

  # Add the test
  if( ITK4OpenCL_TEST_${test} )
    add_test( ${test_name} ${TEST_EXE} -in ${data_in} -out ${data_OUT_CPU} ${data_OUT_GPU} ${write_output}
      ${skip_parameters} -rmse 0.04 )
    set_tests_properties( ${test_name} PROPERTIES RUN_SERIAL true )
  endif()
endforeach()


# #################################################
# BSplineDecompositionImageFilter test
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPUBSplineDecompositionImageFilterTest )

set( TEST_EXE itkGPUBSplineDecompositionImageFilterTest )

# Add all requested tests
set( listOfTests 1D 2D 3D_SMALL 3D_UNEVEN 3D_MEDIUM 3D_BIG )
foreach( test ${listOfTests} )
  # Get input data
  set( data_in ${DATA${test}_IN} )

  # Create test name
  set( test_name GPUBSplineDecompositionImageFilterTest-${test} )

  # Get output names
  set_test_output( ${data_in} data_OUT_CPU data_OUT_GPU test4 none )

  # Add the test
  if( ITK4OpenCL_TEST_${test} )
    add_test( ${test_name} ${TEST_EXE} -in ${data_in} -out ${data_OUT_CPU} ${data_OUT_GPU} ${write_output}
      ${skip_parameters} -rmse 0.001 )
    set_tests_properties( ${test_name} PROPERTIES RUN_SERIAL true )
  endif()
endforeach()


# #################################################
# GPUCastImageFilterTest test
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPUCastImageFilterTest )

set( TEST_EXE itkGPUCastImageFilterTest )

# Add all requested tests
set( listOfTests 1D 2D 3D_SMALL 3D_UNEVEN 3D_MEDIUM 3D_BIG )
foreach( test ${listOfTests} )
  # Get input data
  set( data_in ${DATA${test}_IN} )

  # Create test name
  set( test_name GPUCastImageFilterTest-${test} )

  # Get output names
  set_test_output( ${data_in} data_OUT_CPU data_OUT_GPU test5 none )

  # Add the test
  if( ITK4OpenCL_TEST_${test} )
    add_test( ${test_name} ${TEST_EXE} -in ${data_in} -out ${data_OUT_CPU} ${data_OUT_GPU} ${write_output}
      ${skip_parameters} -rmse 0.0 )
    set_tests_properties( ${test_name} PROPERTIES RUN_SERIAL true )
  endif()
endforeach()


# #################################################
# GPUShrinkImageFilterTest test
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPUShrinkImageFilterTest )

set( TEST_EXE itkGPUShrinkImageFilterTest )

# Add all requested tests
set( listOfTests 1D 2D 3D_SMALL 3D_UNEVEN 3D_MEDIUM 3D_BIG )
foreach( test ${listOfTests} )
  # Get input data
  set( data_in ${DATA${test}_IN} )

  # Create test name
  set( test_name GPUShrinkImageFilterTest-${test} )

  # Get output names
  set_test_output( ${data_in} data_OUT_CPU data_OUT_GPU test6 none )

  # Add the test
  if( ITK4OpenCL_TEST_${test} )
    add_test( ${test_name} ${TEST_EXE} -in ${data_in} -out ${data_OUT_CPU} ${data_OUT_GPU} ${write_output}
      ${skip_parameters} -rmse 0.0 )
    set_tests_properties( ${test_name} PROPERTIES RUN_SERIAL true )
  endif()
endforeach()


# #################################################
# GPUBSplineTransformTest test
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPUBSplineTransformTest )

set( TEST_EXE itkGPUBSplineTransformTest )

# Add all requested tests
set( listOfTests 1D 2D 3D_SMALL 3D_UNEVEN 3D_MEDIUM 3D_BIG )
foreach( test ${listOfTests} )
  # Get input data
  set( data_in ${DATA${test}_IN} )

  # Create test name
  set( test_name GPUBSplineTransformTest-${test} )

  # Get output names
  set_test_output( ${data_in} data_OUT_CPU data_OUT_GPU test7 none )

  # Add the test
  if( ITK4OpenCL_TEST_${test} )
    add_test( ${test_name} ${TEST_EXE} -in ${data_in} -out ${data_OUT_CPU} ${data_OUT_GPU} ${write_output}
      -inpar ${PARAMETERS_IN} ${skip_parameters} -rmse 0.0 )
    set_tests_properties( ${test_name} PROPERTIES RUN_SERIAL true )
  endif()
endforeach()


# #################################################
# ResampleImageFilter tests
# #################################################
ADD_GPU_TEST_EXECUTABLE( GPUResampleImageFilterTest )

# Translation transform tests
ADD_GPU_RESAMPLE_TEST( NearestNeighbor Translation FALSE "2.0;3.1;41.0;0.0;29.0;0.0" )
ADD_GPU_RESAMPLE_TEST( Linear Translation FALSE "0.3;0.2;0.1;0.1;0.2;0.2" )
ADD_GPU_RESAMPLE_TEST( BSpline Translation FALSE "0.0;0.1;0.2;0.1;0.2;0.5" )

# Affine transform tests
ADD_GPU_RESAMPLE_TEST( NearestNeighbor Affine FALSE "2.0;1.8;1.2;4.0;3.0;8.0" )
ADD_GPU_RESAMPLE_TEST( Linear Affine FALSE "0.3;0.8;0.1;0.1;0.2;0.7" )
ADD_GPU_RESAMPLE_TEST( BSpline Affine FALSE "0.0;0.8;0.2;0.1;0.3;0.8" )

# BSpline transform tests
ADD_GPU_RESAMPLE_TEST( NearestNeighbor BSpline FALSE "0.0;0.2;0.5;0.4;0.6;0.6" )
ADD_GPU_RESAMPLE_TEST( Linear BSpline FALSE "0.0;0.0;0.1;0.1;0.1;0.1" )
ADD_GPU_RESAMPLE_TEST( BSpline BSpline FALSE "0.0;0.01;0.2;0.1;0.2;0.1" )

# Combo with single transform
ADD_GPU_RESAMPLE_TEST( NearestNeighbor Translation TRUE "2.0;3.1;41.0;0.0;29.0;0.0" )
ADD_GPU_RESAMPLE_TEST( Linear Translation TRUE "0.3;0.2;0.1;0.1;0.2;0.2" )
ADD_GPU_RESAMPLE_TEST( BSpline Translation TRUE "0.0;0.1;0.2;0.1;0.3;0.3" )

ADD_GPU_RESAMPLE_TEST( NearestNeighbor Affine TRUE "0.0;1.8;1.3;4.0;3.0;8.0" )
ADD_GPU_RESAMPLE_TEST( Linear Affine TRUE "0.0;0.1;0.1;0.1;0.2;0.7" )
ADD_GPU_RESAMPLE_TEST( BSpline Affine TRUE "0.0;0.1;0.2;0.1;0.3;0.8" )

ADD_GPU_RESAMPLE_TEST( NearestNeighbor BSpline TRUE "0.0;0.2;0.5;0.4;0.7;0.7" )
ADD_GPU_RESAMPLE_TEST( Linear BSpline TRUE "0.0;0.0;0.1;0.1;0.1;0.1" )
ADD_GPU_RESAMPLE_TEST( BSpline BSpline TRUE "0.0;0.01;0.2;0.1;0.2;0.1" )

# Combo with multiple transforms
ADD_GPU_RESAMPLE_TEST( NearestNeighbor "Affine;BSpline" TRUE "0.0;0.0;0.1;0.3;0.4;0.7" )
ADD_GPU_RESAMPLE_TEST( Linear "Affine;BSpline" TRUE "0.0;0.01;0.1;0.1;0.1;0.1" )
ADD_GPU_RESAMPLE_TEST( BSpline "Affine;BSpline" TRUE "0.0;0.01;0.2;0.1;0.2;0.1" )

ADD_GPU_RESAMPLE_TEST( NearestNeighbor "Translation;Affine;BSpline;Euler;Similarity" TRUE "0.0;0.0;1.1;0.5;0.7;1.0" )
ADD_GPU_RESAMPLE_TEST( Linear "Translation;Affine;BSpline;Euler;Similarity" TRUE "0.0;0.01;0.1;0.1;0.1;0.1" )
ADD_GPU_RESAMPLE_TEST( BSpline "Translation;Affine;BSpline;Euler;Similarity" TRUE "0.0;0.02;0.2;0.1;0.2;0.1" )

# Combo with multiple transforms of the same type
ADD_GPU_RESAMPLE_TEST( BSpline "Translation;Translation;BSpline;BSpline;BSpline" TRUE "0.0;0.02;0.2;0.1;0.2;0.1" )

