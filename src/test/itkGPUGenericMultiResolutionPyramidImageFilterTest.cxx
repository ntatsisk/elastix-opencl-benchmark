/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//
#include "itkCommandLineArgumentParser.h"
#include "CommandLineArgumentHelper.h"
#include "itkTestHelper.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

// GPU include files
#include "itkGPUImageFactory.h"
#include "itkGPURecursiveGaussianImageFilterFactory.h"
#include "itkGPUCastImageFilterFactory.h"
#include "itkGPUShrinkImageFilterFactory.h"
#include "itkGPUResampleImageFilterFactory.h"
#include "itkGPUIdentityTransformFactory.h"
#include "itkGPULinearInterpolateImageFunctionFactory.h"

#include "itkGenericMultiResolutionPyramidImageFilter.h"

//------------------------------------------------------------------------------
std::string
GetHelpString( void )
{
  std::stringstream ss;

  ss << "Usage:" << std::endl
     << "  -in           input file name" << std::endl
     << "  [-outindex]   output index" << std::endl
     << "  [-outlog]     output log file name, default 'CPUGPULog.txt'" << std::endl
     << "  [-nooutput]   controls where output is created, default write output" << std::endl
     << "  [-runtimes]   controls how many times filter will execute, default 1" << std::endl
     << "  [-skipcpu]    skip running CPU part, default false" << std::endl
     << "  [-skipgpu]    skip running GPU part, default false" << std::endl
     << "  [-numlevels]  number of levels" << std::endl
     << "  [-mrrs]       use multi resolution rescale schedule, default false" << std::endl
     << "  [-mrss]       use multi resolution smoothing schedule, default false" << std::endl
     << "  [-sif]        use shrink image filter, default false" << std::endl
     << "  [-cl]         use current level, default false" << std::endl
     << "  [-l]          current level, default 0" << std::endl
     << "  [-out]        output file names.(outputCPU outputGPU)" << std::endl
     << "  [-rmse]       acceptable rmse error, default 0" << std::endl
     << "  [-threads]    number of threads, default maximum" << std::endl;
  return ss.str();
} // end GetHelpString()


//------------------------------------------------------------------------------
/** run: A macro to call a function. */
#define run( function, type_in, type0, type1, dim )                        \
  if( ComponentType == # type_in && Dimension == dim )                     \
  {                                                                        \
    typedef itk::Image< type0, dim > InputImageType;                       \
    typedef itk::Image< type1, dim > OutputImageType;                      \
    supported = true;                                                      \
    result    = function< InputImageType, OutputImageType >( parameters ); \
  }

//------------------------------------------------------------------------------
namespace
{
class Parameters
{
public:

  // Constructor
  Parameters() :
    logFileName( "CPUGPULog.txt" )
  {
    useCompression = false;
    outputWrite    = true;
    outputLog      = true;
    outputIndex    = 0;

    allowedRMSerror = 0.0;
    runTimes        = 1;
    skipCPU         = false;
    skipGPU         = false;

    // Filter
    numlevels                           = 4;
    currentLevel                        = 0;
    useMultiResolutionRescaleSchedule   = false;
    useMultiResolutionSmoothingSchedule = false;
    useShrinkImageFilter                = false;
    useComputeOnlyForCurrentLevel       = false;
  }


  bool         useCompression;
  bool         outputWrite;
  bool         outputLog;
  bool         skipCPU;
  bool         skipGPU;
  float        allowedRMSerror;
  unsigned int runTimes;

  // Files
  std::string                inputFileName;
  std::vector< std::string > outputFileNames;
  unsigned int               outputIndex;
  std::string                logFileName;

  // Filter
  unsigned int numlevels;
  unsigned int currentLevel;
  bool         useMultiResolutionRescaleSchedule;
  bool         useMultiResolutionSmoothingSchedule;
  bool         useShrinkImageFilter;
  bool         useComputeOnlyForCurrentLevel;
};

}

//----------------------------------------------------------------------------
template< typename GenericPyramidType >
void
SetNumberOfLevels( typename GenericPyramidType::Pointer & genericPyramid,
  const Parameters & _parameters )
{
  const unsigned int numLevels = _parameters.numlevels;

  switch( numLevels )
  {
    case 1:
    {
      itk::Vector< unsigned int, 1 > factors;
      unsigned int                   factor = static_cast< unsigned int >( pow( 2.0, 0 ) );
      for( unsigned int i = 0; i < 1; i++ )
      {
        factors[ i ] = factor;
        factor      /= 2;
      }
      genericPyramid->SetNumberOfLevels( 1 );
      genericPyramid->SetStartingShrinkFactors( factors.Begin() );
    }
    break;
    case 2:
    {
      itk::Vector< unsigned int, 2 > factors;
      unsigned int                   factor = static_cast< unsigned int >( pow( 2.0, 1 ) );
      for( unsigned int i = 0; i < 2; i++ )
      {
        factors[ i ] = factor;
        factor      /= 2;
      }
      genericPyramid->SetNumberOfLevels( 2 );
      genericPyramid->SetStartingShrinkFactors( factors.Begin() );
    }
    break;
    case 3:
    {
      itk::Vector< unsigned int, 3 > factors;
      unsigned int                   factor = static_cast< unsigned int >( pow( 2.0, 2 ) );
      for( unsigned int i = 0; i < 3; i++ )
      {
        factors[ i ] = factor;
        factor      /= 2;
      }
      genericPyramid->SetNumberOfLevels( 3 );
      genericPyramid->SetStartingShrinkFactors( factors.Begin() );
    }
    break;
    case 4:
    {
      itk::Vector< unsigned int, 4 > factors;
      unsigned int                   factor = static_cast< unsigned int >( pow( 2.0, 3 ) );
      for( unsigned int i = 0; i < 4; i++ )
      {
        factors[ i ] = factor;
        factor      /= 2;
      }
      genericPyramid->SetNumberOfLevels( 4 );
      genericPyramid->SetStartingShrinkFactors( factors.Begin() );
    }
    break;
    case 5:
    {
      itk::Vector< unsigned int, 5 > factors;
      unsigned int                   factor = static_cast< unsigned int >( pow( 2.0, 4 ) );
      for( unsigned int i = 0; i < 5; i++ )
      {
        factors[ i ] = factor;
        factor      /= 2;
      }
      genericPyramid->SetNumberOfLevels( 5 );
      genericPyramid->SetStartingShrinkFactors( factors.Begin() );
    }
    break;
    default:
      std::cerr << "Not implemented." << std::endl;
      break;
  }
}


//------------------------------------------------------------------------------
template< typename InputImageType, typename OutputImageType >
int ProcessImage( const Parameters & _parameters );

//------------------------------------------------------------------------------
// This test compares the CPU with the GPU version of the
// GenericMultiResolutionPyramidImageFilter.
// The filter takes an input image and produces an output image.
// We compare the CPU and GPU output image using RMSE and speed.
int
main( int argc, char * argv[] )
{
  // Setup for debugging and create log
  itk::SetupForDebugging();
  itk::CreateOpenCLLogger( "GenericMultiResolutionPyramidImageFilterTest" );

  // Create and check OpenCL context
  if( !itk::CreateContext() )
  {
    return EXIT_FAILURE;
  }

  // Create a command line argument parser
  itk::CommandLineArgumentParser::Pointer parser = itk::CommandLineArgumentParser::New();
  parser->SetCommandLineArguments( argc, argv );
  parser->SetProgramHelpText( GetHelpString() );

  parser->MarkArgumentAsRequired( "-in", "The input filename" );

  const itk::CommandLineArgumentParser::ReturnValue validateArguments = parser->CheckForRequiredArguments();
  if( validateArguments == itk::CommandLineArgumentParser::FAILED )
  {
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }
  else if( validateArguments == itk::CommandLineArgumentParser::HELPREQUESTED )
  {
    itk::ReleaseContext();
    return EXIT_SUCCESS;
  }

  // Create parameters class.
  Parameters parameters;
  parameters.logFileName = itk::GetLogFileName();

  // Get file names arguments
  parser->GetCommandLineArgument( "-in", parameters.inputFileName );
  parameters.outputFileNames.push_back( parameters.inputFileName.substr( 0, parameters.inputFileName.rfind(
    "." ) ) + "-out-cpu.mha" );
  parameters.outputFileNames.push_back( parameters.inputFileName.substr( 0, parameters.inputFileName.rfind(
    "." ) ) + "-out-gpu.mha" );
  parser->GetCommandLineArgument( "-out", parameters.outputFileNames );
  parameters.outputWrite = !( parser->ArgumentExists( "-nooutput" ) );
  parser->GetCommandLineArgument( "-outindex", parameters.outputIndex );
  parser->GetCommandLineArgument( "-outlog", parameters.logFileName );
  parser->GetCommandLineArgument( "-runtimes", parameters.runTimes );

  parser->GetCommandLineArgument( "-rmse", parameters.allowedRMSerror );
  parameters.skipCPU = parser->ArgumentExists( "-skipcpu" );
  parameters.skipGPU = parser->ArgumentExists( "-skipgpu" );

  // Filter parameters
  parser->GetCommandLineArgument( "-numlevels", parameters.numlevels );
  parameters.useMultiResolutionRescaleSchedule   = parser->ArgumentExists( "-mrrs" );
  parameters.useMultiResolutionSmoothingSchedule = parser->ArgumentExists( "-mrss" );
  parameters.useShrinkImageFilter                = parser->ArgumentExists( "-sif" );
  parameters.useComputeOnlyForCurrentLevel       = parser->ArgumentExists( "-cl" );
  parser->GetCommandLineArgument( "-l", parameters.currentLevel );

  // Threads.
  unsigned int maximumNumberOfThreads = itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads();
  parser->GetCommandLineArgument( "-threads", maximumNumberOfThreads );
  itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads( maximumNumberOfThreads );

  // Determine image properties.
  std::string                 ComponentType = "short";
  std::string                 PixelType; //we don't use this
  unsigned int                Dimension          = 2;
  unsigned int                NumberOfComponents = 1;
  std::vector< unsigned int > imagesize( Dimension, 0 );
  int                         retgip = GetImageProperties(
    parameters.inputFileName,
    PixelType,
    ComponentType,
    Dimension,
    NumberOfComponents,
    imagesize );

  if( retgip != 0 )
  {
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Let the user overrule this
  if( NumberOfComponents > 1 )
  {
    std::cerr << "ERROR: The NumberOfComponents is larger than 1!" << std::endl;
    std::cerr << "Vector images are not supported!" << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Get rid of the possible "_" in ComponentType.
  ReplaceUnderscoreWithSpace( ComponentType );

  // Run the program.
  bool supported = false;
  int  result    = EXIT_SUCCESS;
  try
  {
    // 2D
    run( ProcessImage, short, short, float, 2 );
    //run( ProcessImage, short, float, float, 2 );

    // 3D
    run( ProcessImage, short, short, float, 3 );
    //run( ProcessImage, short, float, float, 3 );
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "Caught ITK exception: " << e << std::endl;
    itk::ReleaseContext();
    result = EXIT_FAILURE;
  }
  if( !supported )
  {
    std::cerr << "ERROR: this combination of pixeltype and dimension is not supported!" << std::endl;
    std::cerr
      << "pixel (component) type = " << ComponentType
      << " ; dimension = " << Dimension
      << std::endl;
    itk::ReleaseContext();
    result = EXIT_FAILURE;
  }

  // End program.
  return result;
}


//------------------------------------------------------------------------------
template< typename InputImageType, typename OutputImageType >
int
ProcessImage( const Parameters & _parameters )
{
  // Typedefs
  const unsigned int ImageDim = (unsigned int)InputImageType::ImageDimension;

  typedef itk::Size< InputImageType::ImageDimension > SizeType;

  // Filter typedefs
  typedef itk::GenericMultiResolutionPyramidImageFilter
    < InputImageType, OutputImageType, float > FilterType;
  typedef itk::ImageFileReader< InputImageType >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;

  // Input image size
  SizeType imageSize;
  imageSize.Fill( 0 );

  // CPU part
  bool updateExceptionCPU = false;
  typename ReaderType::Pointer CPUReader;
  typename FilterType::Pointer CPUFilter;
  itk::TimeProbe cputimer;

  if( !_parameters.skipCPU )
  {
    CPUReader = ReaderType::New();
    CPUReader->SetFileName( _parameters.inputFileName );
    try
    {
      CPUReader->Update();
    }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "Caught ITK exception during CPUReader->Update(): " << e << std::endl;
      updateExceptionCPU = updateExceptionCPU || true;
    }

    imageSize = CPUReader->GetOutput()->GetBufferedRegion().GetSize();

    CPUFilter = FilterType::New();
    CPUFilter->SetNumberOfThreads( itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads() );

    cputimer.Start();

    if( !updateExceptionCPU )
    {
      for( unsigned int i = 0; i < _parameters.runTimes; i++ )
      {
        SetNumberOfLevels< FilterType >( CPUFilter, _parameters );
        CPUFilter->SetInput( CPUReader->GetOutput() );

        // Set use properties
        if( !_parameters.useMultiResolutionRescaleSchedule )
        {
          CPUFilter->SetRescaleScheduleToUnity();
        }
        if( !_parameters.useMultiResolutionSmoothingSchedule )
        {
          CPUFilter->SetSmoothingScheduleToZero();
        }
        CPUFilter->SetUseShrinkImageFilter( _parameters.useShrinkImageFilter );

        if( !_parameters.useComputeOnlyForCurrentLevel )
        {
          CPUFilter->SetComputeOnlyForCurrentLevel( _parameters.useComputeOnlyForCurrentLevel );
          CPUFilter->SetCurrentLevel( _parameters.currentLevel );
          CPUFilter->Update();
        }
        else
        {
          CPUFilter->SetComputeOnlyForCurrentLevel( _parameters.useComputeOnlyForCurrentLevel );
          for( unsigned int i = 0; i < CPUFilter->GetNumberOfLevels(); i++ )
          {
            CPUFilter->SetCurrentLevel( i );
            CPUFilter->Update();
          }
        }

        if( i != _parameters.runTimes - 1 )
        {
          CPUFilter->Modified();
        }
      }
    }
  }

  if( !_parameters.skipCPU )
  {
    cputimer.Stop();
    std::cout << "CPU " << CPUFilter->GetNameOfClass() << " took " << cputimer.GetMean() << " seconds with "
              << CPUFilter->GetNumberOfThreads() << " threads. run times " << _parameters.runTimes << std::endl;
  }

  // GPU part
  bool updateExceptionGPU = false;
  typename ReaderType::Pointer GPUReader;
  typename FilterType::Pointer GPUFilter;
  itk::TimeProbe gputimer;

  if( !_parameters.skipGPU )
  {
    typedef typelist::MakeTypeList< short, float >::Type OCLImageTypes;

    // register object factory for GPU image and filter
    itk::GPUImageFactory2< OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();
    itk::GPURecursiveGaussianImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();
    itk::GPUCastImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();

    if( _parameters.useShrinkImageFilter )
    {
      itk::GPUShrinkImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
      ::RegisterOneFactory();
    }
    else
    {
      itk::GPUResampleImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
      ::RegisterOneFactory();
      itk::GPUIdentityTransformFactory2< OCLImageDims >
      ::RegisterOneFactory();
      itk::GPULinearInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >
      ::RegisterOneFactory();
    }

    // GPU reader
    GPUReader = ReaderType::New();
    GPUReader->SetFileName( _parameters.inputFileName );
    try
    {
      GPUReader->Update();
    }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "Caught ITK exception during GPUReader->Update(): " << e << std::endl;
      updateExceptionGPU = updateExceptionGPU || true;
    }

    imageSize = GPUReader->GetOutput()->GetBufferedRegion().GetSize();

    // Construct the filter
    // Use a try/catch, because construction of this filter will trigger
    // OpenCL compilation, which may fail.
    try
    {
      GPUFilter = FilterType::New();
      itk::ITKObjectEnableWarnings( GPUFilter.GetPointer() );
    }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "Caught ITK exception during GPUFilter::New(): " << e << std::endl;
      updateExceptionGPU = updateExceptionGPU || true;
    }

    gputimer.Start();

    if( !updateExceptionGPU )
    {
      for( unsigned int i = 0; i < _parameters.runTimes; i++ )
      {
        SetNumberOfLevels< FilterType >( GPUFilter, _parameters );
        GPUFilter->SetInput( GPUReader->GetOutput() );

        // Set use properties
        if( !_parameters.useMultiResolutionRescaleSchedule )
        {
          GPUFilter->SetRescaleScheduleToUnity();
        }
        if( !_parameters.useMultiResolutionSmoothingSchedule )
        {
          GPUFilter->SetSmoothingScheduleToZero();
        }
        GPUFilter->SetUseShrinkImageFilter( _parameters.useShrinkImageFilter );

        try
        {
          if( !_parameters.useComputeOnlyForCurrentLevel )
          {
            GPUFilter->SetComputeOnlyForCurrentLevel( _parameters.useComputeOnlyForCurrentLevel );
            GPUFilter->SetCurrentLevel( _parameters.currentLevel );
            GPUFilter->Update();
          }
          else
          {
            GPUFilter->SetComputeOnlyForCurrentLevel( _parameters.useComputeOnlyForCurrentLevel );
            for( unsigned int i = 0; i < GPUFilter->GetNumberOfLevels(); i++ )
            {
              GPUFilter->SetCurrentLevel( i );
              GPUFilter->Update();
            }
          }
        }
        catch( itk::ExceptionObject & e )
        {
          std::cerr << "Caught ITK exception during GPUFilter->Update(): " << e << std::endl;
          updateExceptionGPU = updateExceptionGPU || true;
        }

        if( i != _parameters.runTimes - 1 )
        {
          GPUFilter->Modified();
        }
      }
    }
  }

  if( !_parameters.skipGPU )
  {
    gputimer.Stop();
    if( !updateExceptionGPU )
    {
      std::cout << "GPU " << GPUFilter->GetNameOfClass() << " took ";
      std::cout << gputimer.GetMean() / _parameters.runTimes
                << " seconds. run times " << _parameters.runTimes;
    }
  }

  // Get RMS Error check and test passed
  float RMSerror   = 0.0; float RMSrelative = 0.0;
  bool  testPassed = true;
  itk::GetTestFilterResult< float, FilterType, OutputImageType >
    ( CPUFilter, GPUFilter,
    _parameters.allowedRMSerror, RMSerror, RMSrelative, testPassed,
    _parameters.skipCPU, _parameters.skipGPU,
    cputimer.GetMean(), gputimer.GetMean(),
    updateExceptionCPU, updateExceptionGPU,
    _parameters.outputIndex );

  // Write output
  if( _parameters.outputWrite )
  {
    if( !_parameters.skipCPU )
    {
      // Write output CPU image
      typename WriterType::Pointer writerCPU = WriterType::New();
      writerCPU->SetInput( CPUFilter->GetOutput( _parameters.outputIndex ) );
      writerCPU->SetFileName( _parameters.outputFileNames[ 0 ] );
      writerCPU->Update();
    }

    if( !_parameters.skipGPU && !updateExceptionGPU )
    {
      // Write output GPU image
      typename WriterType::Pointer writerGPU = WriterType::New();
      writerGPU->SetInput( GPUFilter->GetOutput( _parameters.outputIndex ) );
      writerGPU->SetFileName( _parameters.outputFileNames[ 1 ] );
      writerGPU->Update();
    }
  }

  // Write log
  if( _parameters.outputLog )
  {
    std::string comments;

    // Add RescaleSchedule comments
    if( _parameters.useMultiResolutionRescaleSchedule )
    {
      comments.append( "RescaleSchedule(On)" );
    }
    else
    {
      comments.append( "RescaleSchedule(Off)" );
    }

    comments.append( ", " );

    // Add SmoothingSchedule comments
    if( _parameters.useMultiResolutionSmoothingSchedule )
    {
      comments.append( "SmoothingSchedule(On)" );
    }
    else
    {
      comments.append( "SmoothingSchedule(Off)" );
    }

    comments.append( ", " );

    // Add ShrinkImageFilter comments
    if( _parameters.useShrinkImageFilter )
    {
      comments.append( "ShrinkImageFilter(On)" );
    }
    else
    {
      comments.append( "ShrinkImageFilter(Off)" );
    }

    if( updateExceptionCPU || updateExceptionGPU )
    {
      comments.append( ", Exception during update" );
    }

    std::string filterName( "na" );
    if( !_parameters.skipCPU || ( !_parameters.skipCPU && !_parameters.skipGPU ) )
    {
      filterName = CPUFilter->GetNameOfClass();
    }
    else if( !_parameters.skipGPU )
    {
      filterName = GPUFilter->GetNameOfClass();
    }

    itk::WriteLog< InputImageType >(
      _parameters.logFileName, ImageDim, imageSize, RMSerror, RMSrelative,
      testPassed, updateExceptionGPU,
      itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads(),
      _parameters.runTimes, filterName,
      cputimer.GetMean(), gputimer.GetMean(), comments );
  }

  itk::ReleaseContext();

  if( testPassed )
  {
    return EXIT_SUCCESS;
  }
  else
  {
    return EXIT_FAILURE;
  }
}
