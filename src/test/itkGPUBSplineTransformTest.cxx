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

#include "itkOpenCLUtil.h"

// GPU include files
#include "itkGPUImageFactory.h"
#include "itkGPUBSplineTransformFactory.h"
#include "itkGPUCastImageFilterFactory.h"

//------------------------------------------------------------------------------
std::string
GetHelpString( void )
{
  std::stringstream ss;

  ss << "Usage:" << std::endl
     << "  -in           input file names" << std::endl
     << "  -inpar        input parameters file name" << std::endl
     << "  [-out]        output file names.(outputCPU outputGPU)" << std::endl
     << "  [-outindex]   output index" << std::endl
     << "  [-outlog]     output log file name, default 'CPUGPULog.txt'" << std::endl
     << "  [-nooutput]   controls where output is created, default write output" << std::endl
     << "  [-runtimes]   controls how many times filter will execute, default 1" << std::endl
     << "  [-skipcpu]    skip running CPU part, default false" << std::endl
     << "  [-skipgpu]    skip running GPU part, default false" << std::endl
     << "  [-rmse]       acceptable rmse error, default 0" << std::endl
     << "  [-threads]    number of threads, default maximum" << std::endl;
  return ss.str();
} // end GetHelpString()


//------------------------------------------------------------------------------
/** run: A macro to call a function. */
#define run( function, type0, type1, dim )                                 \
  if( ComponentType == # type0 && Dimension == dim )                       \
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

    allowedRMSerror = 0.0;
    runTimes        = 1;
    skipCPU         = false;
    skipGPU         = false;
    outputIndex     = 0;
  }


  bool         useCompression;
  bool         outputWrite;
  bool         outputLog;
  bool         skipCPU;
  bool         skipGPU;
  float        allowedRMSerror;
  unsigned int runTimes;

  // Files
  std::vector< std::string > inputFileNames;
  std::string                inputParametersFileName;
  std::vector< std::string > outputFileNames;
  unsigned int               outputIndex;
  std::string                logFileName;

  // Filter
};

}

//------------------------------------------------------------------------------
// This call is a bit different in
// GPUBSplineTransform::CopyCoefficientImagesToGPU( void )
// No going to update it for now
template< typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
CopyCoefficientImagesToGPU(
  itk::BSplineTransform< TScalarType, NDimensions, VSplineOrder > * transform,
  itk::FixedArray< typename itk::GPUImage< TScalarType, NDimensions >::Pointer, NDimensions > & coefficientArray )
{
  // CPU Typedefs
  typedef itk::BSplineTransform< TScalarType, NDimensions, VSplineOrder > BSplineTransformType;
  typedef typename BSplineTransformType::ImageType                        TransformCoefficientImageType;
  typedef typename BSplineTransformType::ImagePointer                     TransformCoefficientImagePointer;
  typedef typename BSplineTransformType::CoefficientImageArray            CoefficientImageArray;

  // GPU Typedefs
  typedef itk::GPUImage< TScalarType, NDimensions >          GPUTransformCoefficientImageType;
  typedef typename GPUTransformCoefficientImageType::Pointer GPUTransformCoefficientImagePointer;

  const CoefficientImageArray coefficientImageArray = transform->GetCoefficientImages();

  // Typedef for caster
  typedef itk::CastImageFilter< TransformCoefficientImageType, GPUTransformCoefficientImageType > CasterType;

  for( unsigned int i = 0; i < coefficientImageArray.Size(); i++ )
  {
    TransformCoefficientImagePointer coefficients = coefficientImageArray[ i ];

    GPUTransformCoefficientImagePointer GPUCoefficients = GPUTransformCoefficientImageType::New();
    GPUCoefficients->CopyInformation( coefficients );
    GPUCoefficients->SetRegions( coefficients->GetBufferedRegion() );
    GPUCoefficients->Allocate();

    // Create caster
    typename CasterType::Pointer caster = CasterType::New();
    caster->SetInput( coefficients );
    caster->GraftOutput( GPUCoefficients );
    caster->Update();

    coefficientArray[ i ] = GPUCoefficients;
  }
}


//------------------------------------------------------------------------------
template< typename BSplineTransformType >
void
DefineBSplineParameters( typename BSplineTransformType::ParametersType & parameters,
  const typename BSplineTransformType::Pointer & transform,
  const std::string & parametersFileName )
{
  const unsigned int numberOfParameters = transform->GetNumberOfParameters();
  const unsigned int Dimension          = BSplineTransformType::SpaceDimension;
  const unsigned int numberOfNodes      = numberOfParameters / Dimension;

  parameters.SetSize( numberOfParameters );

  // Open file and read parameters
  std::ifstream infile;
  infile.open( parametersFileName.c_str() );
  for( unsigned int n = 0; n < numberOfNodes * Dimension; n++ )
  {
    double parValue;
    infile >> parValue;
    parameters[ n ] = parValue;
  }
  infile.close();
}


//------------------------------------------------------------------------------
template< typename InputImageType, typename OutputImageType >
int ProcessImage( const Parameters & _parameters );

//------------------------------------------------------------------------------
// Testing GPU BSplineTransform initialization.
// The filter takes an input image and produces an output image.
// We compare the CPU and GPU output image using RMSE and speed.
int
main( int argc, char * argv[] )
{
  // Setup for debugging and create log
  itk::SetupForDebugging();
  itk::CreateOpenCLLogger( "BSplineTransformTest" );

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
  parser->MarkArgumentAsRequired( "-inpar", "The input parameters file name" );

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
  parser->GetCommandLineArgument( "-in", parameters.inputFileNames );
  parser->GetCommandLineArgument( "-inpar", parameters.inputParametersFileName );
  parameters.outputFileNames.push_back( parameters.inputFileNames[ 0 ].substr( 0, parameters.inputFileNames[ 0 ].rfind(
    "." ) ) + "-out-cpu.mha" );
  parameters.outputFileNames.push_back( parameters.inputFileNames[ 0 ].substr( 0, parameters.inputFileNames[ 0 ].rfind(
    "." ) ) + "-out-gpu.mha" );
  parser->GetCommandLineArgument( "-out", parameters.outputFileNames );
  parameters.outputWrite = !( parser->ArgumentExists( "-nooutput" ) );
  parser->GetCommandLineArgument( "-outindex", parameters.outputIndex );
  parser->GetCommandLineArgument( "-outlog", parameters.logFileName );
  const bool retruntimes = parser->GetCommandLineArgument( "-runtimes", parameters.runTimes );

  parser->GetCommandLineArgument( "-rmse", parameters.allowedRMSerror );
  parameters.skipCPU = parser->ArgumentExists( "-skipcpu" );
  parameters.skipGPU = parser->ArgumentExists( "-skipgpu" );

  // Threads.
  unsigned int maximumNumberOfThreads = itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads();
  parser->GetCommandLineArgument( "-threads", maximumNumberOfThreads );
  itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads( maximumNumberOfThreads );

  // Check if the required arguments are given.
  if( retruntimes && parameters.runTimes < 1 )
  {
    std::cerr << "ERROR: \"-runtimes\" parameter should be more or equal one." << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Determine image properties.
  std::string                 ComponentType = "short";
  std::string                 PixelType; //we don't use this
  unsigned int                Dimension          = 2;
  unsigned int                NumberOfComponents = 1;
  std::vector< unsigned int > imagesize( Dimension, 0 );
  int                         retgip = GetImageProperties(
    parameters.inputFileNames[ 0 ],
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
    // 1D
    //run( ProcessImage, char, float, 1 );
    //run( ProcessImage, unsigned char, float, 1 );
    run( ProcessImage, short, float, 1 );
    //run( ProcessImage, float, float, 1 );

    // 2D
    //run( ProcessImage, char, float, 2 );
    //run( ProcessImage, unsigned char, float, 2 );
    run( ProcessImage, short, float, 2 );
    //run( ProcessImage, float, float, 2 );

    // 3D
    //run( ProcessImage, char, float, 3 );
    //run( ProcessImage, unsigned char, float, 3 );
    run( ProcessImage, short, float, 3 );
    //run( ProcessImage, float, float, 3 );
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
  const unsigned int ImageDim       = (unsigned int)InputImageType::ImageDimension;
  const unsigned int SpaceDimension = ImageDim;

  // Typedefs
  typedef itk::BSplineTransform< float, SpaceDimension, 3 > TransformType;
  typedef typename TransformType::CoefficientImageArray     CoefficientImageArray;
  typedef typename TransformType::ImageType                 CoefficientImageType;
  typedef typename CoefficientImageType::Pointer            CoefficientImagePointer;
  typedef itk::ImageFileReader< InputImageType >            CPUReaderType;

  // CPU Reader
  bool updateExceptionCPU = false;
  typename CPUReaderType::Pointer CPUReader = CPUReaderType::New();
  CPUReader->SetFileName( _parameters.inputFileNames[ 0 ] );
  try
  {
    CPUReader->Update();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "Caught ITK exception during CPUReader->Update(): " << e << std::endl;
    updateExceptionCPU = updateExceptionCPU || true;
  }

  const typename InputImageType::ConstPointer inputImage      = CPUReader->GetOutput();
  const typename InputImageType::SpacingType inputSpacing     = inputImage->GetSpacing();
  const typename InputImageType::PointType inputOrigin        = inputImage->GetOrigin();
  const typename InputImageType::DirectionType inputDirection = inputImage->GetDirection();
  const typename InputImageType::RegionType inputRegion       = inputImage->GetBufferedRegion();
  const typename InputImageType::SizeType inputSize           = inputRegion.GetSize();

  typedef typename TransformType::MeshSizeType MeshSizeType;
  MeshSizeType meshSize;
  meshSize.Fill( 4 );

  typedef typename TransformType::PhysicalDimensionsType PhysicalDimensionsType;
  PhysicalDimensionsType fixedDimensions;
  for( unsigned int d = 0; d < ImageDim; d++ )
  {
    fixedDimensions[ d ] = inputSpacing[ d ] * ( inputSize[ d ] - 1.0 );
  }

  // Create CPUTransform
  typename TransformType::ParametersType parameters;
  typename TransformType::Pointer CPUTransform = TransformType::New();
  CPUTransform->SetTransformDomainOrigin( inputOrigin );
  CPUTransform->SetTransformDomainDirection( inputDirection );
  CPUTransform->SetTransformDomainPhysicalDimensions( fixedDimensions );
  CPUTransform->SetTransformDomainMeshSize( meshSize );

  // Read and set parameters
  DefineBSplineParameters< TransformType >
    ( parameters, CPUTransform, _parameters.inputParametersFileName );

  itk::TimeProbe cputimer;
  cputimer.Start();

  if( !_parameters.skipCPU )
  {
    CPUTransform->SetParameters( parameters );
  }

  cputimer.Stop();
  std::cout << "CPU " << CPUTransform->GetNameOfClass() << " took " << cputimer.GetMean() << " seconds." << std::endl;

  // GPU part
  bool updateExceptionGPU = false;
  typename TransformType::Pointer GPUTransform;
  itk::TimeProbe gputimer;

  // Define BSplineTransformCoefficientImageArray
  typedef itk::GPUImage< float,
    InputImageType::ImageDimension > GPUBSplineTransformCoefficientImageType;
  typedef typename GPUBSplineTransformCoefficientImageType::Pointer
    GPUBSplineTransformCoefficientImagePointer;
  typedef itk::FixedArray< GPUBSplineTransformCoefficientImagePointer,
    InputImageType::ImageDimension > BSplineTransformCoefficientImageArray;

  BSplineTransformCoefficientImageArray coefficientArray;

  if( !_parameters.skipGPU )
  {
    // register object factory for GPU image and filter
    typedef typelist::MakeTypeList< short, float >::Type OCLImageTypes;
    itk::GPUImageFactory2< OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();
    itk::GPUBSplineTransformFactory2< OCLImageDims >
    ::RegisterOneFactory();
    itk::GPUCastImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();

    // Construct the GPUTransform
    // Use a try/catch, because construction of this filter will trigger
    // OpenCL compilation, which may fail.
    try
    {
      GPUTransform = TransformType::New();
    }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "Caught ITK exception during GPUTransform::New(): " << e << std::endl;
      updateExceptionGPU = updateExceptionGPU || true;
    }

    if( !updateExceptionGPU )
    {
      GPUTransform->SetTransformDomainOrigin( inputOrigin );
      GPUTransform->SetTransformDomainDirection( inputDirection );
      GPUTransform->SetTransformDomainPhysicalDimensions( fixedDimensions );
      GPUTransform->SetTransformDomainMeshSize( meshSize );
    }

    gputimer.Start();

    if( !updateExceptionGPU )
    {
      GPUTransform->SetParameters( parameters );
    }

    if( !updateExceptionGPU )
    {
      try
      {
        CopyCoefficientImagesToGPU< float, InputImageType::ImageDimension, 3 >(
          GPUTransform.GetPointer(), coefficientArray );
      }
      catch( itk::ExceptionObject & e )
      {
        std::cerr << "Caught ITK exception during GPUFilter->Update(): " << e << std::endl;
        updateExceptionGPU = updateExceptionGPU || true;
      }
    }

    gputimer.Stop();
    std::cout << "GPU " << GPUTransform->GetNameOfClass() << " took " << gputimer.GetMean() << " seconds." << std::endl;
  }

  CoefficientImageArray cpuCoefficientImageArray;
  if( !_parameters.skipCPU )
  {
    cpuCoefficientImageArray = CPUTransform->GetCoefficientImages();
  }

  bool                  testPassed = true;
  std::vector< double > RMSerrors;
  std::vector< double > RMSrelatives;

  if( updateExceptionCPU || updateExceptionGPU )
  {
    testPassed = false;
  }
  else
  {
    if( !_parameters.skipCPU && !_parameters.skipGPU )
    {
      for( unsigned int i = 0; i < cpuCoefficientImageArray.Size(); i++ )
      {
        CoefficientImagePointer                    cpucoefficients = cpuCoefficientImageArray[ i ];
        GPUBSplineTransformCoefficientImagePointer gpucoefficients = coefficientArray[ i ];

        // RMS Error, RMSE Relative
        double RMSrelative = 0.0;
        double RMSerror
          = itk::ComputeRMSE< double, CoefficientImageType, GPUBSplineTransformCoefficientImageType >(
          cpucoefficients, gpucoefficients, RMSrelative );

        RMSerrors.push_back( RMSerror );
        RMSrelatives.push_back( RMSrelative );

        testPassed |= ( RMSerror <= _parameters.allowedRMSerror );
      }
    }
  }

  if( !_parameters.skipCPU && !_parameters.skipGPU && !updateExceptionGPU )
  {
    std::cout << "RMS Error: " << std::fixed << std::setprecision( 8 )
              << RMSerrors[ _parameters.outputIndex ] << std::endl;
  }

  // Write output
  if( _parameters.outputWrite )
  {
    if( _parameters.outputIndex > cpuCoefficientImageArray.Size() )
    {
      std::cerr << "ERROR: The outputIndex " << _parameters.outputIndex
                << " larger than coefficient array size." << std::endl;
      return EXIT_FAILURE;
    }

    if( !_parameters.skipCPU )
    {
      CoefficientImagePointer cpucoefficients = cpuCoefficientImageArray[ _parameters.outputIndex ];

      // Write output CPU image
      typedef itk::ImageFileWriter< CoefficientImageType > CPUWriterType;
      typename CPUWriterType::Pointer writerCPU = CPUWriterType::New();
      writerCPU->SetInput( cpucoefficients );
      writerCPU->SetFileName( _parameters.outputFileNames[ 0 ] );
      writerCPU->Update();
    }

    if( !_parameters.skipGPU && !updateExceptionGPU )
    {
      GPUBSplineTransformCoefficientImagePointer gpucoefficients = coefficientArray[ _parameters.outputIndex ];

      // Write output GPU image
      typedef itk::ImageFileWriter< GPUBSplineTransformCoefficientImageType > GPUWriterType;
      typename GPUWriterType::Pointer writerGPU = GPUWriterType::New();
      writerGPU->SetInput( gpucoefficients );
      writerGPU->SetFileName( _parameters.outputFileNames[ 1 ] );
      writerGPU->Update();
    }
  }

  // Write log
  if( _parameters.outputLog )
  {
    std::string comments;
    if( updateExceptionCPU || updateExceptionGPU )
    {
      comments.append( ", Exception during update" );
    }

    double RMSError    = 0.0;
    double RMSrelative = 0.0;
    if( RMSerrors.size() > 0 && RMSrelatives.size() > 0 )
    {
      RMSError    = RMSerrors[ _parameters.outputIndex ];
      RMSrelative = RMSrelatives[ _parameters.outputIndex ];
    }

    std::string className( "na" );
    if( !_parameters.skipGPU )
    {
      className = GPUTransform->GetNameOfClass();
    }

    itk::WriteLog< InputImageType >(
      _parameters.logFileName, ImageDim, inputSize,
      RMSError, RMSrelative,
      testPassed, updateExceptionGPU,
      1, _parameters.runTimes,
      className,
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
