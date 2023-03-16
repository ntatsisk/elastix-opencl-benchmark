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

#include "itkContinuousIndex.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkBinaryThresholdImageFilter.h"

// GPU include files
#include "itkGPUResampleImageFilter.h"

// GPU copiers
#include "itkGPUTransformCopier.h"
#include "itkGPUCompositeTransformCopier.h"
#include "itkGPUInterpolatorCopier.h"

// GPU factory includes
#include "itkGPUImageFactory.h"
#include "itkGPUResampleImageFilterFactory.h"
#include "itkGPUCastImageFilterFactory.h"

// GPU transform factory includes
#include "itkGPUAffineTransformFactory.h"
#include "itkGPUTranslationTransformFactory.h"
#include "itkGPUBSplineTransformFactory.h"
#include "itkGPUEuler2DTransformFactory.h"
#include "itkGPUEuler3DTransformFactory.h"
#include "itkGPUSimilarity2DTransformFactory.h"
#include "itkGPUSimilarity3DTransformFactory.h"
#include "itkGPUCompositeTransformFactory.h"

// GPU interpolate factory includes
#include "itkGPUNearestNeighborInterpolateImageFunctionFactory.h"
#include "itkGPULinearInterpolateImageFunctionFactory.h"
#include "itkGPUBSplineInterpolateImageFunctionFactory.h"
#include "itkGPUBSplineDecompositionImageFilterFactory.h"

//------------------------------------------------------------------------------
std::string
GetHelpString( void )
{
  std::stringstream ss;

  ss << "Usage:" << std::endl
     << "  -in           input file name" << std::endl
     << "  [-i]          interpolator, one of {NearestNeighbor, Linear, BSpline}, default NearestNeighbor\n"
     << "  [-t]          transforms, one of {Affine, Translation, BSpline, Euler, Similarity}"
     << " or combinations with option \"-c\", default Affine\n"
     << "  [-c]          use combo transform, default false" << std::endl
     << "  [-p]          parameter file for the B-spline transform" << std::endl
     << "  [-out]        output file names.(outputCPU outputGPU)" << std::endl
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
#define run( function, ttype, etype, stype, type0, type1, dim )                                \
  if( ComponentType == # type0 && Dimension == dim )                                           \
  {                                                                                            \
    typedef itk::Image< type0, dim > InputImageType;                                           \
    typedef itk::Image< type1, dim > OutputImageType;                                          \
    supported = true;                                                                          \
    if( !noSupportFor1D )                                                                      \
    {                                                                                          \
      result = function< ttype, etype, stype, InputImageType, OutputImageType >( parameters ); \
    }                                                                                          \
  }

//------------------------------------------------------------------------------
namespace
{
class Parameters
{
public:

  // Constructor
  Parameters() :
    interpolator( "NearestNeighbor" ),
    parametersFileName( "" ),
    logFileName( "CPUGPULog.txt" )
  {
    useCompression = false;
    outputWrite    = true;
    outputLog      = true;

    allowedRMSerror = 0.0;
    threshold       = 1.0;
    runTimes        = 1;
    skipCPU         = false;
    skipGPU         = false;

    useComboTransform       = false;
    splineOrderInterpolator = 3;
    transforms.push_back( "Affine" );
  }


  bool         useCompression;
  bool         outputWrite;
  bool         outputLog;
  bool         skipCPU;
  bool         skipGPU;
  float        allowedRMSerror;
  float        threshold;
  unsigned int runTimes;

  // Filter
  bool                       useComboTransform;
  std::string                interpolator;
  unsigned int               splineOrderInterpolator;
  std::vector< std::string > transforms;
  std::string                parametersFileName;

  // Files
  std::string                inputFileName;
  std::string                logFileName;
  std::vector< std::string > outputFileNames;
};

}

//------------------------------------------------------------------------------
template< typename TransformType, typename InputImageType >
std::string
GetTransformName( typename TransformType::Pointer & transform )
{
  std::ostringstream ost;

  ost << transform->GetNameOfClass();

  // Try float transform
  typedef itk::CompositeTransform< float, InputImageType::ImageDimension > FloatCompositeTransformType;
  const FloatCompositeTransformType * floatCompositeTransform
    = dynamic_cast< const FloatCompositeTransformType * >( transform.GetPointer() );

  if( floatCompositeTransform )
  {
    ost << " [";
    for( std::size_t i = 0; i < floatCompositeTransform->GetNumberOfTransforms(); i++ )
    {
      ost << floatCompositeTransform->GetNthTransform( i )->GetNameOfClass();
      if( i != floatCompositeTransform->GetNumberOfTransforms() - 1 )
      {
        ost << ", ";
      }
    }
    ost << "]";
  }
  else
  {
    // Try double transform
    typedef itk::CompositeTransform< double, InputImageType::ImageDimension > DoubleCompositeTransformType;
    const DoubleCompositeTransformType * doubleCompositeTransform
      = dynamic_cast< const DoubleCompositeTransformType * >( transform.GetPointer() );

    if( doubleCompositeTransform )
    {
      ost << " [";
      for( std::size_t i = 0; i < doubleCompositeTransform->GetNumberOfTransforms(); i++ )
      {
        ost << doubleCompositeTransform->GetNthTransform( i )->GetNameOfClass();
        if( i != doubleCompositeTransform->GetNumberOfTransforms() - 1 )
        {
          ost << ", ";
        }
      }
      ost << "]";
    }
  }

  return ost.str();
}


//------------------------------------------------------------------------------
template< typename InputImageType >
typename InputImageType::PointType
ComputeCenterOfTheImage( const typename InputImageType::ConstPointer & image )
{
  const unsigned int Dimension = image->GetImageDimension();

  const typename InputImageType::SizeType size   = image->GetLargestPossibleRegion().GetSize();
  const typename InputImageType::IndexType index = image->GetLargestPossibleRegion().GetIndex();

  typedef itk::ContinuousIndex< double, InputImageType::ImageDimension > ContinuousIndexType;
  ContinuousIndexType centerAsContInd;
  for( std::size_t i = 0; i < Dimension; i++ )
  {
    centerAsContInd[ i ]
      = static_cast< double >( index[ i ] )
      + static_cast< double >( size[ i ] - 1 ) / 2.0;
  }

  typename InputImageType::PointType center;
  image->TransformContinuousIndexToPhysicalPoint( centerAsContInd, center );
  return center;
}


//------------------------------------------------------------------------------
template< typename InputImageType, typename OutputImageType >
void
DefineOutputImageProperties(
  const typename InputImageType::ConstPointer & image,
  typename OutputImageType::SpacingType & outputSpacing,
  typename OutputImageType::PointType & outputOrigin,
  typename OutputImageType::DirectionType & outputDirection,
  typename OutputImageType::SizeType & outputSize,
  typename InputImageType::PixelType & minValue,
  typename OutputImageType::PixelType & defaultValue )
{
  typedef typename InputImageType::SizeType::SizeValueType SizeValueType;
  const typename InputImageType::SpacingType inputSpacing     = image->GetSpacing();
  const typename InputImageType::PointType inputOrigin        = image->GetOrigin();
  const typename InputImageType::DirectionType inputDirection = image->GetDirection();
  const typename InputImageType::SizeType inputSize           = image->GetBufferedRegion().GetSize();

  const unsigned int ImageDim   = (unsigned int)InputImageType::ImageDimension;
  const double       scaleFixed = 0.9;
  for( unsigned int i = 0; i < ImageDim; i++ )
  {
    outputSpacing[ i ] = inputSpacing[ i ] * scaleFixed;
    outputOrigin[ i ]  = inputOrigin[ i ] * scaleFixed;

    for( unsigned int j = 0; j < ImageDim; j++ )
    {
      outputDirection[ i ][ j ] = inputDirection[ i ][ j ];
    }
    outputSize[ i ] = itk::Math::Round< SizeValueType >( inputSize[ i ] * scaleFixed );
  }

  typedef itk::MinimumMaximumImageCalculator< InputImageType > MinimumMaximumImageCalculatorType;
  typename MinimumMaximumImageCalculatorType::Pointer calculator = MinimumMaximumImageCalculatorType::New();
  calculator->SetImage( image );
  calculator->ComputeMinimum();

  minValue     = calculator->GetMinimum();
  defaultValue = minValue - 2;
}


//------------------------------------------------------------------------------
template< typename InterpolatorType >
void
DefineInterpolator( typename InterpolatorType::Pointer & interpolator,
  const std::string & interpolatorName,
  const unsigned int splineOrderInterpolator,
  const bool updateException )
{
  // Interpolator typedefs
  typedef typename InterpolatorType::InputImageType InputImageType;
  typedef typename InterpolatorType::CoordRepType   CoordRepType;
  typedef CoordRepType                              CoefficientType;

  // Typedefs for all interpolators
  typedef itk::NearestNeighborInterpolateImageFunction<
    InputImageType, CoordRepType > NearestNeighborInterpolatorType;
  typedef itk::LinearInterpolateImageFunction<
    InputImageType, CoordRepType > LinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction<
    InputImageType, CoordRepType, CoefficientType > BSplineInterpolatorType;

  if( !updateException )
  {
    if( interpolatorName == "NearestNeighbor" )
    {
      typename NearestNeighborInterpolatorType::Pointer tmpInterpolator
                   = NearestNeighborInterpolatorType::New();
      interpolator = tmpInterpolator;
    }
    else if( interpolatorName == "Linear" )
    {
      typename LinearInterpolatorType::Pointer tmpInterpolator
                   = LinearInterpolatorType::New();
      interpolator = tmpInterpolator;
    }
    else if( interpolatorName == "BSpline" )
    {
      typename BSplineInterpolatorType::Pointer tmpInterpolator
        = BSplineInterpolatorType::New();
      tmpInterpolator->SetSplineOrder( splineOrderInterpolator );
      interpolator = tmpInterpolator;
    }
  }
}


//------------------------------------------------------------------------------
template< typename AffineTransformType >
void
DefineAffineParameters( typename AffineTransformType::ParametersType & parameters )
{
  const unsigned int Dimension = AffineTransformType::InputSpaceDimension;

  // Setup parameters
  parameters.SetSize( Dimension * Dimension + Dimension );
  std::size_t par = 0;
  if( Dimension == 2 )
  {
    const double matrix[] =
    {
      0.9, 0.1, // matrix part
      0.2, 1.1, // matrix part
      0.0, 0.0, // translation
    };

    for( std::size_t i = 0; i < 6; i++ )
    {
      parameters[ par++ ] = matrix[ i ];
    }
  }
  else if( Dimension == 3 )
  {
    const double matrix[] =
    {
      1.0, -0.045, 0.02,   // matrix part
      0.0, 1.0, 0.0,       // matrix part
      -0.075, 0.09, 1.0,   // matrix part
      -3.02, 1.3, -0.045   // translation
    };

    for( std::size_t i = 0; i < 12; i++ )
    {
      parameters[ par++ ] = matrix[ i ];
    }
  }
}


//------------------------------------------------------------------------------
template< typename TranslationTransformType >
void
DefineTranslationParameters( const std::size_t transformIndex,
  typename TranslationTransformType::ParametersType & parameters )
{
  const std::size_t Dimension = TranslationTransformType::SpaceDimension;

  // Setup parameters
  parameters.SetSize( Dimension );
  for( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[ i ] = (double)i * (double)transformIndex + (double)transformIndex;
  }
}


//------------------------------------------------------------------------------
template< typename BSplineTransformType >
void
DefineBSplineParameters( const std::size_t transformIndex,
  typename BSplineTransformType::ParametersType & parameters,
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

  // Skip number of elements to make unique coefficients per each transformIndex
  for( std::size_t n = 0; n < transformIndex; n++ )
  {
    double parValue;
    infile >> parValue;
  }

  // Read it
  for( std::size_t n = 0; n < numberOfNodes * Dimension; n++ )
  {
    double parValue;
    infile >> parValue;
    parameters[ n ] = parValue;
  }

  infile.close();
}


//------------------------------------------------------------------------------
template< typename EulerTransformType >
void
DefineEulerParameters( const std::size_t transformIndex,
  typename EulerTransformType::ParametersType & parameters )
{
  const std::size_t Dimension = EulerTransformType::InputSpaceDimension;

  // Setup parameters
  // 2D: angle 1, translation 2
  // 3D: 6 angle, translation 3
  parameters.SetSize( EulerTransformType::ParametersDimension );

  // Angle
  const double angle = (double)transformIndex * -0.05;

  std::size_t par = 0;
  if( Dimension == 2 )
  {
    // See implementation of Rigid2DTransform::SetParameters()
    parameters[ 0 ] = angle;
    ++par;
  }
  else if( Dimension == 3 )
  {
    // See implementation of Rigid3DTransform::SetParameters()
    for( std::size_t i = 0; i < 3; i++ )
    {
      parameters[ par ] = angle;
      ++par;
    }
  }

  for( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[ i + par ] = (double)i * (double)transformIndex + (double)transformIndex;
  }
}


//------------------------------------------------------------------------------
template< typename SimilarityTransformType >
void
DefineSimilarityParameters( const std::size_t transformIndex,
  typename SimilarityTransformType::ParametersType & parameters )
{
  const std::size_t Dimension = SimilarityTransformType::InputSpaceDimension;

  // Setup parameters
  // 2D: 2 translation, angle 1, scale 1
  // 3D: 3 translation, angle 3, scale 1
  parameters.SetSize( SimilarityTransformType::ParametersDimension );

  // Scale, Angle
  const double scale = ( (double)transformIndex + 1.0 ) * 0.05 + 1.0;
  const double angle = (double)transformIndex * -0.06;

  if( Dimension == 2 )
  {
    // See implementation of Similarity2DTransform::SetParameters()
    parameters[ 0 ] = scale;
    parameters[ 1 ] = angle;
  }
  else if( Dimension == 3 )
  {
    // See implementation of Similarity3DTransform::SetParameters()
    for( std::size_t i = 0; i < Dimension; i++ )
    {
      parameters[ i ] = angle;
    }
    parameters[ 6 ] = scale;
  }

  // Translation
  for( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[ i + Dimension ] = -1.0 * ( (double)i * (double)transformIndex + (double)transformIndex );
  }
}


//------------------------------------------------------------------------------
// This helper function completely set the transform
// We are using ITK transforms:
// TransformType, AffineTransformType, TranslationTransformType
// BSplineTransformType, EulerTransformType, SimilarityTransformType,
// CompositeTransform
template< typename TransformType, typename AffineTransformType,
typename TranslationTransformType, typename BSplineTransformType,
typename EulerTransformType, typename SimilarityTransformType,
typename CompositeTransformType, typename InputImageType >
void
SetTransform( const std::size_t transformIndex,
  const std::string & transformName,
  typename TransformType::Pointer & transform,
  typename CompositeTransformType::Pointer & compositeTransform,
  const typename InputImageType::ConstPointer & image,
  std::vector< typename BSplineTransformType::ParametersType > & bsplineParameters,
  const std::string & parametersFileName )
{
  if( transformName == "Affine" )
  {
    // Create Affine transform
    typename AffineTransformType::Pointer affineTransform
      = AffineTransformType::New();

    // Define and set affine parameters
    typename AffineTransformType::ParametersType parameters;
    DefineAffineParameters< AffineTransformType >( parameters );
    affineTransform->SetParameters( parameters );
    if( compositeTransform.IsNull() )
    {
      transform = affineTransform;
    }
    else
    {
      compositeTransform->AddTransform( affineTransform );
    }
  }
  else if( transformName == "Translation" )
  {
    // Create Translation transform
    typename TranslationTransformType::Pointer translationTransform
      = TranslationTransformType::New();

    // Define and set translation parameters
    typename TranslationTransformType::ParametersType parameters;
    DefineTranslationParameters< TranslationTransformType >
      ( transformIndex, parameters );
    translationTransform->SetParameters( parameters );
    if( compositeTransform.IsNull() )
    {
      transform = translationTransform;
    }
    else
    {
      compositeTransform->AddTransform( translationTransform );
    }
  }
  else if( transformName == "BSpline" )
  {
    const unsigned int Dimension = image->GetImageDimension();
    const typename InputImageType::SpacingType inputSpacing     = image->GetSpacing();
    const typename InputImageType::PointType inputOrigin        = image->GetOrigin();
    const typename InputImageType::DirectionType inputDirection = image->GetDirection();
    const typename InputImageType::SizeType inputSize           = image->GetBufferedRegion().GetSize();

    typedef typename BSplineTransformType::MeshSizeType MeshSizeType;
    MeshSizeType gridSize;
    gridSize.Fill( 4 );

    typedef typename BSplineTransformType::PhysicalDimensionsType PhysicalDimensionsType;
    PhysicalDimensionsType gridSpacing;
    for( unsigned int d = 0; d < Dimension; d++ )
    {
      gridSpacing[ d ] = inputSpacing[ d ] * ( inputSize[ d ] - 1.0 );
    }

    // Create BSpline transform
    typename BSplineTransformType::Pointer bsplineTransform
      = BSplineTransformType::New();

    // Set grid properties
    bsplineTransform->SetTransformDomainOrigin( inputOrigin );
    bsplineTransform->SetTransformDomainDirection( inputDirection );
    bsplineTransform->SetTransformDomainPhysicalDimensions( gridSpacing );
    bsplineTransform->SetTransformDomainMeshSize( gridSize );

    // Define and set b-spline parameters
    typename BSplineTransformType::ParametersType parameters;
    DefineBSplineParameters< BSplineTransformType >
      ( transformIndex, parameters, bsplineTransform, parametersFileName );

    // Keep them in memory first by copying to the bsplineParameters array
    bsplineParameters.push_back( parameters );
    const std::size_t indexAt = bsplineParameters.size() - 1;

    // Do not set parameters, the will be destroyed going out of scope
    // instead, set the ones from the bsplineParameters
    bsplineTransform->SetParameters( bsplineParameters[ indexAt ] );
    if( compositeTransform.IsNull() )
    {
      transform = bsplineTransform;
    }
    else
    {
      compositeTransform->AddTransform( bsplineTransform );
    }
  }
  else if( transformName == "Euler" )
  {
    // Create Euler transform
    typename EulerTransformType::Pointer eulerTransform
      = EulerTransformType::New();

    // Compute and set center
    const typename InputImageType::PointType center
      = ComputeCenterOfTheImage< InputImageType >( image );
    eulerTransform->SetCenter( center );

    // Define and set euler parameters
    typename EulerTransformType::ParametersType parameters;
    DefineEulerParameters< EulerTransformType >
      ( transformIndex, parameters );
    eulerTransform->SetParameters( parameters );
    if( compositeTransform.IsNull() )
    {
      transform = eulerTransform;
    }
    else
    {
      compositeTransform->AddTransform( eulerTransform );
    }
  }
  else if( transformName == "Similarity" )
  {
    // Create Similarity transform
    typename SimilarityTransformType::Pointer similarityTransform
      = SimilarityTransformType::New();

    // Compute and set center
    const typename InputImageType::PointType center
      = ComputeCenterOfTheImage< InputImageType >( image );
    similarityTransform->SetCenter( center );

    // Define and set similarity parameters
    typename SimilarityTransformType::ParametersType parameters;
    DefineSimilarityParameters< SimilarityTransformType >
      ( transformIndex, parameters );
    similarityTransform->SetParameters( parameters );
    if( compositeTransform.IsNull() )
    {
      transform = similarityTransform;
    }
    else
    {
      compositeTransform->AddTransform( similarityTransform );
    }
  }
}


//------------------------------------------------------------------------------
// This helper function completely set the transform
// We are using ITK transforms:
// TransformType, AffineTransformType, TranslationTransformType
// BSplineTransformType, EulerTransformType, SimilarityTransformType,
// CompositeTransform
template< typename TransformType, typename AffineTransformType,
typename TranslationTransformType, typename BSplineTransformType,
typename EulerTransformType, typename SimilarityTransformType,
typename CompositeTransformType, typename InputImageType >
void
DefineTransform( typename TransformType::Pointer & transform,
  const Parameters & parameters,
  std::vector< typename BSplineTransformType::ParametersType > & bsplineParameters,
  const typename InputImageType::ConstPointer & image )
{
  if( !parameters.useComboTransform )
  {
    typename CompositeTransformType::Pointer dummy;
    SetTransform<
    TransformType, AffineTransformType, TranslationTransformType,
    BSplineTransformType, EulerTransformType, SimilarityTransformType,
    CompositeTransformType, InputImageType >
      ( 0, parameters.transforms[ 0 ], transform, dummy,
      image, bsplineParameters, parameters.parametersFileName );
  }
  else
  {
    typename CompositeTransformType::Pointer compositeTransform
              = CompositeTransformType::New();
    transform = compositeTransform;

    for( std::size_t i = 0; i < parameters.transforms.size(); i++ )
    {
      SetTransform<
      TransformType, AffineTransformType, TranslationTransformType,
      BSplineTransformType, EulerTransformType, SimilarityTransformType,
      CompositeTransformType, InputImageType >
        ( i, parameters.transforms[ i ], transform, compositeTransform,
        image, bsplineParameters, parameters.parametersFileName );
    }
  }
}


//------------------------------------------------------------------------------
template< typename TransformPrecisionType, typename EulerTransformType, typename SimilarityTransformType,
class InputImageType, typename OutputImageType >
int ProcessImage( const Parameters & _parameters );

//------------------------------------------------------------------------------
// This test compares the CPU with the GPU version of the ResampleImageFilter.
// The filter takes an input image and produces an output image.
// We compare the CPU and GPU output image using RMSE and speed.
//
// The following ITK interpolations are supported:
// itk::NearestNeighborInterpolateImageFunction
// itk::LinearInterpolateImageFunction
// itk::BSplineInterpolateImageFunction
//
// The following ITK transforms are supported:
// itk::CompositeTransform
// itk::AffineTransform
// itk::BSplineTransform
// itk::Euler2DTransform
// itk::Euler3DTransform
// itk::Similarity2DTransform
// itk::Similarity3DTransform
// itk::TranslationTransform
//
int
main( int argc, char * argv[] )
{
  // Setup for debugging and create log
  itk::SetupForDebugging();
  itk::CreateOpenCLLogger( "ResampleImageFilterTest" );

  // Create and check OpenCL context
  if( !itk::CreateContext() )
  {
    return EXIT_FAILURE;
  }

  /** Check for the device 'double' support. The check has 'double' ensures that
   * the OpenCL device is from around 2009 with some decent support for the OpenCL.
   * For NVIDIA that is generation since GT200 and later.
   * for ATI that is generation since HD 4730, 5830, 6930, 7730, R7 240 and later.
   * We are making it minimum requirements for elastix with OpenCL for now. */
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if( !context->GetDefaultDevice().HasDouble() )
  {
    std::cerr << "Your OpenCL device: " << context->GetDefaultDevice().GetName()
              << ", does not support 'double' computations. Consider updating it." << std::endl;
    itk::ReleaseContext();
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
  parser->GetCommandLineArgument( "-outlog", parameters.logFileName );
  const bool retruntimes = parser->GetCommandLineArgument( "-runtimes", parameters.runTimes );

  parser->GetCommandLineArgument( "-rmse", parameters.allowedRMSerror );
  parser->GetCommandLineArgument( "-threshold", parameters.threshold );

  parser->GetCommandLineArgument( "-i", parameters.interpolator );
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

  if( parameters.interpolator != "NearestNeighbor"
    && parameters.interpolator != "Linear"
    && parameters.interpolator != "BSpline" )
  {
    std::cerr << "ERROR: interpolator \"-i\" should be one of {NearestNeighbor, Linear, BSpline}."
              << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Get transform argument
  parameters.useComboTransform = parser->ArgumentExists( "-c" );
  parser->GetCommandLineArgument( "-t", parameters.transforms );

  // check that use combo transform provided when used multiple transforms
  if( parameters.transforms.size() > 1 && !parameters.useComboTransform )
  {
    std::cerr << "ERROR: for multiple transforms option \"-c\" should provided." << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // check for supported transforms
  for( std::size_t i = 0; i < parameters.transforms.size(); i++ )
  {
    const std::string transformName = parameters.transforms[ i ];
    if( transformName != "Affine"
      && transformName != "Translation"
      && transformName != "BSpline"
      && transformName != "Euler"
      && transformName != "Similarity" )
    {
      std::cerr << "ERROR: transforms \"-t\" should be one of "
                << "{Affine, Translation, BSpline, Euler, Similarity}"
                << " or combination of them." << std::endl;
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }
  }
  // Get BSpline transform parameters file
  for( std::size_t i = 0; i < parameters.transforms.size(); i++ )
  {
    if( parameters.transforms[ i ] == "BSpline" )
    {
      const bool retp = parser->GetCommandLineArgument( "-p", parameters.parametersFileName );
      if( !retp )
      {
        std::cerr << "ERROR: You should specify parameters file \"-p\" for the B-spline transform." << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
      }
    }
  }

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

  // Check for transforms that support only 2D/3D not 1D
  bool noSupportFor1D = false;
  if( Dimension == 1 )
  {
    for( std::size_t i = 0; i < parameters.transforms.size(); i++ )
    {
      const std::string transformName = parameters.transforms[ i ];
      if( transformName == "Euler" || transformName == "Similarity" )
      {
        noSupportFor1D = true;
        break;
      }
    }
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

  typedef float TransformPrecisionType;

  // Dummy Euler1DTransformType and Similarity1DTransformType for compiler
  typedef itk::MatrixOffsetTransformBase< TransformPrecisionType, 1, 1 > Euler1DTransformType;
  typedef itk::MatrixOffsetTransformBase< TransformPrecisionType, 1, 1 > Similarity1DTransformType;

  // Typedefs for Euler2D/3D Similarity2D/3D
  typedef itk::Euler2DTransform< TransformPrecisionType >      Euler2DTransformType;
  typedef itk::Similarity2DTransform< TransformPrecisionType > Similarity2DTransformType;
  typedef itk::Euler3DTransform< TransformPrecisionType >      Euler3DTransformType;
  typedef itk::Similarity3DTransform< TransformPrecisionType > Similarity3DTransformType;

  // Run the program.
  bool supported = false;
  int  result    = EXIT_SUCCESS;
  try
  {
    // 1D
    run( ProcessImage, TransformPrecisionType, Euler1DTransformType, Similarity1DTransformType, short, short, 1 );

    // 2D
    run( ProcessImage, TransformPrecisionType, Euler2DTransformType, Similarity2DTransformType, short, short, 2 );

    // 3D
    run( ProcessImage, TransformPrecisionType, Euler3DTransformType, Similarity3DTransformType, short, short, 3 );
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
template< typename TransformPrecisionType, typename EulerTransformType, typename SimilarityTransformType,
class InputImageType, typename OutputImageType >
int
ProcessImage( const Parameters & _parameters )
{
  // Images Typedefs
  const unsigned int ImageDim = (unsigned int)InputImageType::ImageDimension;

  // Transform and Interpolator Typedefs
  typedef TransformPrecisionType                CPUInterpolatorPrecisionType;
  typedef float                                 GPUInterpolatorPrecisionType;
  typedef float                                 GPUTransformPrecisionType;
  typedef typelist::MakeTypeList< short >::Type OCLImageTypes;

  // Filter typedefs
  typedef itk::ResampleImageFilter<
    InputImageType, OutputImageType, CPUInterpolatorPrecisionType > CPUFilterType;
  typedef itk::ResampleImageFilter<
    InputImageType, OutputImageType, GPUInterpolatorPrecisionType > GPUFilterType;

  // Transform typedefs
  typedef itk::Transform< TransformPrecisionType, ImageDim, ImageDim >  CPUTransformType;
  typedef itk::AffineTransform< TransformPrecisionType, ImageDim >      AffineTransformType;
  typedef itk::TranslationTransform< TransformPrecisionType, ImageDim > TranslationTransformType;
  typedef itk::BSplineTransform< TransformPrecisionType, ImageDim, 3 >  BSplineTransformType;
  typedef itk::CompositeTransform< TransformPrecisionType, ImageDim >   CompositeTransformType;

  typedef itk::Transform< GPUTransformPrecisionType, ImageDim, ImageDim > GPUTransformType;

  // Interpolator typedefs
  typedef itk::InterpolateImageFunction< InputImageType, CPUInterpolatorPrecisionType > CPUInterpolatorType;
  typedef itk::InterpolateImageFunction< InputImageType, GPUInterpolatorPrecisionType > GPUInterpolatorType;

  // Typedefs reader/writer
  typedef itk::ImageFileReader< InputImageType >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;

  // Input/Output image properties
  typename InputImageType::RegionType inputRegion;
  typename OutputImageType::SpacingType outputSpacing;
  typename OutputImageType::PointType outputOrigin;
  typename OutputImageType::DirectionType outputDirection;
  typename OutputImageType::SizeType outputSize;

  // Extra parameters
  typename InputImageType::PixelType minValue
    = itk::NumericTraits< typename InputImageType::PixelType >::Zero;
  typename OutputImageType::PixelType defaultValue
    = itk::NumericTraits< typename OutputImageType::PixelType >::Zero;

  // Transform copiers
  typedef itk::GPUCompositeTransformCopier< OCLImageTypes, OCLImageDims, CompositeTransformType,
    GPUTransformPrecisionType >
    CompositeTransformCopierType;
  typedef itk::GPUTransformCopier< OCLImageTypes, OCLImageDims, CPUTransformType, GPUTransformPrecisionType >
    TransformCopierType;

  // Interpolator copier
  typedef itk::GPUInterpolatorCopier< OCLImageTypes, OCLImageDims, CPUInterpolatorType, GPUInterpolatorPrecisionType >
    InterpolateCopierType;

  // Input image size
  typedef itk::Size< ImageDim > SizeType;
  SizeType imageSize;
  imageSize.Fill( 0 );

  // CPU part
  bool updateExceptionCPU = false;
  typename ReaderType::Pointer CPUReader;
  typename CPUFilterType::Pointer CPUFilter;
  typename CPUInterpolatorType::Pointer CPUInterpolator;
  typename CPUTransformType::Pointer CPUTransform;

  // Keep BSpline transform parameters in memory
  typedef typename BSplineTransformType::ParametersType BSplineParametersType;
  std::vector< BSplineParametersType > bsplineParameters;

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

    const typename InputImageType::ConstPointer inputImage = CPUReader->GetOutput();
    imageSize                                              = inputImage->GetBufferedRegion().GetSize();
    inputRegion                                            = inputImage->GetBufferedRegion();

    // Get all properties we need
    DefineOutputImageProperties< InputImageType, OutputImageType >(
      inputImage, outputSpacing, outputOrigin, outputDirection, outputSize,
      minValue, defaultValue );

    // Create CPU Filter
    CPUFilter = CPUFilterType::New();
    CPUFilter->SetNumberOfThreads( itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads() );

    CPUFilter->SetDefaultPixelValue( defaultValue );
    CPUFilter->SetOutputSpacing( outputSpacing );
    CPUFilter->SetOutputOrigin( outputOrigin );
    CPUFilter->SetOutputDirection( outputDirection );
    CPUFilter->SetSize( outputSize );
    CPUFilter->SetOutputStartIndex( inputRegion.GetIndex() );

    // Construct, select and setup transform
    if( !updateExceptionCPU )
    {
      DefineTransform< CPUTransformType, AffineTransformType, TranslationTransformType,
      BSplineTransformType, EulerTransformType, SimilarityTransformType,
      CompositeTransformType, InputImageType >
        ( CPUTransform, _parameters, bsplineParameters, inputImage );
    }

    // Create CPU interpolator here
    DefineInterpolator< CPUInterpolatorType >(
      CPUInterpolator, _parameters.interpolator,
      _parameters.splineOrderInterpolator, updateExceptionCPU );

    // Print info
    if( !updateExceptionCPU )
    {
      std::cout << "Testing " << itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads() << " threads for CPU vs GPU"
                << std::endl;
      std::cout << "Interpolator type: " << CPUInterpolator->GetNameOfClass() << std::endl;
      std::cout << "Transform type: "
                << GetTransformName< CPUTransformType, InputImageType >( CPUTransform ) << std::endl;
    }

    cputimer.Start();

    if( !updateExceptionCPU )
    {
      for( unsigned int i = 0; i < _parameters.runTimes; i++ )
      {
        try
        {
          CPUFilter->SetInput( CPUReader->GetOutput() );
          CPUFilter->SetTransform( CPUTransform );
          CPUFilter->SetInterpolator( CPUInterpolator );
        }
        catch( itk::ExceptionObject & e )
        {
          std::cerr << "Caught ITK exception during initialization of CPUFilter: " << e << std::endl;
          updateExceptionCPU = updateExceptionCPU || true;
        }

        if( !updateExceptionCPU )
        {
          try
          {
             CPUFilter->Update();
          }
          catch( itk::ExceptionObject & e )
          {
            std::cerr << "Caught ITK exception during CPUFilter->Update(): " << e << std::endl;
            itk::ReleaseContext();
            return EXIT_FAILURE;
          }
        }

        // Modify the filter, only not the last iteration
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
    std::cout << "CPU " << CPUFilter->GetNameOfClass() << " took " << cputimer.GetMean() / _parameters.runTimes
              << " seconds with " << CPUFilter->GetNumberOfThreads() << " threads. run times "
              << _parameters.runTimes << std::endl;
  }

  // GPU part
  bool updateExceptionGPU = false;
  typename ReaderType::Pointer GPUReader;
  typename GPUFilterType::Pointer GPUFilter;
  typename GPUInterpolatorType::Pointer GPUInterpolator;
  typename GPUTransformType::Pointer GPUTransform;
  itk::TimeProbe                  gputimer;
  itk::ObjectFactoryBase::Pointer imageFactory;

  if( !_parameters.skipGPU )
  {
    // register object factory for GPU image and filter
    typedef itk::GPUImageFactory2< OCLImageTypes, OCLImageDims > GPUImageFactoryType;
    imageFactory = GPUImageFactoryType::New();
    itk::ObjectFactoryBase::RegisterFactory( imageFactory );

    itk::GPUResampleImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();
    itk::GPUCastImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();

    // Transforms factory registration
    itk::GPUAffineTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUTranslationTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUBSplineTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUEuler2DTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUEuler3DTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUSimilarity2DTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUSimilarity3DTransformFactory2< OCLImageDims >::RegisterOneFactory();

    // Composite transform factory registration
    itk::GPUCompositeTransformFactory2< OCLImageDims >::RegisterOneFactory();

    // Interpolators factory registration
    itk::GPUNearestNeighborInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();
    itk::GPULinearInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();
    itk::GPUBSplineInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();
    itk::GPUBSplineDecompositionImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
    ::RegisterOneFactory();

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

    const typename InputImageType::ConstPointer inputImage = GPUReader->GetOutput();
    imageSize                                              = inputImage->GetBufferedRegion().GetSize();
    inputRegion                                            = inputImage->GetBufferedRegion();

    // Get all properties we need
    DefineOutputImageProperties< InputImageType, OutputImageType >(
      inputImage, outputSpacing, outputOrigin, outputDirection, outputSize,
      minValue, defaultValue );

    // Construct the filter
    // Use a try/catch, because construction of this filter will trigger
    // OpenCL compilation, which may fail.
    try
    {
      GPUFilter = GPUFilterType::New();
      itk::ITKObjectEnableWarnings( GPUFilter.GetPointer() );
    }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "Caught ITK exception during GPUFilter::New(): " << e << std::endl;
      updateExceptionGPU = updateExceptionGPU || true;
    }

    if( !updateExceptionGPU )
    {
      GPUFilter->SetDefaultPixelValue( defaultValue );
      GPUFilter->SetOutputSpacing( outputSpacing );
      GPUFilter->SetOutputOrigin( outputOrigin );
      GPUFilter->SetOutputDirection( outputDirection );
      GPUFilter->SetSize( outputSize );
      GPUFilter->SetOutputStartIndex( inputRegion.GetIndex() );
    }

    // Setup GPU transform
    if( !updateExceptionGPU )
    {
      // if skipCPU was used then create CPU transform the same way as in CPU case.
      if( _parameters.skipCPU && CPUTransform.IsNull() )
      {
        DefineTransform< CPUTransformType, AffineTransformType, TranslationTransformType,
        BSplineTransformType, EulerTransformType, SimilarityTransformType,
        CompositeTransformType, InputImageType >
          ( CPUTransform, _parameters, bsplineParameters, inputImage );
      }

      // Copy CPU transform to GPU using copiers
      if( !_parameters.useComboTransform )
      {
        typename TransformCopierType::Pointer copier = TransformCopierType::New();
        copier->SetInputTransform( CPUTransform );
        copier->SetExplicitMode( false );
        try
        {
          copier->Update();
        }
        catch( itk::ExceptionObject & e )
        {
          std::cerr << "Caught ITK exception during copier->Update(): " << e << std::endl;
          itk::ReleaseContext();
          return EXIT_FAILURE;
        }
        GPUTransform = copier->GetModifiableOutput();
      }
      else
      {
        // Get CPU CompositeTransformType
        CompositeTransformType * CPUCompositeTransform
          = dynamic_cast< CompositeTransformType * >( CPUTransform.GetPointer() );
        if( CPUCompositeTransform )
        {
          typename CompositeTransformCopierType::Pointer compositeCopier = CompositeTransformCopierType::New();
          compositeCopier->SetInputTransform( CPUCompositeTransform );
          compositeCopier->SetExplicitMode( false );
          try
          {
            compositeCopier->Update();
          }
          catch( itk::ExceptionObject & e )
          {
            std::cerr << "Caught ITK exception during compositeCopier->Update(): " << e << std::endl;
            itk::ReleaseContext();
            return EXIT_FAILURE;
          }
          GPUTransform = compositeCopier->GetModifiableOutput();
        }
        else
        {
          std::cerr << "ERROR: Unable to retrieve CPU CompositeTransform." << std::endl;
          itk::ReleaseContext();
          return EXIT_FAILURE;
        }
      }
    }

    // Create GPU copy for interpolator here
    if( !updateExceptionGPU )
    {
      typename InterpolateCopierType::Pointer interpolateCopier = InterpolateCopierType::New();
      interpolateCopier->SetInputInterpolator( CPUInterpolator );
      interpolateCopier->SetExplicitMode( false );
      try
      {
        interpolateCopier->Update();
      }
      catch( itk::ExceptionObject & e )
      {
        std::cerr << "Caught ITK exception during interpolateCopier->Update(): " << e << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
      }
      GPUInterpolator = interpolateCopier->GetModifiableOutput();
    }

    // Print info
    if( !updateExceptionGPU )
    {
      std::cout << "Interpolator type: " << GPUInterpolator->GetNameOfClass() << "\n";
      std::cout << "Transform type: "
                << GetTransformName< GPUTransformType, InputImageType >( GPUTransform ) << std::endl;
    }

    gputimer.Start();

    if( !updateExceptionGPU )
    {
      for( unsigned int i = 0; i < _parameters.runTimes; i++ )
      {
        try
        {
          GPUFilter->SetInput( GPUReader->GetOutput() );
          GPUFilter->SetTransform( GPUTransform );
          GPUFilter->SetInterpolator( GPUInterpolator );
        }
        catch( itk::ExceptionObject & e )
        {
          std::cerr << "Caught ITK exception during initialization of GPUFilter: " << e << std::endl;
          updateExceptionGPU = updateExceptionGPU || true;
        }

        if( !updateExceptionGPU )
        {
          try
          {
            GPUFilter->Update();
          }
          catch( itk::ExceptionObject & e )
          {
            std::cerr << "Caught ITK exception during GPUFilter->Update(): " << e << std::endl;
            updateExceptionGPU = updateExceptionGPU || true;
          }
        }

        // Modify the filter, only not the last iteration
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

  // Print speed up
  if( !_parameters.skipCPU && !_parameters.skipGPU && !updateExceptionGPU )
  {
    std::cout << ". speed up " << cputimer.GetMean() / gputimer.GetMean() << std::endl;
  }
  else
  {
    std::cout << std::endl;
  }

  // RMS Error check
  float RMSerror   = 0.0; float RMSrelative = 0.0;
  bool  testPassed = true;
  if( updateExceptionCPU || updateExceptionGPU )
  {
    testPassed = false;
  }
  else
  {
    if( !_parameters.skipCPU && !_parameters.skipGPU )
    {
      // UnRegister GPUImage before using BinaryThresholdImageFilter,
      // Otherwise GPU memory will be allocated
      itk::ObjectFactoryBase::UnRegisterFactory( imageFactory );

      // Create masks from filter output based on default value,
      // We compute rms error using this masks, otherwise we get false response
      // due to floating errors.
      // MS: what are floating errors?
      typedef itk::Image< unsigned char, OutputImageType::ImageDimension >      MaskImageType;
      typedef itk::BinaryThresholdImageFilter< OutputImageType, MaskImageType > ThresholdType;

      // avoid floating errors
      const typename OutputImageType::PixelType lower = minValue - 1;

      typename ThresholdType::Pointer thresholderCPU = ThresholdType::New();
      thresholderCPU->SetInput( CPUFilter->GetOutput() );
      thresholderCPU->SetInsideValue( itk::NumericTraits< typename MaskImageType::PixelType >::One );
      thresholderCPU->SetLowerThreshold( lower );
      thresholderCPU->SetUpperThreshold( itk::NumericTraits< typename OutputImageType::PixelType >::max() );
      thresholderCPU->Update();

      typename ThresholdType::Pointer thresholderGPU = ThresholdType::New();
      thresholderGPU->SetInput( GPUFilter->GetOutput() );
      thresholderGPU->SetInsideValue( itk::NumericTraits< typename MaskImageType::PixelType >::One );
      thresholderGPU->SetLowerThreshold( lower );
      thresholderGPU->SetUpperThreshold( itk::NumericTraits< typename OutputImageType::PixelType >::max() );
      thresholderGPU->Update();

      RMSerror = itk::ComputeRMSE< float, OutputImageType, OutputImageType, MaskImageType >(
        CPUFilter->GetOutput(), GPUFilter->GetOutput(),
        thresholderCPU->GetOutput(), thresholderGPU->GetOutput(), RMSrelative );
      std::cout << std::fixed << std::setprecision( 6 );
      std::cout << "Maximum allowed RMS Error: " << _parameters.allowedRMSerror << std::endl;
      std::cout << "Computed real   RMS Error: " << RMSerror << std::endl;
      std::cout << "Computed real  nRMS Error: " << RMSrelative << std::endl;
      testPassed = ( RMSerror <= _parameters.allowedRMSerror );

	  // KN: My change
      // RMSrelative = 0.0;
      // RMSerror    = itk::ComputeRMSE2< float, OutputImageType, OutputImageType, MaskImageType >(
        // CPUFilter->GetOutput(), GPUFilter->GetOutput(),
        // thresholderCPU->GetOutput(), thresholderGPU->GetOutput(),
        // _parameters.threshold, RMSrelative );
      // std::cout << "Computed real   RMS Error (t = " << _parameters.threshold << "): " << RMSerror << std::endl;
      // std::cout << "Computed real  nRMS Error (t = " << _parameters.threshold << "): " << RMSrelative << std::endl;
      // testPassed = ( RMSerror <= _parameters.allowedRMSerror );
    }
  }

  // Write output
  if( _parameters.outputWrite )
  {
    if( !_parameters.skipCPU )
    {
      // Write output CPU image
      typename WriterType::Pointer writerCPU = WriterType::New();
      writerCPU->SetInput( CPUFilter->GetOutput() );
      writerCPU->SetFileName( _parameters.outputFileNames[ 0 ] );
      writerCPU->Update();
    }

    if( !_parameters.skipGPU && !updateExceptionGPU )
    {
      // Write output GPU image
      typename WriterType::Pointer writerGPU = WriterType::New();
      writerGPU->SetInput( GPUFilter->GetOutput() );
      writerGPU->SetFileName( _parameters.outputFileNames[ 1 ] );
      writerGPU->Update();
    }
  }

  // Write log
  if( _parameters.outputLog )
  {
    std::string filterName( "na" );
    std::string interpolatorName( "na" );
    std::string transformName( "na" );
    if( !_parameters.skipCPU || ( !_parameters.skipCPU && !_parameters.skipGPU ) )
    {
      filterName       = CPUFilter->GetNameOfClass();
      interpolatorName = CPUInterpolator->GetNameOfClass();
      transformName    = GetTransformName< CPUTransformType, InputImageType >( CPUTransform );
    }
    else if( !_parameters.skipGPU )
    {
      filterName       = GPUFilter->GetNameOfClass();
      interpolatorName = GPUInterpolator->GetNameOfClass();
      transformName    = GetTransformName< GPUTransformType, InputImageType >( GPUTransform );
    }

    std::string comments;
    comments.append( "Interpolator : " );
    comments.append( interpolatorName );
    comments.append( ", Transform : " );
    comments.append( transformName );

    if( updateExceptionCPU || updateExceptionGPU )
    {
      comments.append( ", Exception during update" );
    }

    itk::WriteLog< InputImageType >(
      _parameters.logFileName, ImageDim, imageSize, RMSerror, RMSrelative,
      testPassed, updateExceptionGPU,
      itk::MultiThreaderBase::GetGlobalMaximumNumberOfThreads(),
      _parameters.runTimes, filterName,
      cputimer.GetMean(), gputimer.GetMean(),
      comments );
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
