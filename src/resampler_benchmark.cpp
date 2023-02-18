#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTimeProbe.h"
// #include "itkImageRegionConstIterator.h"
#include "itkBSplineTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

// GPU
#include "itkGPUResampleImageFilter.h"

// GPU Factories
#include "itkGPUImageFactory.h"
#include "itkGPUResampleImageFilterFactory.h"
#include "itkGPUBSplineTransformFactory.h"
#include "itkGPUBSplineInterpolateImageFunctionFactory.h"
#include "itkGPUBSplineDecompositionImageFilterFactory.h"

// GPU Copiers
#include "itkGPUTransformCopier.h"
#include "itkGPUInterpolatorCopier.h"

#include "itkTestHelper.h"

using ImageType = itk::Image<float, 3>; // create a 3D image of floats
// using ImageType = itk::Image<short, 3>; // create a 3D image of floats
using MaskType = itk::Image<unsigned char, 3>; 

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
// This helper function completely set the transform
template< typename BSplineTransformType, typename InputImageType >
typename BSplineTransformType::Pointer  
CreateTransform( const std::size_t transformIndex,
  const typename InputImageType::ConstPointer & image,
  const std::string & parametersFileName )
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
    // bsplineParameters.push_back( parameters );
    // const std::size_t indexAt = bsplineParameters.size() - 1;

    // Do not set parameters, the will be destroyed going out of scope
    // instead, set the ones from the bsplineParameters
    // bsplineTransform->SetParameters( bsplineParameters[ indexAt ] );
    bsplineTransform->SetParameters(parameters);

    return bsplineTransform;
}


int main()
{
    // Create and check OpenCL context
    if( !itk::CreateContext() )
    {
        return EXIT_FAILURE;
    }

    // Read image
    const unsigned int ImageDim = (unsigned int)ImageType::ImageDimension;
    const std::string image_filepath = "./data/image-512x512x256-3D.mha";
    ImageType::Pointer image = itk::ReadImage<ImageType>(image_filepath);

    // Setup BSplines transform
    const std::string parametersFileName = "./data/BSplineDisplacements.txt";
    using BSplineTransformType = itk::BSplineTransform< float, ImageDim, 3>;
    BSplineTransformType::Pointer bsplineTransform = 
        CreateTransform< BSplineTransformType, ImageType >(0, image, parametersFileName);

    // Setup the interpolator
    using BSplineInterpolatorType = itk::BSplineInterpolateImageFunction<
        ImageType, float, float>;
    BSplineInterpolatorType::Pointer bsplineInterpolator = BSplineInterpolatorType::New();
    bsplineInterpolator->SetSplineOrder(3);

    // std::cout << bsplineTransform->GetParameters();
    // Create Resampler
    itk::TimeProbe cputimer;
    cputimer.Start();
    using ResamplerType = itk::ResampleImageFilter<ImageType, ImageType, float>;
    ResamplerType::Pointer CPUResampler = ResamplerType::New();
    //unsigned int maximumNumberOfThreads = itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads();
    //itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(maximumNumberOfThreads);
    //std::cout << maximumNumberOfThreads << std::endl;
    std::cout << "Threads: " << CPUResampler->GetNumberOfThreads() << std::endl;
    std::cout << "Work Units: " << CPUResampler->GetNumberOfWorkUnits() << std::endl;
    CPUResampler->SetOutputParametersFromImage(image);
    CPUResampler->SetInput(image);
    CPUResampler->SetTransform(bsplineTransform);
    CPUResampler->SetInterpolator(bsplineInterpolator);
    CPUResampler->Update();
    cputimer.Stop();


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // GPU Factories
    using OCLImageTypes = typelist::MakeTypeList<float>::Type;
    itk::GPUImageFactory2<OCLImageTypes, OCLImageDims>::RegisterOneFactory();
    itk::GPUResampleImageFilterFactory2<OCLImageTypes, OCLImageTypes, OCLImageDims>::RegisterOneFactory();
    itk::GPUBSplineTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUBSplineInterpolateImageFunctionFactory2<OCLImageTypes, OCLImageDims>::RegisterOneFactory();
    itk::GPUBSplineDecompositionImageFilterFactory2<OCLImageTypes, OCLImageTypes, OCLImageDims>::RegisterOneFactory();

    // GPU Transform
    using GPUTransformType = itk::Transform<float, ImageDim, ImageDim>;
    BSplineTransformType::Pointer gpuBsplineTransform = BSplineTransformType::New();
    using TransformCopierType = itk::GPUTransformCopier<OCLImageTypes, OCLImageDims, BSplineTransformType, float>;
    TransformCopierType::Pointer transformCopier = TransformCopierType::New();
    transformCopier->SetInputTransform(bsplineTransform);
    transformCopier->SetExplicitMode(false);
    try
    {
        transformCopier->Update();
    }
    catch (itk::ExceptionObject& e)
    {
        std::cerr << "Caught ITK exception during copier->Update(): " << e << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
    }
    GPUTransformType::Pointer GPUTransform = transformCopier->GetModifiableOutput();

    // GPU Interpolator
    using InterpolateCopierType = itk::GPUInterpolatorCopier< OCLImageTypes, OCLImageDims, BSplineInterpolatorType, float>;
    InterpolateCopierType::Pointer interpolateCopier = InterpolateCopierType::New();
    interpolateCopier->SetInputInterpolator(bsplineInterpolator);
    interpolateCopier->SetExplicitMode(false);
    try
    {
        interpolateCopier->Update();
    }
    catch (itk::ExceptionObject& e)
    {
        std::cerr << "Caught ITK exception during interpolateCopier->Update(): " << e << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
    }
    using GPUInterpolatorType = itk::InterpolateImageFunction<ImageType, float>;
    GPUInterpolatorType::Pointer GPUInterpolator = interpolateCopier->GetModifiableOutput();


    // GPU Resampler
    ImageType::Pointer gpuImage = itk::ReadImage<ImageType>(image_filepath);
    ResamplerType::Pointer GPUResampler = ResamplerType::New();
    GPUResampler->SetInput(gpuImage);
    GPUResampler->SetTransform(GPUTransform);
    GPUResampler->SetInterpolator(GPUInterpolator);
    GPUResampler->SetOutputParametersFromImage(image);

    itk::TimeProbe gputimer;
    gputimer.Start();
    try
    {
        GPUResampler->Update();
    }
    catch (itk::ExceptionObject& e)
    {
        std::cerr << "ERROR: " << e << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
    }
    gputimer.Stop();


    std::cout << "CPU Resampler: " << cputimer.GetMean() << std::endl;
    std::cout << "GPU Resampler: " << gputimer.GetMean() << std::endl;

    // Compare CPU and GPU images
    auto cpuresult = CPUResampler->GetOutput();
    auto gpuresult = GPUResampler->GetOutput();
    float rmse = 0.0;
    float rmsRelative = 0.0;

    rmse = itk::ComputeRMSE<float, ImageType, ImageType>(cpuresult, gpuresult, rmsRelative);
    std::cout << "RMSE: " << rmse << " RMSRelative: " << rmsRelative << std::endl;

    rmse = itk::ComputeRMSE<float, ImageType, ImageType>(cpuresult, image, rmsRelative);
    std::cout << "RMSE: " << rmse << " RMSRelative: " << rmsRelative << std::endl;

    rmse = itk::ComputeRMSE<float, ImageType, ImageType>(image, gpuresult, rmsRelative);
    std::cout << "RMSE: " << rmse << " RMSRelative: " << rmsRelative << std::endl;


    itk::WriteImage(cpuresult, "cpu_result.mha");
    itk::WriteImage(gpuresult, "gpu_result.mha");

    std::cout << "Finished!" << std::endl;
    
    return EXIT_SUCCESS;

}

