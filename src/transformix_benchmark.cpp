#include "itkImage.h"
#include "itkImageFileReader.h"
// #include "itkElastixRegistrationMethod.h"
#include "itkTransformixFilter.h"
#include "elxParameterObject.h"
#include "itkTimeProbe.h"
#include "itkImageRegionConstIterator.h"

using ImageType = itk::Image<float, 3>; // create a 3D image of floats
using MaskType = itk::Image<unsigned char, 3>; 
using TransformixFilterType = itk::TransformixFilter<ImageType>;

ImageType::Pointer test_resampler(ImageType::Pointer moving_image, elx::ParameterObject::Pointer parameter_object)
{
    TransformixFilterType::Pointer transformixFilter = TransformixFilterType::New();
    transformixFilter->SetMovingImage(moving_image);
    transformixFilter->SetTransformParameterObject(parameter_object);
    transformixFilter->SetLogToConsole(false);
    transformixFilter->SetOutputDirectory("./");
    transformixFilter->Update();

    return transformixFilter->GetOutput();
}


namespace itk
{
// Helper function to compute RMSE
template< typename TScalarType, typename CPUImageType, typename GPUImageType >
TScalarType
ComputeRMSE( const CPUImageType * cpuImage, const GPUImageType * gpuImage,
  TScalarType & rmsRelative )
{
  ImageRegionConstIterator< CPUImageType > cit(
    cpuImage, cpuImage->GetLargestPossibleRegion() );
  ImageRegionConstIterator< GPUImageType > git(
    gpuImage, gpuImage->GetLargestPossibleRegion() );

  TScalarType rmse          = 0.0;
  TScalarType sumCPUSquared = 0.0;

  for( cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git )
  {
    TScalarType cpu = static_cast< TScalarType >( cit.Get() );
    TScalarType err = cpu - static_cast< TScalarType >( git.Get() );
    rmse          += err * err;
    sumCPUSquared += cpu * cpu;
  }

  rmse        = std::sqrt( rmse / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
  rmsRelative = rmse / std::sqrt( sumCPUSquared / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );

  return rmse;
} // end ComputeRMSE()
}

int main()
{
    const unsigned char run_times = 1;//0;

    // ImageType::Pointer moving_image = itk::ReadImage<ImageType>("./data/CT_3D_lung_moving.mha");
    ImageType::Pointer moving_image = itk::ReadImage<ImageType>("./data/image-512x512x256-3D.mha");

    // Read the transform parameters
    elx::ParameterObject::Pointer parameter_object = elx::ParameterObject::New();
    parameter_object->AddParameterFile("./data/transform_parameters_affine.txt");
    parameter_object->AddParameterFile("./data/transform_parameters_bspline.txt");

    // CPU
    ImageType::Pointer cpuresult;
    for (int i=0; i < parameter_object->GetNumberOfParameterMaps(); i++){
        parameter_object->SetParameter(i, "Resampler", "DefaultResampler");
    }

    itk::TimeProbe cputimer;
    cputimer.Start();
    try
    {
        for (int i=0; i < run_times; i++){
            cpuresult = test_resampler(moving_image, parameter_object);
        }
    }
    catch (const itk::ExceptionObject & err)
    {
        std::cerr << "Exception: " << err << std::endl;
        return EXIT_SUCCESS;
    }
    cputimer.Stop();

    // std::cout << parameter_object << std::endl;

    // GPU
    ImageType::Pointer gpuresult;
    for (int i=0; i < parameter_object->GetNumberOfParameterMaps(); i++){
        parameter_object->SetParameter(i, "Resampler", "OpenCLResampler");
    }

    itk::TimeProbe gputimer;
    gputimer.Start();
    try
    {
        for (int i=0; i < run_times; i++){
            gpuresult = test_resampler(moving_image, parameter_object);
        } 
    }
    catch (const itk::ExceptionObject & err)
    {
        std::cerr << "Exception: " << err << std::endl;
        return EXIT_SUCCESS;
    }
    gputimer.Stop();

    // std::cout << parameter_object << std::endl;

    // Print results
    std::cout << "CPU: " << cputimer.GetMean() / run_times << std::endl;
    std::cout << "GPU: " << gputimer.GetMean() / run_times << std::endl;

    // Compare CPU and GPU images
    float rmse = 0.0;
    float rmsRelative = 0.0;
    rmse = itk::ComputeRMSE<float, ImageType, ImageType>(cpuresult, gpuresult, rmsRelative);
    std::cout << "RMSE: " << rmse << " RMSRelative: " << rmsRelative << std::endl;

    itk::WriteImage(cpuresult, "cpu_result.mha");
    itk::WriteImage(gpuresult, "gpu_result.mha");
    
    return EXIT_SUCCESS;

}

