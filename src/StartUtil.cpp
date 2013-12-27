#include "CL/cl.h"
#include "CL/cl_ext.h"
#include <string.h>
#include <malloc.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#define NUMDEVS 1
using namespace std;

cl_uint num_devices;
cl_uint num_platforms;
cl_platform_id platform_id;
cl_platform_id* platform_ids;
cl_device_id *device_ids;
cl_context context;
cl_command_queue command_queue;

char * ReadSources(const char *fileName) {

  FILE *file = fopen(fileName, "rb");
  if (!file)
    {
      printf("ERROR: Failed to open file '%s'\n", fileName);
      return NULL;
    }

  if (fseek(file, 0, SEEK_END))
    {
      printf("ERROR: Failed to seek file '%s'\n", fileName);
      fclose(file);
      return NULL;
    }

  long size = ftell(file);
  if (size == 0)
    {
      printf("ERROR: Failed to check position on file '%s'\n", fileName);
      fclose(file);
      return NULL;
    }

  rewind(file);

  char *src = (char *)malloc(sizeof(char) * size + 1);
  if (!src)
    {
      printf("ERROR: Failed to allocate memory for file '%s'\n", fileName);
      fclose(file);
      return NULL;
    }

  printf("Reading file '%s' (size %ld bytes)\n", fileName, size);
  size_t res = fread(src, 1, sizeof(char) * size, file);
  if (res != sizeof(char) * size)
    {
      printf("ERROR: Failed to read file '%s'\n", fileName);
      fclose(file);
      free(src);
      return NULL;
    }

  src[size] = '\0'; /* NULL terminated */
  fclose(file);

  return src;
      
}


void oclCheckErr(cl_int err, const char * function) {
  if (err != CL_SUCCESS)
    { 
      printf("Error: Failure %s: %d\n", function, err);
      exit(-1);
    }
}


void StartUpGPU() {
  std::cout << "StartUpGPU" << std::endl;
  cl_int err = CL_SUCCESS;
  err |= clGetPlatformIDs(0, NULL, &num_platforms);
  oclCheckErr(err, "clGetPlatformIDs1");
  
  platform_ids = new cl_platform_id[num_platforms];
  // get available platforms
  std::cout << "numPLAT: " << num_platforms << std::endl;
  err |= clGetPlatformIDs(num_platforms, platform_ids, NULL);
  oclCheckErr(err, "clGetPlatformIDs2");

  platform_id = platform_ids[0];
  err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  oclCheckErr(err, "clGetDeviceIDs1");
  
  num_devices = std::min<unsigned>(num_devices, NUMDEVS);
  std::cout << "numDEV: " << num_devices << std::endl;
      
  device_ids = new cl_device_id[num_devices];
  
  err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, device_ids, NULL);
  oclCheckErr(err, "clGetDeviceIDs2");

  context = clCreateContext(0, num_devices, device_ids, NULL, NULL, &err);
  oclCheckErr(err, "clCreateContext");

    command_queue = clCreateCommandQueue(context, device_ids[0],
					    CL_QUEUE_PROFILING_ENABLE, &err);
    oclCheckErr(err, "clCreateCommandQueue");
}


void compileKernelFromFile(std::string kernel_name,
			   const char *filename,
			   cl_kernel* kernel,
			   std::string options) {

  cl_int err = CL_SUCCESS;
  const char* source2 = ReadSources(filename);
  cout << source2 << endl;
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");

  std::stringstream buildOptions;
  // Okay for most programs
  buildOptions << "-cl-fast-relaxed-math " << options;
  
  
  err = clBuildProgram(program, 0, NULL, buildOptions.str().c_str(), NULL, NULL);
  if (err != CL_SUCCESS)
    {
      std::cout << "OCL Error: OpenCL Build Error. Error Code: " << err << std::endl;

      size_t len;
      char buffer[10000];

      // get the build log
      clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

      std::cout << "--- Build Log ---" << std::endl << buffer << std::endl;
    }
  oclCheckErr(err, "clBuildProgram");


  kernel[0] = clCreateKernel(program, kernel_name.c_str(), &err);
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
  free((void *)source2);
} 

template <class T>
void transpose(T * sink, T* source, size_t source_dim1, size_t source_dim2) {

  for (size_t i = 0; i < source_dim2; i++) {
    for (size_t j = 0; j < source_dim1; j++) {
      source[j * source_dim2 + i] = sink[i * source_dim1 + j];
    }
  }

}
