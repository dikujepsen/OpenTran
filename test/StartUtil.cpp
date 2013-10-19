#include "CL/cl.h"
#include "CL/cl_ext.h"
//#include <string.h>
#include <malloc.h>
#include <cstdlib>
#include <iostream>
#define NUMDEVS 1
using namespace std;

cl_uint num_devices;
cl_uint num_platforms;
cl_platform_id platform_id;
cl_platform_id* platform_ids;
cl_device_id *device_ids;
cl_context context;
cl_command_queue command_queue[NUMDEVS];

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

  for (size_t i = 0; i < num_devices; i++){
    command_queue[i] = clCreateCommandQueue(context, device_ids[i],
					    CL_QUEUE_PROFILING_ENABLE, &err);
    oclCheckErr(err, "clCreateCommandQueue");
  }
}

