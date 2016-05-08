#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <string.h>
#include <malloc.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include "helper.hpp"
#define NUMDEVS 1

using namespace std;


#include "Stopwatch.cpp"


class OCLContext
{

    cl_uint num_devices;
    cl_uint num_platforms;
    cl_platform_id platform_id;
    cl_platform_id* platform_ids;
    cl_device_id *device_ids;
    cl_context context;
    cl_command_queue command_queue;


  public:

    cl_context getContext()
    {
      return context;
    }

    cl_command_queue getCommandQueue()
    {
      return command_queue;
    }

    void StartUpOCL(std::string ocl_type)
    {

      cl_int err = CL_SUCCESS;
      err |= clGetPlatformIDs(0, NULL, &num_platforms);
      helper::oclCheckErr(err, "clGetPlatformIDs Count");

      platform_ids = new cl_platform_id[num_platforms];

      err |= clGetPlatformIDs(num_platforms, platform_ids, NULL);
      helper::oclCheckErr(err, "clGetPlatformIDs Allocate");

      cl_device_type devtype = CL_DEVICE_TYPE_GPU;
      if (ocl_type == "cpu")
      {
        devtype = CL_DEVICE_TYPE_CPU;
      }


      for (unsigned i = 0; i < num_platforms; i++)
      {
        platform_id = platform_ids[i];
        err |= clGetDeviceIDs(platform_id, devtype, 0, NULL, &num_devices);
        if (err == CL_DEVICE_NOT_FOUND)
        {
          err = 0;
        }
        else
        {
          break;
        }
      }

      helper::oclCheckErr(err, "clGetDeviceIDs num_devices");
      num_devices = std::min<unsigned>(num_devices, NUMDEVS);

      device_ids = new cl_device_id[num_devices];

      err |= clGetDeviceIDs(platform_id, devtype, num_devices, device_ids, NULL);
      helper::oclCheckErr(err, "clGetDeviceIDs allocate");
      size_t len;
      for(size_t i = 0; i < num_devices; i++)
      {
        clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, 0, NULL, &len);
        char * buff = new char[len];
        clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(char)*len, buff, NULL);
        free(buff);
      }

      context = clCreateContext(0, num_devices, device_ids, NULL, NULL, &err);
      helper::oclCheckErr(err, "clCreateContext Allocate");

      command_queue = clCreateCommandQueue(context, device_ids[0],
                                           CL_QUEUE_PROFILING_ENABLE, &err);
      helper::oclCheckErr(err, "clCreateCommandQueue Allocate");
    }



    void compileKernel(std::string kernel_name,
                       std::string kernelstr,
                       cl_kernel* kernel,
                       std::string options)
    {

      cl_int err = CL_SUCCESS;
      const char* source2;
      
        source2 = kernelstr.c_str();
      

      cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2, NULL, &err);
      helper::oclCheckErr(err, "clCreateProgramWithSource");

      std::stringstream buildOptions;

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
      helper::oclCheckErr(err, "clBuildProgram");


      kernel[0] = clCreateKernel(program, kernel_name.c_str(), &err);
      helper::oclCheckErr(err, "clCreateKernel");

      err |= clReleaseProgram(program);
      helper::oclCheckErr(err, "clReleaseProgram");

    }
private:


};











