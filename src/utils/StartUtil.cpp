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
      std::cout << "StartUpGPU" << std::endl;
      cl_int err = CL_SUCCESS;
      err |= clGetPlatformIDs(0, NULL, &num_platforms);
      helper::oclCheckErr(err, "clGetPlatformIDs Count");

      platform_ids = new cl_platform_id[num_platforms];
      // get available platforms
      std::cout << "$numPLAT: " << num_platforms << std::endl;
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
      std::cout << "$numDEV " << num_devices << std::endl;
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
        cout << "$Devicename " << buff << endl;
        free(buff);
      }

      context = clCreateContext(0, num_devices, device_ids, NULL, NULL, &err);
      helper::oclCheckErr(err, "clCreateContext");

      command_queue = clCreateCommandQueue(context, device_ids[0],
                                           CL_QUEUE_PROFILING_ENABLE, &err);
      helper::oclCheckErr(err, "clCreateCommandQueue");
    }



    void compileKernel(std::string kernel_name,
                       const char *filename,
                       std::string kernelstr,
                       bool useFile,
                       cl_kernel* kernel,
                       std::string options)
    {

      cl_int err = CL_SUCCESS;
      const char* source2;
      if (useFile)
      {
        source2 = ReadSources(filename);
      }
      else
      {
        source2 = kernelstr.c_str();
        std::ofstream file;
        file.open(filename);
        file << kernelstr;
        file.close();
      }

      cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2, NULL, &err);
      helper::oclCheckErr(err, "clCreateProgramWithSource");

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
      helper::oclCheckErr(err, "clBuildProgram");


      kernel[0] = clCreateKernel(program, kernel_name.c_str(), &err);
      helper::oclCheckErr(err, "clCreateKernel");

      err |= clReleaseProgram(program);
      helper::oclCheckErr(err, "clReleaseProgram");
      if (useFile)
      {
        free((void *)source2);
      }
    }
private:

    char * ReadSources(const char *fileName)
    {

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

};











