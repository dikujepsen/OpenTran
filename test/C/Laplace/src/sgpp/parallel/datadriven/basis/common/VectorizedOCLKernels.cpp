#include "CL/cl.h"
#include "CL/cl_ext.h"
#include <string.h>
#include <malloc.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include "base/grid/Grid.hpp"
#include "base/grid/type/LinearGrid.hpp"
#include "base/grid/type/LinearStretchedGrid.hpp"
#include "base/datatypes/DataVector.hpp"
#include "base/tools/SGppStopwatch.hpp"
using namespace sg::base;

#define NUMDEVS 1
#define REAL double

#define MAXBYTES 2100000000
#define LSIZE 16


cl_uint num_devices;
cl_uint num_platforms;
cl_platform_id platform_id;
cl_platform_id* platform_ids;
cl_device_id *device_ids;
cl_context context;
cl_command_queue command_queue[NUMDEVS];



cl_kernel LaplaceBoundKernel[NUMDEVS];
cl_kernel LaplaceInnerKernel[NUMDEVS];

cl_kernel LTwoDotBoundKernel[NUMDEVS];
cl_kernel LTwoDotInnerKernel[NUMDEVS];

cl_kernel ReduceBoundKernel[NUMDEVS];
cl_kernel ReduceInnerKernel[NUMDEVS];


cl_mem d_ptrLevelBound[NUMDEVS];
cl_mem d_ptrIndexBound[NUMDEVS];
cl_mem d_ptrLevel_intBound[NUMDEVS];
cl_mem d_ptrResultBound[NUMDEVS];
cl_mem d_ptrParResultBound[NUMDEVS];
cl_mem d_ptrAlphaBound[NUMDEVS];
cl_mem d_ptrLevelIndexLevelintconBound[NUMDEVS]; // constant memory buffer holding all three components
cl_mem d_ptrLambdaBound[NUMDEVS];
cl_mem d_ptrLcl_qBound[NUMDEVS]; // Also holds q_inverse


cl_mem d_ptrLevelInner[NUMDEVS];
cl_mem d_ptrIndexInner[NUMDEVS];
cl_mem d_ptrLevel_intInner[NUMDEVS];
cl_mem d_ptrResultInner[NUMDEVS];
cl_mem d_ptrParResultInner[NUMDEVS];
cl_mem d_ptrAlphaInner[NUMDEVS];
cl_mem d_ptrLevelIndexLevelintconInner[NUMDEVS]; // constant memory buffer holding all three components
cl_mem d_ptrLambdaInner[NUMDEVS];
cl_mem d_ptrLcl_qInner[NUMDEVS]; // Also holds q_inverse


unsigned * offsetBound;
REAL * ptrResultBound;


REAL * ptrLevelTBound;
REAL * ptrIndexTBound;
REAL * ptrLevel_intTBound;
// ptrResult;
REAL * ptrParResultBound;
REAL * ptrAlphaEndBound;          
REAL * ptrLevelIndexLevelintBound; // for the constant memory buffer holding all three components
REAL * ptrLcl_qBound;            // Also holds q_inverse
REAL * ptrResultTempBound;
REAL * ptrResultZeroBound;


REAL * ptrLevelTInner;
REAL * ptrIndexTInner;
REAL * ptrLevel_intTInner;
// ptrResult;
REAL * ptrParResultInner;
REAL * ptrAlphaEndInner;          
REAL * ptrLevelIndexLevelintInner; // for the constant memory buffer holding all three components
REAL * ptrLcl_qInner;            // Also holds q_inverse
REAL * ptrResultTemp;
REAL * ptrResultZero;


size_t dims;
size_t padding_size = NUMDEVS*LSIZE;

// Sizes for Laplace + LTwo Inner
size_t storageSize;
size_t storageSizePadded;
size_t num_groups;
size_t par_result_size;
size_t lcl_q_size;
size_t alphaend_size;

// size_t storageInnerSizePadded;
// size_t storageInnerSize;
// size_t InnerSize;
// size_t Inner_num_groups;
// size_t Inner_par_result_size;
// size_t Inner_par_result_max_size;
// size_t Inner_result_size;
size_t max_buffer_size;
size_t par_result_max_size;

// Sizes for Laplace + LTwo Bound
size_t storageSizeBound;
size_t storageSizePaddedBound;
size_t num_groupsBound;
size_t par_result_sizeBound;
size_t lcl_q_sizeBound;
size_t alphaend_sizeBound;

size_t storageInnerSizePaddedBound;
size_t storageInnerSizeBound;
size_t InnerSizeBound;
size_t Inner_num_groupsBound;
size_t Inner_par_result_sizeBound;
size_t Inner_par_result_max_sizeBound;
size_t Inner_result_sizeBound;
size_t par_result_max_sizeBound;


size_t constant_mem_size;
size_t constant_buffer_size;
size_t constant_buffer_iterations;

size_t constant_buffer_size_noboundary;
size_t constant_buffer_iterations_noboundary;


double MultTimeLaplaceInner = 0.0;
double ReduTimeLaplaceInner = 0.0;
double CounterLaplaceInner = 0.0;


double MultTimeLTwoDotInner = 0.0;
double ReduTimeLTwoDotInner = 0.0;
double CounterLTwoDotInner = 0.0;


double MultTimeLaplaceBound = 0.0;
double ReduTimeLaplaceBound = 0.0;
double CounterLaplaceBound = 0.0;


double MultTimeLTwoDotBound = 0.0;
double ReduTimeLTwoDotBound = 0.0;
double CounterLTwoDotBound = 0.0;

SGppStopwatch* myStopwatch = new SGppStopwatch();

#define GPUREDUCE 1
#define TOTALTIMING 1
#define GPUTIMING 0
#define PRINTOCL  0
#define TWODKERNEL 0
#define PRINTPARRESULT 0
#define PRINTBUFFERSIZES 0

/****************************
 * PREFERRED PERFORMANCE SETTINGS 16-04-13
 * GPUREDUCE 1
 * FILEKERNEL 0
 * INLINEKERNEL 0
 * TWODKERNEL 1
 * INDEXES 1
 * UNROLLLSIZE 0
 * IINGLOBAL 1
 *****************************/



// Need this because of the software architecture
unsigned isVeryFirstTime = 1;
unsigned isFirstTimeLaplaceInner = 1;
unsigned isFirstTimeLaplaceBound = 1;
unsigned isFirstTimeLTwoDotInner = 1;
unsigned isFirstTimeLTwoDotBound = 1;
unsigned isCleanedUp = 0;


void transposer(REAL* sink, REAL* source, size_t dim1, size_t dim2pad, size_t dim2) {
  for (unsigned i = 0; i < dim2; i++) {
    for (unsigned j = 0; j < dim1; j++) {
      sink[j*dim2pad + i] = source[i*dim1 + j];
    }
  }
}

void boundarytransposer(REAL* sink, REAL* source, size_t dim1, size_t dim2, sg::base::GridStorage* storage) {

  unsigned k = 0;
  for (unsigned i = 0; i < dim2; i++) {
    sg::base::GridIndex* curPoint = (*storage)[i];
    if (curPoint->isInnerPoint()) {
      for (unsigned j = 0; j < dim1; j++) {
	sink[j*storageInnerSizePaddedBound + k] = source[i*dim1 + j];
      }
      k++;
    }
  }
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
 

void compileKernelInner(int id, std::string kernel_src, const char *filename, cl_kernel* kernel) {

  cl_int err = CL_SUCCESS;
  const char* source2 = ReadSources(filename);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");
  char buildOptions[256];
  int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DINNERSTORAGEPAD=%lu -DDIMS=%lu -cl-finite-math-only  -cl-fast-relaxed-math ", storageSize, storageSizePadded, num_groups, 0L, dims);
  if (ierr < 0) {
    printf("Error in Build Options");
    exit(-1);
  }
  err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
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


  kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
  free((void *)source2);
} 

void compileKernelBound(int id, std::string kernel_src, const char *filename, cl_kernel* kernel) {

  cl_int err = CL_SUCCESS;
  const char* source2 = ReadSources(filename);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");
  char buildOptions[256];
  int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DINNERSTORAGEPAD=%lu -DDIMS=%lu -cl-finite-math-only  -cl-fast-relaxed-math ", storageSizeBound, storageSizePaddedBound, num_groupsBound, storageInnerSizePaddedBound, dims);
  if (ierr < 0) {
    printf("Error in Build Options");
    exit(-1);
  }
  err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
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


  kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
  free((void *)source2);
} 


const char* ReduceBoundKernelStr(){

  return "__kernel void ReduceBoundKernel(\
			   __global  REAL* ptrResultBound,\
			   __global  REAL* ptrParResultBound,\
			   ulong overallMultOffset,\
			   ulong num_groups\
\
			   )\
{\
  unsigned j = get_global_id(0);\
  \
  REAL res = 0.0;\
  for (unsigned k = 0; k < num_groups; k++) {\
    res += ptrParResultBound[k*get_global_size(0) + j];\
  }\
  \
  ptrResultBound[j] += res;\
\
}"; //"


}


void compileReduceBound(int id, std::string kernel_src, cl_kernel* kernel) {

  cl_int err = CL_SUCCESS;
  const char* source2 = ReduceBoundKernelStr();
  
  std::stringstream stream_program_src;

  stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
  stream_program_src << "#define REAL double" << std::endl;
  stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
  stream_program_src << source2 << std::endl;
  std::string program_src = stream_program_src.str(); 
  source2 = program_src.c_str();
  //  std::cout << "SOURCE " << source2 << std::endl;
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");
  char buildOptions[256];
  int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DINNERSTORAGEPAD=%lu -DDIMS=%lu -cl-finite-math-only  -cl-fast-relaxed-math ", storageSizeBound, storageSizePaddedBound, num_groupsBound, storageInnerSizePaddedBound, dims);
  if (ierr < 0) {
    printf("Error in Build Options");
    exit(-1);
  }
  err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
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


  kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
  
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
} 



const char * ReduceInnerKernelStr() {
  return "__kernel void ReduceInnerKernel(__global  REAL* ptrResult, \n\
			   __global  REAL* ptrParResult,\n\
			   ulong overallParOffset, \n\
			   ulong num_groups \n\
\n\
			   )\n\
{\n\
  unsigned j = get_global_id(0);\n\
  \n\
  REAL res = 0.0;\n\
  for (unsigned k = 0; k < num_groups; k++) {\n\
    res += ptrParResult[k*get_global_size(0) + j];\n\
  }\n\
  \n\
  ptrResult[j] += res;\n\
\n\
}\n\
"; //"

}


void compileReduceInner(int id, std::string kernel_src, cl_kernel* kernel) {

  cl_int err = CL_SUCCESS;
  const char* source2 = ReduceInnerKernelStr();
  
  std::stringstream stream_program_src;

  stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
  stream_program_src << "#define REAL double" << std::endl;
  stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
  stream_program_src << source2 << std::endl;
  std::string program_src = stream_program_src.str(); 
  source2 = program_src.c_str();

  //  std::cout << source2 << std::endl;

  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");
  char buildOptions[256];
  int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DINNERSTORAGEPAD=%lu -DDIMS=%lu -cl-finite-math-only  -cl-fast-relaxed-math ", storageSize, storageSizePadded, num_groups, 0L, dims);
  if (ierr < 0) {
    printf("Error in Build Options");
    exit(-1);
  }
  err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
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


  kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
  
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
} 


const char* InnerLTwoDotFunction() {

return "REAL l2dot(REAL lid, \n\
	   REAL iid, \n\
	   REAL in_lid, \n\
	   REAL ljd, \n\
	   REAL ijd,\n\
	   REAL in_ljd,\n\
	   REAL lcl_q)\n\
\n\
{\n\
  double res_one = select(0.0,(2.0/3.0) * in_lid, (ulong)(iid == ijd));\n\
\n\
  ulong selector = (lid > ljd);\n\
  double i1d = select(ijd, iid, selector);\n\
  double in_l1d = select(in_ljd, in_lid,selector);\n\
  double i2d = select(iid, ijd, selector);\n\
  double l2d = select(lid, ljd, selector);\n\
  double in_l2d = select(in_lid, in_ljd, selector);\n\
\n\
  double q = fma(i1d, in_l1d, -in_l1d); //(i1d-1)*in_l1d;\n\
  double p = fma(i1d, in_l1d,  in_l1d); //(i1d+1)*in_l1d;\n\
\n\
  ulong overlap = (max(q, fma(i2d, in_l2d, -in_l2d)) < min(p, fma(i2d, in_l2d,in_l2d)));\n\
\n\
  double temp_res = fma((0.5*in_l1d), (- fabs(fma(l2d,q,-i2d)) - fabs(fma(l2d,p,-i2d))), in_l1d);\n\
\n\
 double res_two = select(0.0,temp_res,overlap); // Now mask result	\n\
 return (select(res_two, res_one, (ulong)(lid == ljd)))*lcl_q;\n	\
}\n"; //"

}

const char* LaplaceInnerHeader() {
  return "REAL gradient(REAL i_level, \n\
	      REAL i_index, \n\
	      REAL j_level, \n\
	      REAL j_index,\n\
	      REAL lcl_q_inv)\n\
{\n\
  REAL grad;\n\
  ulong doGrad = (ulong)((i_level == j_level) && (i_index == j_index));\n\
  grad = select(0.0, i_level * 2.0 * lcl_q_inv,doGrad);\n\
\n\
  return grad;\n\
}\n\
\n\
__kernel void multKernel(__global  REAL* ptrLevel, // 64\n\
			 __constant  REAL* ptrLevelIndexLevelintcon,\n\
			 __global  REAL* ptrIndex, // 64\n\
			 __global  REAL* ptrLevel_int, // 64\n\
			 __global  REAL* ptrAlpha, // 64\n\
			 __global  REAL* ptrParResult, // 64\n\
			 __constant  REAL* ptrLcl_q,\n\
			 __constant  REAL* ptrLambda,\n\
			 ulong overallMultOffset,\n	\
                        ulong ConstantMemoryOffset)\n	\
{\n;"; //"

}

void compileLTwoDotInner(int id, std::string kernel_src, cl_kernel* kernel) {
  cl_int err = CL_SUCCESS;
  const char* source2 = "";
  const char* l2dotfunction = InnerLTwoDotFunction();

  std::stringstream stream_program_src;

  stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
  stream_program_src << "#define REAL double" << std::endl;

  stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
  stream_program_src << l2dotfunction << std::endl;
  stream_program_src << "__kernel void multKernel(__global  REAL* ptrLevel, " << std::endl;
  stream_program_src << "			 __constant  REAL* ptrLevelIndexLevelintcon," << std::endl;
  stream_program_src << "			 __global  REAL* ptrIndex," << std::endl;
  stream_program_src << "			 __global  REAL* ptrLevel_int," << std::endl;
  stream_program_src << "			 __global  REAL* ptrAlpha," << std::endl;
  stream_program_src << "			 __global  REAL* ptrParResult," << std::endl;
  stream_program_src << "			 __constant  REAL* ptrLcl_q," << std::endl;
  stream_program_src << "			 ulong overallMultOffset," << std::endl;
  stream_program_src << "			 ulong ConstantMemoryOffset)" << std::endl;
  stream_program_src << "{" << std::endl;


  stream_program_src << "__local REAL alphaTemp["<<LSIZE<<"];" << std::endl;

  stream_program_src << "ptrLevel += get_local_id(0);" << std::endl;
  stream_program_src << "ptrIndex += get_local_id(0);" << std::endl;
  stream_program_src << "ptrLevel_int += get_local_id(0);" << std::endl;

  stream_program_src << "alphaTemp[get_local_id(0)]   = ptrAlpha[get_global_id(0) + ConstantMemoryOffset*LSIZE];" << std::endl;
  stream_program_src << " barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
 
  stream_program_src << "REAL res = 0.0;" << std::endl;
  const char* typeREAL = "";

  stream_program_src << "for (unsigned k = 0; k < "<<LSIZE<<"; k++) {" << std::endl;
  typeREAL = "REAL ";
  stream_program_src << typeREAL<< "element = alphaTemp["<<"k"<<"];" << std::endl;

  for(size_t d_inner = 0; d_inner < dims; d_inner++) {
    if (d_inner == 0)
      {
	typeREAL = "REAL ";
      } else {
      typeREAL = "";
    }
    stream_program_src << typeREAL << "i_level =         ptrLevel["<<d_inner*storageSizePadded<<" + (get_global_id(1)+overallMultOffset)*LSIZE];" << std::endl;
    stream_program_src << typeREAL << "i_index =         ptrIndex["<<d_inner*storageSizePadded<<" + (get_global_id(1)+overallMultOffset)*LSIZE];" << std::endl;            
    stream_program_src << typeREAL << "i_level_int =     ptrLevel_int["<<d_inner*storageSizePadded<<" + (get_global_id(1)+overallMultOffset)*LSIZE];" << std::endl;

    stream_program_src << typeREAL << "j_levelreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3<<"]; " << std::endl;
    stream_program_src << typeREAL << "j_indexreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 1<<"]; " << std::endl;
    stream_program_src << typeREAL << "j_level_intreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 2<<"]; " << std::endl;

    stream_program_src << "element *= (l2dot(i_level,i_index,i_level_int, j_levelreg,j_indexreg,j_level_intreg,ptrLcl_q["<<d_inner<<"]));" << std::endl;

  }
  stream_program_src << "res +=  element;" << std::endl;
  
  stream_program_src << "}" << std::endl;
  stream_program_src << "ptrParResult[((get_group_id(0)+ConstantMemoryOffset)*get_global_size(1) + get_global_id(1))*"<<LSIZE<<" + get_local_id(0)] = res; " << std::endl;
  stream_program_src << "}" << std::endl;

  std::string program_src = stream_program_src.str(); 
  source2 = program_src.c_str();
#if PRINTOCL
  std::cout <<  source2 << std::endl;
#endif
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");
  char buildOptions[256];
  int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DDIMS=%lu -cl-finite-math-only -cl-fast-relaxed-math ", storageSize, storageSizePadded, num_groups, dims);
  if (ierr < 0) {
    printf("Error in Build Options");
    exit(-1);
  }

  err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
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


  kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
  
}


void compileMultKernel2DIndexesJInReg2(int id, std::string kernel_src, cl_kernel* kernel) {
  cl_int err = CL_SUCCESS;
  const char* source2 = LaplaceInnerHeader();
  const char* l2dotfunction = InnerLTwoDotFunction();

  std::stringstream stream_program_src;

  stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
  stream_program_src << "#define REAL double" << std::endl;

  stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
  stream_program_src << l2dotfunction << std::endl;
  stream_program_src << source2 << std::endl;
  stream_program_src << "__local REAL alphaTemp["<<LSIZE<<"];" << std::endl;
  unsigned dcon = dims; 
  stream_program_src << "__local REAL l2dotTemp["<<dcon*LSIZE<<"];" << std::endl;
  stream_program_src << "__local REAL gradTemp["<<dcon*LSIZE<<"];" << std::endl;

  stream_program_src << "ptrLevel += get_local_id(0);" << std::endl;
  stream_program_src << "ptrIndex += get_local_id(0);" << std::endl;
  stream_program_src << "ptrLevel_int += get_local_id(0);" << std::endl;

  stream_program_src << "alphaTemp[get_local_id(0)]   = ptrAlpha[get_global_id(0) + overallMultOffset + ConstantMemoryOffset * LSIZE];" << std::endl;

  stream_program_src << "REAL res = 0.0;" << std::endl;
  const char* typeREAL = "";

  stream_program_src << "for (unsigned k = 0; k < "<<LSIZE<<"; k++) {" << std::endl;
  for(size_t d_inner = 0; d_inner < dims; d_inner++) {
    if (d_inner == 0)
      {
	typeREAL = "REAL ";
      } else {
      typeREAL = "";
    }
    if (0 && d_inner == 0) {
      stream_program_src << typeREAL << "i_level =         Leveld1Temp[get_local_id(0)];" << std::endl;
      stream_program_src << typeREAL << "i_index =         Indexd1Temp[get_local_id(0)];" << std::endl;
      stream_program_src << typeREAL << "i_level_int = Levelintd1Temp[get_local_id(0)];" << std::endl;
    } else if (0 && d_inner == 1 ) {
      stream_program_src << typeREAL << "i_level =         Leveld2Temp[get_local_id(0)];" << std::endl;
      stream_program_src << typeREAL << "i_index =         ptrIndex["<<d_inner*storageSizePadded<<" + (get_global_id(1)+overallMultOffset)*LSIZE];" << std::endl;            
      stream_program_src << typeREAL << "i_level_int =     ptrLevel_int["<<d_inner*storageSizePadded<<" + (get_global_id(1)+overallMultOffset)*LSIZE];" << std::endl;

    } else {
      stream_program_src << typeREAL << "i_level =         ptrLevel["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;
      stream_program_src << typeREAL << "i_index =         ptrIndex["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;            
      stream_program_src << typeREAL << "i_level_int =     ptrLevel_int["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;
    }

    stream_program_src << typeREAL << "j_levelreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3<<"]; " << std::endl;
    stream_program_src << typeREAL << "j_indexreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 1<<"]; " << std::endl;
    stream_program_src << typeREAL << "j_level_intreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 2<<"]; " << std::endl;
    const char* extra = "";
    const char* extra2 = "";
    if (d_inner < dcon) {
      stream_program_src << "l2dotTemp["<<d_inner*LSIZE<<" + get_local_id(0)] = (l2dot(i_level"<<extra2<<",i_index"<<extra<<",i_level_int"<<extra<<", j_levelreg,j_indexreg,j_level_intreg,"<< "ptrLcl_q["<<(d_inner+1)*2-2<<"]));" << std::endl;

      stream_program_src << "gradTemp["<<d_inner*LSIZE<<" + get_local_id(0)] = (gradient(i_level"<<extra2<<",i_index"<<extra<<", j_levelreg,j_indexreg,"<< "ptrLcl_q["<<(d_inner+1)*2-1<<"]));" << std::endl;
    } else {
      stream_program_src << "REAL l2dotreg_"<<d_inner<<" = (l2dot(i_level"<<extra2<<",i_index"<<extra<<",i_level_int"<<extra<<", j_levelreg,j_indexreg,j_level_intreg,"<< "ptrLcl_q["<<(d_inner+1)*2-2<<"]));" << std::endl;

      stream_program_src << "REAL gradreg_"<<d_inner<<" = (gradient(i_level"<<extra2<<",i_index"<<extra<<", j_levelreg,j_indexreg,"<< "ptrLcl_q["<<(d_inner+1)*2-1<<"]));" << std::endl;
    }

  }

  stream_program_src << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  typeREAL = "REAL ";
  stream_program_src << typeREAL<< "alphaTempReg = alphaTemp["<<"k"<<"];" << std::endl;

  for(unsigned d_outer = 0; d_outer < dims; d_outer++)
    {
      if (d_outer == 0)
	{
	  typeREAL = "REAL ";
	} else {
	typeREAL = "";
      }

      stream_program_src << typeREAL<< "element = alphaTempReg;" << std::endl;
      for(unsigned d_inner = 0; d_inner < dims; d_inner++)
	{
	  stream_program_src << "element *= ";
	  if (d_outer == d_inner) {
	    if (d_inner < dcon) {
	      stream_program_src << "(gradTemp["<<d_inner*LSIZE<<" + get_local_id(0)]);";
	    } else {
	      stream_program_src << "(gradreg_"<<d_inner<<");";
	    }
	  } else {
	    if (d_inner < dcon) {
	      stream_program_src << "(l2dotTemp["<<d_inner*LSIZE<<" + get_local_id(0)]);";
	    } else {
	      stream_program_src << "(l2dotreg_"<<d_inner<<");";
	    }
	  }
	  stream_program_src << std::endl;
	}
      stream_program_src << "res += ptrLambda["<<d_outer<<"] * element;" << std::endl;
    }
  stream_program_src << "}" << std::endl;
  stream_program_src << "ptrParResult[((get_group_id(0)+ConstantMemoryOffset)*get_global_size(1) + get_global_id(1))*"<<LSIZE<<" + get_local_id(0)] = res; " << std::endl;
  stream_program_src << "}" << std::endl;

  std::string program_src = stream_program_src.str(); 
  source2 = program_src.c_str();
  #if PRINTOCL
  std::cout <<  source2 << std::endl;
  #endif
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");
  char buildOptions[256];
  int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DDIMS=%lu -cl-finite-math-only  -cl-fast-relaxed-math ", storageSize, storageSizePadded, num_groups, dims);
  if (ierr < 0) {
    printf("Error in Build Options");
    exit(-1);
  }

  err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
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


  kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
  
}


const char* BoundLTwoDotFunction() {

  return "REAL l2dot(REAL lid, \n\
	   REAL iid, \n\
	   REAL in_lid, \n\
	   REAL ljd, \n\
	   REAL ijd,\n\
	   REAL in_ljd,\n\
	   REAL lcl_q)\n\
\n\
{\n\
  double res_one = select(0.0,(2.0/3.0) * in_lid, (ulong)((iid == ijd) && (ljd != 1)));\n\
  ulong selector = (lid > ljd);\n\
  double i1d = select(ijd, iid, selector);\n\
  double in_l1d = select(in_ljd, in_lid,selector);\n\
  double i2d = select(iid, ijd, selector);\n\
  double l2d = select(lid, ljd, selector);\n\
  double in_l2d = select(in_lid, in_ljd, selector);\n\
\n\
  double q = fma(i1d, in_l1d, -in_l1d); \n\
  double p = fma(i1d, in_l1d,  in_l1d); \n\
\n\
  ulong overlap = (max(q, fma(i2d, in_l2d, -in_l2d)) < min(p, fma(i2d, in_l2d,in_l2d)));\n\
  \n\
  double temp_res_inner = 2.0 - fabs(fma(l2d,q,-i2d)) - fabs(fma(l2d,p,-i2d));\n\
\n\
  double temp_res_rightbound = p + q;\n\
  double temp_res_leftbound = 2.0 - temp_res_rightbound;\n\
//  double temp_res = select(0.0, (temp_res_inner), (ulong) (l2d != 1)) + select(0.0, temp_res_leftbound, (ulong)((l2d == 1) && (i2d == 0))) + select(0.0, temp_res_rightbound, (ulong)((l2d == 1) && (i2d == 1)));\n\
  double temp_res = select((temp_res_inner), \n\
     select(temp_res_rightbound, temp_res_leftbound, (ulong)(i2d == 0))\n\
			   , (ulong) (l2d == 1));\n\
\n\
\n\
  temp_res *= (0.5 * in_l1d);\n\
\n\
  double res_two = select(0.0,temp_res,  overlap);	\n\
  return (select(res_two, res_one, (ulong)(lid == ljd))) * lcl_q;\n	\
\n\
}\n"; //"

   }


const char* LaplaceBoundHeader() {
  return "REAL gradient(REAL i_level, \n\
	      REAL i_index, \n\
	      REAL j_level, \n\
	      REAL j_index,\n\
	      REAL lcl_q_inv)\n\
{\n\
  REAL grad;\n\
     \n\
  ulong doGrad = (ulong)((i_level == j_level) && (i_index == j_index) && (i_level != 1.0));\n\
  grad = select(0.0, i_level * 2.0 * lcl_q_inv, doGrad);\n\
//grad = i_level * 2.0 * lcl_q_inv;\n\
\n\
  return grad;\n\
}\n\
\n\
\n\
\n\
\n\
__kernel void multKernelBound(__global  REAL* ptrLevel, \n\
				 __constant  REAL* ptrLevelIndexLevelintcon,\n\
				 __global  REAL* ptrIndex, \n\
				 __global  REAL* ptrLevel_int, \n\
				 __global  REAL* ptrParResult, \n\
				 __global  REAL* ptrAlpha, \n\
				 __constant  REAL* ptrLcl_q,\n\
				 __constant  REAL* ptrLambda,\n\
				 ulong overallMultOffset,\n\
				 ulong ConstantMemoryOffset)\n\
{\n\
    __local REAL alphaTemp[LSIZE];"; //"
				    }

void compileLTwoDotBound(int id, std::string kernel_src, cl_kernel* kernel) {
  cl_int err = CL_SUCCESS;
  const char* source2 = "";
  const char* l2dotfunction = BoundLTwoDotFunction();

  std::stringstream stream_program_src;

  stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
  stream_program_src << "#define REAL double" << std::endl;
  stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
  stream_program_src << l2dotfunction << std::endl;

  stream_program_src << "__kernel void multKernel(__global  REAL* ptrLevel," << std::endl;
  stream_program_src << "			 __constant  REAL* ptrLevelIndexLevelintcon," << std::endl;
  stream_program_src << "			 __global  REAL* ptrIndex, " << std::endl;
  stream_program_src << "			 __global  REAL* ptrLevel_int," << std::endl;
  stream_program_src << "			 __global  REAL* ptrAlpha," << std::endl;
  stream_program_src << "			 __global  REAL* ptrParResult," << std::endl;
  stream_program_src << "			 __constant  REAL* ptrLcl_q," << std::endl;
  stream_program_src << "			 ulong overallMultOffset," << std::endl;
  stream_program_src << "			 ulong ConstantMemoryOffset)" << std::endl;
  stream_program_src << "{" << std::endl;


  stream_program_src << "  __local REAL alphaTemp[LSIZE];" << std::endl;

  stream_program_src << "alphaTemp[get_local_id(0)]   = ptrAlpha[get_global_id(0) + ConstantMemoryOffset*LSIZE];" << std::endl;
  stream_program_src << "ptrLevel += get_local_id(0);" << std::endl;
  stream_program_src << "ptrIndex += get_local_id(0);" << std::endl;
  stream_program_src << "ptrLevel_int += get_local_id(0);" << std::endl;
  stream_program_src << "REAL res = 0.0;" << std::endl;
  
  stream_program_src << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  const char* typeREAL = "";

  stream_program_src << "for (unsigned k = 0; k < "<<LSIZE<<"; k++) {" << std::endl;
  stream_program_src << "REAL element = alphaTemp["<<"k"<<"];" << std::endl;

  for(size_t d_inner = 0; d_inner < dims; d_inner++) {
    if (d_inner == 0)
      {
	typeREAL = "REAL ";
      } else {
      typeREAL = "";
    }
    
    stream_program_src << typeREAL << "i_level =         ptrLevel["<<d_inner*storageInnerSizePaddedBound<<"+ (get_global_id(1)+overallMultOffset)*"<<LSIZE<<"];" << std::endl;
    stream_program_src << typeREAL << "i_index =         ptrIndex["<<d_inner*storageInnerSizePaddedBound<<"+ (get_global_id(1)+overallMultOffset)*"<<LSIZE<<"];" << std::endl;
    stream_program_src << typeREAL << "i_level_int =     ptrLevel_int["<<d_inner*storageInnerSizePaddedBound<<"+ (get_global_id(1)+overallMultOffset)*"<<LSIZE<<"];" << std::endl;

    
    stream_program_src << typeREAL << "j_levelreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3<<"]; " << std::endl;
    stream_program_src << typeREAL << "j_indexreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<< d_inner*3 + 1<<"]; " << std::endl;
    stream_program_src << typeREAL << "j_level_intreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 2<<"]; " << std::endl;
    stream_program_src << "element *= (l2dot(i_level,i_index,i_level_int, j_levelreg,j_indexreg,j_level_intreg,ptrLcl_q["<<d_inner<<"]));" << std::endl;
  }

  stream_program_src << "res += element;" << std::endl;
  stream_program_src << "}" << std::endl;
  stream_program_src << "ptrParResult[((get_group_id(0) + ConstantMemoryOffset)*get_global_size(1) + get_global_id(1))*"<<LSIZE<<" + get_local_id(0)] = res; " << std::endl;
  stream_program_src << "}" << std::endl;

  std::string program_src = stream_program_src.str(); 
  source2 = program_src.c_str();
#if PRINTOCL
  std::cout <<  source2 << std::endl;
#endif
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");
  char buildOptions[256];
  int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DDIMS=%lu -cl-finite-math-only  -cl-fast-relaxed-math ", storageSizeBound, storageSizePaddedBound, num_groupsBound, dims);
  if (ierr < 0) {
    printf("Error in Build Options");
    exit(-1);
  }

  err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
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


  kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
  
}



void compileMultKernel2DBound2Indexes(int id, std::string kernel_src, cl_kernel* kernel) {
  cl_int err = CL_SUCCESS;
  const char* source2 = LaplaceBoundHeader();
  // std::cout << "SOURCE2 " << source2 << std::endl;
  const char* l2dotfunction = BoundLTwoDotFunction();

  std::stringstream stream_program_src;

  stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
  stream_program_src << "#define REAL double" << std::endl;
  stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
  stream_program_src << l2dotfunction << std::endl;
  stream_program_src << source2 << std::endl;
  stream_program_src << "alphaTemp[get_local_id(0)]   = ptrAlpha[get_global_id(0) + overallMultOffset + ConstantMemoryOffset*LSIZE];" << std::endl;
  unsigned dcon = dims;

  stream_program_src << "__local REAL l2dotTemp["<<dcon*LSIZE<<"];" << std::endl ;
  stream_program_src << "__local REAL gradTemp["<<dcon*LSIZE<<"];" << std::endl;

 
  stream_program_src << "ptrLevel += get_local_id(0);" << std::endl;
  stream_program_src << "ptrIndex += get_local_id(0);" << std::endl;
  stream_program_src << "ptrLevel_int += get_local_id(0);" << std::endl;
  stream_program_src << "REAL res = 0.0;" << std::endl;
  
  const char* typeREAL = "";

  stream_program_src << "for (unsigned k = 0; k < "<<LSIZE<<"; k++) {" << std::endl;

  for(size_t d_inner = 0; d_inner < dims; d_inner++) {
    if (d_inner == 0)
      {
	typeREAL = "REAL ";
      } else {
      typeREAL = "";
    }
    
    stream_program_src << typeREAL << "i_level =         ptrLevel["<<d_inner*storageInnerSizePaddedBound<<"+ (get_global_id(1))*"<<LSIZE<<"];" << std::endl;
    stream_program_src << typeREAL << "i_index =         ptrIndex["<<d_inner*storageInnerSizePaddedBound<<"+ (get_global_id(1))*"<<LSIZE<<"];" << std::endl;
    stream_program_src << typeREAL << "i_level_int =     ptrLevel_int["<<d_inner*storageInnerSizePaddedBound<<"+ (get_global_id(1))*"<<LSIZE<<"];" << std::endl;
    
    stream_program_src << typeREAL << "j_levelreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3<<"]; " << std::endl;
    stream_program_src << typeREAL << "j_indexreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<< d_inner*3 + 1<<"]; " << std::endl;
    stream_program_src << typeREAL << "j_level_intreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 2<<"]; " << std::endl;

      stream_program_src << "l2dotTemp["<<d_inner*LSIZE<<" + get_local_id(0)] = (l2dot(i_level,i_index,i_level_int, j_levelreg,j_indexreg,j_level_intreg,"<< "ptrLcl_q["<<(d_inner+1)*2-2<<"]));" << std::endl;

      stream_program_src << "gradTemp["<<d_inner*LSIZE<<" + get_local_id(0)] = (gradient(i_level,i_index, j_levelreg,j_indexreg,"<< "ptrLcl_q["<<(d_inner+1)*2-1<<"]));" << std::endl;
  }

  stream_program_src << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  typeREAL = "REAL ";
  stream_program_src << typeREAL<< "alphaTempReg = alphaTemp["<<"k"<<"];" << std::endl;

  for(unsigned d_outer = 0; d_outer < dims; d_outer++)
    {
      if (d_outer == 0)
	{
	  typeREAL = "REAL ";
	} else {
	typeREAL = "";
      }

      stream_program_src << typeREAL<< "element = alphaTempReg;" << std::endl;
      for(unsigned d_inner = 0; d_inner < dims; d_inner++)
	{
	  stream_program_src << "element *= ";
	  if (d_outer == d_inner) {
	    if (d_inner < dcon) {
	      stream_program_src << "(gradTemp["<<d_inner*LSIZE<<" + get_local_id(0)]);";
	    } else {
	      stream_program_src << "(gradreg_"<<d_inner<<");";
	    }
	  } else {
	    if (d_inner < dcon) {
	      stream_program_src << "(l2dotTemp["<<d_inner*LSIZE<<" + get_local_id(0)]);";
	    } else {
	      stream_program_src << "(l2dotreg_"<<d_inner<<");";
	    }
	  }
	  stream_program_src << std::endl;
	}
      stream_program_src << "res += ptrLambda[" << d_outer << "] * element;" << std::endl;
    }
  stream_program_src << "}" << std::endl;
  stream_program_src << "ptrParResult[((get_group_id(0) + ConstantMemoryOffset)*get_global_size(1) + get_global_id(1))*"<<LSIZE<<" + get_local_id(0)] = res; " << std::endl;
  stream_program_src << "}" << std::endl;

  std::string program_src = stream_program_src.str(); 
  source2 = program_src.c_str();
  #if PRINTOCL
  std::cout <<  source2 << std::endl;
  #endif
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
  oclCheckErr(err, "clCreateProgramWithSource");
  char buildOptions[256];
  int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DDIMS=%lu -cl-finite-math-only  -cl-fast-relaxed-math ", storageSizeBound, storageSizePaddedBound, num_groupsBound, dims);
  if (ierr < 0) {
    printf("Error in Build Options");
    exit(-1);
  }

  err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
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


  kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
  oclCheckErr(err, "clCreateKernel");

  err |= clReleaseProgram(program);
  oclCheckErr(err, "clReleaseProgram");
  
}


void printInfo() {

  char name[1000];
  clGetDeviceInfo(device_ids[0], CL_DEVICE_NAME, sizeof(char)*1000, &name, 0);

  //std::cout << name << std::endl;

  cl_ulong size;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);

  std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << size << std::endl;
  
  cl_int size2;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &size2, 0);

  std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << size2 << std::endl;

  cl_ulong size3;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &size3, 0);

  std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << size3 << std::endl;

}


void SetBuffersBound(REAL * ptrLevel,
		     REAL * ptrIndex,
		     REAL * ptrLevel_int,
		     unsigned localStorageSize,
		     unsigned localdim, sg::base::GridStorage* storage) {
  storageSizeBound = localStorageSize;
  int pad = padding_size - (storageSizeBound % padding_size);
  storageSizePaddedBound = storageSizeBound + pad;
  dims = localdim;
  cl_ulong sizec;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &sizec, 0);
  constant_mem_size = sizec;

  num_groupsBound = (storageSizePaddedBound) / LSIZE / num_devices;      
  int level_size = storageSizePaddedBound * dims;
  int index_size = storageSizePaddedBound * dims;
  int level_int_size = storageSizePaddedBound * dims;
  int alpha_size = storageSizePaddedBound;
  // int result_size = storageSizePaddedBound;
  int lambda_size = dims;
  par_result_sizeBound = storageSizePaddedBound*num_groupsBound * num_devices;
  lcl_q_sizeBound      = dims+dims;
  constant_mem_size = constant_mem_size - lcl_q_sizeBound * sizeof(REAL)
    -lambda_size * sizeof(REAL);
  alphaend_sizeBound = pad;


  
#if PRINTBUFFERSIZES
  std::cout << "storageSizeBound " << storageSizeBound << std::endl;
  std::cout << "storageSizePaddedBound " << storageSizePaddedBound << std::endl;
  std::cout << "dims " << dims << std::endl;
  std::cout << "num_groupsBound " << num_groupsBound << std::endl;
  std::cout << "index_size " << index_size << std::endl;
  std::cout << "level_int_size " << level_int_size << std::endl;
  std::cout << "alpha_size " << alpha_size << std::endl;
  std::cout << "par_result_sizeBound " << par_result_sizeBound << std::endl;
  std::cout << "lcl_q_sizeBound " << lcl_q_sizeBound << std::endl;
#endif
  //       std::cout << level_size << " " << index_size << std::endl;
  cl_int ciErrNum = CL_SUCCESS;

  ptrLcl_qBound = (REAL *)calloc(lcl_q_sizeBound,sizeof(REAL));
  ptrAlphaEndBound = (REAL *)calloc(alphaend_sizeBound,sizeof(REAL));

  unsigned innerpoints = storage->getNumInnerPoints();
  unsigned pad2 = padding_size - (innerpoints % padding_size);
  storageInnerSizeBound = innerpoints;
  storageInnerSizePaddedBound = innerpoints + pad2;

  InnerSizeBound = storageInnerSizePaddedBound*dims;
  Inner_num_groupsBound = (storageSizePaddedBound) / LSIZE / num_devices;      
  Inner_par_result_sizeBound = storageInnerSizePaddedBound*Inner_num_groupsBound * num_devices;
  Inner_result_sizeBound = storageInnerSizePaddedBound;


  ptrLevelTBound = (REAL *)calloc(InnerSizeBound,sizeof(REAL));
  ptrIndexTBound = (REAL *)calloc(InnerSizeBound,sizeof(REAL));
  ptrLevel_intTBound = (REAL *)calloc(InnerSizeBound,sizeof(REAL));
  offsetBound = (unsigned *)calloc(storageInnerSizePaddedBound,sizeof(unsigned));
  ptrResultBound = (REAL *)calloc(Inner_result_sizeBound, sizeof(REAL));
  ptrResultTempBound = (REAL *)calloc(Inner_result_sizeBound * num_devices,sizeof(REAL));
  ptrResultZeroBound = (REAL *)calloc(Inner_result_sizeBound,sizeof(REAL));

  unsigned num = 0;
  for (unsigned i = 0; i < storage->size(); i++) {
    sg::base::GridIndex* curPoint = (*storage)[i];
    if (curPoint->isInnerPoint()) {
      offsetBound[num] = i;
      num++;
    }
  }


  boundarytransposer(ptrLevelTBound, ptrLevel, dims, storageSizeBound,storage);
  boundarytransposer(ptrIndexTBound, ptrIndex, dims, storageSizeBound,storage);
  boundarytransposer(ptrLevel_intTBound, ptrLevel_int, dims, storageSizeBound,storage);


  cl_ulong size3;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size3, 0);

  cl_ulong size4;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(cl_ulong), &size4, 0);

  size_t sizedoubles = size3 / sizeof(REAL);
  size_t sizedoubles64 = sizedoubles;
  size_t gpu_max_buffer_size = size4 / sizeof(REAL);
  gpu_max_buffer_size = gpu_max_buffer_size - (gpu_max_buffer_size % (storageSizePaddedBound));

  size_t memoryOther = Inner_result_sizeBound + 3* InnerSizeBound + lcl_q_sizeBound;
  size_t memoryNonParResult = level_size + index_size + level_int_size + alpha_size + lambda_size;
  size_t memoryLeftover = sizedoubles64 - memoryOther - memoryNonParResult;
  Inner_par_result_max_sizeBound = memoryLeftover - (memoryLeftover % storageSizePaddedBound);


    
  max_buffer_size = MAXBYTES/sizeof(REAL) - memoryOther - memoryNonParResult;
  max_buffer_size = max_buffer_size - ((max_buffer_size) % (storageSizePaddedBound));

  Inner_par_result_max_sizeBound = std::min(Inner_par_result_max_sizeBound, max_buffer_size);

  Inner_par_result_max_sizeBound = std::min(Inner_par_result_max_sizeBound, Inner_par_result_sizeBound);
  Inner_par_result_max_sizeBound = std::min(Inner_par_result_max_sizeBound, gpu_max_buffer_size) / num_devices;

  ptrParResultBound = (REAL *)calloc(Inner_par_result_max_sizeBound, sizeof(REAL));

  ptrLevelIndexLevelintBound = (REAL *)calloc(level_size+index_size+level_int_size,sizeof(REAL));
  unsigned three_d = 0;
  unsigned threedims = 3*dims;
  for(unsigned i = 0; i < storageSizeBound; i++) {
    for(unsigned d = 0; d < dims; d++) {
      ptrLevelIndexLevelintBound[i * threedims + three_d] = ptrLevel[i * dims + d];
      three_d += 1;
      ptrLevelIndexLevelintBound[i * threedims + three_d] = ptrIndex[i * dims + d];
      three_d += 1;
      ptrLevelIndexLevelintBound[i * threedims + three_d] = ptrLevel_int[i * dims + d];
      three_d += 1;
    }
    three_d = 0;
  }

  constant_buffer_size = (constant_mem_size / sizeof(REAL));
  unsigned mod_res = constant_buffer_size % (3*dims*padding_size); // Constant buffer has the normal padding
  constant_buffer_size = constant_buffer_size - mod_res;
  constant_buffer_iterations = constant_buffer_size / (3*dims);

  

  for(unsigned int i=0; i < num_devices; ++i) 
    {
      d_ptrLevelBound[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					  InnerSizeBound* sizeof(REAL), ptrLevelTBound, &ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrLevel");
      d_ptrLevelIndexLevelintconBound[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
							  constant_buffer_size*sizeof(REAL), NULL, &ciErrNum);
      oclCheckErr(ciErrNum, "clCreateBuffer ptrLevelIndexLevelint");

      d_ptrIndexBound[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					  InnerSizeBound* sizeof(REAL), ptrIndexTBound, &ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrIndex");

      d_ptrLevel_intBound[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					      InnerSizeBound* sizeof(REAL), ptrLevel_intTBound, &ciErrNum);
      oclCheckErr(ciErrNum, "clCreateBuffer ptrLevel_int");

      d_ptrAlphaBound[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
					  alpha_size*sizeof(REAL), NULL, &ciErrNum);
  

      oclCheckErr(ciErrNum, "clCreateBuffer ptrAlpha");

     


      //RED
      d_ptrResultBound[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY ,
					   Inner_result_sizeBound*sizeof(REAL), NULL,&ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrResult");
      d_ptrParResultBound[i] = clCreateBuffer(context, CL_MEM_READ_WRITE ,
					      Inner_par_result_max_sizeBound*sizeof(REAL), NULL,&ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrParResult");

      d_ptrLcl_qBound[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
					  lcl_q_sizeBound*sizeof(REAL), NULL, &ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrLcl_q");

    }

}

void SetLambdaBufferLaplaceBound(
		     REAL * ptrLambda,
		     unsigned localdim) {
  int lambda_size = localdim;
  cl_int ciErrNum = CL_SUCCESS;

  for(unsigned int i=0; i < num_devices; ++i) 
    {

      d_ptrLambdaBound[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   lambda_size*sizeof(REAL), ptrLambda, &ciErrNum);
      oclCheckErr(ciErrNum, "clCreateBuffer");

    }

}


double AccumulateTiming(cl_event *GPUExecution, const char* kernelname, size_t gpuid) {
  cl_int ciErrNum = CL_SUCCESS;

  cl_ulong startTime, endTime;
  ciErrNum = clGetEventProfilingInfo(GPUExecution[gpuid], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
  oclCheckErr(ciErrNum, "clGetEventProfilingInfo1");
  ciErrNum = clGetEventProfilingInfo(GPUExecution[gpuid], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
  oclCheckErr(ciErrNum, "clGetEventProfilingInfo2");
  return (double)(endTime - startTime);

}

void SetBuffersInner(REAL * ptrLevel,
		     REAL * ptrIndex,
		     REAL * ptrLevel_int,
		     unsigned localStorageSize,
		     unsigned localdim, sg::base::GridStorage* storage) {
  storageSize = localStorageSize;
  unsigned pad = padding_size - (storageSize % padding_size);
  storageSizePadded = storageSize + pad;
  dims = localdim;
  cl_ulong sizec;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &sizec, 0);
  constant_mem_size = sizec;


  num_groups = (storageSizePadded) / LSIZE / num_devices;      
  unsigned level_size = storageSizePadded * dims;
  unsigned index_size = storageSizePadded * dims;
  unsigned level_int_size = storageSizePadded * dims;
  unsigned alpha_size = storageSizePadded;
  unsigned lambda_size = dims;
  unsigned result_size = storageSizePadded; 
  par_result_size = storageSizePadded * num_groups *num_devices;
  lcl_q_size      = dims+dims;
  constant_mem_size = constant_mem_size - lcl_q_size * sizeof(REAL)
    -lambda_size * sizeof(REAL);
  alphaend_size = pad;

#if PRINTBUFFERSIZES
  std::cout << "storageSize " << storageSize << std::endl;
  std::cout << "storageSizePadded " << storageSizePadded << std::endl;
  std::cout << "dims " << dims << std::endl;
  std::cout << "num_groups " << num_groups << std::endl;
  std::cout << "index_size " << index_size << std::endl;
  std::cout << "level_int_size " << level_int_size << std::endl;
  std::cout << "alpha_size " << alpha_size << std::endl;
  std::cout << "result_size " << result_size << std::endl;
  std::cout << "par_result_size " << par_result_size << std::endl;
  std::cout << "lcl_q_size " << lcl_q_size << std::endl;

#endif
  //       std::cout << level_size << " " << index_size << std::endl;
  cl_int ciErrNum = CL_SUCCESS;
  ptrLevelTInner = (REAL *)calloc(level_size,sizeof(REAL));
  ptrIndexTInner = (REAL *)calloc(index_size,sizeof(REAL));
  ptrLevel_intTInner = (REAL *)calloc(level_int_size,sizeof(REAL));
  transposer(ptrLevelTInner, ptrLevel, dims, storageSizePadded,storageSize);
  transposer(ptrIndexTInner, ptrIndex, dims, storageSizePadded,storageSize);
  transposer(ptrLevel_intTInner, ptrLevel_int, dims, storageSizePadded,storageSize);

  ptrResultTemp = (REAL *)calloc(result_size * num_devices,sizeof(REAL));
  ptrResultZero = (REAL *)calloc(result_size,sizeof(REAL));
  ptrLcl_qInner = (REAL *)calloc(lcl_q_size,sizeof(REAL));
  ptrAlphaEndInner = (REAL *)calloc(alphaend_size,sizeof(REAL));
  
  ptrLevelIndexLevelintInner = (REAL *)calloc(level_size+index_size+level_int_size,sizeof(REAL));

  unsigned three_d = 0;
  unsigned threedims = 3*dims;
  for(unsigned i = 0; i < storageSize; i++) {
    for(unsigned d = 0; d < dims; d++) {
      ptrLevelIndexLevelintInner[i * threedims + three_d] = ptrLevel[i * dims + d];
      three_d += 1;
      ptrLevelIndexLevelintInner[i * threedims + three_d] = ptrIndex[i * dims + d];
      three_d += 1;
      ptrLevelIndexLevelintInner[i * threedims + three_d] = ptrLevel_int[i * dims + d];
      three_d += 1;
    }
    three_d = 0;
  }

  cl_ulong size3;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size3, 0);

  cl_ulong size4;
  clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(cl_ulong), &size4, 0);


  size_t sizedoubles64 = size3 / sizeof(REAL);
  size_t gpu_max_buffer_size = size4 / sizeof(REAL);
  gpu_max_buffer_size = gpu_max_buffer_size - (gpu_max_buffer_size % (storageSizePadded));
  size_t memoryNonParResult = level_size + index_size + level_int_size + alpha_size + result_size;
  size_t memoryLeftover = sizedoubles64 - memoryNonParResult;
  par_result_max_size = memoryLeftover - (memoryLeftover % storageSizePadded);
  par_result_max_size = std::min(par_result_max_size, par_result_size);
  par_result_max_size = std::min(par_result_max_size, gpu_max_buffer_size) / num_devices;
//   std::cout << " num_groups " << num_groups << std::endl;  
//   std::cout << " par_result_max_size " << par_result_max_size << std::endl;
  ptrParResultInner = (REAL *)calloc(par_result_max_size,sizeof(REAL));


  constant_buffer_size_noboundary = (constant_mem_size / sizeof(REAL));

  unsigned mod_res = constant_buffer_size_noboundary % (3*dims*padding_size);
  constant_buffer_size_noboundary = constant_buffer_size_noboundary - mod_res;
  constant_buffer_iterations_noboundary = constant_buffer_size_noboundary / (3*dims);
  
//   std::cout << "constant_buffer_size_noboundary " << constant_buffer_size_noboundary << std::endl;
//   std::cout << "constant_buffer_iterations_noboundary " << constant_buffer_iterations_noboundary << std::endl;
  
  for(unsigned int i=0; i < num_devices; ++i) 
    {

      d_ptrLevelInner[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					  level_size* sizeof(REAL), ptrLevelTInner, &ciErrNum);


      d_ptrLevelIndexLevelintconInner[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
							  constant_buffer_size_noboundary*sizeof(REAL), NULL, &ciErrNum);
      oclCheckErr(ciErrNum, "clCreateBuffer ptrLevelIndexLevelint");



      d_ptrIndexInner[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					  index_size* sizeof(REAL), ptrIndexTInner, &ciErrNum);
      oclCheckErr(ciErrNum, "clCreateBuffer ptrIndex");
      
      d_ptrLevel_intInner[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					      level_int_size* sizeof(REAL), ptrLevel_intTInner , &ciErrNum);
      oclCheckErr(ciErrNum, "clCreateBuffer ptrLevel_int");

      d_ptrAlphaInner[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
					  alpha_size*sizeof(REAL), NULL, &ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrAlpha");



      //RED
      d_ptrResultInner[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY ,
					   result_size*sizeof(REAL), NULL,&ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrResult");

      d_ptrParResultInner[i] = clCreateBuffer(context, CL_MEM_READ_WRITE ,
					      par_result_max_size*sizeof(REAL), NULL,&ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrParResult");



      d_ptrLcl_qInner[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
					  lcl_q_size*sizeof(REAL), NULL, &ciErrNum);

      oclCheckErr(ciErrNum, "clCreateBuffer ptrLcl_q");

    }

}

void SetLambdaBufferLaplaceInner(
			  REAL * ptrLambda,
			  unsigned localdim) {
  unsigned lambda_size = localdim;
  cl_int ciErrNum = CL_SUCCESS;
  
  for(unsigned int i=0; i < num_devices; ++i) 
    {

      d_ptrLambdaInner[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   lambda_size*sizeof(REAL), ptrLambda, &ciErrNum);
      oclCheckErr(ciErrNum, "clCreateBuffer ptrLambda");

    }

}

void SetArgumentsLaplaceInner() {
  cl_int ciErrNum = CL_SUCCESS;
  int counter = 0;

  for(unsigned int i=0; i < num_devices; ++i) 
    {
      ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelInner[i]);
      ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelIndexLevelintconInner[i]);

      ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrIndexInner[i]);
      ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevel_intInner[i]);

      ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrAlphaInner[i]);
      ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultInner[i]);
      ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLcl_qInner[i]);
      ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLambdaInner[i]);
      oclCheckErr(ciErrNum, "clSetKernelArg1 Kernel Construct");

      counter = 0;
      ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrResultInner[i]);
      ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultInner[i]);
      counter = 0;
    }
      
  //printInfo();

}


void SetArgumentsLaplaceBound() {
  cl_int ciErrNum = CL_SUCCESS;
  int counter = 0;
  for(unsigned int i=0; i < num_devices; ++i) 
    {
      ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelBound[i]);
      ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelIndexLevelintconBound[i]);

      ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrIndexBound[i]);
      ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevel_intBound[i]);
      ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultBound[i]);
      ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrAlphaBound[i]);
      ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLcl_qBound[i]);
      ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLambdaBound[i]);
      oclCheckErr(ciErrNum, "clSetKernelArg1 Kernel Construct");

      counter = 0;
      ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrResultBound[i]);
      ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultBound[i]);
      counter = 0;
    }
      
  //printInfo();
  //   std::cout << "Using " << LSIZE*(5*dims+1)*sizeof(REAL) << std::endl;
  //exit(-1);


}


void PrintGFLOPS(double LaplaceInnerTime, double LaplaceBoundTime,double LTwoDotInnerTime, double LTwoDotBoundTime) {

  double TotalCalculationTime = LaplaceInnerTime + LaplaceBoundTime + LTwoDotInnerTime + LTwoDotBoundTime;
  double LaplaceInnerWeight = LaplaceInnerTime / TotalCalculationTime;
  double LaplaceBoundWeight = LaplaceBoundTime / TotalCalculationTime;
  double LTwoDotInnerWeight = LTwoDotInnerTime / TotalCalculationTime;
  double LTwoDotBoundWeight = LTwoDotBoundTime / TotalCalculationTime;
  double AverageGFLOPS = 0.0;

  double StorD= (double)storageSize;
  double dimsD= (double)dims;

  double StorDInnersize= (double)storageInnerSizeBound;
  double StorDBound = (double)storageSizeBound;

//   double GPUNVDMAXFLOPS = 660 * 0.75;
//   double GPUNVDMAXFLOPSHALF = 660 * 0.375;

  // Laplace Bound
  
  double numberOfFlopBound = StorDInnersize*StorDBound*(dimsD*(28.0 + dimsD));

  // Calculation of FMA
//   double FMAFLOPBound = StorDInnersize*StorDBound*(dimsD*6.0 + dimsD);
//   // Calculation of Select, AND, EQ
//   double OTHEROPSBound = StorDInnersize*StorDBound*(dimsD*(26.0 + dimsD));

//   double TOTALBound = numberOfFlopBound + OTHEROPSBound;
  

//   double procentFMABound = FMAFLOPBound / TOTALBound; // FLOP running at full speed
//   double procentFLOPBound = ((numberOfFlopBound - FMAFLOPBound) / TOTALBound); // FLOP running at half speed, since there is no FMA.

//   double THEORETICALBound = (GPUNVDMAXFLOPS * procentFMABound + GPUNVDMAXFLOPSHALF *  procentFLOPBound);

  if (!isFirstTimeLaplaceBound) {
    double LaplaceBoundGFLOPS = ((numberOfFlopBound*CounterLaplaceBound)/LaplaceBoundTime)*(1e-9);
    AverageGFLOPS += LaplaceBoundGFLOPS * LaplaceBoundWeight;
//     std::cout << "THEORETICAL LaplaceBound GFLOPS: " << (THEORETICALBound) << std::endl;
//     std::cout << "ACTUAL      LaplaceBound GFLOPS: " << LaplaceBoundGFLOPS << std::endl;
//     std::cout << "TIMESCALLED LaplaceBound:        " << CounterLaplaceBound << std::endl;
  }

  // LTwoDot Bound
  
  numberOfFlopBound = StorDInnersize*StorDBound*(dimsD*(24.0 + 1) + 1);

  // Calculation of FMA
//   FMAFLOPBound = StorDInnersize*StorDBound*(dimsD*6.0);
//   // Calculation of Select, AND, EQ
//   OTHEROPSBound = StorDInnersize*StorDBound*(dimsD*(20.0 + dimsD));

//   TOTALBound = numberOfFlopBound + OTHEROPSBound;

//   procentFMABound = FMAFLOPBound / TOTALBound; // FLOP running at full speed
//   procentFLOPBound = ((numberOfFlopBound - FMAFLOPBound) / TOTALBound); // FLOP running at half speed, since there is no FMA.

//   THEORETICALBound = (GPUNVDMAXFLOPS * procentFMABound + GPUNVDMAXFLOPSHALF *  procentFLOPBound);
  if (!isFirstTimeLTwoDotBound) {
    double LTwoDotBoundGFLOPS = ((numberOfFlopBound*CounterLTwoDotBound)/LTwoDotBoundTime)*(1e-9);
    AverageGFLOPS += LTwoDotBoundGFLOPS * LTwoDotBoundWeight;
//     std::cout << "THEORETICAL LTwoDotBound GFLOPS: " << (THEORETICALBound) << std::endl;
//     std::cout << "ACTUAL      LTwoDotBound GFLOPS: " << LTwoDotBoundGFLOPS << std::endl;
//     std::cout << "TIMESCALLED LTwoDotBound:        " << CounterLTwoDotBound << std::endl;
  }



  // Laplace Inner
  double numberOfFlop = StorD*StorD*(dimsD*(26.0 + dimsD));

  // Calculation of FMA
//   double FMAFLOP = StorD*StorD*(dimsD*7.0 + dimsD);
//   // Calculation of Select, AND, EQ
//   double OTHEROPS = StorD*StorD*(dimsD*(16.0 + dimsD));

//   double TOTAL = numberOfFlop + OTHEROPS;

//   double procentFMA = FMAFLOP / TOTAL; // FLOP running at full speed
//   double procentFLOP = (numberOfFlop - FMAFLOP) / TOTAL; // FLOP running at half speed, since there is no FMA.


//   double THEORETICAL = (GPUNVDMAXFLOPS * procentFMA + GPUNVDMAXFLOPSHALF * procentFLOP);
  if (!isFirstTimeLaplaceInner) {
    double LaplaceInnerGFLOPS = ((numberOfFlop*CounterLaplaceInner)/LaplaceInnerTime)*(1e-9);
    AverageGFLOPS += LaplaceInnerGFLOPS * LaplaceInnerWeight;
//     std::cout << "THEORETICAL LaplaceInner GFLOPS: " << (THEORETICAL) << std::endl;
//     std::cout << "ACTUAL      LaplaceInner GFLOPS: " << LaplaceInnerGFLOPS << std::endl;
//     std::cout << "TIMESCALLED LaplaceInner:        " << CounterLaplaceInner << std::endl;
  }

  // LTwoDot Inner
  numberOfFlop = StorD*StorD*(dimsD*(22.0 + 1) + 1);

  // Calculation of FMA
//   FMAFLOP = StorD*StorD*(dimsD*7.0);
//   // Calculation of Select, AND, EQ
//   OTHEROPS = StorD*StorD*(dimsD*(12.0 + dimsD));

//   TOTAL = numberOfFlop + OTHEROPS;

//   procentFMA = FMAFLOP / TOTAL; // FLOP running at full speed
//   procentFLOP = (numberOfFlop - FMAFLOP) / TOTAL; // FLOP running at half speed, since there is no FMA.

//   THEORETICAL = (GPUNVDMAXFLOPS * procentFMA + GPUNVDMAXFLOPSHALF * procentFLOP);

  if (!isFirstTimeLTwoDotInner) {
    double LTwoDotInnerGFLOPS = ((numberOfFlop*CounterLTwoDotInner)/LTwoDotInnerTime)*(1e-9);
    AverageGFLOPS += LTwoDotInnerGFLOPS * LTwoDotInnerWeight;
//     std::cout << "THEORETICAL LTwoDotInner GFLOPS: " << (THEORETICAL) << std::endl;
//     std::cout << "ACTUAL      LTwoDotInner GFLOPS: " << LTwoDotInnerGFLOPS << std::endl;
//     std::cout << "TIMESCALLED LTwoDotInner:        " << CounterLTwoDotInner << std::endl;

  }

    std::cout << "Average GFLOPS: " << AverageGFLOPS << std::endl;


}



void CompileLaplaceBoundKernels() {
  for(unsigned int i=0; i < num_devices; ++i) 
    {
      compileMultKernel2DBound2Indexes(i, "multKernelBound", LaplaceBoundKernel);

      compileReduceBound(i, "ReduceBoundKernel",ReduceBoundKernel);
    }
}

void CompileLaplaceInnerKernels() {
  for(unsigned int i=0; i < num_devices; ++i) 
    {
      compileMultKernel2DIndexesJInReg2(i, "multKernel", LaplaceInnerKernel);

      compileReduceInner(i, "ReduceInnerKernel", ReduceInnerKernel);
    }
}

void ExecLaplaceBound(REAL * ptrAlpha, 
		      REAL * ptrResult,
		      REAL * lcl_q,
		      REAL * lcl_q_inv, 
		      REAL * ptrLevel, 
		      REAL * ptrIndex, 
		      REAL *ptrLevel_int, 
		      unsigned argStorageSize, 
		      unsigned argStorageDim) {
  cl_int ciErrNum = CL_SUCCESS;
  cl_event GPUDone[NUMDEVS];
  cl_event GPUDoneLcl[NUMDEVS];
  cl_event GPUExecution[NUMDEVS];
  
  size_t idx = 0;
  for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
    ptrLcl_qBound[idx++] = lcl_q[d_outer];
    ptrLcl_qBound[idx++] = lcl_q_inv[d_outer];
  }

  for(size_t i = 0; i < num_devices; i++) {

    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaBound[i], CL_FALSE, 0,
				     storageSizeBound*sizeof(REAL), ptrAlpha, 0, 0, &GPUDone[i]);
    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qBound[i], CL_FALSE, 0,
				     lcl_q_sizeBound*sizeof(REAL), ptrLcl_qBound, 0, 0, &GPUDoneLcl[i]);

  }
  oclCheckErr(ciErrNum, "clEnqueueWriteBuffer L2225");
  clWaitForEvents(num_devices, GPUDone);
  clWaitForEvents(num_devices, GPUDoneLcl);

  for(size_t i = 0; i < num_devices; i++) {
    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrResultBound[i], 
				     CL_FALSE, 0,
				     Inner_result_sizeBound*sizeof(REAL), 
				     ptrResultZeroBound, 0, 0, &GPUDone[i]);

    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaBound[i], CL_FALSE, storageSizeBound*sizeof(REAL),
				     alphaend_sizeBound*sizeof(REAL), ptrAlphaEndBound, 0, 0, &GPUExecution[i]);
  }
  oclCheckErr(ciErrNum, "clEnqueueWriteBuffer LapB L1778");
  clWaitForEvents(num_devices, GPUDone);
  // 	clWaitForEvents(num_devices, GPUExecution);

  {
    size_t multglobalworksize[2];
    // 2D Bound
    //     std::cout << " Inner_par_result_max_sizeBound " << Inner_par_result_max_sizeBound << std::endl;
    size_t storageSizePaddedStep = std::min(storageSizePaddedBound / num_devices, Inner_par_result_max_sizeBound / (storageSizePaddedBound) * LSIZE);
    multglobalworksize[0] = std::min(storageSizePaddedBound,storageSizePaddedStep);
    multglobalworksize[1] = storageInnerSizePaddedBound/ LSIZE;
    //     std::cout << " storageSizePaddedStep " << storageSizePaddedStep << std::endl;
    //     std::cout << " multglobalworksize[1] " << multglobalworksize[1] << std::endl;
    size_t multglobal =  storageSizePaddedBound / num_devices;
    //     std::cout << " multglobal " << multglobal << std::endl;

    for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[0], multglobal - overallMultOffset)) {
      multglobalworksize[0] = std::min(multglobalworksize[0], multglobal-overallMultOffset);

      for(unsigned int i = 0; i < num_devices; i++) {
	size_t overallMultOffset2 = i*multglobal + overallMultOffset;
	ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], 8, sizeof(cl_ulong), (void *) &overallMultOffset2);
	oclCheckErr(ciErrNum, "clSetKernelArgL1660");
      }
      
      size_t constantglobalworksize[2];
      size_t constantlocalworksize[2];
      size_t constantglobal = multglobalworksize[0];
      constantglobalworksize[0] = std::min(constant_buffer_iterations,constantglobal);
      constantglobalworksize[1] = multglobalworksize[1]; // already LSIZE pad
      constantlocalworksize[0] = LSIZE;
      constantlocalworksize[1] = 1;
      for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < constantglobal; ConstantMemoryOffset+= std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset)) {
	constantglobalworksize[0] = std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset);

	for(unsigned int i = 0; i < num_devices; i++) {
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], 
					   d_ptrLevelIndexLevelintconBound[i], 
					   CL_TRUE, 0 ,
					   constantglobalworksize[0]*3*dims*sizeof(REAL), 
					   ptrLevelIndexLevelintBound+(overallMultOffset + i*multglobal + ConstantMemoryOffset)*3*dims, 
					   1, 
					   &GPUExecution[i], NULL);
	  oclCheckErr(ciErrNum, "clEnqueueWriteBufferOCLLapBoundL2157");
	  size_t jj =  ConstantMemoryOffset / LSIZE ;
	  ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], 9, sizeof(cl_ulong), (void *) &jj);
	  oclCheckErr(ciErrNum, "clSetKernelArgOCLLapBoundL75");
	  ciErrNum = clEnqueueNDRangeKernel(command_queue[i], LaplaceBoundKernel[i], 2, 0, constantglobalworksize, constantlocalworksize,
					    0, NULL, &GPUExecution[i]);
	  oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2195");

	}
      }
      //       for(unsigned int i = 0; i < num_devices; i++) {
      // 	ciErrNum = clFinish(command_queue[i]);
      // 	oclCheckErr(ciErrNum, "clFinish");
      //       }


      size_t overallReduceOffset = 0;
      for(unsigned int i = 0; i < num_devices; i++) {
	ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
	oclCheckErr(ciErrNum, "clSetKernelArgL1201");

	size_t newnum_groups = multglobalworksize[0] / LSIZE ;

	ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], 3, sizeof(cl_ulong), (void *) &newnum_groups);
	oclCheckErr(ciErrNum, "clSetKernelArgL1205");
	size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
	size_t local2[] = {LSIZE,1};
	ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], ReduceBoundKernel[i], 2, 0, reduceglobalworksize2, local2,
					   0, NULL, &GPUExecution[i]);
	oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
      }
      for(unsigned int i = 0; i < num_devices; i++) {
	ciErrNum = clFinish(command_queue[i]);
	oclCheckErr(ciErrNum, "clFinish");      
      }
    }
  }


  {
    if (num_devices > 1) {
      for(unsigned int i = 0;i < num_devices; i++) 	{    
	clEnqueueReadBuffer(command_queue[i], d_ptrResultBound[i], CL_FALSE, 0,
			    storageInnerSizeBound*sizeof(REAL), 
			    ptrResultTempBound + i*storageInnerSizePaddedBound, 0, NULL, &GPUDone[i]);
      }
      oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL2145");
      ciErrNum |= clWaitForEvents(num_devices, GPUDone);
      oclCheckErr(ciErrNum, "clWaitForEventsLapIL2147");
      for (size_t i = 0; i < storageInnerSizeBound; i++) {
	ptrResultBound[i] = 0.0;
      }

      for (size_t j = 0; j < num_devices; j++) {
	for (size_t i = 0; i < storageInnerSizeBound; i++) {
	  ptrResultBound[i] += ptrResultTempBound[j*storageInnerSizePaddedBound + i];

	}
      }

    } else {
      for(unsigned int i = 0;i < num_devices; i++) 
	{    
	  clEnqueueReadBuffer(command_queue[i], d_ptrResultBound[i], CL_FALSE, 0,
			      storageInnerSizeBound*sizeof(REAL), ptrResultBound, 0, NULL, &GPUDone[i]);
	}
      clWaitForEvents(num_devices, GPUDone);
    
    }
    for (unsigned i = 0; i < storageInnerSizeBound; i++) {
      ptrResult[offsetBound[i]] = ptrResultBound[i];
    }


  }
#if TOTALTIMING
  CounterLaplaceBound += 1.0;
#endif	

  for(size_t i = 0; i < num_devices; i++) {
    clReleaseEvent(GPUExecution[i]);
    clReleaseEvent(GPUDone[i]);
    clReleaseEvent(GPUDoneLcl[i]);
  }  
} // ExecLaplaceBound
// void ExecLaplaceBound(REAL * ptrAlpha, REAL * ptrResult,REAL * lcl_q,REAL * lcl_q_inv, REAL *ptrLevel, REAL *ptrIndex, REAL *ptrLevel_int, unsigned argStorageSize, unsigned argStorageDim) {
//   cl_int ciErrNum = CL_SUCCESS;
//   cl_event GPUDone[NUMDEVS];
//   cl_event GPUDoneLcl[NUMDEVS];
//   cl_event GPUExecution[NUMDEVS];
  
//   size_t idx = 0;
//   for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
//     ptrLcl_qBound[idx++] = lcl_q[d_outer];
//     ptrLcl_qBound[idx++] = lcl_q_inv[d_outer];
//   }

//   for(size_t i = 0; i < num_devices; i++) {

//     ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaBound[i], CL_FALSE, 0,
// 				     storageSizeBound*sizeof(REAL), ptrAlpha, 0, 0, &GPUDone[i]);
//     ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qBound[i], CL_FALSE, 0,
// 				     lcl_q_sizeBound*sizeof(REAL), ptrLcl_qBound, 0, 0, &GPUDoneLcl[i]);

//   }
//   oclCheckErr(ciErrNum, "clEnqueueWriteBuffer L2225");
//   clWaitForEvents(num_devices, GPUDone);
//   clWaitForEvents(num_devices, GPUDoneLcl);

//   for(size_t i = 0; i < num_devices; i++) {
//     ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaBound[i], CL_FALSE, storageSizeBound*sizeof(REAL),
// 				     alphaend_sizeBound*sizeof(REAL), ptrAlphaEndBound, 0, 0, &GPUDone[i]);
//   }
//   oclCheckErr(ciErrNum, "clEnqueueWriteBuffer mult");
//   clWaitForEvents(num_devices, GPUDone);

//   {
//     size_t multglobalworksize[2];
//     // 2D Bound
//     size_t storageInnerSizePaddedStep = std::min(storageInnerSizePaddedBound, Inner_par_result_max_sizeBound / Inner_num_groupsBound) / LSIZE;
//     multglobalworksize[0] = storageSizePaddedBound;
//     multglobalworksize[1] = std::min(storageInnerSizePaddedBound/LSIZE,storageInnerSizePaddedStep);

//     size_t multglobal = storageInnerSizePaddedBound/ LSIZE / num_devices;

//     for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[1], multglobal - overallMultOffset)) {
//       multglobalworksize[1] = std::min(multglobalworksize[1], multglobal-overallMultOffset);

//       for(unsigned int i = 0; i < num_devices; i++) {
// 	size_t overallMultOffset2 = i*multglobal + overallMultOffset;
// 	ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], 8, sizeof(cl_ulong), (void *) &overallMultOffset2);
// 	oclCheckErr(ciErrNum, "clSetKernelArgL1660");
//       }
      
//       size_t constantglobalworksize[2];
//       size_t constantlocalworksize[2];
//       constantglobalworksize[0] = std::min(constant_buffer_iterations,storageSizePaddedBound);
//       constantglobalworksize[1] = multglobalworksize[1]; // already LSIZE pad
//       constantlocalworksize[0] = LSIZE;
//       constantlocalworksize[1] = 1;
//       for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < storageSizePaddedBound; ConstantMemoryOffset+= std::min(constantglobalworksize[0],storageSizePaddedBound-ConstantMemoryOffset)) {
// 	constantglobalworksize[0] = std::min(constantglobalworksize[0],storageSizePaddedBound-ConstantMemoryOffset);
// 	size_t jj = ConstantMemoryOffset / LSIZE;

// 	for(unsigned int i = 0; i < num_devices; i++) {
// 	  ciErrNum |= clSetKernelArg(LaplaceBoundKernel[i], 9, sizeof(cl_ulong), (void *) &jj);
// 	  oclCheckErr(ciErrNum, "clSetKernelArgL2150");
// 	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLevelIndexLevelintconBound[i], CL_TRUE, 0 ,
// 					   constantglobalworksize[0]*3*dims*sizeof(REAL), ptrLevelIndexLevelintBound+ConstantMemoryOffset*3*dims, 0, 0, NULL);
// 	  oclCheckErr(ciErrNum, "clEnqueueWriteBufferL2157");
// 	  ciErrNum = clEnqueueNDRangeKernel(command_queue[i], LaplaceBoundKernel[i], 2, 0, constantglobalworksize, constantlocalworksize,
// 					    0, NULL, &GPUExecution[i]);
// 	  oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2195");

// 	}
// #if TOTALTIMING
// 	ciErrNum = clFinish(command_queue[0]);
// 	oclCheckErr(ciErrNum, "clFinish");
// 	MultTimeLaplaceBound += AccumulateTiming(GPUExecution, "MULT", 0);
// #endif	
//       }
//       //       for(unsigned int i = 0; i < num_devices; i++) {
//       // 	ciErrNum = clFinish(command_queue[i]);
//       // 	oclCheckErr(ciErrNum, "clFinish");
//       //       }


//       size_t overallReduceOffset = overallMultOffset*LSIZE;
//       for(unsigned int i = 0; i < num_devices; i++) {
// 	ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
// 	oclCheckErr(ciErrNum, "clSetKernelArgL1205");
      
// 	size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
// 	size_t local2[] = {LSIZE,1};
// 	ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], ReduceBoundKernel[i], 2, 0, reduceglobalworksize2, local2,
// 					   0, NULL, &GPUExecution[i]);
// 	oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
// #if TOTALTIMING
// 	ciErrNum = clFinish(command_queue[0]);
// 	oclCheckErr(ciErrNum, "clFinish");
// 	ReduTimeLaplaceBound += AccumulateTiming(GPUExecution, "REDUCE", 0);
// #endif
//       }
//       for(unsigned int i = 0; i < num_devices; i++) {
// 	ciErrNum = clFinish(command_queue[i]);
// 	oclCheckErr(ciErrNum, "clFinish");
      
//       }
//     }
//   }


//   {
//     if (num_devices > 1) {
//       size_t slice_per_GPU = storageInnerSizePaddedBound/num_devices;
//       size_t leftover = storageInnerSizeBound;
//       for(unsigned int i = 0;i < num_devices; i++) 	{    
// 	size_t offsetGPU = std::min(leftover,slice_per_GPU);
// 	clEnqueueReadBuffer(command_queue[i], d_ptrResultBound[i], CL_FALSE, 0,
// 			    offsetGPU*sizeof(REAL), ptrResultBound + i*slice_per_GPU, 0, NULL, &GPUDone[i]);
// 	leftover -= offsetGPU;
//       }
//       clWaitForEvents(num_devices, GPUDone);
  

//     } else {
//       for(unsigned int i = 0;i < num_devices; i++) 
// 	{    
// 	  clEnqueueReadBuffer(command_queue[i], d_ptrResultBound[i], CL_FALSE, 0,
// 			      storageInnerSizeBound*sizeof(REAL), ptrResultBound, 0, NULL, &GPUDone[i]);
// 	}
//       clWaitForEvents(num_devices, GPUDone);
    
//     }
//     for (unsigned i = 0; i < storageInnerSizeBound; i++) {
//       ptrResult[offsetBound[i]] = ptrResultBound[i];
//     }


//   }
// #if TOTALTIMING
//   CounterLaplaceBound += 1.0;
// #endif	

//   for(size_t i = 0; i < num_devices; i++) {
//     clReleaseEvent(GPUExecution[i]);
//     clReleaseEvent(GPUDone[i]);
//     clReleaseEvent(GPUDoneLcl[i]);
//   }  
// }

void ExecLaplaceInner(REAL * ptrAlpha,
		      REAL * ptrResult,
		      REAL * lcl_q,
		      REAL * lcl_q_inv,
		      REAL *ptrLevel,
		      REAL *ptrIndex,
		      REAL *ptrLevel_int,
		      unsigned argStorageSize,
		      unsigned argStorageDim) {

  cl_int ciErrNum = CL_SUCCESS;
  cl_event GPUDone[NUMDEVS];
  cl_event GPUDoneLcl[NUMDEVS];
  cl_event GPUExecution[NUMDEVS];
  
  size_t idx = 0;
  for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
    ptrLcl_qInner[idx++] = lcl_q[d_outer];
    ptrLcl_qInner[idx++] = lcl_q_inv[d_outer];
  }


  for(size_t i = 0; i < num_devices; i++) {

    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaInner[i], CL_FALSE, 0,
				     storageSize*sizeof(REAL), ptrAlpha, 0, 0, &GPUDone[i]);
    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qInner[i], CL_FALSE, 0,
				     lcl_q_size*sizeof(REAL), ptrLcl_qInner, 0, 0, &GPUDoneLcl[i]);
  }
  clWaitForEvents(num_devices, GPUDone);
  clWaitForEvents(num_devices, GPUDoneLcl);
  for(size_t i = 0; i < num_devices; i++) {
    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrResultInner[i], 
				     CL_FALSE, 0,
				     storageSizePadded*sizeof(REAL), 
				     ptrResultZero, 0, 0, &GPUDone[i]);

    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaInner[i], CL_FALSE, storageSize*sizeof(REAL),
				     alphaend_size*sizeof(REAL), ptrAlphaEndInner, 0, 0, &GPUExecution[i]);
  }
  oclCheckErr(ciErrNum, "clEnqueueWriteBuffer LapBoundmult");
  clWaitForEvents(num_devices, GPUDone);
  //clWaitForEvents(num_devices, GPUExecution);

  {
    size_t multglobalworksize[2];
    size_t storageSizePaddedStep = std::min(storageSizePadded  / num_devices, par_result_max_size / (storageSizePadded) * LSIZE);

    multglobalworksize[0] = std::min(storageSizePadded,storageSizePaddedStep);
    multglobalworksize[1] = storageSizePadded/LSIZE;
    size_t multglobal = storageSizePadded / num_devices;

    for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[0], multglobal - overallMultOffset)) {
      multglobalworksize[0] =  std::min(multglobalworksize[0], multglobal-overallMultOffset);
//       std::cout << " overallMultOffset " << overallMultOffset << std::endl;
//       std::cout << " multglobalworksize[0] " << multglobalworksize[0] << std::endl;
//       std::cout << " multglobalworksize[1] " << multglobalworksize[1] << std::endl;
      for(unsigned int i = 0; i < num_devices; i++) {
	size_t overallMultOffset2 = overallMultOffset + i*multglobal;
	ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], 8, sizeof(cl_ulong), (void *) &overallMultOffset2);
      }
      size_t constantglobalworksize[2];
      size_t constantlocalworksize[2];
      size_t constantglobal = multglobalworksize[0];
      constantglobalworksize[0] = std::min(constant_buffer_iterations_noboundary,constantglobal);
      constantglobalworksize[1] = multglobalworksize[1];
      constantlocalworksize[0] = LSIZE;
      constantlocalworksize[1] = 1;
      
      for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < constantglobal; ConstantMemoryOffset+= std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset)) {
	constantglobalworksize[0] = std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset);
//  	std::cout << " constantglobalworksize[0] " << constantglobalworksize[0]<< std::endl;
	
	for(unsigned int i = 0; i < num_devices; i++) {
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], 
					   d_ptrLevelIndexLevelintconInner[i], 
					   CL_TRUE, 0 ,
					   constantglobalworksize[0]*3*dims*sizeof(REAL), 
					   ptrLevelIndexLevelintInner+(overallMultOffset + i*multglobal + ConstantMemoryOffset)*3*dims, 
					   1,
					   &GPUExecution[i] , NULL);
	  oclCheckErr(ciErrNum, "clEnqueueWriteBufferL306");
	  
	  size_t jj =  ConstantMemoryOffset / LSIZE ;
// 	  std::cout << " jj " << jj << std::endl;
	  ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], 9, sizeof(cl_ulong), (void *) &jj);
	  oclCheckErr(ciErrNum, "clEnqueueWriteBufferL302");

	  ciErrNum = clEnqueueNDRangeKernel(command_queue[i], LaplaceInnerKernel[i], 2, 0, constantglobalworksize, constantlocalworksize,
					    0, NULL, &GPUExecution[i]);
	  // 	  std::cout << " i " << i << std::endl;
	  oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2314");
	}

      }
//       for(unsigned int i = 0; i < num_devices; i++) {
// 	ciErrNum |= clFinish(command_queue[i]);
// 	oclCheckErr(ciErrNum, "clFinish");
//       }

      //REDUCE
      size_t overallReduceOffset = 0;//overallMultOffset*LSIZE; 
      for(unsigned int i = 0; i < num_devices; i++) {

	ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
	oclCheckErr(ciErrNum, "clSetKernelArgLapIL2199");
	size_t newnum_groups = multglobalworksize[0] / LSIZE ;
// 	std::cout << " newnum_groups " << newnum_groups << std::endl;
// 	std::cout << " num_groups " << num_groups << std::endl;
	ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 3, sizeof(cl_ulong), (void *) &newnum_groups);
	oclCheckErr(ciErrNum, "clSetKernelArgLapIL340");

	size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
	size_t local2[] = {LSIZE,1};
	ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], ReduceInnerKernel[i], 2, 0, reduceglobalworksize2, local2,
					   0, NULL, &GPUExecution[i]);
	oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
      }
      for(unsigned int i = 0; i < num_devices; i++) {
	ciErrNum |= clFinish(command_queue[i]);
	oclCheckErr(ciErrNum, "clFinishLapIL355");
      }
    }
  }

  if (num_devices > 1) {
	    
    for(unsigned int i = 0;i < num_devices; i++) 
      {    
	ciErrNum |= clEnqueueReadBuffer(command_queue[i], d_ptrResultInner[i], CL_FALSE, 0,
					storageSize*sizeof(REAL), ptrResultTemp + i * storageSizePadded, 0, NULL, &GPUDone[i]);

      }
    oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL2145");
    ciErrNum |= clWaitForEvents(num_devices, GPUDone);
    oclCheckErr(ciErrNum, "clWaitForEventsLapIL2147");


    for (size_t j = 0; j < num_devices; j++) {
      for (size_t i = 0; i < storageSize; i++) {
	ptrResult[i] += ptrResultTemp[j*storageSizePadded + i];

      }
    }

  } else {
    for(unsigned int i = 0;i < num_devices; i++) 
      {    
	clEnqueueReadBuffer(command_queue[i], d_ptrResultInner[i], CL_FALSE, 0,
			    storageSize*sizeof(REAL), ptrResult, 0, NULL, &GPUDone[i]);

      }
    clWaitForEvents(num_devices, GPUDone);
  }

#if TOTALTIMING
  CounterLaplaceInner += 1.0;
#endif	
  for(size_t i = 0; i < num_devices; i++) {
    clReleaseEvent(GPUExecution[i]);
    clReleaseEvent(GPUDone[i]);
    clReleaseEvent(GPUDoneLcl[i]);
  }  

}

// void ExecLaplaceInner(REAL * ptrAlpha, REAL * ptrResult,REAL * lcl_q,REAL * lcl_q_inv, REAL *ptrLevel, REAL *ptrIndex, REAL *ptrLevel_int, unsigned argStorageSize, unsigned argStorageDim) {

//   cl_int ciErrNum = CL_SUCCESS;
//   cl_event GPUDone[NUMDEVS];
//   cl_event GPUDoneLcl[NUMDEVS];
//   cl_event GPUExecution[NUMDEVS];
  
//   size_t idx = 0;
//   for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
//     ptrLcl_qInner[idx++] = lcl_q[d_outer];
//     ptrLcl_qInner[idx++] = lcl_q_inv[d_outer];
//   }


//   for(size_t i = 0; i < num_devices; i++) {

//     ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaInner[i], CL_FALSE, 0,
// 				     storageSize*sizeof(REAL), ptrAlpha, 0, 0, &GPUDone[i]);
//     ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qInner[i], CL_FALSE, 0,
// 				     lcl_q_size*sizeof(REAL), ptrLcl_qInner, 0, 0, &GPUDoneLcl[i]);
//   }
//   clWaitForEvents(num_devices, GPUDone);
//   clWaitForEvents(num_devices, GPUDoneLcl);
//   for(size_t i = 0; i < num_devices; i++) {
//     ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaInner[i], CL_FALSE, storageSize*sizeof(REAL),
// 				     alphaend_size*sizeof(REAL), ptrAlphaEndInner, 0, 0, &GPUDone[i]);
//   }
//   oclCheckErr(ciErrNum, "clEnqueueWriteBuffer mult");
//   clWaitForEvents(num_devices, GPUDone);

//   {
//     size_t multglobalworksize[2];
//     size_t storageSizePaddedStep = std::min(storageSizePadded, par_result_max_size / num_groups) / LSIZE;
//     multglobalworksize[0] = storageSizePadded;

//     multglobalworksize[1] = std::min(storageSizePadded/LSIZE,storageSizePaddedStep);
//     size_t multglobal = storageSizePadded/LSIZE / num_devices;
    

//     for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[1], multglobal - overallMultOffset)) {
//       multglobalworksize[1] =  std::min(multglobalworksize[1], multglobal-overallMultOffset);

//       for(unsigned int i = 0; i < num_devices; i++) {
// 	size_t overallMultOffset2 = i*multglobal + overallMultOffset;
// 	ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], 8, sizeof(cl_ulong), (void *) &overallMultOffset2);
//       }
//       size_t constantglobalworksize[2];
//       size_t constantlocalworksize[2];
//       constantglobalworksize[0] = std::min(constant_buffer_iterations_noboundary,storageSizePadded);
//       constantglobalworksize[1] = multglobalworksize[1];
//       constantlocalworksize[0] = LSIZE;
//       constantlocalworksize[1] = 1;
//       for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < storageSizePadded; ConstantMemoryOffset+= std::min(constantglobalworksize[0],storageSizePadded-ConstantMemoryOffset)) {
// 	constantglobalworksize[0] = std::min(constantglobalworksize[0],storageSizePadded-ConstantMemoryOffset);

// 	size_t jj = ConstantMemoryOffset / LSIZE;
	
// 	for(unsigned int i = 0; i < num_devices; i++) {
// 	  ciErrNum |= clSetKernelArg(LaplaceInnerKernel[i], 9, sizeof(cl_ulong), (void *) &jj);
	  
// 	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLevelIndexLevelintconInner[i], CL_TRUE, 0 ,
// 					   constantglobalworksize[0]*3*dims*sizeof(REAL), ptrLevelIndexLevelintInner+ConstantMemoryOffset*3*dims, 0, 0, NULL);
// 	  oclCheckErr(ciErrNum, "clEnqueueWriteBufferL2344");
// 	  ciErrNum = clEnqueueNDRangeKernel(command_queue[i], LaplaceInnerKernel[i], 2, 0, constantglobalworksize, constantlocalworksize,
// 					    0, NULL, &GPUExecution[i]);
// 	  oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2314");
// 	}
// #if TOTALTIMING
// 	ciErrNum = clFinish(command_queue[0]);
// 	oclCheckErr(ciErrNum, "clFinish");
// 	MultTimeLaplaceInner += AccumulateTiming(GPUExecution, "MULT", 0);
// #endif	

//       }
// //       for(unsigned int i = 0; i < num_devices; i++) {
// // 	ciErrNum = clFinish(command_queue[i]);
// // 	oclCheckErr(ciErrNum, "clFinish");
// //       }


//       //REDUCE
//       size_t overallReduceOffset = overallMultOffset*LSIZE; 
//       for(unsigned int i = 0; i < num_devices; i++) {

// 	ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);


// 	size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
// 	size_t local2[] = {LSIZE,1};
// 	ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], ReduceInnerKernel[i], 2, 0, reduceglobalworksize2, local2,
// 					   0, NULL, &GPUExecution[i]);
// 	oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
// #if TOTALTIMING
// 	ciErrNum = clFinish(command_queue[0]);
// 	oclCheckErr(ciErrNum, "clFinish");
// 	ReduTimeLaplaceInner += AccumulateTiming(GPUExecution, "REDUCE", 0);
// #endif
//       }
//       for(unsigned int i = 0; i < num_devices; i++) {
// 	ciErrNum = clFinish(command_queue[i]);
// 	oclCheckErr(ciErrNum, "clFinish");
//       }
//     }
//   }


//   {
//     if (num_devices > 1) {

//       size_t leftover = storageSize;
//       size_t slice_per_GPU = storageSizePadded/num_devices;
//       for(unsigned int i = 0;i < num_devices; i++) 
// 	{    
// 	  size_t offsetGPU = std::min(leftover,slice_per_GPU);
	  
// 	  clEnqueueReadBuffer(command_queue[i], d_ptrResultInner[i], CL_FALSE, 0,
// 			      offsetGPU*sizeof(REAL), ptrResult+ i * slice_per_GPU, 0, NULL, &GPUDone[i]);
// 	  leftover -= offsetGPU;

// 	}
//       clWaitForEvents(num_devices, GPUDone);
//     } else {
//       for(unsigned int i = 0;i < num_devices; i++) 
// 	{    
// 	  clEnqueueReadBuffer(command_queue[i], d_ptrResultInner[i], CL_FALSE, 0,
// 			      storageSize*sizeof(REAL), ptrResult, 0, NULL, &GPUDone[i]);

// 	}
//       clWaitForEvents(num_devices, GPUDone);
//     }
//   }

// #if TOTALTIMING
//   CounterLaplaceInner += 1.0;
// #endif	
//   for(size_t i = 0; i < num_devices; i++) {
//     clReleaseEvent(GPUExecution[i]);
//     clReleaseEvent(GPUDone[i]);
//     clReleaseEvent(GPUDoneLcl[i]);
//   }  

// }


void CleanUpGPU() {
  if (isCleanedUp == 0) {

#if TOTALTIMING
    double LaplaceBoundTime = (MultTimeLaplaceBound + ReduTimeLaplaceBound);
    double LaplaceInnerTime = (MultTimeLaplaceInner + ReduTimeLaplaceInner);
    double LTwoDotBoundTime = (MultTimeLTwoDotBound + ReduTimeLTwoDotBound);
    double LTwoDotInnerTime = (MultTimeLTwoDotInner + ReduTimeLTwoDotInner);
    
    if (!isFirstTimeLaplaceBound) {
      std::cout << "Time for calculating LaplaceBound: " << LaplaceBoundTime << std::endl;
    }
    if (!isFirstTimeLTwoDotBound) {
      std::cout << "Time for calculating LTwoDotBound: " << LTwoDotBoundTime << std::endl;
    }
    if (!isFirstTimeLaplaceInner) {
      std::cout << "Time for calculating LaplaceInner: " << LaplaceInnerTime << std::endl;
    }
    if (!isFirstTimeLTwoDotInner) {
      std::cout << "Time for calculating LTwoDotInner: " << LTwoDotInnerTime << std::endl;
    }
    std::cout << "Time to solve: " << (LaplaceBoundTime + LaplaceInnerTime + LTwoDotBoundTime + LTwoDotInnerTime) << std::endl;

    PrintGFLOPS(LaplaceInnerTime, LaplaceBoundTime, LTwoDotInnerTime, LTwoDotBoundTime);
#endif
    

    delete device_ids;
    delete platform_ids;
    //Laplace + Ltwo Bound
    cl_int ciErrNum = CL_SUCCESS;

    if (!isFirstTimeLaplaceBound || !isFirstTimeLTwoDotBound) {
      for(unsigned int i = 0; i < num_devices; i++) 
	{
	  ciErrNum |= clReleaseMemObject(d_ptrLevelBound[i]);   
	  ciErrNum |= clReleaseMemObject(d_ptrIndexBound[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrLevel_intBound[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrResultBound[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrParResultBound[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrLevelIndexLevelintconBound[i]);

	  ciErrNum |= clReleaseMemObject(d_ptrAlphaBound[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrLcl_qBound[i]);
	  if (!isFirstTimeLaplaceBound) {
	    ciErrNum |= clReleaseMemObject(d_ptrLambdaBound[i]);
	    ciErrNum |= clReleaseKernel( LaplaceBoundKernel[i] );
	  }
	  oclCheckErr(ciErrNum, "clReleaseMemObject");
	  if (!isFirstTimeLTwoDotBound)
	    ciErrNum |= clReleaseKernel( LTwoDotBoundKernel[i] );
	  ciErrNum |= clReleaseKernel( ReduceBoundKernel[i] );
	  oclCheckErr(ciErrNum, "clReleaseKernel");
	}
      free(ptrLevelTBound);
      free(ptrIndexTBound);
      free(ptrLevel_intTBound);
      free(ptrParResultBound);
      free(ptrLcl_qBound);
      free(ptrResultBound);
      free(offsetBound);
    }
    // Laplace + LtwoInner
    if (!isFirstTimeLaplaceInner || !isFirstTimeLTwoDotInner) {
      for(unsigned int i = 0; i < num_devices; i++) 
	{
	  ciErrNum |= clReleaseMemObject(d_ptrLevelInner[i]);   
	  ciErrNum |= clReleaseMemObject(d_ptrIndexInner[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrLevel_intInner[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrResultInner[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrParResultInner[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrLevelIndexLevelintconInner[i]);

	  ciErrNum |= clReleaseMemObject(d_ptrAlphaInner[i]);
	  ciErrNum |= clReleaseMemObject(d_ptrLcl_qInner[i]);
	  if (!isFirstTimeLaplaceInner) {
	    ciErrNum |= clReleaseMemObject(d_ptrLambdaInner[i]);
	    ciErrNum |= clReleaseKernel( LaplaceInnerKernel[i] );
	  }
	  if (!isFirstTimeLTwoDotInner)
	    ciErrNum |= clReleaseKernel( LTwoDotInnerKernel[i] );
	  ciErrNum |= clReleaseKernel( ReduceInnerKernel[i] );
	  oclCheckErr(ciErrNum, "clReleaseMemObject & clReleaseKernel");
	}
      free(ptrLevelTInner);
      free(ptrIndexTInner);
      free(ptrLevel_intTInner);
      free(ptrParResultInner);
      free(ptrLcl_qInner);
      for(unsigned int i = 0; i < num_devices; i++)  {
	clReleaseCommandQueue(command_queue[i]);
      }
      clReleaseContext(context);
    }
    isCleanedUp = 1;
  }
}


void RunOCLKernelLaplaceBound2(REAL * ptrAlpha, REAL * ptrResult,REAL * lcl_q,REAL * lcl_q_inv, REAL *ptrLevel, REAL *ptrIndex, REAL *ptrLevel_int, REAL *ptrLambda,size_t argStorageSize, size_t argStorageDim, sg::base::GridStorage* storage) {
  //std::cout << "EXEC RunOCLKernelBound" << std::endl;
  if (isVeryFirstTime) {
    StartUpGPU();
  } 
  if (isFirstTimeLaplaceBound) {
    if (isFirstTimeLTwoDotBound) {
      SetBuffersBound(ptrLevel,
			     ptrIndex,
			     ptrLevel_int,
			     argStorageSize,
			     argStorageDim,storage);
      SetLambdaBufferLaplaceBound(
				  ptrLambda,
		      
				  argStorageDim);

    } else {
      SetLambdaBufferLaplaceBound(
				  ptrLambda,
				  argStorageDim);
    }
    CompileLaplaceBoundKernels(); // Need to compile separate kernels here
    SetArgumentsLaplaceBound();
    isVeryFirstTime = 0;
    isFirstTimeLaplaceBound = 0;
  }
  myStopwatch->start();
  ExecLaplaceBound(ptrAlpha, ptrResult,lcl_q,lcl_q_inv, ptrLevel, ptrIndex, ptrLevel_int, argStorageSize, argStorageDim);
  MultTimeLaplaceBound += myStopwatch->stop();
  //std::cout << "END RunOCLKernelBound" << std::endl;
    
}

void RunOCLKernelLaplaceInner2(REAL * ptrAlpha, REAL * ptrResult, REAL * lcl_q, REAL * lcl_q_inv, REAL *ptrLevel, REAL *ptrIndex, REAL *ptrLevel_int,REAL *ptrLambda, size_t argStorageSize, size_t argStorageDim, sg::base::GridStorage* storage) {
  //std::cout << "EXEC RunOCLKernelInner" << std::endl;
  if (isVeryFirstTime) {
    StartUpGPU();
  } 
  if (isFirstTimeLaplaceInner) {
    if (isFirstTimeLTwoDotInner) {

      SetBuffersInner(ptrLevel,
			     ptrIndex,
			     ptrLevel_int,
			     argStorageSize,
			     argStorageDim,storage);
      SetLambdaBufferLaplaceInner(
				  ptrLambda,
				  argStorageDim);
    } else {
      SetLambdaBufferLaplaceInner(
				  ptrLambda,
				  argStorageDim);

    }

    CompileLaplaceInnerKernels(); 
    SetArgumentsLaplaceInner();
    isVeryFirstTime = 0;
    isFirstTimeLaplaceInner = 0;
  }
  myStopwatch->start();
  ExecLaplaceInner(ptrAlpha, ptrResult,lcl_q,lcl_q_inv, ptrLevel, ptrIndex, ptrLevel_int, argStorageSize, argStorageDim);
  MultTimeLaplaceInner += myStopwatch->stop();
  //std::cout << "EXIT RunOCLKernelInner" << std::endl;

}


  void CompileLTwoDotInnerKernels() {
    for(unsigned int i=0; i < num_devices; ++i) 
      {
	compileLTwoDotInner(i, "multKernel", LTwoDotInnerKernel);

	compileReduceInner(i, "ReduceInnerKernel", ReduceInnerKernel);
      }
  }

  void CompileLTwoDotBoundKernels() {
    for(unsigned int i=0; i < num_devices; ++i) 
      {
	compileLTwoDotBound(i, "multKernel", LTwoDotBoundKernel);

	compileReduceBound(i, "ReduceBoundKernel",ReduceBoundKernel);
      }
  }


void SetArgumentsLTwoDotInner() {
  cl_int ciErrNum = CL_SUCCESS;
  int counter = 0;

  for(unsigned int i=0; i < num_devices; ++i) 
    {
      ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelInner[i]);
      ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelIndexLevelintconInner[i]);

      ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrIndexInner[i]);
      ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevel_intInner[i]);

      ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrAlphaInner[i]);
      ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultInner[i]);
      ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLcl_qInner[i]);
      oclCheckErr(ciErrNum, "clSetKernelArg1 Kernel Construct");

      counter = 0;
      ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrResultInner[i]);
      ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultInner[i]);
      counter = 0;
    }
      
}



void ExecLTwoDotInner(REAL * ptrAlpha, REAL * ptrResult, REAL * lcl_q, REAL *ptrLevel, REAL *ptrIndex, REAL *ptrLevel_int, size_t argStorageSize, size_t argStorageDim) {

  cl_int ciErrNum = CL_SUCCESS;
  cl_event GPUDone[NUMDEVS];
  cl_event GPUDoneLcl[NUMDEVS];
  cl_event GPUExecution[NUMDEVS];

  for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
    ptrLcl_qInner[d_outer] = lcl_q[d_outer];
  }

  for(size_t i = 0; i < num_devices; i++) {

    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaInner[i], CL_FALSE, 0,
				     storageSize*sizeof(REAL), ptrAlpha, 0, 0, &GPUDone[i]);
    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qInner[i], CL_FALSE, 0,
				     lcl_q_size*sizeof(REAL), ptrLcl_qInner, 0, 0, &GPUDoneLcl[i]);


  }
  clWaitForEvents(num_devices, GPUDone);
  clWaitForEvents(num_devices, GPUDoneLcl);
  oclCheckErr(ciErrNum, "clEnqueueWriteBuffer LapI L2588");

  for(size_t i = 0; i < num_devices; i++) {
    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaInner[i], CL_FALSE, storageSize*sizeof(REAL),
				     alphaend_size*sizeof(REAL), ptrAlphaEndInner, 0, 0, &GPUDone[i]);
  }
  oclCheckErr(ciErrNum, "clEnqueueWriteBuffer mult");
  clWaitForEvents(num_devices, GPUDone);


  {
    size_t multglobalworksize[2];
    size_t storageSizePaddedStep = std::min(storageSizePadded, par_result_max_size / num_groups) / LSIZE;
    multglobalworksize[0] = storageSizePadded;
    multglobalworksize[1] = std::min(storageSizePadded/LSIZE,storageSizePaddedStep);
    size_t multglobal = storageSizePadded / LSIZE / num_devices;

    for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[1], multglobal - overallMultOffset)) {
      multglobalworksize[1] = std::min(multglobalworksize[1], multglobal-overallMultOffset);

      for(unsigned int i = 0; i < num_devices; i++) {
	size_t overallMultOffset2 = i*multglobal + overallMultOffset;
	ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], 7, sizeof(cl_ulong), (void *) &overallMultOffset2);
	oclCheckErr(ciErrNum, "clSetKernelArgL1660");
      }

      size_t constantglobalworksize[2];
      size_t constantlocalworksize[2];
      constantglobalworksize[0] = std::min(constant_buffer_iterations_noboundary,storageSizePadded);
      constantglobalworksize[1] = multglobalworksize[1];
      constantlocalworksize[0] = LSIZE;
      constantlocalworksize[1] = 1;
      for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < storageSizePadded; ConstantMemoryOffset+= std::min(constantglobalworksize[0],storageSizePadded-ConstantMemoryOffset)) {
	constantglobalworksize[0] = std::min(constantglobalworksize[0],storageSizePadded-ConstantMemoryOffset);

	size_t jj = ConstantMemoryOffset / LSIZE;
	for(unsigned int i = 0; i < num_devices; i++) {
	  ciErrNum |= clSetKernelArg(LTwoDotInnerKernel[i], 8, sizeof(cl_ulong), (void *) &jj);
	  
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLevelIndexLevelintconInner[i], CL_TRUE, 0 ,
					   constantglobalworksize[0]*3*dims*sizeof(REAL), ptrLevelIndexLevelintInner+ConstantMemoryOffset*3*dims, 0, 0, NULL);
	  oclCheckErr(ciErrNum, "clEnqueueWriteBufferL2026");
	  ciErrNum = clEnqueueNDRangeKernel(command_queue[i], LTwoDotInnerKernel[i], 2, 0, constantglobalworksize, constantlocalworksize,
					    0, NULL, &GPUExecution[i]);
	  oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2745");
	}
#if TOTALTIMING
	ciErrNum = clFinish(command_queue[0]);
	oclCheckErr(ciErrNum, "clFinish");
	MultTimeLTwoDotInner += AccumulateTiming(GPUExecution, "MULT", 0);
#endif
      }
//       for(unsigned int i = 0; i < num_devices; i++) {
// 	ciErrNum = clFinish(command_queue[i]);
// 	oclCheckErr(ciErrNum, "clFinish");
//       }

      //REDUCE
      size_t overallReduceOffset = overallMultOffset*LSIZE; 
      for(unsigned int i = 0; i < num_devices; i++) {
	ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
	size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
	size_t local2[] = {LSIZE,1};
	ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], ReduceInnerKernel[i], 2, 0, reduceglobalworksize2, local2,
					   0, NULL, &GPUExecution[i]);

#if TOTALTIMING
	ciErrNum = clFinish(command_queue[0]);
	oclCheckErr(ciErrNum, "clFinish");
	ReduTimeLTwoDotInner += AccumulateTiming(GPUExecution, "REDUCE", 0);
#endif
      }
      for(unsigned int i = 0; i < num_devices; i++) {
	ciErrNum = clFinish(command_queue[i]);
	oclCheckErr(ciErrNum, "clFinish");
      }
    }
  }

  {
    if (num_devices > 1) {
      size_t slice_per_GPU = storageSizePadded/num_devices;
      size_t leftover = storageSize;

      for(unsigned int i = 0;i < num_devices; i++) 
	{    
size_t offsetGPU = std::min(leftover,slice_per_GPU);
	  clEnqueueReadBuffer(command_queue[i], d_ptrResultInner[i], CL_FALSE, 0,
			      offsetGPU*sizeof(REAL), ptrResult+ i*slice_per_GPU, 0, NULL, &GPUDone[i]);
leftover -= offsetGPU;
	}
      clWaitForEvents(num_devices, GPUDone);
    } else {

      for(unsigned int i = 0;i < num_devices; i++) 
	{    
	  clEnqueueReadBuffer(command_queue[i], d_ptrResultInner[i], CL_FALSE, 0,
			      storageSize*sizeof(REAL), ptrResult, 0, NULL, &GPUDone[i]);
	}
      clWaitForEvents(num_devices, GPUDone);
    }
  }
#if TOTALTIMING
  CounterLTwoDotInner += 1.0;
#endif	
  for(size_t i = 0; i < num_devices; i++) {
    clReleaseEvent(GPUExecution[i]);
    clReleaseEvent(GPUDone[i]);
    clReleaseEvent(GPUDoneLcl[i]);
  }  
}


void RunOCLKernelLTwoDotInner2(REAL * ptrAlpha, REAL * ptrResult,  REAL * lcl_q, REAL *ptrLevel, REAL *ptrIndex, REAL *ptrLevel_int, size_t argStorageSize, size_t argStorageDim, sg::base::GridStorage* storage) {
  //std::cout << "EXE RunOCLKernelLTwoDotInner" << std::endl;
  if (isVeryFirstTime) {
    StartUpGPU();
  } 
  if (isFirstTimeLTwoDotInner) {

    if (isFirstTimeLaplaceInner) {
      SetBuffersInner(ptrLevel,
			     ptrIndex,
			     ptrLevel_int,
			     argStorageSize,
			     argStorageDim,storage
			     );
    }
    CompileLTwoDotInnerKernels(); 
    SetArgumentsLTwoDotInner();

    isVeryFirstTime = 0;
    isFirstTimeLTwoDotInner = 0;
  }
  ExecLTwoDotInner(ptrAlpha, ptrResult, lcl_q, ptrLevel, ptrIndex, ptrLevel_int, argStorageSize, argStorageDim);
  //   std::cout << "END RunOCLKernelLTwoDot" << std::endl;
}


void SetArgumentsLTwoDotBound() {
  cl_int ciErrNum = CL_SUCCESS;
  int counter = 0;
  for(unsigned int i=0; i < num_devices; ++i) 
    {
      ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelBound[i]);
      ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelIndexLevelintconBound[i]);

      ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrIndexBound[i]);
      ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevel_intBound[i]);
      ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrAlphaBound[i]);
      ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultBound[i]);
      ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLcl_qBound[i]);
      oclCheckErr(ciErrNum, "clSetKernelArg1 Kernel Construct");

      counter = 0;
      ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrResultBound[i]);
      ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultBound[i]);
      counter = 0;
    }
      
}

void ExecLTwoDotBound(REAL * ptrAlpha, REAL * ptrResult,REAL * lcl_q, REAL *ptrLevel, REAL *ptrIndex, REAL *ptrLevel_int, size_t argStorageSize, size_t argStorageDim) {

  cl_int ciErrNum = CL_SUCCESS;
  cl_event GPUDone[NUMDEVS];
  cl_event GPUDoneLcl[NUMDEVS];
  cl_event GPUExecution[NUMDEVS];
  
  for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
    ptrLcl_qBound[d_outer] = lcl_q[d_outer];
  }


  for(size_t i = 0; i < num_devices; i++) {

    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaBound[i], CL_FALSE, 0,
				     storageSizeBound*sizeof(REAL), ptrAlpha, 0, 0, &GPUDone[i]);
    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qBound[i], CL_FALSE, 0,
				     lcl_q_sizeBound*sizeof(REAL), ptrLcl_qBound, 0, 0, &GPUDoneLcl[i]);
  }
  clWaitForEvents(num_devices, GPUDone);
  clWaitForEvents(num_devices, GPUDoneLcl);
  oclCheckErr(ciErrNum, "clEnqueueWriteBuffer mult");

  for(size_t i = 0; i < num_devices; i++) {
    ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaBound[i], CL_FALSE, storageSizeBound*sizeof(REAL),
				     alphaend_sizeBound*sizeof(REAL), ptrAlphaEndBound, 0, 0, &GPUDone[i]);
  }
  oclCheckErr(ciErrNum, "clEnqueueWriteBuffer mult");  
  clWaitForEvents(num_devices, GPUDone);

  {
    size_t multglobalworksize[2];
    // 2D Bound
    size_t storageInnerSizePaddedStep = std::min(storageInnerSizePaddedBound, Inner_par_result_max_sizeBound / Inner_num_groupsBound) / LSIZE;
    multglobalworksize[0] = storageSizePaddedBound;
    multglobalworksize[1] = std::min(storageInnerSizePaddedBound/LSIZE,storageInnerSizePaddedStep);

    size_t multglobal = storageInnerSizePaddedBound / LSIZE / num_devices;
    for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[1], multglobal - overallMultOffset)) {
      multglobalworksize[1] = std::min(multglobalworksize[1], multglobal-overallMultOffset);

      for(unsigned int i = 0; i < num_devices; i++) {
	size_t overallMultOffset2 = i*multglobal + overallMultOffset;
	ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], 7, sizeof(cl_ulong), (void *) &overallMultOffset2);
	oclCheckErr(ciErrNum, "clSetKernelArgL1660");
      }
	  
      size_t constantglobalworksize[2];
      size_t constantlocalworksize[2];
      constantglobalworksize[0] = std::min(constant_buffer_iterations,storageSizePaddedBound);
      constantglobalworksize[1] = multglobalworksize[1]; // already LSIZE pad
      constantlocalworksize[0] = LSIZE;
      constantlocalworksize[1] = 1;
      for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < storageSizePaddedBound; ConstantMemoryOffset+= std::min(constantglobalworksize[0],storageSizePaddedBound-ConstantMemoryOffset)) {
	constantglobalworksize[0] = std::min(constantglobalworksize[0],storageSizePaddedBound-ConstantMemoryOffset);
	size_t jj = ConstantMemoryOffset / LSIZE;
	for(unsigned int i = 0; i < num_devices; i++) {
	  ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], 8, sizeof(cl_ulong), (void *) &jj);
	  oclCheckErr(ciErrNum, "clSetKernelArgL2150");
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLevelIndexLevelintconBound[i], CL_TRUE, 0 ,
					   constantglobalworksize[0]*3*dims*sizeof(REAL), ptrLevelIndexLevelintBound+ConstantMemoryOffset*3*dims, 0, 0, NULL);
	  oclCheckErr(ciErrNum, "clEnqueueWriteBufferL2157");
	  ciErrNum = clEnqueueNDRangeKernel(command_queue[i], LTwoDotBoundKernel[i], 2, 0, constantglobalworksize, constantlocalworksize,
					    0, NULL, &GPUExecution[i]);
	  oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel3115");

	}
#if TOTALTIMING
	ciErrNum = clFinish(command_queue[0]);
	oclCheckErr(ciErrNum, "clFinish");
	MultTimeLTwoDotBound += AccumulateTiming(GPUExecution, "MULT", 0);
#endif	
      }
      //       for(unsigned int i = 0; i < num_devices; i++) {
      // 	ciErrNum = clFinish(command_queue[i]);
      // 	oclCheckErr(ciErrNum, "clFinish");
      //       }

      size_t overallReduceOffset = overallMultOffset*LSIZE;
      for(unsigned int i = 0; i < num_devices; i++) {
	ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
	oclCheckErr(ciErrNum, "clSetKernelArgL1205");
      
	size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
	size_t local2[] = {LSIZE,1};

	ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], ReduceBoundKernel[i], 2, 0, reduceglobalworksize2, local2,
					   0, NULL, &GPUExecution[i]);
	oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
#if TOTALTIMING
	ciErrNum |= clFinish(command_queue[0]);
	oclCheckErr(ciErrNum, "clFinishlL1214");
	ReduTimeLTwoDotBound += AccumulateTiming(GPUExecution, "REDUCE", 0);
#endif
      }
      for(unsigned int i = 0; i < num_devices; i++) {
	ciErrNum = clFinish(command_queue[i]);
	oclCheckErr(ciErrNum, "clFinish");
      
      }
    }
  }


  {
    if (num_devices > 1) {
      size_t slice_per_GPU = storageInnerSizePaddedBound/num_devices;
      size_t leftover = storageInnerSizeBound;

      for(unsigned int i = 0;i < num_devices; i++) 	{    
size_t offsetGPU = std::min(leftover,slice_per_GPU);
	clEnqueueReadBuffer(command_queue[i], d_ptrResultBound[i], CL_FALSE, 0,
			    offsetGPU*sizeof(REAL), ptrResultBound + i*slice_per_GPU, 0, NULL, &GPUDone[i]);
	leftover -= offsetGPU;
      }
      clWaitForEvents(num_devices, GPUDone);
  

    } else {
      for(unsigned int i = 0;i < num_devices; i++) 
	{    
	  clEnqueueReadBuffer(command_queue[i], d_ptrResultBound[i], CL_FALSE, 0,
			      storageInnerSizeBound*sizeof(REAL), ptrResultBound, 0, NULL, &GPUDone[i]);
	}
      clWaitForEvents(num_devices, GPUDone);
    
    }
    for (unsigned i = 0; i < storageInnerSizeBound; i++) {
      ptrResult[offsetBound[i]] = ptrResultBound[i];
    }
  }
#if TOTALTIMING
  CounterLTwoDotBound += 1.0;
#endif	

  for(size_t i = 0; i < num_devices; i++) {
    clReleaseEvent(GPUExecution[i]);
    clReleaseEvent(GPUDone[i]);
    clReleaseEvent(GPUDoneLcl[i]);
  }  
}


void RunOCLKernelLTwoDotBound2(REAL * ptrAlpha, REAL * ptrResult,  REAL * lcl_q, REAL *ptrLevel, REAL *ptrIndex, REAL *ptrLevel_int, size_t argStorageSize, size_t argStorageDim, sg::base::GridStorage* storage) {
  //   std::cout << "EXE RunOCLKernelLTwoDotBound" << std::endl;
  if (isVeryFirstTime) {
    StartUpGPU();
  } 
  if (isFirstTimeLTwoDotBound) {
    if (isFirstTimeLaplaceBound) {
      SetBuffersBound(ptrLevel,
			     ptrIndex,
			     ptrLevel_int,
			     argStorageSize,
			     argStorageDim,storage);
    }
    CompileLTwoDotBoundKernels(); 
    SetArgumentsLTwoDotBound();
    isVeryFirstTime = 0; 
    isFirstTimeLTwoDotBound = 0;
  }
  ExecLTwoDotBound(ptrAlpha, ptrResult,lcl_q, ptrLevel, ptrIndex, ptrLevel_int, argStorageSize, argStorageDim);
}
