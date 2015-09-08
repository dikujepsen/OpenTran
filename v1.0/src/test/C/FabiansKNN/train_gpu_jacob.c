// problem parameters
int NTRAIN = 16384;
int NTEST = 3*16384;
int dim = 16;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>

#include <assert.h>
#include <float.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>

long get_system_time_in_microseconds(void){
	struct timeval tempo;
	gettimeofday(&tempo, NULL);
	return tempo.tv_sec * 1000000 + tempo.tv_usec;	
}

#define TIMING

#ifdef TIMING
#define DEFINE_TIMER(num) long start_time##num = 0; double elapsed_time##num = 0.0f;
#define DECLARE_TIMER(num) extern long start_time##num; extern double elapsed_time##num;
#define START_TIMER(num) start_time##num = get_system_time_in_microseconds();
#define STOP_TIMER(num) elapsed_time##num = (((double)get_system_time_in_microseconds())-((double)start_time##num));
#define GET_TIME(num) (double)(1.0*elapsed_time##num / 1000000.0)
#else
#define DEFINE_TIMER(num) 
#define DECLARE_TIMER(num)
#define START_TIMER(num) 
#define STOP_TIMER(num) 
#define GET_TIME(num)
#endif



DEFINE_TIMER(1);
DEFINE_TIMER(2);

void check_cl_error(cl_int err, const char *file, int line)
{
    if (err != CL_SUCCESS) {
        printf("Error with errorcode: %d in file %s in line %d \n", err, file, line);
        exit(1);
    }
}

// READ ME READ ME READ ME READ ME READ ME READ ME READ ME READ ME READ ME
// Below, we need to define DIM as dim manually ...
// READ ME READ ME READ ME READ ME READ ME READ ME READ ME READ ME READ ME
const char *kernelSource =                                                                 "\n" \
"#define DIM 16                                                                            \n" \
"__kernel void compute_distances_caching(__global float *dist_matrix,                       \n" \
"                                        __global float *train_patterns,                    \n" \
"                                        __global float *test_patterns,                     \n" \
"                                        int NTRAIN,                                        \n" \
"                                        int NTEST,                                         \n" \
"                                        int dim)                                           \n" \
"{                                                                                          \n" \
"    int tid = get_global_id(0);                                                            \n" \
"    if (tid>=NTEST) return;                                                                \n" \
"    int j,k;                                                                               \n" \
"    // generate private copy of test pattern                                               \n" \
"    float test_patt_private[DIM];                                                          \n" \
"    for (k=0;k<DIM; k++){test_patt_private[k]=test_patterns[k*NTEST + tid];}               \n" \
"    for(j = 0; j < NTRAIN; j++) {                                                          \n" \
"        float d = 0.0;                                                                     \n" \
"        for(k = 0; k < DIM; k++) {                                                         \n" \
"            float tmp = (test_patt_private[k] - train_patterns[j*DIM+k]);                  \n" \
"            d += tmp*tmp;                                                                  \n" \
"        }                                                                                  \n" \
"    dist_matrix[j*NTEST + tid] = d;                                                        \n" \
"    }                                                                                      \n" \
"}                                                                                          \n" \
                                                                                           "\n" ;
 
int main( int argc, char* argv[] )
{
    int i,j,k;

    // local and global sizes
    size_t globalSize, localSize;
    localSize = 256; 
    globalSize = ceil(NTEST/(float)localSize)*localSize;

    // training and test patterns
    float *train_patterns = (float*)malloc(NTRAIN*dim*sizeof(float));
    float *test_patterns = (float*)malloc(NTEST*dim*sizeof(float));
    // distance matrix (result)
    float *dist_matrix = (float*)malloc(NTRAIN*NTEST*sizeof(float));

    // initialize with some values ...
    for (i=0;i<NTRAIN;i++){
        for (k=0;k<dim;k++){
            train_patterns[i*dim + k] = (float)cos(i*k);
        }
    }
    for (i=0;i<NTEST;i++){
        for (k=0;k<dim;k++){
            test_patterns[k*NTEST + i] = (float)sin(i*k);
        }
    }

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
    cl_int err;
    cl_event event;
 
    // platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    check_cl_error(err, __FILE__, __LINE__);
 
    // id of device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    check_cl_error(err, __FILE__, __LINE__);

    // context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
 
    // build program
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // device buffers
    cl_mem device_train_patterns;
    cl_mem device_test_patterns;
    cl_mem device_dist_matrix;

    // train patterns
    device_train_patterns = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       NTRAIN*dim*sizeof(float), train_patterns, &err);
    check_cl_error(err, __FILE__, __LINE__);

    // test patterns
    device_test_patterns = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       NTEST*dim*sizeof(float), test_patterns, &err);
    check_cl_error(err, __FILE__, __LINE__);

    // allocate space for distances
    device_dist_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, NTEST*NTRAIN*sizeof(float), NULL, NULL);

    // create kernel
    kernel = clCreateKernel(program, "compute_distances_caching", &err);    

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_dist_matrix);
    check_cl_error(err, __FILE__, __LINE__);
    err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_train_patterns);
    check_cl_error(err, __FILE__, __LINE__);
    err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_test_patterns);
    check_cl_error(err, __FILE__, __LINE__);
    err = clSetKernelArg(kernel, 3, sizeof(int), &NTRAIN);
    check_cl_error(err, __FILE__, __LINE__);
    err = clSetKernelArg(kernel, 4, sizeof(int), &NTEST);
    check_cl_error(err, __FILE__, __LINE__);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dim);
    check_cl_error(err, __FILE__, __LINE__);

    START_TIMER(1); 
    // run kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &event);
    check_cl_error(err, __FILE__, __LINE__);
    // wait for kernel
    clFinish(queue);
    err = clWaitForEvents(1, &event);
    check_cl_error(err, __FILE__, __LINE__);
    STOP_TIMER(1)
    printf("Computation done.\n");

    START_TIMER(2);
    // transfer results back to host
    printf("Transferring data back to host system ...\n");
    err = clEnqueueReadBuffer(queue, device_dist_matrix, CL_TRUE, 0, NTEST*NTRAIN*sizeof(float), dist_matrix, 0, NULL, &event );
    check_cl_error(err, __FILE__, __LINE__);
    err = clWaitForEvents(1, &event);
    check_cl_error(err, __FILE__, __LINE__);
    STOP_TIMER(2)

    // print matrix
    for (i=0;i<20;i++){     
        for (j=0;j<10;j++){
            // NOTE: different indexing for GPU matrix here
            printf("%f ", dist_matrix[j*NTEST+i]);
        }
        printf("\n");
    }
    // sanity check: first and last row
    float sum = 0.0;
    for (j=0;j<NTRAIN;j++){
        sum += (float)dist_matrix[j*NTEST + 0];
        sum += (float)dist_matrix[j*NTEST + (NTEST-1)];
    }
    printf("Sum of first and last row=%f\n", sum);

    // final timing results
    printf("\nElapsed time for kernel execution=%f\n", GET_TIME(1));
    printf("Elapsed time for matrix transfer=%f\n", GET_TIME(2));

    // release OpenCL resources
    clReleaseMemObject(device_train_patterns);
    clReleaseMemObject(device_test_patterns);
    clReleaseMemObject(device_dist_matrix);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context); 

    free(train_patterns);
    free(test_patterns);
    free(dist_matrix);
    return 0;
}
