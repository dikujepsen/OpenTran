// problem parameters
int NTRAIN = 16384;
int NTEST = 3*NTRAIN;
int dim = 16;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
 
int main( int argc, char* argv[] )
{
    int i,j,k;

    // training and test patterns
    float *train_patterns = (float*)malloc(NTRAIN*dim*sizeof(float));
    float *test_patterns = (float*)malloc(NTEST*dim*sizeof(float));
    // result
    float *dist_matrix = (float*)malloc(NTRAIN*NTEST*sizeof(float));
    // initialize with some values ...
    for (i=0;i<NTRAIN;i++){
        for (k=0;k<dim;k++){
            train_patterns[i*dim + k] = (float)cos(i*k);
        }
    }
    for (i=0;i<NTEST;i++){
        for (k=0;k<dim;k++){
            test_patterns[i*dim + k] = (float)sin(i*k);
        }
    }
    START_TIMER(1);
    float d,tmp;
    int jdim, idim;
    /********************************* SPEED UP THESE COMPUTATIONS *********************************/
    // for each test pattern
    for (i=0;i<NTEST;i++){
      idim = i*dim;
      // compute distances to all training patterns                           
      for (j=0;j<NTRAIN;j++){
	d = 0.0;
	jdim = j*dim;
	// for each feature                                                 
	for (k=0;k<dim;k++){
	  tmp = test_patterns[idim+k]-train_patterns[jdim+k];
	  d += tmp*tmp;
	}
	dist_matrix[i*NTRAIN + j] = d;
      }
    }

    /********************************* SPEED UP THESE COMPUTATIONS *********************************/
    STOP_TIMER(1);
    printf("Computation done.\n");

    // print matrix
    for (i=0;i<20;i++){    
        for (j=0;j<10;j++){
            printf("%f ", dist_matrix[i*NTRAIN + j]);
        }
        printf("\n");
    }
    // sanity check: first and last row
    float sum = 0.0;
    for (j=0;j<NTRAIN;j++){
        sum += (float)dist_matrix[0*NTRAIN + j];
        sum += (float)dist_matrix[(NTEST-1)*NTRAIN + j];
    }
    printf("Sum of first and last row=%f\n", sum);

    // final timing results
    printf("\nElapsed time=%f\n", GET_TIME(1));
    
    return 0;
}
