#include <sstream>
#include <iostream>


#include "helper.hpp"

using namespace std;


namespace helper
{

  template <class T>
  bool AlmostEqualRelative(T A, T B,
                           T maxRelDiff = FLT_EPSILON)
  {
    // Calculate the difference.
    T diff = fabs(A - B);
    A = fabs(A);
    B = fabs(B);
    // Find the largest
    T largest = (B > A) ? B : A;

    if (diff <= largest * maxRelDiff)
      return true;
    return false;
  }

  template <class T>
  bool check_matrices(T* cpu_mat, T* ocl_mat, size_t size)
  {
    bool retval = true;
    for(int i = 0; i < size; i++)
    {
      T cpu_num = cpu_mat[i];
      T ocl_num = ocl_mat[i];
      T diff = abs(cpu_num - ocl_num);
      if (!AlmostEqualRelative(cpu_num, ocl_num, (T)1e-3))
      {
        std::cout << "Error in calculated matrix: CPU: " << cpu_num << " OCL: " << ocl_num;
        std::cout << " Diff: " << diff << std::endl;
        retval = false;
      }

    }
    if (!retval)
    {
      std::cout << "$Error" << endl;
    }
    return retval;
  }

      void oclCheckErr(cl_int err, const char * function)
    {
      if (err != CL_SUCCESS)
      {
        printf("Error: Failure %s: %d\n", function, err);
        exit(-1);
      }
    }


  template <class T>
  void transpose(T * sink, T* source, size_t source_dim1, size_t source_dim2)
  {

    for (size_t i = 0; i < source_dim2; i++)
    {
      for (size_t j = 0; j < source_dim1; j++)
      {
        source[j * source_dim2 + i] = sink[i * source_dim1 + j];
      }
    }

  }

}
