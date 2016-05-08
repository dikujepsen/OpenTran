#include <sstream>
#include <iostream>


#include "helper.hpp"

using namespace std;


namespace helper
{

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
