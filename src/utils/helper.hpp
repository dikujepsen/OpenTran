#include <cfloat>
#include <cmath>

namespace helper
{

  template <class T>
  bool AlmostEqualRelative(T A, T B,
                           T maxRelDiff = FLT_EPSILON);

   template <class T>
  bool check_matrices(T* cpu_mat, T* ocl_mat, size_t size);

  void oclCheckErr(cl_int err, const char * function);

  template <class T>
  void transpose(T * sink, T* source, size_t source_dim1, size_t source_dim2);

}
