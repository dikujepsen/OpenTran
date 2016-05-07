#include <cfloat>
#include <cmath>

namespace helper
{


  void oclCheckErr(cl_int err, const char * function);

  template <class T>
  void transpose(T * sink, T* source, size_t source_dim1, size_t source_dim2);

}
