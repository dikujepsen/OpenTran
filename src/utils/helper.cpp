#include <sstream>
#include <iostream>
#include <cmath>
#include <float.h>

namespace helper {

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
    bool check_matrices(T* cpu_mat, T* ocl_mat, size_t size) {
        bool retval = true;
        for(int i = 0; i < size; i++) {
            T cpu_num = cpu_mat[i];
            T ocl_num = ocl_mat[i];
            T diff = abs(cpu_num - ocl_num);
            if (!AlmostEqualRelative(cpu_num, ocl_num, (T)1e-3)) {
                std::cout << "Error in calculated matrix: CPU: " << cpu_num << " OCL: " << ocl_num;
                std::cout << " Diff: " << diff << std::endl;
                retval = false;
            }

        }
	if (!retval) {
	  std::cout << "$Error" << endl;
	}
        return retval;
    }


}
