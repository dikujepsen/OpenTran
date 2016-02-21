#include <sstream>
#include <iostream>
#include <cmath>

namespace helper {

    template <class T>
    bool check_matrices(T* cpu_mat, T* ocl_mat, size_t size) {
        bool retval = true;
        for(int i = 0; i < size; i++) {
            T cpu_num = cpu_mat[i];
            T ocl_num = ocl_mat[i];
            T diff = abs(cpu_num - ocl_num);
            if (diff > 1e-3) {
                std::cout << "Error in calculated matrix: CPU: " << cpu_num << " OCL: " << ocl_num;
                std::cout << " Diff: " << diff << std::endl;
                retval = false;
            }

        }
        return retval;
    }

}