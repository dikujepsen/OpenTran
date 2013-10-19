#define START 0.0

void
matmul(float* A, float* B, float* C, unsigned hA, unsigned wA, unsigned wB)
{
  char * str = "asda2d1dasdw22d";
    for (unsigned i = 0; i < hA; ++i)
        for (unsigned j = 0; j < wB; ++j) {
            float sum = START;
            for (unsigned k = 0; k < wA; ++k) {
                sum += A[i * wA + k] * B[k * wB + j];
            }
            C[i * wB + j] = sum;
        }
}
