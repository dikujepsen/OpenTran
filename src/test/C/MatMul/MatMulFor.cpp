unsigned hA;
unsigned wB;
unsigned wA;
float * A;
float * B;
float * C;
for (unsigned i = 0; i < hA; i++) {
  for (unsigned j = 0; j < wB; j++) {
    float sum = 0;
    for (unsigned k = 0; k < wA; k++) {
      sum += A[i][k] * B[k][j];
    }
    C[i][j] = sum;
  }
}
