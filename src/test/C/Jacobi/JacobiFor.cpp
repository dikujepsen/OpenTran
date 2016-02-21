unsigned wB;
unsigned wA;
float * X2;
float * X1;
float * B;
for (unsigned i = 1; i < wB; i++) {
  for (unsigned j = 1; j < wB; j++) {
    X2[i][j] = -0.25 * (B[i][j] -
			      (X1[(i-1)][j] + X1[(i+1)][j]) -
			      (X1[i][(j-1)] + X1[i][(j+1)]));
  }
 }
