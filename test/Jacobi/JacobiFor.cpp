for (unsigned i = 1; i < wB; i++) {
  for (unsigned j = 1; j < wB; j++) {

    X2[i * wA + j] = -0.25 * (B[i * wB + j] -
			      (X1[(i-1) * wA + j] + X1[(i+1) * wA + j]) -
			      (X1[i * wA + (j-1)] + X1[i * wA + (j+1)]));
  }
 }