unsigned NTEST;
unsigned NTRAIN;
unsigned dim;
float * test_patterns;
float * train_patterns;
float * dist_matrix;
for (size_t i = 0; i < NTEST; i++)
  {
    for (size_t j = 0; j < NTRAIN; j++)
      {
	float d = 0.0;
	for (size_t k = 0; k < dim; k++)
	  {
	    float tmp = test_patterns[i][k] - train_patterns[j][k];
	    d += tmp * tmp;
	  }
	dist_matrix[j][i] = d;
      }
  }
