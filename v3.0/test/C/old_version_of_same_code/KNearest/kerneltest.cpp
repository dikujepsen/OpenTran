void KNearestFor(
	float * train_patterns, size_t hst_ptrtrain_patterns_dim1, size_t hst_ptrtrain_patterns_dim2, 
	float * test_patterns, size_t hst_ptrtest_patterns_dim1, size_t hst_ptrtest_patterns_dim2, 
	float * dist_matrix, size_t hst_ptrdist_matrix_dim1, size_t hst_ptrdist_matrix_dim2, 
	unsigned dim, unsigned NTEST, unsigned NTRAIN
	)
{
  for (size_t i = 0; i < NTEST; i++)
    {
      for (size_t j = 0; j < NTRAIN; j++)
        {
          float d = 0.0;
          for (size_t k = 0; k < dim; k++)
            {
              float tmp = test_patterns[k][i] - train_patterns[j][k];
              d += tmp * tmp;
            }
          dist_matrix[j][i] = d;
        }
    }
}

