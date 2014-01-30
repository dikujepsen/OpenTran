void LaplaceFor(
	double * index, size_t hst_ptrindex_dim1, size_t hst_ptrindex_dim2, 
	double * level_int, size_t hst_ptrlevel_int_dim1, size_t hst_ptrlevel_int_dim2, 
	double * level, size_t hst_ptrlevel_dim1, size_t hst_ptrlevel_dim2, 
	double * lcl_q, size_t hst_ptrlcl_q_dim1, double * result, 
	size_t hst_ptrresult_dim1, double * lcl_q_inv, size_t hst_ptrlcl_q_inv_dim1, 
	double * alpha, size_t hst_ptralpha_dim1, double * lambda, 
	size_t hst_ptrlambda_dim1, size_t dim, size_t storagesize
	)
{
  for (unsigned i = 0; i < storagesize; i++)
    {
      for (unsigned j = 0; j < storagesize; j++)
        {
          float gradient_temp[dim];
          float dot_temp[dim];
          for (unsigned d = 0; d < dim; d++)
            {
              float level_i = level[i][d];
              float level_j = level[j][d];
              float level_int_i = level_int[i][d];
              float level_int_j = level_int[j][d];
              float index_i = index[i][d];
              float index_j = index[j][d];
              gradient_temp[d] = gradient(
	level_i, index_i, level_j, 
	index_j, lcl_q_inv[d]);
              dot_temp[d] = l2dot(
	level_i, level_j, index_i, 
	index_j, level_int_i, level_int_j, 
	lcl_q[d]);
            }
          float sub = 0.0;
          for (size_t d_outer = 0; d_outer < dim; d_outer++)
            {
              float element = alpha[j];
              for (size_t d_inner = 0; d_inner < dim; d_inner++)
                {
                  element *= (dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner));
                }
              sub += lambda[d_outer] * element;
            }
          result[i] += sub;
        }
    }
}

