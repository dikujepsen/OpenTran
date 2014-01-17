for (unsigned i = 0; i < storagesize; i++) {
  for (unsigned j = 0; j < storagesize; j++) {
    float gradient_temp[dim];
    float dot_temp[dim];
    for (unsigned d = 0; d < dim; d++) {
      float level_i = level[i * dim + d];
      float level_j = level[j * dim + d];
      float level_int_i = level_int[i * dim + d];
      float level_int_j = level_int[j * dim + d];
      float index_i = index[i * dim + d];
      float index_j = index[j * dim + d];
      gradient_temp[d] = gradient(level_i,index_i,
				   level_j,index_j, lcl_q_inv[d]);
      dot_temp[d] = l2dot(level_i,
			   level_j,
			   index_i,
			   index_j,
			   level_int_i,
			   level_int_j,
			   lcl_q[d]);
    }
    float sub = 0.0;
    for (size_t d_outer = 0; d_outer < dim; d_outer++) {
      float element = alpha[j];

      for (size_t d_inner = 0; d_inner < dim; d_inner++) {
	element *= ((dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner)));
      }
      sub += lambda[d_outer] * element;
    }
    result[i] += sub;
  }
 }