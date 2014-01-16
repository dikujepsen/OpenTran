float gradient(
		float i_level_grad,
		float i_index_grad,
		float j_level_grad,
		float j_index_grad,
		float lcl_q_inv
		) {
  float grad;
  
  unsigned doGrad = ((i_level_grad == j_level_grad) && (i_index_grad == j_index_grad));
  grad = i_level_grad * 2.0 * (float)(doGrad);

  return (grad)* lcl_q_inv;
}



float l2dot(float lid,
	     float ljd,
	     float iid,
	     float ijd,
	     float in_lid,
	     float in_ljd,
	     float lcl_q
	     ) {

  float res_one = (2.0 / 3.0) * in_lid * (iid == ijd);

  unsigned selector = (lid > ljd);
  float i1d = iid * (selector) + ijd * (!selector);
  float in_l1d = in_lid * (selector) + in_ljd * (!selector);
  float i2d = ijd * (selector) + iid * (!selector);
  float l2d = ljd * (selector) + lid * (!selector);
  float in_l2d = in_ljd * (selector) + in_lid * (!selector);

  float q = (i1d - 1) * in_l1d;
  float p = (i1d + 1) * in_l1d;
  unsigned overlap = (max(q, (i2d - 1) * in_l2d) < min(p, (i2d + 1) * in_l2d));


  float temp_res = 2.0 - fabs(l2d * q - i2d) - fabs(l2d * p - i2d);
  temp_res *= (0.5 * in_l1d);
  float res_two = temp_res * overlap;

  return (res_one * (lid == ljd) + res_two * (lid != ljd)) * lcl_q;
}


