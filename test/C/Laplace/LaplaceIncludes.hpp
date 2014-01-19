double gradient(
		double i_level_grad,
		double i_index_grad,
		double j_level_grad,
		double j_index_grad,
		double lcl_q_inv
		) {
  double grad;
  
  unsigned doGrad = ((i_level_grad == j_level_grad) && (i_index_grad == j_index_grad));
  grad = i_level_grad * 2.0 * (double)(doGrad);

  return (grad)* lcl_q_inv;
}



double l2dot(double lid,
	     double ljd,
	     double iid,
	     double ijd,
	     double in_lid,
	     double in_ljd,
	     double lcl_q
	     ) {

  double res_one = (2.0 / 3.0) * in_lid * (iid == ijd);

  unsigned selector = (lid > ljd);
  double i1d = iid * (selector) + ijd * (!selector);
  double in_l1d = in_lid * (selector) + in_ljd * (!selector);
  double i2d = ijd * (selector) + iid * (!selector);
  double l2d = ljd * (selector) + lid * (!selector);
  double in_l2d = in_ljd * (selector) + in_lid * (!selector);

  double q = (i1d - 1) * in_l1d;
  double p = (i1d + 1) * in_l1d;
  unsigned overlap = (max(q, (i2d - 1) * in_l2d) < min(p, (i2d + 1) * in_l2d));


  double temp_res = 2.0 - fabs(l2d * q - i2d) - fabs(l2d * p - i2d);
  temp_res *= (0.5 * in_l1d);
  double res_two = temp_res * overlap;

  return (res_one * (lid == ljd) + res_two * (lid != ljd)) * lcl_q;
}


