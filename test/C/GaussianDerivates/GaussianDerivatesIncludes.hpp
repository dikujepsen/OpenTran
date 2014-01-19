float He(const int n,
	 const float x) {
  float v;

  switch(n) {
  case 0:
    v = 1;
    break;
  case 1:
    v = x;
    break;
  case 2:
    v = x*x-1;
    break;
  case 3:
    v = native_powr(x,3)-3*x;
    break;
  case 4:
    v = native_powr(x,4)-6*x*x+3;
    break;
  case 5:
    v = native_powr(x,5)-10*native_powr(x,3)+15*x;
    break;
  default:
    ;
  }

  return v;
}


float DaKs(int da[3], float xmy[3], float r, float ks) {
  float z[3];
  float sum_da = 0.0;
  for (int k = 0; k < 3; k++) {
    z[k] = native_sqrt(2.0f)/r*xmy[k];

    sum_da += da[k];
  }
  
  float res;
  res = native_powr(-native_sqrt(2.0f)/r, sum_da)*He(da[0],z[0])*He(da[1],z[1])*He(da[2],z[2])*ks;
  return res;
}


float gamma(float xmy[3], float r2, float sw2) {
  float dot1 = 0.0;
  for (int k = 0; k < 3; k++) {
    dot1 += xmy[k] * xmy[k];
  }
  return 1.0f/sw2*native_exp(dot1/r2);
}
