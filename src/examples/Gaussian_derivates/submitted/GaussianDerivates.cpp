#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>

using namespace std;

float He(const unsigned n,
	 const float x) {
  float v = 0.0;

  switch(n) {
  case 0:
    v = 1.0;
    break;
  case 1:
    v = x;
    break;
  case 2:
    v = x*x-1;
    break;
  case 3:
    v = pow(x,3)-3*x;
    break;
  case 4:
    v = pow(x,4)-6*x*x+3;
    break;
  case 5:
    v = pow(x,5)-10*pow(x,3)+15*x;
    break;
  default:
    printf("INVALID HERMITE\n");
  }

  return v;
}


float DaKs(unsigned da[3], float xmy[3], float r, float ks) {
  float z[3];
  float sum_da = 0.0;
  for (unsigned k = 0; k < 3; k++) {
    z[k] = sqrt(2)/r*xmy[k];

    sum_da += da[k];
  }
  
  float res;
  res = pow(-sqrt(2)/r, sum_da)*He(da[0],z[0])*He(da[1],z[1])*He(da[2],z[2])*ks;
  return res;
}


float gamma(float xmy[3], float r2, float sw2) {
  float dot1 = 0.0;
  for (unsigned k = 0; k < 3; k++) {
    dot1 += xmy[k] * xmy[k];
  }
  return 1.0/sw2*exp(dot1/r2);
}

void
GaussianDerivates(unsigned Lp, unsigned Lq, unsigned dim,
		  float * p_a_i_x, unsigned p_a_i_rows,
		  float * q_a_i_x, unsigned q_a_i_rows,
		  float * K__ij_x, unsigned K__ij_rows,
		  float scales2_x, float scaleweight2_x,
		  float * D1Ks__ijb_x,   unsigned * D1Ks__ijb_dimsI,
		  float * D2Ks__ijbg_x,  unsigned * D2Ks__ijbg_dimsI,
		  float * D3Ks__ijbgd_x, unsigned * D3Ks__ijbgd_dimsI
		  )
{
  
  for (unsigned j=0; j<Lp; j++) {
    for (unsigned i=0; i<Lq; i++) {
      float xj[3];
      float xi[3];
      for (unsigned k = 0; k < dim; k++) {
	xj[k] = p_a_i_x[j*p_a_i_rows + k];
      }

      for (unsigned k = 0; k < dim; k++) {
	xi[k] = q_a_i_x[i*q_a_i_rows + k];
      }


      float ximxj[3];
      for (unsigned k = 0; k < dim; k++) {
	ximxj[k] = xi[k] - xj[k];
      }
      
      float r = sqrt(scales2_x);

      float ks = gamma(ximxj, scales2_x, scaleweight2_x);
      K__ij_x[i+K__ij_rows*j] = ks;

      unsigned da[3];
      unsigned db[3];
      unsigned dc[3];

      for (unsigned b=0; b<dim; b++) {

	for (unsigned k = 0; k < dim; k++) {
	  da[k] = 1;
	}

	D1Ks__ijb_x[i +
		    D1Ks__ijb_dimsI[0] * j +
		    D1Ks__ijb_dimsI[1] * b
		    ] = DaKs(da, ximxj,r,ks);


	// nargout 2
	for (unsigned g=0; g<dim; g++) {
	  // Vector3<unsigned> db = da;
	  for (unsigned k = 0; k < dim; k++) {
	    db[k] = da[k] + 1;
	  }

	  
	  D2Ks__ijbg_x[i +
		       D2Ks__ijbg_dimsI[0] * j +
		       D2Ks__ijbg_dimsI[1] * b +
		       D2Ks__ijbg_dimsI[2] * g ] =
	    DaKs(db,ximxj,r,ks);


	  for (unsigned d=0; d<dim; d++) {
	    // Vector3<unsigned> dc = db; dc.set(dc[d]+1,d);
	    for (unsigned k = 0; k < dim; k++) {
	      dc[k] = db[k] + 1;
	    }
	    
	    D3Ks__ijbgd_x[i +
			  D3Ks__ijbgd_dimsI[0] * j +
			  D3Ks__ijbgd_dimsI[1] * b +
			  D3Ks__ijbgd_dimsI[2] * g +
			  D3Ks__ijbgd_dimsI[3] * d] =
	      DaKs(dc,ximxj,r,ks);

	  }
	}
      }
    }
  }
}

void
randMat(float* mat, unsigned mat_size)
{
  for (unsigned i = 0; i < mat_size; ++i) {
    mat[i] = (float)((rand() % 10)/10.0);
  }
}


void
copyMat(float* mat1, float* mat2, unsigned mat_size)
{
  for (unsigned i = 0; i < mat_size; i++) {
      mat1[i] = mat2[i];
  }
}


int main(int argc, char** argv)
{
  unsigned Lp = 2048;
  unsigned Lq = 2048;
  unsigned dim = 3;

  unsigned p_a_i_rows = dim;
  unsigned q_a_i_rows = dim;
  unsigned K__ij_rows = Lp;
  float scales2_x = 2.1f;
  float scaleweight2_x = 1.7f;

  unsigned p_a_i_x_size = Lp * p_a_i_rows;
  unsigned q_a_i_x_size = Lq * q_a_i_rows;
  unsigned K__ij_x_size = Lq * K__ij_rows; 
  float * p_a_i_x = new float[p_a_i_x_size];
  float * q_a_i_x = new float[q_a_i_x_size];
  float * K__ij_x_cpu = new float[K__ij_x_size];
  float * K__ij_x = new float[K__ij_x_size];

  unsigned * D1Ks__ijb_dimsI = new unsigned[2];
  unsigned * D2Ks__ijbg_dimsI = new unsigned[3];
  unsigned * D3Ks__ijbgd_dimsI  = new unsigned[4];

  // Insert some values as dimensions
  D1Ks__ijb_dimsI[0] = Lq;
  D2Ks__ijbg_dimsI[0] = Lq;
  D3Ks__ijbgd_dimsI[0] = Lq;

  D1Ks__ijb_dimsI[1] = Lq*Lp;
  D2Ks__ijbg_dimsI[1] = Lq*Lp;
  D3Ks__ijbgd_dimsI[1] = Lq*Lp;

  D2Ks__ijbg_dimsI[2] = Lq*Lp*dim;
  D3Ks__ijbgd_dimsI[2] = Lq*Lp*dim;

  D3Ks__ijbgd_dimsI[3] = Lq*Lp*dim*dim;

  
  unsigned D1Ks__ijb_x_size = D1Ks__ijb_dimsI[1] * dim;
  unsigned D2Ks__ijbg_x_size = D2Ks__ijbg_dimsI[2] * dim;
  unsigned D3Ks__ijbgd_x_size = D3Ks__ijbgd_dimsI[3] * dim;
  
  float * D1Ks__ijb_x =   new float[D1Ks__ijb_x_size];
  float * D2Ks__ijbg_x =  new float[D2Ks__ijbg_x_size];
  float * D3Ks__ijbgd_x = new float[D3Ks__ijbgd_x_size];

  
  srand(2013);

  randMat(p_a_i_x, p_a_i_x_size);
  randMat(q_a_i_x, q_a_i_x_size);
  randMat(K__ij_x_cpu, K__ij_x_size);
  copyMat(K__ij_x, K__ij_x_cpu, K__ij_x_size);



  GaussianDerivates( Lp,  Lq,  dim,
		     p_a_i_x,  p_a_i_rows,
		     q_a_i_x,  q_a_i_rows,
		     K__ij_x_cpu,  K__ij_rows,
		     scales2_x,  scaleweight2_x,
		     D1Ks__ijb_x,     D1Ks__ijb_dimsI,
		     D2Ks__ijbg_x,    D2Ks__ijbg_dimsI,
		     D3Ks__ijbgd_x,   D3Ks__ijbgd_dimsI);


  
}




