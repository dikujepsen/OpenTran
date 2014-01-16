#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define REAL double
#define LSIZE 64

REAL gradient(unsigned i, unsigned j, unsigned dim,
		__global  REAL* ptrLevel,
		__global  REAL* ptrIndex,
		__global  REAL* ptrLevel_int)
{
  REAL grad;
     
  REAL i_level_grad = ptrLevel[dim*STORAGEPAD + i];
  REAL j_level_grad = ptrLevel[dim*STORAGEPAD + j];
  REAL i_index_grad = ptrIndex[dim*STORAGEPAD + i];
  REAL j_index_grad = ptrIndex[dim*STORAGEPAD + j];

  //only affects the diagonal of the stiffness matrix
  REAL doGrad = (REAL)((i_level_grad == j_level_grad) && (i_index_grad == j_index_grad));
  grad = i_level_grad*2.0*(doGrad);

  return grad;
}

REAL l2dot(unsigned i, unsigned j, unsigned dim,
	     __global  REAL* ptrLevel,
	     __global  REAL* ptrIndex,
	     __global  REAL* ptrLevel_int)
{
  REAL i_level = ptrLevel[dim*STORAGEPAD + i];
  REAL j_level = ptrLevel[dim*STORAGEPAD + j];
  REAL i_index = ptrIndex[dim*STORAGEPAD + i];
  REAL j_index = ptrIndex[dim*STORAGEPAD + j];
  REAL i_level_int = ptrLevel_int[dim*STORAGEPAD + i];
  REAL j_level_int = ptrLevel_int[dim*STORAGEPAD + j];

  // start and end of domain
  REAL st_i = i_level_int*(i_index-1.0);
  REAL en_i = i_level_int*(i_index+1.0);
  REAL st_j = j_level_int*(j_index-1.0);
  REAL en_j = j_level_int*(j_index+1.0);
  REAL mi_i = i_level_int*(i_index);
  REAL mi_j = j_level_int*(j_index);

  {
    REAL a_st = max(st_i, st_j);
    REAL b_st = max(st_i, mi_j);
    REAL c_st = max(mi_i, st_j);
    REAL d_st = max(mi_i, mi_j);
    REAL a_en = max(min(mi_i, mi_j), a_st);
    REAL b_en = max(min(mi_i, en_j), b_st);
    REAL c_en = max(min(en_i, mi_j), c_st);
    REAL d_en = max(min(en_i, en_j), d_st);

    REAL i_indexm1 = (i_index-1.0);
    REAL j_indexm1 = (j_index-1.0);
    REAL a = (a_en-a_st);
    REAL b = (b_en-b_st);
    REAL c = (c_en-c_st);
    REAL d = (d_en-d_st);
	
    REAL a_ensq = a_en*a_en;
    REAL a_stsq = a_st*a_st;
    REAL b_ensq = b_en*b_en;
    REAL b_stsq = b_st*b_st;
    REAL c_ensq = c_en*c_en;
    REAL c_stsq = c_st*c_st;
    REAL d_ensq = d_en*d_en;
    REAL d_stsq = d_st*d_st;
    REAL a_sq = (a_ensq-a_stsq);
    REAL b_sq = (b_ensq-b_stsq);
    REAL c_sq = (c_ensq-c_stsq);
    REAL d_sq = (d_ensq-d_stsq);
    // power to 2
    REAL l2dotA = i_level*(b_sq-d_sq) + j_level*(c_sq-d_sq);
    // 23 ops, 1 FMA
	
    // power to 1
    l2dotA += 2*((i_index+j_index)*(d)- i_indexm1*(b) - j_indexm1*(c) );
    // 31 ops, 4 FMA
    l2dotA += 0.5*(i_level*j_indexm1 + j_level*i_indexm1)*
      ( b_sq-a_sq+c_sq-d_sq);
    l2dotA += i_indexm1*j_indexm1*(a-b-c+d);
    // 46 ops, 7 FMA
    // l2dotA += i_level*(b_sq)
    //   + j_level*(c_sq)
    //   + (i_level+j_level)*(-d_sq);
    // l2dotA += -d_sq*(i_level*(b_sq)
    // 		 + j_level*(c_sq));
    // REAL a_enqu = (a_ensq*a_en);
    // REAL a_qu = (a_enqu-(a_stsq*a_st));
    REAL b_enqu = (b_ensq*b_en);
    REAL b_qu = (b_enqu-(b_stsq*b_st));
    REAL c_enqu = (c_ensq*c_en);
    REAL  c_qu = (c_enqu-(c_stsq*c_st));
    // 52 ops, 9 FMA
    // power to 3
    l2dotA += i_level*((a_ensq*a_en - b_qu - (a_stsq*a_st)
			- (d_stsq*d_st) + d_ensq*d_en - c_qu
			)*0.333333333333333333333*j_level);
    // 65 ops, 14 FMA
    return l2dotA;
  }
}
    




__kernel void multKernel(__global  REAL* ptrLevel, // 64
			 __global  REAL* ptrIndex, // 64
			 __global  REAL* ptrLevel_int, // 64
			 __global  REAL* ptrAlpha, // 64
			 __global  REAL* ptrResult, // 64
			 __global  REAL* ptrParResult // 64
			 )
{
  unsigned i = get_global_id(1);
  unsigned lj = get_local_id(0);
  unsigned gj = get_group_id(0);
  
  REAL element;
  REAL res = 0.0;
  __local REAL threadParResult[LSIZE];
  __local REAL l2dotTemp[LSIZE*DIMS];

  for (unsigned k = 0; k < GRAIN; k++) {
    for(size_t d_inner = 0; d_inner < DIMS; d_inner++) {
      l2dotTemp[d_inner*LSIZE + lj] = (l2dot(i, ((gj*GRAIN+k)*LSIZE+lj), d_inner,ptrLevel,ptrIndex,ptrLevel_int));
    }
    for(unsigned d_outer = 0; d_outer < DIMS; d_outer++)
      {
	element = ptrAlpha[(gj*GRAIN+k)*LSIZE + lj];
	for(unsigned d_inner = 0; d_inner < DIMS; d_inner++)
	  {
	   
	    element *= (l2dotTemp[d_inner*LSIZE+lj]*(d_outer != d_inner)) + gradient(i, ((gj*GRAIN+k)*LSIZE+lj), d_inner,ptrLevel,ptrIndex,ptrLevel_int)*(d_outer == d_inner);
	  }
	res += element;
      }
  }

  threadParResult[lj] = res;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lj+get_local_id(1) == 0) {
    res = 0.0;
    for (unsigned k = 0; k < LSIZE; k++) {
      res += threadParResult[k];
    }
    ptrParResult[gj*STORAGEPAD + i] = res;
  }
}
