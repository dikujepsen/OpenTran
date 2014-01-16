#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define REAL double
#define LSIZE 64

REAL gradient(REAL i_level, 
	      REAL i_index, 
	      REAL j_level, 
	      REAL j_index,
	      REAL lcl_q_inv)
{
  REAL grad;
     
  /* REAL i_level_grad = ptrLevel[dim*STORAGEPAD + i]; */
  /* REAL j_level_grad = ptrLevel[dim*STORAGEPAD + j]; */
  /* REAL i_index_grad = ptrIndex[dim*STORAGEPAD + i]; */
  /* REAL j_index_grad = ptrIndex[dim*STORAGEPAD + j]; */

  //only affects the diagonal of the stiffness matrix
  ulong doGrad = (ulong)((i_level == j_level) && (i_index == j_index));
  grad = select(0.0, i_level * 2.0 * lcl_q_inv,doGrad);

  return grad;
}

REAL l2dot(REAL lid, 
	   REAL iid, 
	   REAL in_lid, 
	   REAL ljd, 
	   REAL ijd,
	   REAL in_ljd,
	   REAL lcl_q)

{

  //////////
  /// First case: lid == ljd
  /// we mask this in the end in the last line
  //////////
	
  // use textbook formular if both operands are identical
  // ansatz function on the same level but with different indecies
  // don't overlap!
  REAL res_one = select(0.0,(2.0/3.0) * in_lid, (ulong)(iid == ijd));
  // 4 ops
  //////////
  /// Second case: lid != ljd
  /// we mask this in the end in the last line
  //////////

  // now we select the 1st as the "narrow" basisfunction (it has a higher level)
  // --> we know can regard 2nd function as linear function and therefore
  // apply the wellknown formular: 1/2 * (f_l + f_r) * 2^{-l}
  ulong selector = (lid > ljd);
  REAL i1d = select(ijd, iid, selector);
  REAL l1d = select(ljd, lid,selector);
  REAL in_l1d = select(in_ljd, in_lid,selector);
  REAL i2d = select(iid, ijd, selector);
  REAL l2d = select(lid, ljd, selector);
  REAL in_l2d = select(in_lid, in_ljd, selector);
		
  // check if Ansatz functions on different
  // levels do not overlap and neg.
  // overlap is 1 if the functions overlap
  // if they don't overlap result is zero.
  // we mask the l2 scalar product in the end
  REAL q = fma(i1d, in_l1d, -in_l1d); //(i1d-1)*in_l1d;
  REAL p = fma(i1d, in_l1d, in_l1d); //(i1d+1)*in_l1d;
  // NEW 5 ops
  // 23 ops
  ulong overlap = (max(q, fma(i2d, in_l2d, -in_l2d)) < min(p, fma(i2d, in_l2d,in_l2d)));
  // 13 ops
  // wie determine fl and fr by plugging them
  // into the sparse grids basis functions given by l2d and i2d.
  // Then we use the formular from above
  REAL temp_res = fma((0.5*in_l1d), (- fabs(fma(l2d,q,-i2d)) - fabs(fma(l2d,p,-i2d))), in_l1d);
  // NEW 1 + 2 + 2 + 2 + 2 + 2 = 11

  // temp_res *= (0.5*in_l1d);
  // 23 ops
  REAL res_two = select(0.0,temp_res,overlap); // Now mask result
  // NEW 1 
  return (select(res_two*lcl_q, res_one*lcl_q, (ulong)(lid == ljd)));
  // 46 ops
  // NEW TOTAL 17 flop

}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define REAL double
#define LSIZE 64
#define GRAIN 1
#define WARP_SIZE 32

REAL gradient(REAL i_level, 
	      REAL i_index, 
	      REAL j_level, 
	      REAL j_index,
	      REAL lcl_q_inv)
{
  REAL grad;
     
  /* REAL i_level_grad = ptrLevel[dim*STORAGEPAD + i]; */
  /* REAL j_level_grad = ptrLevel[dim*STORAGEPAD + j]; */
  /* REAL i_index_grad = ptrIndex[dim*STORAGEPAD + i]; */
  /* REAL j_index_grad = ptrIndex[dim*STORAGEPAD + j]; */

  //only affects the diagonal of the stiffness matrix
  ulong doGrad = (ulong)((i_level == j_level) && (i_index == j_index));
  grad = select(0.0, i_level * 2.0 * lcl_q_inv,doGrad);

  return grad;
}

REAL l2dot(REAL lid, 
	   REAL iid, 
	   REAL in_lid, 
	   REAL ljd, 
	   REAL ijd,
	   REAL in_ljd,
	   REAL lcl_q)

{

  //////////
  /// First case: lid == ljd
  /// we mask this in the end in the last line
  //////////
	
  // use textbook formular if both operands are identical
  // ansatz function on the same level but with different indecies
  // don't overlap!
  REAL res_one = select(0.0,(2.0/3.0) * in_lid, (ulong)(iid == ijd));
  // 4 ops
  //////////
  /// Second case: lid != ljd
  /// we mask this in the end in the last line
  //////////

  // now we select the 1st as the "narrow" basisfunction (it has a higher level)
  // --> we know can regard 2nd function as linear function and therefore
  // apply the wellknown formular: 1/2 * (f_l + f_r) * 2^{-l}
  ulong selector = (lid > ljd);
  REAL i1d = select(ijd, iid, selector);
  REAL l1d = select(ljd, lid,selector);
  REAL in_l1d = select(in_ljd, in_lid,selector);
  REAL i2d = select(iid, ijd, selector);
  REAL l2d = select(lid, ljd, selector);
  REAL in_l2d = select(in_lid, in_ljd, selector);
		
  // check if Ansatz functions on different
  // levels do not overlap and neg.
  // overlap is 1 if the functions overlap
  // if they don't overlap result is zero.
  // we mask the l2 scalar product in the end
  REAL q = fma(i1d, in_l1d, -in_l1d); //(i1d-1)*in_l1d;
  REAL p = fma(i1d, in_l1d, in_l1d); //(i1d+1)*in_l1d;
  // NEW 5 ops
  // 23 ops
  ulong overlap = (max(q, fma(i2d, in_l2d, -in_l2d)) < min(p, fma(i2d, in_l2d,in_l2d)));
  // 13 ops
  // wie determine fl and fr by plugging them
  // into the sparse grids basis functions given by l2d and i2d.
  // Then we use the formular from above
  REAL temp_res = fma((0.5*in_l1d), (- fabs(fma(l2d,q,-i2d)) - fabs(fma(l2d,p,-i2d))), in_l1d);
  // NEW 1 + 2 + 2 + 2 + 2 + 2 = 11

  // temp_res *= (0.5*in_l1d);
  // 23 ops
  REAL res_two = select(0.0,temp_res,overlap); // Now mask result
  // NEW 1 
  return (select(res_two*lcl_q, res_one*lcl_q, (ulong)(lid == ljd)));
  // 46 ops
  // NEW TOTAL 17 flop

}
    




__kernel void multKernelIndexes(__global  REAL* ptrLevel, // 64
				__global  REAL* ptrIndex, // 64
				__global  REAL* ptrLevel_int, // 64
				__global  REAL* ptrAlpha, // 64
				__global  REAL* ptrResult, // 64
				__global  REAL* ptrParResult, // 64
				__constant  REAL* ptrLcl_q)
{
  
  __local REAL alphaTemp[LSIZE];
  __local REAL l2dotTemp[LSIZE*DIMS];
  __local REAL gradTemp[LSIZE*DIMS];
  __local REAL j_level[LSIZE*DIMS]; 
  __local REAL j_index[LSIZE*DIMS]; 
  __local REAL j_level_int[LSIZE*DIMS]; 
  alphaTemp[get_local_id(0)]   = ptrAlpha[get_global_id(0)];
 

  for(size_t d_inner = 0; d_inner < DIMS; d_inner++) {
    j_level[d_inner*LSIZE + get_local_id(0)] =         ptrLevel[d_inner*STORAGEPAD + get_global_id(0)];
    j_index[d_inner*LSIZE + get_local_id(0)] =         ptrIndex[d_inner*STORAGEPAD + get_global_id(0)];
    j_level_int[d_inner*LSIZE + get_local_id(0)] = ptrLevel_int[d_inner*STORAGEPAD + get_global_id(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  ptrLevel += get_global_id(1)*LSIZE + get_local_id(0);
  ptrIndex += get_global_id(1)*LSIZE + get_local_id(0);
  ptrLevel_int += get_global_id(1)*LSIZE + get_local_id(0);
  __local REAL * l2dotTempPtr = l2dotTemp + get_local_id(0);
  __local REAL * gradTempPtr = gradTemp + get_local_id(0);
  REAL res = 0.0;
  for (unsigned k = 0; k < LSIZE; k++) {
    for(size_t d_inner = 0; d_inner < DIMS; d_inner++) {
      REAL i_level =         ptrLevel[d_inner*STORAGEPAD];
      REAL i_index =         ptrIndex[d_inner*STORAGEPAD];
      REAL i_level_int = ptrLevel_int[d_inner*STORAGEPAD];
      REAL j_levelreg = j_level[d_inner*LSIZE + k];
      REAL j_indexreg = j_index[d_inner*LSIZE + k];
      l2dotTempPtr[d_inner*LSIZE] = (l2dot(i_level,i_index,i_level_int, j_levelreg,j_indexreg,j_level_int[d_inner*LSIZE + k],ptrLcl_q[(d_inner+1)*2-2]));
      gradTempPtr[d_inner*LSIZE] = (gradient(i_level,i_index, j_levelreg,j_indexreg,ptrLcl_q[(d_inner+1)*2-1]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned d_outer = 0; d_outer < DIMS; d_outer++)
      {
	REAL element = alphaTemp[k];
	for(unsigned d_inner = 0; d_inner < DIMS; d_inner++)
	  {	   
	    element *= (l2dotTempPtr[d_inner*LSIZE]*(d_outer != d_inner)) + gradTempPtr[d_inner*LSIZE]*(d_outer == d_inner);
	  }
	res += element;
      }
    /* thrresult[local_id(0)] = PartialRes*alpha[i]* jj <ii; */
    /* if local_id(0) == 0 */
    /* 	pres=	 sum(thrsubresult) */
    /* localsubres[k] = pres; */
    
    
  }
  /* ptrSubResult[jj] = localsubres[local_id(0)] */
  ptrParResult[get_group_id(0)*STORAGEPAD + get_global_id(1)*LSIZE + get_local_id(0)] = res;

}

