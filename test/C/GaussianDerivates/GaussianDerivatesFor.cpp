for (int i=0; i<Lq; i++) { 
  for (int j=0; j<Lp; j++) {
    float xj[3];
    float xi[3];
    for (int k = 0; k < dim; k++) {
      xj[k] = p_a_i_x[j*p_a_i_rows + k];
    }

    for (int k = 0; k < dim; k++) {
      xi[k] = q_a_i_x[i*q_a_i_rows + k];
    }


      
    // Vector3<scalar> xi(&q_a_i.x[q_a_i.rows*i]);
    float ximxj[3];
    for (int k = 0; k < dim; k++) {
      ximxj[k] = xi[k]-xj[k];
    }
      
    float r = sqrt(scales2_x);

    float ks = gamma(ximxj, scales2_x, scaleweight2_x);
    K__ij_x[i+K__ij_rows*j] = ks;

    int da[3];
    int db[3];
    int dc[3];
    // nargout 1
    for (int b=0; b<dim; b++) {
      // da.set(1,b);
      for (int k = 0; k < dim; k++) {
	da[k] = 1;
      }

      D1Ks__ijb_x[i +
		  D1Ks__ijb_dimsI[0] * j +
		  D1Ks__ijb_dimsI[1] * b] =
	DaKs(da,ximxj,r,ks);


      // nargout 2
      for (int g=0; g<dim; g++) {
	// Vector3<int> db = da;
	for (int k = 0; k < dim; k++) {
	  db[k] = da[k];
	}
	// db.set(db[g]+1,g) ?
	// db[g] = db[g] + 1;
	for (int k = 0; k < dim; k++) {
	  db[k] = db[k] + 1;
	}
	  
	D2Ks__ijbg_x[i +
		     D2Ks__ijbg_dimsI[0] * j +
		     D2Ks__ijbg_dimsI[1] * b +
		     D2Ks__ijbg_dimsI[2] * g ] =
	  DaKs(db,ximxj,r,ks);


	for (int d=0; d<dim; d++) {
	  // Vector3<int> dc = db; dc.set(dc[d]+1,d);
	  for (int k = 0; k < dim; k++) {
	    dc[k] = db[k];
	  }
	  for (int k = 0; k < dim; k++) {
	    dc[d] = dc[d] + 1;
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



