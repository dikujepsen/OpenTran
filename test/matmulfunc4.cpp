for (unsigned i = 0; i < hA; i++)
    {
      int b;
      int a = 4;
      for (unsigned j = 0; j < wB; j++)
        {
          float sum = 0;
          for (unsigned k = 0; k < wA; k++)
            {
              sum += A[i * wA + k] * B[j + k * wB] + D;
            }
          C[wB * i + j] = sum * E;
        }
    }
