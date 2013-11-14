void
matmul(float* A, float* B, float* C, unsigned hA, unsigned wA, unsigned wB)
{

  for (unsigned i = 0; i < hA; i++)
    {
      for (unsigned j = 0; j < wB; j++)
        {
          float sum = 0;
          for (unsigned k = 0; k < wA; k++)
            {
              sum += A[i * wA + k] * B[j + k * wB];
            }
          C[wB * i + j] = sum;
        }
    }

  for (unsigned t = 0; t < wB; t++)
    {
      float sum = 0;
      for (unsigned h = 0; h < wA; h++)
        {
          sum += A[wA * h + h+1+23+1] * B[h + wB * t];
          sum += A[h + wA + 1] * B[h + wB * t + t];
        }
      C[i + wB * i] = sum;
    }
  a = i * (wA + 1);
}
