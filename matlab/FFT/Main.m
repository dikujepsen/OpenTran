
N = 22;

NFFT = 32;

X = zeros(1,NFFT);

for i = 1:N
    X(i) = i-1;
     
end

res = fft(X);
res'