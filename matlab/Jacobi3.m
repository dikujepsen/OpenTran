function [ X ] = Jacobi3(b, N_x, N_y)

X = zeros(N_x+2,N_y+2);
X2 = zeros(N_x+2,N_y+2);

N = N_x * N_y;
iter = 2;

while (iter < 1000)
    for i=2:(N_x+1)
        for j=2:(N_y+1)
            X2(i,j) = -0.25 * (b(i,j) - (X(i-1,j) + X(i+1,j)) -...
                (X(i,j-1) + X(i,j+1)));
        end
    end
    X = X2 ;
    iter = iter + 1;
end

iter
% % AX
% iter
% grid = reshape(AX(2:(N_x+1),2:(N_y+1)),N,1)

end

