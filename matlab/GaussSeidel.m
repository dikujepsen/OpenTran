function [ X ] = GaussSeidel(b, N_x, N_y)

X = zeros(N_x+2,N_y+2);
h_xsq = 1/((N_x + 1)^2);
h_ysq = 1/((N_y + 1)^2);
N = N_x * N_y;
grid = zeros(N);
AX = zeros(N_x+2,N_y+2);
m = 30;
err = ones(100000,1).*10;
% err = 10;
iter = 2;

while (iter < 1000)
    for i=2:(N_x+1)
        for j=2:(N_y+1)
            X(i,j) = (b(i*N_x-2*N_x+j-1) - (X(i-1,j) + X(i+1,j))/h_xsq -...
                (X(i,j-1) + X(i,j+1))/h_ysq) / -(2/h_xsq + 2/h_ysq);
        end
    end
%     for i=2:(N_x+1)
%         for j=2:(N_y+1)
%             AX(i,j) = ((X(i-1,j) + X(i+1,j))/h_xsq + ...
%                 (X(i,j-1) + X(i,j+1))/h_ysq   ...
%                 -2*X(i,j)/h_xsq-2*X(i,j)/h_ysq);
%             
%         end
%     end
% 
%     grid = reshape(AX(2:(N_x+1),2:(N_y+1)),1,N);
%     err(iter) = sqrt((sum((b - grid).^2)/N));
%     err = sqrt((sum((b - grid).^2)/N));
    iter = iter + 1;
end

% % AX
% iter
% grid = reshape(AX(2:(N_x+1),2:(N_y+1)),N,1)

end

