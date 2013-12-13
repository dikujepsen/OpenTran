function [ b ] = createB( N_x,N_y )

h_x = 1/(N_x+1);
h_y = 1/(N_y+1);

N = N_x*N_y;
b = zeros(1,N);

for i = 1:N_y
    for j = 1:N_x
    b((i-1)*N_x + j) = -2*(pi^2)*sin(pi*(j)*h_x)*sin(pi*(i)*h_y);
    end
end

end

