function [ b ] = createB2( N_x,N_y )

h_x = 1/(N_x+1);
h_y = 1/(N_y+1);
h_xsq = 1/((N_x + 1)^2);

b = zeros(N_x+1,N_y+1);

for i = 2:(N_y+1)
    for j = 2:(N_x+1)
    b(i,j) = -2*(pi^2)*sin(pi*(j-1)*h_x)*sin(pi*(i-1)*h_y)*h_xsq;
    end
end

end

