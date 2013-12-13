function forces = mycomputeForces(n, positions,masses);


forces = zeros(size(positions));
for h = 1:n
xA = positions(:,h);
m1 = masses(h);
f_i = [0;0];
for i = 1:n
    xB = positions(:,i);
    r_ij_vec = xB-xA;
    d = r_ij_vec;
    d2 = d'*d ;
    deno = sqrt(d2 * d2 * d2) + (i == h);
    f_i = (f_i + ((m1*masses(i)/deno) * r_ij_vec) * (i ~= h));
end
forces(:,h) = f_i;
end




end