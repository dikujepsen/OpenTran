function [positions_new, velocities_new, forces_new] = ...
    myvelocityStoermerVerlet(n, positions, velocities,...
                             forcesOld, dt, masses);


positions_new = zeros(size(positions));
velocities_new = zeros(size(velocities));


for i = 1:n
F_1 = forcesOld(:,i);
m_1 = masses(i);
a = F_1/m_1;

v_1 = velocities(:,i);
v_1_hnew = v_1 + 0.5*dt*a;

r_1 = positions(:,i);
r_1_new = r_1 + dt*v_1_hnew;
positions_new(:,i) = r_1_new;
velocities_new(:,i) = v_1_hnew;

end
forces_new = mycomputeForces(n,positions_new,masses);

for i = 1:n
v_1 = velocities_new(:,i);
F_1 = forces_new(:,i);
m_1 = masses(i);

a = F_1/m_1;
v_1_new = v_1 + 0.5*dt*a;

velocities_new(:,i) = v_1_new;
end

end