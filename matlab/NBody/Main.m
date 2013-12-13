clear;
[masses, positions, velocities] = myinitialise();

rng(2553);

n = 8;

masses = rand(n) * 500;
positions = rand(2,n)*100;
% positions = (mod(randi(100,2,n),100) / 100.0) * 100.0;
velocities = rand(2,n);

% C Example n = 8:
masses = [405 390 235 365 455 300 375 310];
positions = [25 51 37 97 91 50 83 63 ; 88 53 90 49 72 8 76 70 ];
velocities = [0.99 0.06 0.3 0.66 0.89 0.6 0.58 0.7 ; 0.9 0.06 0.96 0.82 0.18 0.71 0.96 0.95];

forces = mycomputeForces(n, positions,masses);

dt = 0.015;
masses
positions
velocities
forces
N = 1;
for i = 1:N
[positions, velocities, forces] = myvelocityStoermerVerlet(n, positions, velocities, forces, dt, masses);
end
positions
velocities
forces

% 68.1596   54.4214  -24.2704   96.4048  -19.0230   52.3993  214.6312   66.3753
%    93.8045   57.3262   82.8663   56.3066  185.0335   12.5061  -64.2480   71.1904
   