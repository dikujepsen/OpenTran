clear;

N_x = [7 15 31 63];
N_y = [7 15 31 63];

%% System solved without storing the matrix (using Gauss-Seidel instead)
t_gs = zeros(length(N_x),1);
storage_gs = zeros(length(N_x),1);
for k=1 % :length(N_x)
    B = createB(N_x(k),N_y(k));
    
    t_start = tic;
    T_GS = Jacobi2(B,N_x(k),N_y(k));
    t_gs(k) = toc(t_start);
    T_GS 
    storage_gs(k) = numel(B) + numel(T_GS);
    [X,Y] = meshgrid(0:1/(N_x(k)+1):1,0:1/(N_x(k)+1):1);
    figure(k)

    subplot(1,2,1)
    surf(X,Y,T_GS)
    subplot(1,2,2)
    contour(X,Y,T_GS)
end

%%
t_gs = zeros(length(N_x),1);
storage_gs = zeros(length(N_x),1);
for k=2 % :length(N_x)
    B = createB2(N_x(k),N_y(k));
    B
    t_start = tic;
    T_GS = Jacobi3(B,N_x(k),N_y(k));
    t_gs(k) = toc(t_start);
    storage_gs(k) = numel(B) + numel(T_GS);
    [X,Y] = meshgrid(0:1/(N_x(k)+1):1,0:1/(N_x(k)+1):1);
    figure(k)
    T_GS
    subplot(1,2,1)
    surf(X,Y,T_GS)
    subplot(1,2,2)
    contour(X,Y,T_GS)
end

