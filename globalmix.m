% 2D Global Inversion for NMR (e.g., T1-T2 or D-T2)
% Inputs:
%   K1, K2 - Kernels for dimension 1 (e.g., T1) and 2 (e.g., T2)
%   S      - Measured NMR signal (vector or matrix)
%   alpha  - Regularization parameter (controls smoothness)
%   T1_range, T2_range - Logarithmic time ranges (e.g., logspace(-3, 1, 50))
% Outputs:
%   F      - 2D distribution (e.g., T1-T2 map)
%   T1, T2 - Axis values for plotting

%% 
% Simulate synthetic T1-T2 data
T1_range = logspace(-3, 1, 50); % T1 bins (10^-3 to 10^1 s)
T2_range = logspace(-3, 0, 50); % T2 bins (10^-3 to 10^0 s)
t1 = linspace(0, 2, 100);       % IR acquisition times
t2 = linspace(0, 1, 100);       % CPMG acquisition times

% Kernels
K1 = exp(-t1' * (1./T1_range)); % T1 kernel (inversion recovery)
K2 = exp(-t2' * (1./T2_range)); % T2 kernel (CPMG)

% Simulate a T1-T2 distribution (oil + water)
F_true = zeros(50, 50);
F_true(10:15, 20:25) = 1; % Water (short T1, medium T2)
F_true(30:35, 35:40) = 2; % Oil (longer T1, longer T2)

% Generate synthetic signal
S = K1 * F_true * K2' + 0.01 * randn(100, 100); % Add noise

% Run global inversion
alpha = 0.1; % Regularization parameter
% [F_inverted, T1_axis, T2_axis] = global_2D_inversion(K1, K2, S(:), alpha, T1_range, T2_range);

%%
%% Step 1: Combine kernels into a single global kernel
% K1 = exp(-t/T1) for inversion recovery
% K2 = exp(-t/T2) for CPMG
[Nt1, NT1] = size(K1); % Nt1 = # of T1 decay points, NT1 = # of T1 bins
[Nt2, NT2] = size(K2); % Nt2 = # of T2 decay points, NT2 = # of T2 bins

% Construct the 2D kernel (Kronecker product)
K_global = kron(K2, K1); % Combines K1 and K2 into a single operator

%% Step 2: Vectorize the signal and regularization
S_vec = S(:); % Convert signal to column vector

%% Step 3: Regularization (Tikhonov smoothing in 2D)
% 1D Laplacian operators for T1 and T2
L1 = spdiags([-ones(NT1,1) 2*ones(NT1,1) -ones(NT1,1)], [-1 0 1], NT1, NT1);
L2 = spdiags([-ones(NT2,1) 2*ones(NT2,1) -ones(NT2,1)], [-1 0 1], NT2, NT2);

% 2D Laplacian (Kronecker sum)
L_2D = kron(speye(NT2), L1) + kron(L2, speye(NT1));

%% Step 4: Solve the global inverse problem
% System matrix: [K_global; alpha * L_2D]
A = [K_global; alpha * L_2D];
b = [S_vec; zeros(size(L_2D, 1), 1)];

% Solve using non-negative least squares (lsqnonneg)
F_vec = lsqnonneg(A, b);

%% Step 5: Reshape into 2D distribution
F = reshape(F_vec, NT1, NT2);

%% Step 6: Generate axis for plotting
T1 = log10(T1_range);
T2 = log10(T2_range);

%% Plot results
figure;
contourf(T2, T1, F, 20, 'LineColor', 'none');
colormap('jet');
colorbar;
xlabel('log_{10}(T2) [s]');
ylabel('log_{10}(T1) [s]');
title('2D Global Inversion (T1-T2 Map)');
