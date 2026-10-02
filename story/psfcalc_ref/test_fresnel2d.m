% 参数设置
lambda = 632.8e-9; % 波长 (m)
k = 2*pi/lambda; % 波数
z = 0.1; % 传播距离 (m)
L = 0.01; % 光场尺寸 (m)
N = 1024; % 采样点数
dx = L/N; % 空间采样间隔 (m)
x = (-N/2:N/2-1)*dx; % 空间坐标
[X, Y] = meshgrid(x, x); % 网格坐标

% 输入光场 (假设是高斯光束)
w0 = 0.001; % 光束 waist (m)
E0 = exp(-(X.^2 + Y.^2)/w0^2); % 高斯光束

% 频域计算
fx = (-N/2:N/2-1)/(N*dx); % 频率坐标
[FX, FY] = meshgrid(fx, fx); % 频率网格
H = exp(1i*k*z)*exp(-1i*pi*lambda*z*(FX.^2 + FY.^2)); % 菲涅尔传递函数

E0_fft = fft2(fftshift(E0)); % 输入光场的傅里叶变换
E_fft = E0_fft .* fftshift(H); % 应用传递函数
E = ifftshift(ifft2(E_fft)); % 逆傅里叶变换得到衍射场

% 显示结果
figure;
subplot(1, 2, 1);
imagesc(abs(E0));
title('输入光场');
axis square;

subplot(1, 2, 2);
imagesc(abs(E));
title('菲涅尔衍射 (频域)');
axis square;

% 参数设置
lambda = 632.8e-9; % 波长 (m)
k = 2*pi/lambda; % 波数
z = 0.1; % 传播距离 (m)
L = 0.01; % 光场尺寸 (m)
N = 1024; % 采样点数
dx = L/N; % 空间采样间隔 (m)
x = (-N/2:N/2-1)*dx; % 空间坐标
[X, Y] = meshgrid(x, x); % 网格坐标

% 输入光场 (假设是高斯光束)
w0 = 0.001; % 光束 waist (m)
E0 = exp(-(X.^2 + Y.^2)/w0^2); % 高斯光束

% 空间域计算
h = exp(1i*k*z)/(1i*lambda*z) .* exp(1i*k/(2*z)*(X.^2 + Y.^2)); % 菲涅尔冲激响应
E = dx^2 * conv2(E0, h, 'same'); % 卷积计算

% 显示结果
figure;
subplot(1, 2, 1);
imagesc(abs(E0));
title('输入光场');
axis square;

subplot(1, 2, 2);
imagesc(abs(E));
title('菲涅尔衍射 (空间域)');
axis square;