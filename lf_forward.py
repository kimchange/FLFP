import torch
import torch.nn.functional as F

def conv_and_downsample_fast(V, PSF):
    """
    输入:
        V: [d, 15h, 15w] 的三维体
        PSF: [d, kh, kw] 的三维卷积核 (要求 kh, kw 为奇数)
    输出:
        out: [1, 3h, 3w] 的结果
    """
    # 1. 调整维度以适配 F.conv2d (N, C, H, W) 和 (Out_C, In_C, kH, kW)
    V_input = V.unsqueeze(0)    # [1, d, 15h, 15w]
    P_weight = PSF.unsqueeze(0) # [1, d, kh, kw]
    
    # 获取 kernel size 用于 padding，保证中心对齐
    kh, kw = PSF.shape[1], PSF.shape[2]
    pad_h, pad_w = kh // 2, kw // 2
    
    # 2. 核心加速：一步完成 (多通道卷积求和 + 步长为 5 的直接定点采样)
    # 计算量直接减小到原来的 1/25
    out = F.conv2d(V_input, P_weight, bias=None, stride=5, padding=(pad_h, pad_w))
    
    return out # Shape: [1, 1, 3h, 3w]

import torch

def conv_and_downsample_fft(V, PSF, precomputed_PSF_f=None):
    """
    高效的 FFT 多通道卷积与降采样实现。
    
    输入:
        V: [d, 15h, 15w] 的三维体 (实数张量)
        PSF: [d, kh, kw] 的三维核 (实数张量)
        precomputed_PSF_f: 如果 PSF 是固定的，强烈建议在外部预计算传入，能省下近一半算力。
    输出:
        out: [1, 3h, 3w] 的最终结果
    """
    d, H, W = V.shape
    kh, kw = PSF.shape[1], PSF.shape[2]
    
    # =======================================================
    # 1. 前向实数傅里叶变换 (RFFT)
    # RFFT 只需要计算一半的频谱，速度提升近 1 倍
    # =======================================================
    V_f = torch.fft.rfft2(V, s=(H, W))  # Shape: [d, H, W//2 + 1]
    
    # 如果没有预计算 PSF 的频谱，则现场计算
    if precomputed_PSF_f is None:
        # 1.1 创建全尺寸的零矩阵，将 PSF 填入中心以保证空间相位对齐 (同 F.conv2d 的 padding='same')
        psf_padded = torch.zeros((d, H, W), dtype=V.dtype, device=V.device)
        start_h = (H - kh) // 2
        start_w = (W - kw) // 2
        psf_padded[:, start_h:start_h+kh, start_w:start_w+kw] = PSF
        
        # 1.2 极其关键：将 PSF 的几何中心通过 ifftshift 移到图像原点 [0,0]，消除寄生相移
        psf_shifted = torch.fft.ifftshift(psf_padded, dim=(-2, -1))
        
        # 1.3 计算 PSF 频谱
        PSF_f = torch.fft.rfft2(psf_shifted, s=(H, W))
    else:
        PSF_f = precomputed_PSF_f

    # =======================================================
    # 2. 频域点乘 与 极速降维求和
    # 在频域提前进行 dim=0 的求和，将极其昂贵的 d 次 IFFT 降为仅仅 1 次！
    # =======================================================
    # V_f * PSF_f (点乘), 然后沿 d 维度相加
    out_f = torch.sum(V_f * PSF_f, dim=0, keepdim=True) # Shape: [1, H, W//2 + 1]
    
    # =======================================================
    # 3. 反向实数傅里叶变换 (IRFFT)
    # =======================================================
    out_full = torch.fft.irfft2(out_f, s=(H, W)) # 恢复出全分辨率图像 [1, 15h, 15w]
    
    # =======================================================
    # 4. 空间域间隔降采样 (Stride = 5)
    # 取出每 5 个像素的中心点
    # =======================================================
    # PyTorch 切片操作是零拷贝的 (Zero-copy)，极快
    out_downsampled = out_full[:, ::5, ::5]  # Shape: [1, 3h, 3w]
    
    return out_downsampled

# ---------------------------------------------------------
# 强烈建议的优化用法：预计算静态核的频谱
# 如果你的 PSF 在很多次迭代/不同批次的数据中是不变的：
# 
# psf_padded = torch.zeros((d, 15h, 15w), device=device)
# psf_padded[:, start_h:start_h+kh, start_w:start_w+kw] = PSF
# psf_shifted = torch.fft.ifftshift(psf_padded, dim=(-2, -1))
# precomputed_PSF_f = torch.fft.rfft2(psf_shifted)
#
# 然后在主循环中只调用：
# out = conv_and_downsample_fft(V, PSF=None, precomputed_PSF_f=precomputed_PSF_f)
# ---------------------------------------------------------

import torch
import torch.nn.functional as F

def apply_decoupled_psf(V, psf_blur, shift, stride=5, method='spatial'):
    """
    将解耦的 PSF (Blur + Shift) 高效作用于三维体积 V，并直接输出降采样结果。
    
    输入:
        V: [B, D, H, W] - 批次的三维体积
        psf_blur: [A, D, kh, kw] - 解耦出的中心化微观模糊核
        shift: [A, D, 2] - 解耦出的宏观整数平移量 (y, x)
        stride: int - 降采样步长 (默认 5)
        method: 'spatial' 或 'fft'
        
    输出:
        out: [B, A, H_out, W_out] - 每个视角的最终低分辨率图像
    """
    B, D, H, W = V.shape
    A, D_p, kh, kw = psf_blur.shape
    assert D == D_p, "V 和 PSF 的深度 D 必须一致！"
    
    out_views = []
    
    if method == 'spatial':
        # ==========================================================
        # 模式 A: 空间域极致跨步法 (适用于 kh, kw <= 63)
        # ==========================================================
        # 预计算 V 的全图采样网格
        gy = torch.arange(H, device=V.device).view(1, 1, H, 1)
        gx = torch.arange(W, device=V.device).view(1, 1, 1, W)
        b_idx = torch.arange(B, device=V.device).view(B, 1, 1, 1)
        d_idx = torch.arange(D, device=V.device).view(1, D, 1, 1)
        
        # 遍历视角 A (A 通常较小，如 9 或 25，循环极快且彻底杜绝 OOM)
        for a in range(A):
            # 1. 提取当前视角的 Shift [D, 2]
            dy = shift[a, :, 0].view(1, D, 1, 1)
            dx = shift[a, :, 1].view(1, D, 1, 1)
            
            # 2. 坐标网格平移反演
            # 数学上：卷积核向右平移 dx，等价于输入图像向左偏移读取 (x - dx)
            idx_y = gy - dy
            idx_x = gx - dx
            
            # 3. 边界遮罩 (Zero-padding 逻辑)
            valid_mask = (idx_y >= 0) & (idx_y < H) & (idx_x >= 0) & (idx_x < W)
            idx_y = torch.clamp(idx_y, 0, H - 1)
            idx_x = torch.clamp(idx_x, 0, W - 1)
            
            # 4. 零拷贝快速重组 V (Zero-copy Gathering)
            V_shifted = V[b_idx, d_idx, idx_y, idx_x]
            V_shifted = V_shifted * valid_mask # 越界像素置零
            
            # 5. 多通道跨步卷积 (D个通道合并为 1，直接抽样)
            weight = psf_blur[a].unsqueeze(0)  # Shape: (1, D, kh, kw)
            out_a = F.conv2d(V_shifted, weight, bias=None, stride=stride, padding=(kh//2, kw//2))
            
            out_views.append(out_a) # out_a: (B, 1, H_out, W_out)

    elif method == 'fft':
        # ==========================================================
        # 模式 B: 频域折叠法 (适用于 kh, kw > 64 的巨型 PSF)
        # ==========================================================
        # 对 V 提前进行全局 FFT (省去 A 次重复计算)
        V_f = torch.fft.rfft2(V, s=(H, W)) # Shape: [B, D, H, W//2 + 1]
        
        # PSF 移位网格
        gy = torch.arange(H, device=V.device).view(1, H, 1)
        gx = torch.arange(W, device=V.device).view(1, 1, W)
        d_idx = torch.arange(D, device=V.device).view(D, 1, 1)
        
        for a in range(A):
            # 1. 补零填充 (Padding)
            psf_padded = torch.zeros((D, H, W), dtype=V.dtype, device=V.device)
            start_y, start_x = (H - kh) // 2, (W - kw) // 2
            psf_padded[:, start_y:start_y+kh, start_x:start_x+kw] = psf_blur[a]
            
            # 2. 空间循环移位 (FFT 的位移定理严格对应 modulo 循环)
            dy = shift[a, :, 0].view(D, 1, 1)
            dx = shift[a, :, 1].view(D, 1, 1)
            idx_y = (gy - dy) % H
            idx_x = (gx - dx) % W
            psf_shifted = psf_padded[d_idx, idx_y, idx_x]
            
            # 3. 中心化与频域转换
            psf_centered = torch.fft.ifftshift(psf_shifted, dim=(-2, -1))
            PSF_f = torch.fft.rfft2(psf_centered, s=(H, W)).unsqueeze(0) # [1, D, H, W//2 + 1]
            
            # 4. 频域点乘并提前求和 (消灭 D 维度，仅需 1 次 IFFT)
            out_f_a = torch.sum(V_f * PSF_f, dim=1, keepdim=True) # [B, 1, H, W//2 + 1]
            
            # 5. 反变换与直接降采样
            out_full_a = torch.fft.irfft2(out_f_a, s=(H, W))
            out_a = out_full_a[:, :, ::stride, ::stride]
            
            out_views.append(out_a)
    else:
        raise ValueError("Method must be 'spatial' or 'fft'")

    # 将 A 个视角的图像拼接返回
    return torch.cat(out_views, dim=1) # Shape: [B, A, H_out, W_out]


import torch
import torch.nn.functional as F

def apply_decoupled_psf_v3(V, psf_blur, shift, stride=5, psf_downsample=1, method='spatial'):
    """
    大一统解耦 PSF 融合引擎。
    支持 PSF 尺度缩放 (Dilated)，支持多种降采样物理等效逻辑。
    
    输入:
        V: [B, D, H, W] - 批次的三维体积 (原图高分辨率尺度)
        psf_blur: [A, D, kh_d, kw_d] - 解耦且可能已经降采样的微观模糊核
        shift: [A, D, 2] - 宏观整数平移量 (y, x)
        stride: int - 降采样倍率 (决定最终输出分辨率)
        psf_downsample: int - PSF 预先降采样的倍率
        method: 'spatial' (跨步空洞卷积), 'fft' (频域加速+空间跨步), 'fft_crop' (频域中心裁剪低通滤波)
    """
    B, D, H, W = V.shape
    A, D_p, kh_d, kw_d = psf_blur.shape
    assert D == D_p, "V 和 PSF 的深度 D 必须一致！"
    
    # 无论哪种模式，PSF 映射回原图物理网格的等效尺寸不变
    phys_kh = (kh_d - 1) * psf_downsample + 1
    phys_kw = (kw_d - 1) * psf_downsample + 1
    
    out_views = []
    
    if method == 'spatial':
        # ==========================================================
        # 模式 A: 空间域空洞跨步法 (保留奈奎斯特混叠，还原传感器物理)
        # ==========================================================
        gy = torch.arange(H, device=V.device).view(1, 1, H, 1)
        gx = torch.arange(W, device=V.device).view(1, 1, 1, W)
        b_idx = torch.arange(B, device=V.device).view(B, 1, 1, 1)
        d_idx = torch.arange(D, device=V.device).view(1, D, 1, 1)
        
        pad_h = psf_downsample * (kh_d - 1) // 2
        pad_w = psf_downsample * (kw_d - 1) // 2
        
        for a in range(A):
            dy = shift[a, :, 0].view(1, D, 1, 1)
            dx = shift[a, :, 1].view(1, D, 1, 1)
            
            idx_y = torch.clamp(gy - dy, 0, H - 1)
            idx_x = torch.clamp(gx - dx, 0, W - 1)
            valid_mask = ((gy - dy) >= 0) & ((gy - dy) < H) & ((gx - dx) >= 0) & ((gx - dx) < W)
            
            V_shifted = V[b_idx, d_idx, idx_y, idx_x] * valid_mask
            weight = psf_blur[a].unsqueeze(0)
            
            out_a = F.conv2d(V_shifted, weight, bias=None, stride=stride, padding=(pad_h, pad_w), dilation=psf_downsample)
            out_views.append(out_a)

    elif method == 'fft':
        # ==========================================================
        # 模式 B: 频域折叠法 (巨型 PSF 的提速方案，等效于模式 A)
        # ==========================================================
        V_f = torch.fft.rfft2(V, s=(H, W))
        
        gy = torch.arange(H, device=V.device).view(1, H, 1)
        gx = torch.arange(W, device=V.device).view(1, 1, W)
        d_idx = torch.arange(D, device=V.device).view(D, 1, 1)
        
        for a in range(A):
            psf_padded = torch.zeros((D, H, W), dtype=V.dtype, device=V.device)
            start_y, start_x = (H - phys_kh) // 2, (W - phys_kw) // 2
            
            psf_padded[:, start_y:start_y+phys_kh:psf_downsample, start_x:start_x+phys_kw:psf_downsample] = psf_blur[a]
            
            dy = shift[a, :, 0].view(D, 1, 1)
            dx = shift[a, :, 1].view(D, 1, 1)
            psf_shifted = psf_padded[d_idx, (gy - dy) % H, (gx - dx) % W]
            
            PSF_f = torch.fft.rfft2(torch.fft.ifftshift(psf_shifted, dim=(-2, -1)), s=(H, W)).unsqueeze(0)
            
            out_f_a = torch.sum(V_f * PSF_f, dim=1, keepdim=True)
            out_full_a = torch.fft.irfft2(out_f_a, s=(H, W))
            
            out_a = out_full_a[:, :, ::stride, ::stride]
            out_views.append(out_a)

    elif method == 'fft_crop':
        # ==========================================================
        # 模式 C: 频域裁剪法 (完美理想低通滤波，纯净无混叠伪影)
        # ==========================================================
        H_out = H // stride
        W_out = W // stride
        
        # 因为后续要做对称中心裁剪，这里使用全尺寸复数 fft2 会更稳妥，避免 rfft2 切片对称性的麻烦
        V_f = torch.fft.fft2(V, s=(H, W))
        
        gy = torch.arange(H, device=V.device).view(1, H, 1)
        gx = torch.arange(W, device=V.device).view(1, 1, W)
        d_idx = torch.arange(D, device=V.device).view(D, 1, 1)
        
        for a in range(A):
            # 1. 构建系数 PSF 并在空间移位 (与模式 B 相同)
            psf_padded = torch.zeros((D, H, W), dtype=V.dtype, device=V.device)
            start_y, start_x = (H - phys_kh) // 2, (W - phys_kw) // 2
            
            psf_padded[:, start_y:start_y+phys_kh:psf_downsample, start_x:start_x+phys_kw:psf_downsample] = psf_blur[a]
            
            dy = shift[a, :, 0].view(D, 1, 1)
            dx = shift[a, :, 1].view(D, 1, 1)
            psf_shifted = psf_padded[d_idx, (gy - dy) % H, (gx - dx) % W]
            
            # 2. 转换到频域并乘加
            PSF_f = torch.fft.fft2(torch.fft.ifftshift(psf_shifted, dim=(-2, -1)), s=(H, W)).unsqueeze(0)
            out_f_a = torch.sum(V_f * PSF_f, dim=1, keepdim=True) # [B, 1, H, W]
            
            # 3. 核心魔法：频域移位并切除所有外围高频
            out_f_shifted = torch.fft.fftshift(out_f_a, dim=(-2, -1))
            
            crop_start_y = (H - H_out) // 2
            crop_start_x = (W - W_out) // 2
            out_f_cropped = out_f_shifted[:, :, crop_start_y:crop_start_y+H_out, crop_start_x:crop_start_x+W_out]
            
            # 4. 移回原点并执行【微型极速 IFFT】
            out_f_cropped_centered = torch.fft.ifftshift(out_f_cropped, dim=(-2, -1))
            
            # 能量补偿: 频域面积缩小了 stride^2 倍，IFFT 结果需要按比例缩放以保持光强守恒
            out_a = torch.fft.ifft2(out_f_cropped_centered).real / (stride ** 2)
            
            out_views.append(out_a)

    else:
        raise ValueError("Method must be 'spatial', 'fft', or 'fft_crop'")

    return torch.cat(out_views, dim=1) # Shape: [B, A, H_out, W_out]