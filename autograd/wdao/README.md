# wDAO 对照

```bash
CUDA_VISIBLE_DEVICES=0 python autograd/demo_wdao.py
```

使用主图 3 的同一 Sample 2 输入：10 次阶段波前估计，再固定波前进行 2300 次物体更新。
输出到 `outputs/demo_wdao/`，使用 24 GiB GPU。

## 适配数据

光学参数、体网格和噪声配置共用 `demo_wavefront.py`。
`--data` 指定观测，`--output` 指定输出，`--intensity-scale` 设置强度换算。
自定义输入默认输出光子单位，评价参考可在文件顶部设置。
当前标定使用 81 个视角、每微透镜 3×3 空间采样和 21 个 OSA 模式。

## 算法

阶段估计采用 ISRA 物体更新、NCC 位移估计、物理梯度换算与相位积分。
首轮遍历视角 5 次，后续每轮 3 次；最终波前用于固定波前物体重建。
物体重建共用 Joint 的初始化、噪声目标、正则和 Adam 参数。

| 程序 | 用途 |
| --- | --- |
| `estimate.py` | 阶段估计循环 |
| `optical_operator.py` | 缓存光学前向/伴随和 ISRA 更新 |
| `rush_registration.py` | NCC 位移估计与离焦去除 |
| `phase_integration.py` | 位移标定、梯度积分与 OSA 拟合 |
| `reconstruct.py` | 固定波前物体重建 |

这是基于 RUSH3D 的 wDAO adaptation，采用本模型的物理标定、积分和模式拟合。
参考：[阶段循环](https://github.com/yuanlong-o/RUSH3D/blob/13923b3fbf3c0afec950e7645b4e604000fae6d9/v0.1/reconstruction/Loop_Phase_Reconstruction)、
[重建模块](https://github.com/yuanlong-o/RUSH3D/blob/4e78e6cbaba3245ad82516526fed329e2a6f3d49/va.31/reconstruction_module/recon_module.m)。
