# FLFP

快速、可微分的光场 PSF 模型，用于三维物体与波前联合重建。
本仓库对应论文 *A Fast, Differentiable Representation of Light-Field Point-Spread Functions*

## 快速开始

使用 Python 3.10 和支持 CUDA 的 PyTorch，在本目录运行：

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python autograd/demo_wavefront.py
```

**从 [demo_wavefront.py](autograd/demo_wavefront.py) 开始。** 它将配置、前向模型、损失和优化集中在一个文件中，
复现主图 3(b) 的 Joint 重建：Sample 2、峰值 100 光子、81 个视角、2300 步。
体数据、波前系数、LF 重投影和相位数组保存到 `outputs/demo_wavefront_sample02/`。

## 适配自己的任务

修改 `demo_wavefront.py` 顶部的配置：

| 配置                                       | 设置内容                                                   |
| ------------------------------------------ | ---------------------------------------------------------- |
| `DATA`、`PHYSICAL_VIEWS`               | 原始 LF`[view,y,x]` 和视角顺序；保留传感器偏置与原始强度 |
| `VOLUME_SHAPE`、`OPTICS`               | 重建网格与光学参数，光学长度单位为 m                       |
| `PSF_SIZE`、`SAMPLE_STRIDE`、`MODES` | PSF 支持、采样间隔和波前模式数                             |
| `BIAS`、`READ_VARIANCE`                | 传感器偏置和读出噪声方差                                   |
| `STEPS`、`LR_*`、`REGULARIZATION`    | 优化步数、学习率和正则强度                                 |
| `TRUE_WF`、`CLEAN_LF`                  | 可选评价参考；自己的数据可设为`None`                     |
| `INTENSITY_SCALE`                        | 输出单位换算系数；设为`1` 时保留光子单位                 |

自定义观测模型或先验，分别修改 `data_loss()` 和 `spatial_penalty()`。体数据与 LF 默认保存为 float32。

## 文件结构

```text
github/
├── autograd/                              # 光学模型和重建
│   ├── demo_wavefront.py                   # 单文件 Joint demo，修改任务的入口
│   ├── demo_wdao.py                        # 同一观测的 wDAO 对照
│   ├── batch_points_poisson_gaussian.py    # 50 例 Joint 批量实验
│   ├── lfpsf_torch_shift_blur_batched.py    # 默认使用的可微 PSF 模型
│   ├── lfpsf_torch_batch5.py               # 原光学模型，供参考
│   ├── wdao/                              # 位移估计、相位积分和固定波前重建
│   └── psfsim_timetest/                    # PSF 实现与计时
│       └── benchmark_psf.py               # GPU 计时入口
├── data/                                  # 数据说明见 data/README.md
│   ├── points/                            # 散点真值体数据和坐标
│   ├── reference/                         # Sample 原始 LF、干净 LF 和波前真值
│   ├── benchmarks/                        # PyTorch / MATLAB 的逐次 PSF 耗时
│   └── metrics/                           # 50 对 Joint/wDAO 数值结果、定位坐标和参数
├── scripts/
│   └── evaluate_localization.py           # 发射体定位评价
├── outputs/                               # 运行时生成的数值结果
└── requirements.txt                       # Python 依赖
```

## 其他实验

```bash
# wDAO 
CUDA_VISIBLE_DEVICES=0 python autograd/demo_wdao.py

# 10 组像差 × 5 个光子等级
CUDA_VISIBLE_DEVICES=0 python autograd/batch_points_poisson_gaussian.py

# PSF 计时
CUDA_VISIBLE_DEVICES=0 python autograd/psfsim_timetest/benchmark_psf.py \
  --output outputs/psf_benchmark --gpu 0 --threads 4 --warmups 10 --repeats 10 \
  --methods traditional shift_fresnel_precomputed shift_blur_batched
```

方法与参数：[wDAO](autograd/wdao/README.md) · [PSF 计时](autograd/psfsim_timetest/README_benchmark.md) · [数据说明](data/README.md)

## 三维定位评估

```bash
python scripts/evaluate_localization.py
```

定位默认评估 Joint 输出，接受条件为每个坐标误差 ≤6 µm；`--volume` 可选择其他重建体。
