# 数据

| 目录 | 内容 |
| --- | --- |
| `points/` | 100 个散点的真值体数据和 `(z,y,x)` 坐标 |
| `reference/points_photons100_sample02/` | 主图 3 的 Sample 2 输入与真值 |
| `benchmarks/pytorch/`、`benchmarks/matlab_gpu/` | 主图 2 的逐次耗时 |
| `metrics/batch_points_poisson_gaussian_output/` | 50 例 Joint 结果和真值波前 |
| `metrics/wdao_comparison_20260914/` | 50 对 wDAO/Joint 指标、估计波前和定位数据 |

数据表中的路径相对于仓库根目录；`catalog.json` 内的路径相对于该文件所在目录。

## Demo 输入

Sample 2 对应零基 index 1、峰值 100 光子。

| 文件 | 用途 |
| --- | --- |
| `lf.tiff` | 原始观测，`[view,y,x]=[81,135,135]` |
| `lf_clean.tiff` | 干净 LF，用于重投影评价 |
| `wf_true.txt` | 21 项真值波前系数，单位 wave |
| `noise.json` | 曝光系数和噪声参数 |

体数据为 `[z,y,x]=[101,675,675]`，体素步长 `[5,0.47898089,0.47898089]` µm。
观测满足 `LF = Poisson(c × clean_LF) + N(105,9)`，本例 `c=0.5829866912756507`。
`lf_clean.tiff` 使用原始光学强度单位。

## 结果

`results.csv` / `paired_metrics.csv` 保存逐例波前误差与 LF 误差。
Joint 的定位配对坐标位于 `metrics/batch_points_poisson_gaussian_output/localization/`。
wDAO 的 `estimates/`、`fixed_results/` 按 `p*/sample*/` 组织；`p100/sample01/` 即当前 demo。
`localization_box6_v2/` 保存两方法的定位计数和接受掩膜。

波前 RMS 单位为 wave，LF relative L2 相对全部 81 个干净通道计算。
定位使用最强 100 个未平滑局部峰，先按物理距离一一匹配，再判断每个坐标误差是否 ≤6 µm。
主图 2 使用含初始化的总时间；解耦角度方法的单视角时间为总时间除以 225。
