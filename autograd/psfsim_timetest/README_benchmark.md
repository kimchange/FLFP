# PSF 计时

从仓库根目录运行主图 2 的三方法对照：

```bash
CUDA_VISIBLE_DEVICES=0 python autograd/psfsim_timetest/benchmark_psf.py \
  --output outputs/psf_benchmark --gpu 0 --threads 4 --warmups 10 --repeats 10 \
  --methods traditional shift_fresnel_precomputed shift_blur_batched
```

程序对两套光学参数分别执行 10 次预热和 10 次测量，每次计算 101 层 × 225 个视角。
`raw_timings.csv` 保存逐次耗时，`summary.csv` / `summary.json` 保存汇总。
光学参数见 `benchmark_psf.py` 的 `config()`。

重新汇总：

```bash
python autograd/psfsim_timetest/benchmark_psf.py \
  --output outputs/psf_benchmark --summarize-only
```

## 计时定义

总时间为光学初始化加 PSF 生成时间，阶段前后同步 CUDA。
计算采用 float32/complex64，关闭梯度；预热与正式测量分别统计，标准差采用 `ddof=1`。
CPU 传输和文件写入在计时区间之外。
主图 2 中，传统方法的单视角时间取总时间，解耦角度方法取总时间除以 225。

## 程序

| 文件 | 用途 |
| --- | --- |
| `lfpsf_torch_traditional.py` | 传统 PSF 模型 |
| `lfpsf_torch_shift_fresnel.py`、`lfpsf_torch_shift_fresnel_precomputed.py` | Shift–Fresnel 基础与预计算实现 |
| `lfpsf_torch_shift_blur.py`、`lfpsf_torch_shift_blur_batched.py` | Shift–blur 基础与分批实现 |
| `validate_precomputed.py`、`validate_blur_batched.py` | 两种优化实现与基础实现的数值对照 |

实验数据位于 `data/benchmarks/pytorch/` 和 `data/benchmarks/matlab_gpu/`。
