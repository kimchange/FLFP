#!/usr/bin/env python3
"""Reproducible, synchronized timing of the three supplied PSF implementations.

The source modules are imported unchanged. Outputs are computed in the supplied
batching schemes and streamed on GPU; timings, rather than huge PSF stacks, are
retained. Run --help for the single-scenario and suite interfaces.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
METHODS = ("traditional", "shift_fresnel", "shift_blur")
SUPPORTED_METHODS = METHODS + ("shift_fresnel_precomputed", "shift_blur_batched")
SOURCE_NAMES = {m: f"lfpsf_torch_{m}.py" for m in SUPPORTED_METHODS}
OPTICS = {
    1: dict(M=20, n=1.406, na_detection=1.05, MLPitch=56.4e-6,
            fml=536.4e-6, lam_detection=525e-9, dz=0.6e-6),
    2: dict(M=7.85, n=1, na_detection=0.5, MLPitch=56.4e-6,
            fml=444.15e-6, lam_detection=525e-9, dz=5e-6),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def gpu_snapshot(device):
    fields = ("name", "driver_version", "pstate", "temperature.gpu",
              "power.draw", "clocks.current.sm", "clocks.current.memory",
              "memory.used", "utilization.gpu")
    result = subprocess.run([
        "nvidia-smi", "-i", str(device),
        "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits",
    ], text=True, capture_output=True)
    if result.returncode:
        return dict(returncode=result.returncode)
    snapshot = dict(zip(fields, next(csv.reader(result.stdout.splitlines()), [])))
    for key, value in snapshot.items():
        value = value.strip()
        if key not in ("name", "driver_version", "pstate"):
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
        snapshot[key] = value
    return snapshot


def config(method, parameter, device):
    # These values deliberately follow the two parameter blocks of each source.
    # Shift-blur uses Nz=1 in the constructor, with 101 physical depth
    # coordinates supplied at evaluation time.
    is_blur = method.startswith("shift_blur")
    side = 375 if parameter == 1 else (525 if is_blur else 765)
    kwargs = dict(OPTICS[parameter], Nnum=15, OSR=3, device=device,
                  lfpsf_shape=(15, 15, 1 if is_blur else 101,
                               side, side))
    if method == "traditional":
        kwargs.update(shift_mode="zero")
    elif method.startswith("shift_fresnel"):
        kwargs.update(depth_batch_size=101 // 8)
    else:
        kwargs.update(xy_downsample=3, input_views=list(range(225)))
    return dict(
        method=method, parameter=parameter, constructor=kwargs,
        generated_depths=101, generated_views=225,
        depth_coordinates_in_dz=list(range(-50, 51)),
        view_order="row-major, flat view = u*15+v",
        outer_depth_batch=3 if method == "traditional" else
                          (11 if is_blur else 101),
        outer_view_batch=3 if method.startswith("shift_fresnel") else 225,
        phase_input="21 zero OSA coefficients" if is_blur else "None",
        normalized=True, piston_tip_tilt=is_blur,
        psf_binning=1 if is_blur else None,
        output_side=225 if is_blur else side,
        output_representation="integer shift + centered intensity PSF"
                              if is_blur else "full intensity PSF",
    )


def load_generator(method, source_dir):
    path = source_dir / SOURCE_NAMES[method]
    spec = importlib.util.spec_from_file_location(f"benchmark_{method}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PsfGenerator5D


def requests(cfg, device):
    """Prepare loop inputs outside the timed generation interval."""
    method = cfg["method"]
    if method == "traditional":
        u, v = torch.meshgrid(torch.arange(15), torch.arange(15), indexing="ij")
        depths = torch.arange(101)
        return [(0, 225, i, min(i + 3, 101),
                 (u, v, depths[i:i + 3], None),
                 dict(normalized=True, piston_tip_tilt=False))
                for i in range(0, 101, 3)]
    if method.startswith("shift_fresnel"):
        u, v = np.unravel_index(np.arange(225), (15, 15))
        return [(i, min(i + 3, 225), 0, 101,
                 (u[i:i + 3], v[i:i + 3], slice(None), None),
                 dict(normalized=True, piston_tip_tilt=False))
                for i in range(0, 225, 3)]
    depths = torch.arange(101, dtype=torch.float32, device=device) - 50
    # The example resets wf to 21 zeros immediately before generation.
    phi = torch.zeros(21)
    return [(0, 225, i, min(i + 11, 101),
             (list(range(225)), depths[i:i + 11], phi),
             dict(normalized=True, piston_tip_tilt=True, psf_binning=1))
            for i in range(0, 101, 11)]


def validate_last(output, method, save_path):
    """Finite/nonnegative check and small output extract, OUTSIDE the timer."""
    if method.startswith("shift_blur"):
        shift, psf = output
    else:
        shift, psf = None, output
    checks = dict(shape=list(psf.shape), dtype=str(psf.dtype),
                  all_finite=bool(torch.isfinite(psf).all().item()),
                  min=float(psf.min().item()), max=float(psf.max().item()),
                  last_psf_sum=float(psf[-1, -1].sum().item()))
    assert checks["all_finite"] and checks["min"] >= 0, checks
    if shift is not None:
        checks.update(shift_shape=list(shift.shape), shift_dtype=str(shift.dtype),
                      shift_min=int(shift.min().item()), shift_max=int(shift.max().item()))
    if save_path is not None:
        # Last completed output block includes physical view 224, depth +50*dz
        # in all three methods. This is a diagnostic, not an equivalence test.
        data = dict(psf=psf[-1, -1].cpu().numpy(), view=np.array(224),
                    depth_index=np.array(100), depth_in_dz=np.array(50))
        if shift is not None:
            data["shift_yx"] = shift[-1, -1].cpu().numpy()
        np.savez_compressed(save_path, **data)
    return checks


def run_scenario(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_dir = out / f"p{args.parameter}_{args.method}"
    result_dir.mkdir(exist_ok=True)
    if (result_dir / "complete.json").exists():
        completed = json.loads((result_dir / "complete.json").read_text())
        if (completed.get("warmup_runs", 1), completed["measured_repeats"]) != (args.warmups, args.repeats):
            raise ValueError("Existing result uses different run counts; use a new output directory")
        print(f"Already complete: {result_dir.name}", flush=True)
        return
    raw_path = result_dir / "runs.jsonl"
    if raw_path.exists() and raw_path.stat().st_size:
        raise ValueError("Incomplete scenario retained: use a new output directory to keep the warm-up sequence continuous")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.gpu}")
    # Common CUDA runtime / library startup is separately timed and excluded
    # from the optical constructor and PSF generation measurements.
    t0 = time.perf_counter()
    torch.cuda.set_device(device)
    torch.cuda.init()
    warm = torch.ones((64, 64), dtype=torch.complex64, device=device)
    torch.fft.fft2(warm)
    torch.fft.ifft2(warm)
    torch.matmul(warm, warm)
    torch.cuda.synchronize(device)
    startup_s = time.perf_counter() - t0
    del warm
    torch.cuda.empty_cache()

    source_dir = out / "sources"
    cls = load_generator(args.method, source_dir)
    cfg = config(args.method, args.parameter, str(device))
    props = torch.cuda.get_device_properties(device)
    write_json(result_dir / "config.json", dict(cfg,
        constructor=dict(cfg["constructor"], device=device.type)))
    write_json(result_dir / "environment.json", dict(
        python=platform.python_version(),
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        gpu_name=props.name, gpu_total_memory=props.total_memory,
        gpu_compute_capability=[props.major, props.minor],
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        runtime_startup_s=startup_s,
        gpu_snapshot=gpu_snapshot(args.gpu),
    ))
    for iteration in range(args.warmups + args.repeats):
        phase = "warmup" if iteration < args.warmups else "measurement"
        phase_iteration = iteration + 1 if phase == "warmup" else iteration - args.warmups + 1
        tag = f"warmup_{phase_iteration:02d}" if phase == "warmup" else f"repeat_{phase_iteration:02d}"
        print(f"{now()} START p{args.parameter} {args.method} {tag}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        before = gpu_snapshot(args.gpu)
        cfg_inputs = requests(cfg, device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        init_begin = torch.cuda.Event(enable_timing=True)
        init_end = torch.cuda.Event(enable_timing=True)
        init_begin.record()
        start = time.perf_counter()
        with torch.no_grad():
            generator = cls(**cfg["constructor"])
        init_end.record()
        torch.cuda.synchronize(device)
        init_s = time.perf_counter() - start
        init_gpu_ms = init_begin.elapsed_time(init_end)
        init_peak = torch.cuda.max_memory_allocated(device)
        init_reserved_peak = torch.cuda.max_memory_reserved(device)
        init_resident = torch.cuda.memory_allocated(device)
        print(f"{now()} INIT p{args.parameter} {args.method} {tag}: {init_s:.6f} s", flush=True)

        # Events and coverage bookkeeping are prepared before timing. Each
        # call retains exactly one output block; no GPU-to-CPU transfer occurs
        # during generation. The source examples also do not assemble stacks.
        events = [(torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True)) for _ in cfg_inputs]
        coverage = np.zeros((225, 101), dtype=np.uint8)
        batches = []
        output = None
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        gen_start = time.perf_counter()
        with torch.no_grad():
            for bi, (vs, ve, ds, de, call_args, call_kwargs) in enumerate(cfg_inputs):
                del output
                events[bi][0].record()
                output = generator.incoherent_psf(*call_args, **call_kwargs)
                events[bi][1].record()
                psf_shape = tuple(output[1].shape if args.method.startswith("shift_blur") else output.shape)
                expected_shape = (ve - vs, de - ds, cfg["output_side"], cfg["output_side"])
                assert psf_shape == expected_shape, (psf_shape, expected_shape)
                if args.method.startswith("shift_blur"):
                    assert tuple(output[0].shape) == (ve - vs, de - ds, 2)
                coverage[vs:ve, ds:de] += 1
                batches.append(dict(batch=bi, view_start=vs, view_stop=ve,
                                    depth_start=ds, depth_stop=de, shape=list(psf_shape)))
        torch.cuda.synchronize(device)
        generation_s = time.perf_counter() - gen_start
        generation_peak = torch.cuda.max_memory_allocated(device)
        generation_reserved_peak = torch.cuda.max_memory_reserved(device)
        assert np.all(coverage == 1), "Every view-depth pair must be generated exactly once"
        for batch, (begin, end) in zip(batches, events):
            batch["cuda_elapsed_ms"] = begin.elapsed_time(end)
        diagnostic = validate_last(
            output, args.method,
            result_dir / "last_psf_warmup.npz" if iteration == 0 else None,
        )
        record = dict(
            status="ok", method=args.method, parameter=args.parameter,
            iteration=iteration, phase=phase, phase_iteration=phase_iteration,
            included_in_summary=phase == "measurement",
            initialization_s=init_s, generation_s=generation_s, total_s=init_s + generation_s,
            initialization_cuda_ms=init_gpu_ms,
            generation_cuda_batch_sum_ms=sum(b["cuda_elapsed_ms"] for b in batches),
            init_peak_allocated_bytes=init_peak, init_peak_reserved_bytes=init_reserved_peak,
            init_resident_bytes=init_resident,
            generation_peak_allocated_bytes=generation_peak,
            generation_peak_reserved_bytes=generation_reserved_peak,
            generated_view_depth_pairs=int(coverage.sum()), batch_count=len(batches),
            validation_last_batch=diagnostic, gpu_before=before,
            gpu_after=gpu_snapshot(args.gpu),
        )
        if args.method == "shift_fresnel_precomputed":
            record["precomputed_view_count"] = generator.precomputed_view_count
            record["precomputed_kernel_bytes"] = generator.precomputed_kernel_bytes
            record["angle_batch_cache_hits"] = generator.angle_batch_cache_hits
            record["angle_subset_cache_hits"] = generator.angle_subset_cache_hits
            record["full_depth_spectrum_reuses"] = generator.full_depth_spectrum_reuses
            assert generator.precomputed_view_count == 225
            assert generator.angle_batch_cache_hits == len(cfg_inputs) == 75
            assert generator.angle_subset_cache_hits == 0
            assert generator.full_depth_spectrum_reuses == 75
        if args.method == "shift_blur_batched":
            record["precomputed_view_count"] = generator.precomputed_view_count
            record["precompute_view_batch_size"] = generator.precompute_view_batch_size
            assert generator.precomputed_view_count == 225
            assert generator.precompute_view_batch_size == 3
        write_json(result_dir / f"batches_{tag}.json", batches)
        with raw_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"{now()} DONE p{args.parameter} {args.method} {tag}: "
              f"init={init_s:.6f} s generation={generation_s:.6f} s "
              f"total={init_s + generation_s:.6f} s "
              f"peak={max(init_peak, generation_peak)/2**30:.3f} GiB", flush=True)
        del output, generator, cfg_inputs, events
    write_json(result_dir / "complete.json", dict(warmup_runs=args.warmups,
                                                  measured_repeats=args.repeats))


def summarize(out, phase="measurement"):
    if phase not in ("measurement", "warmup"):
        raise ValueError(phase)
    records = []
    for path in sorted(out.glob("p*_*/runs.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    for r in records:
        r.setdefault("phase", "measurement" if r["included_in_summary"] else "warmup")
    csv_columns = ["parameter", "method", "iteration", "phase", "phase_iteration", "included_in_summary",
                   "initialization_s", "generation_s", "total_s",
                   "initialization_cuda_ms", "generation_cuda_batch_sum_ms",
                   "init_peak_allocated_bytes", "init_peak_reserved_bytes", "init_resident_bytes",
                   "generation_peak_allocated_bytes", "generation_peak_reserved_bytes",
                   "generated_view_depth_pairs", "batch_count", "status"]
    if phase == "measurement":
        with (out / "raw_timings.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    summary = []
    for parameter in (1, 2):
        for method in SUPPORTED_METHODS:
            selected = [r for r in records if r["parameter"] == parameter
                        and r["method"] == method and r["phase"] == phase
                        and r["status"] == "ok"]
            if not selected:
                continue
            row = dict(parameter=parameter, method=method, phase=phase, n=len(selected))
            for field in ("initialization_s", "generation_s", "total_s"):
                values = [r[field] for r in selected]
                for stat, value in dict(mean=statistics.mean(values),
                                        std=statistics.stdev(values) if len(values) > 1 else 0,
                                        min=min(values), max=max(values),
                                        median=statistics.median(values)).items():
                    row[f"{field}_{stat}"] = value
            row["max_peak_allocated_gib"] = max(
                max(r["init_peak_allocated_bytes"], r["generation_peak_allocated_bytes"])
                for r in selected) / 2**30
            divisor = 1 if method == "traditional" else 225
            row["single_view_divisor"] = divisor
            row["single_view_derived_s_mean"] = row["generation_s_mean"] / divisor
            row["single_view_derived_s_std"] = row["generation_s_std"] / divisor
            summary.append(row)
    for row in summary:
        baseline = next((r for r in summary if r["parameter"] == row["parameter"]
                         and r["method"] == "traditional"), None)
        row["generation_speedup_vs_traditional"] = (
            baseline["generation_s_mean"] / row["generation_s_mean"] if baseline else None)
        row["total_speedup_vs_traditional"] = (
            baseline["total_s_mean"] / row["total_s_mean"] if baseline else None)
    prefix = "warmup_" if phase == "warmup" else ""
    write_json(out / f"{prefix}summary.json", summary)
    if summary:
        with (out / f"{prefix}summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    lines = [f"# PSF simulation timing: {phase}", "", "Mean ± sample standard deviation, seconds. "
             "Warm-up and measurement runs are summarized separately; no runs are discarded.", "",
             "| Parameter | Method | n | Initialization / s | 101 depths × 225 views / s | Total / s | Peak allocated / GiB |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        cells = [str(r["parameter"]), r["method"], str(r["n"])]
        for field in ("initialization_s", "generation_s", "total_s"):
            cells.append(f"{r[field + '_mean']:.6f} ± {r[field + '_std']:.6f}")
        cells.append(f"{r['max_peak_allocated_gib']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "See protocol.json, source snapshots, per-scenario config.json and environment.json,",
                  "raw_timings.csv, runs.jsonl, and batches_*.json for the complete provenance.", ""])
    (out / f"{prefix.upper()}SUMMARY.md").write_text("\n".join(lines))
    if phase == "measurement":
        summarize(out, phase="warmup")
    return summary


def prepare(out, args):
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "sources"
    snapshot.mkdir(exist_ok=True)
    for name in list(SOURCE_NAMES.values()) + [Path(__file__).name, "validate_precomputed.py", "validate_blur_batched.py", "README_benchmark.md"]:
        path = ROOT / name
        target = snapshot / name
        if not target.exists():
            shutil.copy2(path, target)
    protocol = dict(
        gpu_name=torch.cuda.get_device_name(args.gpu),
        suite_methods=[args.method] if args.method else args.methods,
        measured_repeats=args.repeats, full_warmup_runs_per_scenario=args.warmups,
        statistics_phases=["warmup", "measurement"],
        run_sequence="all warm-up runs followed by all measurement runs within the same process",
        constructor_rebuilt_every_run=True, source_modules_modified=False,
        cuda_runtime_warmed_before_optical_timing=True,
        empty_cuda_allocator_before_each_constructor=True,
        process_isolation="one child process per method/parameter, serial on same GPU",
        timing="time.perf_counter with CUDA synchronize before and after each phase",
        initialization="supplied constructor only, including any eager kernel precomputation",
        generation="all 101 depths and 225 views, including per-call kernel preparation; "
                   "shift-blur includes both displacement estimation and local intensity kernels",
        precomputed_variant="all 225 angle kernels prepared at initialization; unused kprop released; "
                            "full-depth zero-phase requests reuse kbase without copying",
        blur_batched_variant="only angle-kernel initialization changed: fixed batches of 3 and "
                             "preallocated cropped bank replace growing concatenations; "
                             "generation methods unchanged",
        single_view_statistic="derived from generation only: traditional / 1, "
                              "shift_fresnel and shift_fresnel_precomputed / 225, shift_blur / 225; "
                              "not separately measured single-view latency",
        total="initialization + generation; excludes runtime startup, loop-input setup, "
              "validation, telemetry and file I/O",
        outputs="streamed in source-example batch sizes, released between calls, "
                "no assembly of full stack, no CPU transfer or PSF disk export in timer",
        raw_data="every run timing incl. all warm-ups, every batch CUDA event timing, "
                 "memory, configurations, source snapshots, environment, GPU telemetry, "
                 "and one diagnostic final PSF per scenario",
        validation="all 22725 view-depth pairs covered exactly once and all output shapes checked; "
                   "last output block checked finite and nonnegative outside timing",
        wavefront="zero aberration; source examples use None for traditional/Fresnel "
                  "and 21 zero coefficients for shift-blur",
        representation="supplied full PSF versus shift + compact PSF settings retained; "
                       "no assertion of numeric equivalence or equal truncation error from timing alone",
        precision="source float32/complex64, no gradients, PyTorch default TF32 setting recorded",
        statistics="warm-up and measurement runs separately: arithmetic mean and sample std (ddof=1)",
    )
    if not (out / "protocol.json").exists():
        write_json(out / "protocol.json", protocol)
    else:
        previous = json.loads((out / "protocol.json").read_text())
        for key in ("measured_repeats", "full_warmup_runs_per_scenario"):
            if previous[key] != protocol[key]:
                raise ValueError(f"Existing protocol has a different {key}; use a new output directory")
        if previous.get("gpu_name", protocol["gpu_name"]) != protocol["gpu_name"]:
            raise ValueError("Existing protocol uses a different GPU model; use a new output directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--method", choices=SUPPORTED_METHODS)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS, default=list(METHODS))
    parser.add_argument("--parameter", type=int, choices=[1, 2])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.summarize_only:
        print(json.dumps(summarize(args.output), indent=2))
        return
    if args.repeats < 2 or args.warmups < 1:
        parser.error("At least one warm-up and two measured repeats are required")
    prepare(args.output, args)
    if args.method:
        if args.parameter is None:
            parser.error("--method requires --parameter")
        run_scenario(args)
        summarize(args.output)
        return
    if args.parameter is not None:
        parameters = [args.parameter]
    else:
        parameters = [1, 2]
    for parameter in parameters:
        for method in args.methods:
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--output", str(args.output), "--gpu", str(args.gpu),
                       "--threads", str(args.threads), "--repeats", str(args.repeats),
                       "--warmups", str(args.warmups),
                       "--method", method, "--parameter", str(parameter)]
            result = subprocess.run(command)
            summarize(args.output)
            if result.returncode:
                raise RuntimeError(f"Scenario failed ({result.returncode}): p{parameter} {method}")
    write_json(args.output / "complete.json", dict(scenarios=len(parameters) * len(args.methods)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
