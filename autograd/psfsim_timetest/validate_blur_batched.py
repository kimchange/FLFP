#!/usr/bin/env python3
"""Validate batched shift--blur initialization against the unchanged reference."""

import argparse
import ast
import gc
from pathlib import Path

import torch

from benchmark_psf import ROOT, config, load_generator, requests, write_json


def check(reference, actual):
    assert reference.shape == actual.shape
    equal = torch.equal(reference, actual)
    if equal:
        return dict(bitwise_equal=True, max_absolute_error=0.0, relative_l2=0.0)
    difference = (actual - reference).abs()
    maximum = float(difference.max())
    relative = float(torch.linalg.vector_norm(difference.flatten())
                     / torch.linalg.vector_norm(reference.abs().flatten()))
    torch.testing.assert_close(actual, reference, rtol=1e-5,
                               atol=1e-7 * float(reference.abs().max()))
    return dict(bitwise_equal=False, max_absolute_error=maximum, relative_l2=relative)


def unchanged_generation():
    trees = []
    for name in ("shift_blur", "shift_blur_batched"):
        tree = ast.parse((ROOT / f"lfpsf_torch_{name}.py").read_text())
        if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            tree.body.pop(0)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "PsfGenerator5D":
                node.body = [item for item in node.body
                             if not (isinstance(item, ast.FunctionDef) and item.name == "__init__")]
        trees.append(ast.dump(tree))
    assert trees[0] == trees[1], "Changes outside the constructor are not allowed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    unchanged_generation()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    cases = []
    for parameter in (1, 2):
        with torch.no_grad():
            baseline = load_generator("shift_blur", ROOT)(
                **config("shift_blur", parameter, device)["constructor"])
            variant = load_generator("shift_blur_batched", ROOT)(
                **config("shift_blur_batched", parameter, device)["constructor"])
            kernel = check(baseline.uv_mla_fresnel, variant.uv_mla_fresnel)
            cases.append(dict(parameter=parameter, case="all_225_angle_kernels", **kernel))
            print(cases[-1], flush=True)
            covered = 0
            for _, _, start, stop, call_args, kwargs in requests(config("shift_blur", parameter, device), device):
                ref_shift, ref_psf = baseline.incoherent_psf(*call_args, **kwargs)
                shift, psf = variant.incoherent_psf(*call_args, **kwargs)
                assert torch.equal(ref_shift, shift), "Integer displacements changed"
                result = check(ref_psf, psf)
                covered += (stop - start) * 225
                cases.append(dict(parameter=parameter, case="full_generation_batch",
                                  depth_start=start, depth_stop=stop, shifts_equal=True, **result))
                print(cases[-1], flush=True)
                del ref_shift, ref_psf, shift, psf
            assert covered == 22725
            phase = torch.zeros(21)
            phase[3], phase[5], phase[7] = 0.13, -0.07, 0.09
            for name, views, depths, phi, binning in (
                ("nonzero_aberration", [0, 112, 224], [-50, 0, 50], phase, 1),
                ("single_view_no_phase", [112], [0], None, 1),
                ("binning_two", [1, 73, 213], [-25, 0, 25], phase, 2),
            ):
                depths = torch.tensor(depths, device=device, dtype=torch.float32)
                kwargs = dict(normalized=True, piston_tip_tilt=True, psf_binning=binning)
                ref_shift, ref_psf = baseline.incoherent_psf(views, depths, phi, **kwargs)
                shift, psf = variant.incoherent_psf(views, depths, phi, **kwargs)
                assert torch.equal(ref_shift, shift)
                cases.append(dict(parameter=parameter, case=name, shifts_equal=True, **check(ref_psf, psf)))
                print(cases[-1], flush=True)
                if name == "nonzero_aberration":
                    _, ref_fixed = baseline.incoherent_psf(views, depths, phi, shift_fixed=ref_shift, **kwargs)
                    _, fixed = variant.incoherent_psf(views, depths, phi, shift_fixed=ref_shift, **kwargs)
                    cases.append(dict(parameter=parameter, case="fixed_shifts", **check(ref_fixed, fixed)))
                    print(cases[-1], flush=True)
                    del ref_fixed, fixed
                del ref_shift, ref_psf, shift, psf
            del baseline, variant
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "validation.json", dict(
        all_passed=True, gpu_name=torch.cuda.get_device_name(args.gpu), generation_source_ast_identical=True,
        all_bitwise_equal=all(row["bitwise_equal"] for row in cases), cases=cases,
        full_generation_view_depth_pairs_per_parameter=22725,
    ))


if __name__ == "__main__":
    main()
