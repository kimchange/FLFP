#!/usr/bin/env python3
"""Check the prepared kernels against the original full-grid PSF calculation."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_psf import ROOT, config, load_generator, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    phase = torch.zeros(21)
    phase[3], phase[5], phase[7] = 0.13, -0.07, 0.09
    cases = [
        ("full_depths_contiguous_views", [111, 112, 113], slice(None), None),
        ("full_depths_arbitrary_views", [0, 112, 224], slice(None), None),
        ("three_depths_nonzero_aberration", [1, 73, 213], [0, 50, 100], phase),
        ("single_view", [112], [50], None),
    ]
    results = []
    for parameter in (1, 2):
        references = []
        cls = load_generator("shift_fresnel", ROOT)
        with torch.no_grad():
            generator = cls(**config("shift_fresnel", parameter, device)["constructor"])
            for name, views, depths, phi in cases:
                u, v = np.unravel_index(views, (15, 15))
                output = generator.incoherent_psf(u, v, depths, phi, normalized=True, piston_tip_tilt=True)
                references.append(output.cpu().numpy())
                del output
            del generator
        gc.collect()
        torch.cuda.empty_cache()
        cls = load_generator("shift_fresnel_precomputed", ROOT)
        with torch.no_grad():
            generator = cls(**config("shift_fresnel_precomputed", parameter, device)["constructor"])
            assert generator.precomputed_view_count == 225
            cache_keys = set(generator._angle_kernel_cache)
            for (name, views, depths, phi), reference in zip(cases, references):
                u, v = np.unravel_index(views, (15, 15))
                output = generator.incoherent_psf(u, v, depths, phi, normalized=True, piston_tip_tilt=True)
                actual = output.cpu().numpy()
                equal = bool(np.array_equal(actual, reference))
                if equal:
                    max_abs = relative_l2 = 0.0
                else:
                    diff = actual - reference
                    max_abs = float(np.max(np.abs(diff)))
                    relative_l2 = float(np.sqrt(np.sum(diff.astype(np.float64)**2)
                                                 / np.sum(reference.astype(np.float64)**2)))
                    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)
                result = dict(parameter=parameter, case=name, views=views,
                              output_shape=list(actual.shape), bitwise_equal=equal,
                              max_absolute_error=max_abs, relative_l2=relative_l2)
                results.append(result)
                print(json.dumps(result), flush=True)
                del output, actual
            assert set(generator._angle_kernel_cache) == cache_keys
            assert generator.angle_batch_cache_hits == 1
            assert generator.angle_subset_cache_hits == 3
            assert generator.full_depth_spectrum_reuses == 2
            del generator, references, reference
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "validation.json", dict(
        gpu_name=torch.cuda.get_device_name(args.gpu), cases=results,
        all_passed=True, cache_did_not_grow_during_evaluation=True,
        full_depth_copy_elimination_verified=True,
        description="Two parameter sets; 101 depths for standard and arbitrary view batches, "
                    "nonzero aberration at three depths, and a single requested view.",
    ))


if __name__ == "__main__":
    main()
