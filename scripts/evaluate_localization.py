"""Evaluate an estimated volume using the manuscript's coordinate-wise 6 µm rule.

Ground-truth emitter coordinates are used only for this post-fit evaluation.
The known emitter count sets the number of strongest unsmoothed local maxima.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import tifffile

ROOT = Path(__file__).resolve().parents[1]


def evaluate(volume, truth_zyx, spacing_zyx_um, tolerance_um=6.):
    truth = np.asarray(truth_zyx, dtype=np.float64)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    if volume.ndim != 3 or not np.isfinite(volume).all() or volume.min() < 0:
        raise ValueError('Volume must be a finite, nonnegative [z,y,x] array.')
    if truth.ndim != 2 or truth.shape[1] != 3 or len(truth) == 0 or not np.isfinite(truth).all():
        raise ValueError('Truth must contain finite voxel coordinates with columns z,y,x.')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Provide three positive voxel spacings in micrometres.')
    if not np.isfinite(tolerance_um) or tolerance_um <= 0:
        raise ValueError('The per-axis tolerance must be positive.')
    local = maximum_filter(volume, size=3, mode='constant', cval=-np.inf)
    indices = np.flatnonzero((volume == local) & (volume > volume.min()))
    del local
    intensity = volume.ravel()[indices]
    # Match the archived deterministic tie break: lower flat voxel index first.
    selected = np.lexsort((indices, -intensity))[:len(truth)]
    if len(selected) != len(truth):
        raise ValueError('Fewer distinct local maxima than the known emitter count.')
    peaks = np.array(np.unravel_index(indices[selected], volume.shape)).T
    cost = cdist(truth*spacing, peaks*spacing)
    rows, columns = linear_sum_assignment(cost)
    estimated = peaks[columns]
    delta = (estimated-truth)*spacing
    accepted = (np.abs(delta) <= tolerance_um).all(axis=1)
    return dict(truth_zyx=truth, estimated_zyx=estimated,
                spacing_zyx_um=spacing, distance_um=cost[rows, columns],
                delta_zyx_um=delta, intensity=intensity[selected][columns],
                accepted_box=accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume', type=Path,
                        default=ROOT/'outputs/demo_wavefront_sample02/volume_estimated.tiff')
    parser.add_argument('--truth', type=Path, default=ROOT/'data/points/random_points.csv',
                        help='CSV with a header and z,y,x voxel-coordinate columns')
    parser.add_argument('--spacing-zyx-um', type=float, nargs=3,
                        default=[5., 56.4/15/7.85, 56.4/15/7.85])
    parser.add_argument('--tolerance-um', type=float, default=6.)
    parser.add_argument('--output', type=Path, help='Output .npz; JSON is written alongside')
    args = parser.parse_args()
    result = evaluate(tifffile.imread(args.volume),
                      np.loadtxt(args.truth, delimiter=',', skiprows=1, ndmin=2),
                      args.spacing_zyx_um, args.tolerance_um)
    output = args.output or args.volume.parent/'localization.npz'
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    # Serialized paths are relative to the repository root.
    summary = dict(volume=Path(os.path.relpath(args.volume.resolve(), ROOT)).as_posix(),
                   truth=Path(os.path.relpath(args.truth.resolve(), ROOT)).as_posix(),
                   candidates=len(result['accepted_box']),
                   accepted=int(result['accepted_box'].sum()),
                   tolerance_each_axis_um=args.tolerance_um,
                   spacing_zyx_um=args.spacing_zyx_um,
                   peak_selection='strongest unsmoothed 3x3x3 local maxima; known emitter count',
                   assignment='one-to-one minimum total physical Euclidean distance',
                   acceptance='all absolute coordinate errors <= tolerance (box)')
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
