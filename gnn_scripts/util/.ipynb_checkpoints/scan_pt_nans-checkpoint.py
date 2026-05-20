"""
scan_pt_nans.py
---------------
Scans a PyTorch .pt file (a list of PyG Data objects) for NaN and Inf values.

Usage:
    python scan_pt_nans.py <path_to_file.pt>
"""

import sys
import torch
import numpy as np
from torch_geometric.data import Data

def scan_tensor(tensor, field_name, graph_idx):
    """Check a single tensor for NaN/Inf. Returns a list of issue strings."""
    issues = []
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return issues
    if torch.isnan(tensor).any():
        n = torch.isnan(tensor).sum().item()
        issues.append(f"  [Graph {graph_idx}] '{field_name}': {n} NaN value(s)")
    if torch.isinf(tensor).any():
        n = torch.isinf(tensor).sum().item()
        issues.append(f"  [Graph {graph_idx}] '{field_name}': {n} Inf value(s)")
    return issues


def scan_file(path):
    print(f"\n{'='*60}")
    print(f"  Scanning: {path}")
    print(f"{'='*60}\n")

    dataset = torch.load(path, weights_only=False)

    # Handle both a single Data object and a list/dataset of them
    if isinstance(dataset, Data):
        dataset = [dataset]

    print(f"  Total graphs: {len(dataset)}\n")

    all_issues = []
    field_nan_counts = {}   # field -> total NaN count across all graphs
    field_inf_counts = {}   # field -> total Inf count across all graphs
    graphs_with_issues = set()

    for idx, data in enumerate(dataset):
        for field in data.keys():
            tensor = data[field]
            if not isinstance(tensor, torch.Tensor):
                continue
            # Only check floating point tensors
            if not tensor.is_floating_point():
                continue

            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()

            if nan_count > 0:
                field_nan_counts[field] = field_nan_counts.get(field, 0) + nan_count
                graphs_with_issues.add(idx)
                all_issues.append(f"  [Graph {idx:>6}] '{field}': {nan_count} NaN(s)")

            if inf_count > 0:
                field_inf_counts[field] = field_inf_counts.get(field, 0) + inf_count
                graphs_with_issues.add(idx)
                all_issues.append(f"  [Graph {idx:>6}] '{field}': {inf_count} Inf(s)")

    # ── Summary ──────────────────────────────────────────────────────────────
    if not all_issues:
        print("  ✅  No NaN or Inf values found. Your file looks clean!\n")
    else:
        print(f"  ⚠️  Found issues in {len(graphs_with_issues)} / {len(dataset)} graphs\n")

        print("  Per-field summary:")
        all_fields = set(list(field_nan_counts.keys()) + list(field_inf_counts.keys()))
        for field in sorted(all_fields):
            nans = field_nan_counts.get(field, 0)
            infs = field_inf_counts.get(field, 0)
            parts = []
            if nans: parts.append(f"{nans} NaN(s)")
            if infs: parts.append(f"{infs} Inf(s)")
            print(f"    '{field}': {', '.join(parts)}")

        print(f"\n  First 20 per-graph issues:")
        for line in all_issues[:20]:
            print(line)
        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more (suppressed)")

    # ── Basic feature stats for 'x' ──────────────────────────────────────────
    print("\n  Feature stats for 'x' (ignoring NaN/Inf):")
    try:
        all_x = torch.cat([d.x for d in dataset if d.x is not None], dim=0)
        finite_mask = torch.isfinite(all_x)
        finite_x = all_x[finite_mask.all(dim=1)]  # rows where all features are finite
        if finite_x.numel() > 0:
            print(f"    Shape (finite rows): {finite_x.shape}")
            for i in range(finite_x.shape[1]):
                col = finite_x[:, i]
                print(f"    Feature {i:>2}: min={col.min():.4f}  max={col.max():.4f}"
                      f"  mean={col.mean():.4f}  std={col.std():.4f}")
        else:
            print("    No fully-finite rows found.")
    except Exception as e:
        print(f"    Could not compute stats: {e}")

    print(f"\n{'='*60}\n")
    return len(all_issues) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scan_pt_nans.py <path_to_file.pt>")
        sys.exit(1)

    path = sys.argv[1]
    clean = scan_file(path)
    sys.exit(0 if clean else 1)
