"""
validateBDT.py
==============
Scores validation/test ROOT files with a trained combined ECal+HCal BDT.
Saves per-event scores with file/event metadata for later merging.

Supports:
  - Multiple background directories
  - Run-number filtering (--bkg_run_ranges)
  - Job splitting for parallel execution (--job_index / --n_jobs)
  - Per-event metadata tracking (filename, event index, BDT score)

Usage (single job):
    python3 validateBDT.py --model model.pkl --sig_dir ... --bkg_dir dir1 dir2 --out_dir ...

Usage (parallel, e.g. 6 jobs):
    python3 validateBDT.py --model model.pkl ... --job_index 0 --n_jobs 6
    python3 validateBDT.py --model model.pkl ... --job_index 1 --n_jobs 6
    ...
    Then: python3 mergeBDTResults.py --input_dir ... --out_dir ...
"""

import argparse
import glob
import os
import subprocess
import sys
import re
import pickle as pkl

import awkward as ak
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import trainCombinedBDT as tcb


# ═══════════════════════════════════════════════════════════════════════════
#  MASS POINT / RUN NUMBER PARSING
# ═══════════════════════════════════════════════════════════════════════════

def extract_mass_point(filename):
    match = re.search(r'mAMeV_(\d+)', filename)
    return match.group(1) if match else None


def extract_run_number(filename):
    match = re.search(r'run_(\d+)', filename)
    return int(match.group(1)) if match else None


def group_files_by_mass(directory):
    root_files = sorted(glob.glob(os.path.join(directory, "*.root")))
    groups = {}
    for fp in root_files:
        mass = extract_mass_point(os.path.basename(fp))
        if mass is None:
            print(f"  WARNING: Could not extract mass from {os.path.basename(fp)}, skipping")
            continue
        groups.setdefault(mass, []).append(fp)
    return groups


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION WITH EVENT TRACKING
# ═══════════════════════════════════════════════════════════════════════════

def process_file_with_tracking(filepath, is_signal, ecal_threshold, hcal_threshold, roc_data):
    """
    Process one ROOT file. Returns (features, labels, filenames, event_indices)
    or (None, None, None, None) on failure.
    """
    pass_name = 'eat_vis' if is_signal else 'eat'

    try:
        with uproot.open(filepath) as f:
            tree = f["LDMX_Events"]
            total = tree.num_entries

            event_mask = tcb.apply_pre_discriminator_cuts(
                tree, pass_name, pass_name, 'is_noise_', ecal_threshold, hcal_threshold
            )
            n_pass = int(ak.sum(event_mask))
            if n_pass == 0:
                return None, None, None, None

            print(f"  {os.path.basename(filepath)}: {n_pass}/{total} pass cuts ({100.*n_pass/total:.2f}%)")

            # Get indices of passing events
            passing_indices = np.where(ak.to_numpy(event_mask))[0]

            layer_z = tcb.build_layer_z_lookup(tree, pass_name)
            ecal_feats = tcb.extract_ecal_features(tree, pass_name, event_mask, roc_data, layer_z)
            hcal_feats = tcb.extract_hcal_features(tree, pass_name, event_mask)

            combined = np.hstack([ecal_feats, hcal_feats])
            labels = np.full(n_pass, int(is_signal), dtype=np.float32)
            basename = os.path.basename(filepath)
            filenames = np.array([basename] * n_pass)
            event_ids = passing_indices.astype(np.int64)

            return combined, labels, filenames, event_ids

    except Exception as e:
        print(f"  ERROR processing {os.path.basename(filepath)}: {e}")
        return None, None, None, None


def extract_features_with_tracking(file_list, is_signal, ecal_threshold, hcal_threshold, roc_data):
    """
    Extract features from a list of ROOT files with metadata tracking.
    Returns (features, labels, filenames_array, event_indices_array) or all None.
    """
    all_features, all_labels, all_filenames, all_event_ids = [], [], [], []

    for fp in file_list:
        feats, labs, fnames, eids = process_file_with_tracking(
            fp, is_signal, ecal_threshold, hcal_threshold, roc_data
        )
        if feats is not None:
            all_features.append(feats)
            all_labels.append(labs)
            all_filenames.append(fnames)
            all_event_ids.append(eids)

    if not all_features:
        return None, None, None, None

    return (np.vstack(all_features), np.concatenate(all_labels),
            np.concatenate(all_filenames), np.concatenate(all_event_ids))


# ═══════════════════════════════════════════════════════════════════════════
#  FILE COLLECTION & FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def collect_bkg_files(bkg_dirs, run_ranges):
    """
    Collect background ROOT files from multiple directories with optional
    run-number filtering. Uses `find` to handle large directories.
    """
    all_files = []
    for bkg_dir, (rmin, rmax) in zip(bkg_dirs, run_ranges):
        print(f"\nScanning: {bkg_dir}")
        result = subprocess.run(
            ['find', bkg_dir, '-maxdepth', '1', '-name', '*.root'],
            capture_output=True, text=True
        )
        found = sorted([f.strip() for f in result.stdout.strip().split('\n') if f.strip()])
        print(f"  Found {len(found)} ROOT files")

        if rmin is not None or rmax is not None:
            filtered = []
            for fp in found:
                run = extract_run_number(os.path.basename(fp))
                if run is None:
                    continue
                if rmin is not None and run < rmin:
                    continue
                if rmax is not None and run > rmax:
                    continue
                filtered.append(fp)
            print(f"  After run filter [{rmin}–{rmax}]: {len(filtered)} files")
            all_files.extend(filtered)
        else:
            all_files.extend(found)

    all_files = sorted(all_files)
    print(f"\nTotal background files: {len(all_files)}")
    return all_files


# ═══════════════════════════════════════════════════════════════════════════
#  BDT SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_events(model, features):
    features_clean = features.copy()
    features_clean[np.isnan(features_clean)] = 0.0
    probs = model.predict_proba(features_clean)
    return probs[:, 1]


# ═══════════════════════════════════════════════════════════════════════════
#  ROC & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def make_roc(sig_scores, bkg_scores):
    labels = np.concatenate([np.ones(len(sig_scores)), np.zeros(len(bkg_scores))])
    scores = np.concatenate([sig_scores, bkg_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_val


def plot_single_roc(fpr, tpr, auc_val, mass_label, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(tpr, 1.0 / (fpr + 1e-12), lw=2, label=f'AUC = {auc_val:.6f}')
    ax.set_xlabel('Signal Efficiency (TPR)', fontsize=14)
    ax.set_ylabel('Background Rejection (1/FPR)', fontsize=14)
    ax.set_title(f'ROC Curve — mA = {mass_label} MeV', fontsize=16)
    ax.set_yscale('log')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([1, 1e6])
    ax.legend(loc='upper right', fontsize=12); ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_combined_roc(roc_results, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    for mass_label, (fpr, tpr, auc_val) in sorted(roc_results.items()):
        ax.plot(tpr, 1.0 / (fpr + 1e-12), lw=2,
                label=f'mA = {mass_label} MeV (AUC = {auc_val:.6f})')
    ax.set_xlabel('Signal Efficiency (TPR)', fontsize=14)
    ax.set_ylabel('Background Rejection (1/FPR)', fontsize=14)
    ax.set_title('Combined ROC Curves — All Mass Points', fontsize=16)
    ax.set_yscale('log')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([1, 1e6])
    ax.legend(loc='upper right', fontsize=12); ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_combined_roc_standard(roc_results, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    for mass_label, (fpr, tpr, auc_val) in sorted(roc_results.items()):
        ax.plot(fpr, tpr, lw=2,
                label=f'mA = {mass_label} MeV (AUC = {auc_val:.6f})')
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curves — All Mass Points', fontsize=16)
    ax.set_xlim([1e-6, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right', fontsize=12); ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_score_distributions(sig_scores, bkg_scores, mass_label, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bkg_scores, bins=100, range=(0, 1), density=True, alpha=0.6, color='red', label='Background')
    ax.hist(sig_scores, bins=100, range=(0, 1), density=True, alpha=0.6, color='blue',
            label=f'Signal (mA = {mass_label} MeV)')
    ax.set_xlabel('BDT Score', fontsize=14); ax.set_ylabel('Normalized Counts', fontsize=14)
    ax.set_title(f'BDT Score Distribution — mA = {mass_label} MeV', fontsize=16)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Validate combined ECal+HCal BDT with event tracking.")
    parser.add_argument('--model', required=True)
    parser.add_argument('--sig_dir', default='/standard/ldmxuva/gnn_files/note_stuff/validation_root/signal')
    parser.add_argument('--bkg_dir', nargs='+', required=True,
                        help='One or more background directories')
    parser.add_argument('--bkg_run_ranges', nargs='+', default=None,
                        help='Run number ranges per bkg_dir as min:max (use *:* for no filter). '
                             'E.g. --bkg_run_ranges *:* *:* 51000:66000')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--roc_file', default='/home/yun5pc/ldmx/ldmx-sw/Ecal/data/RoC_v14_8gev.csv')
    parser.add_argument('--ecal_threshold', type=int, default=3160)
    parser.add_argument('--hcal_threshold', type=int, default=4440)
    parser.add_argument('--job_index', type=int, default=0, help='This job index (0-based)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Total number of parallel jobs')
    opts = parser.parse_args()

    os.makedirs(opts.out_dir, exist_ok=True)

    # ── Parse run ranges ──
    n_dirs = len(opts.bkg_dir)
    if opts.bkg_run_ranges:
        if len(opts.bkg_run_ranges) != n_dirs:
            print(f"ERROR: Got {len(opts.bkg_run_ranges)} run ranges but {n_dirs} bkg dirs")
            sys.exit(1)
        run_ranges = []
        for rr in opts.bkg_run_ranges:
            parts = rr.split(':')
            rmin = None if parts[0] == '*' else int(parts[0])
            rmax = None if parts[1] == '*' else int(parts[1])
            run_ranges.append((rmin, rmax))
    else:
        run_ranges = [(None, None)] * n_dirs

    # ── Load model & RoC ──
    print(f"Loading model: {opts.model}")
    with open(opts.model, 'rb') as f:
        model = pkl.load(f)
    print(f"Loading RoC: {opts.roc_file}")
    roc_data = tcb.load_roc_file(opts.roc_file)

    # ── Collect & split background files ──
    all_bkg_files = collect_bkg_files(opts.bkg_dir, run_ranges)

    chunk_size = len(all_bkg_files) // opts.n_jobs
    remainder = len(all_bkg_files) % opts.n_jobs
    start = opts.job_index * chunk_size + min(opts.job_index, remainder)
    end = start + chunk_size + (1 if opts.job_index < remainder else 0)
    my_bkg_files = all_bkg_files[start:end]

    print(f"\nJob {opts.job_index+1}/{opts.n_jobs}: background files {start}–{end-1} "
          f"({len(my_bkg_files)} files)")

    # ── Process background ──
    bkg_features, bkg_labels, bkg_filenames, bkg_event_ids = \
        extract_features_with_tracking(
            my_bkg_files, is_signal=False,
            ecal_threshold=opts.ecal_threshold,
            hcal_threshold=opts.hcal_threshold,
            roc_data=roc_data
        )

    if bkg_features is not None:
        bkg_scores = score_events(model, bkg_features)
        print(f"\n  Background events scored: {len(bkg_scores)}")
        print(f"  Mean={np.mean(bkg_scores):.4f}, Median={np.median(bkg_scores):.4f}")

        bkg_out = os.path.join(opts.out_dir, f"bkg_scores_job{opts.job_index}.npz")
        np.savez(bkg_out, scores=bkg_scores, filenames=bkg_filenames, event_indices=bkg_event_ids)
        print(f"  Saved: {bkg_out}")
    else:
        print("  WARNING: No background events passed cuts in this chunk")
        bkg_scores = np.array([])

    # ── Process signal (only in job 0) ──
    if opts.job_index == 0:
        print(f"\nProcessing signal from: {opts.sig_dir}")
        mass_groups = group_files_by_mass(opts.sig_dir)
        print(f"  Found mass points: {sorted(mass_groups.keys())}")

        for mass_str in sorted(mass_groups.keys()):
            files = mass_groups[mass_str]
            print(f"\n  ── Mass point: mA = {mass_str} MeV ({len(files)} files) ──")

            sig_features, sig_labels, sig_filenames, sig_event_ids = \
                extract_features_with_tracking(
                    files, is_signal=True,
                    ecal_threshold=opts.ecal_threshold,
                    hcal_threshold=opts.hcal_threshold,
                    roc_data=roc_data
                )
            if sig_features is None:
                print(f"  WARNING: No signal events for mass {mass_str}")
                continue

            sig_scores = score_events(model, sig_features)
            print(f"    Signal events: {len(sig_scores)}")
            print(f"    Mean={np.mean(sig_scores):.4f}, Median={np.median(sig_scores):.4f}")

            sig_out = os.path.join(opts.out_dir, f"sig_scores_mA_{mass_str}.npz")
            np.savez(sig_out, scores=sig_scores, filenames=sig_filenames, event_indices=sig_event_ids)
            print(f"    Saved: {sig_out}")

    # ── If single job, produce plots directly ──
    if opts.n_jobs == 1 and len(bkg_scores) > 0 and opts.job_index == 0:
        print("\n── Generating ROC curves ──")
        mass_groups = group_files_by_mass(opts.sig_dir)
        roc_results = {}

        for mass_str in sorted(mass_groups.keys()):
            sig_file = os.path.join(opts.out_dir, f"sig_scores_mA_{mass_str}.npz")
            if not os.path.exists(sig_file):
                continue
            sig_scores = np.load(sig_file)['scores']
            fpr, tpr, thresholds, auc_val = make_roc(sig_scores, bkg_scores)
            roc_results[mass_str] = (fpr, tpr, auc_val)
            print(f"  mA = {mass_str} MeV: AUC = {auc_val:.6f}")

            np.savez(os.path.join(opts.out_dir, f"roc_mA_{mass_str}.npz"),
                     fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc_val)
            plot_single_roc(fpr, tpr, auc_val, mass_str,
                           os.path.join(opts.out_dir, f"roc_mA_{mass_str}.png"))
            plot_score_distributions(sig_scores, bkg_scores, mass_str,
                                    os.path.join(opts.out_dir, f"scores_mA_{mass_str}.png"))

        if roc_results:
            all_sig = np.concatenate([np.load(os.path.join(opts.out_dir, f"sig_scores_mA_{m}.npz"))['scores']
                                      for m in sorted(roc_results.keys())])
            fpr_all, tpr_all, thresh_all, auc_all = make_roc(all_sig, bkg_scores)
            print(f"  Combined AUC = {auc_all:.6f}")
            np.savez(os.path.join(opts.out_dir, "roc_combined.npz"),
                     fpr=fpr_all, tpr=tpr_all, thresholds=thresh_all, auc=auc_all)
            plot_combined_roc(roc_results, os.path.join(opts.out_dir, "roc_all_masses_rejection.png"))
            plot_combined_roc_standard(roc_results, os.path.join(opts.out_dir, "roc_all_masses_standard.png"))

    print(f"\n{'='*60}")
    print(f"Job {opts.job_index+1}/{opts.n_jobs} complete.")
    if opts.n_jobs > 1:
        print("Run mergeBDTResults.py after all jobs finish to produce final ROC curves.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()