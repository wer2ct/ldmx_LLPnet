"""
mergeBDTResults.py
==================
Merges background score files from parallel validateBDT.py jobs,
combines with signal scores, and produces final ROC curves + plots.

Usage:
    python3 mergeBDTResults.py --input_dir /scratch/yun5pc/bdt_3e13/ --out_dir /scratch/yun5pc/bdt_3e13/
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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


def main():
    parser = argparse.ArgumentParser(description="Merge parallel BDT validation results.")
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing bkg_scores_job*.npz and sig_scores_mA_*.npz')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory for plots and final ROC (default: same as input_dir)')
    opts = parser.parse_args()

    out_dir = opts.out_dir or opts.input_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Load and merge background scores ──
    bkg_files = sorted(glob.glob(os.path.join(opts.input_dir, "bkg_scores_job*.npz")))
    if not bkg_files:
        print("ERROR: No bkg_scores_job*.npz files found."); sys.exit(1)

    print(f"Found {len(bkg_files)} background score files:")
    all_bkg_scores = []
    all_bkg_filenames = []
    all_bkg_event_ids = []
    for bf in bkg_files:
        d = np.load(bf, allow_pickle=True)
        scores = d['scores']
        all_bkg_scores.append(scores)
        all_bkg_filenames.append(d['filenames'])
        all_bkg_event_ids.append(d['event_indices'])
        print(f"  {os.path.basename(bf)}: {len(scores)} events")

    bkg_scores = np.concatenate(all_bkg_scores)
    bkg_filenames = np.concatenate(all_bkg_filenames)
    bkg_event_ids = np.concatenate(all_bkg_event_ids)
    print(f"\nTotal background events: {len(bkg_scores)}")
    print(f"  Mean score={np.mean(bkg_scores):.6f}, Median={np.median(bkg_scores):.6f}")

    # Save merged background metadata
    np.savez(os.path.join(out_dir, "bkg_scores_merged.npz"),
             scores=bkg_scores, filenames=bkg_filenames, event_indices=bkg_event_ids)

    # ── Load signal scores ──
    sig_files = sorted(glob.glob(os.path.join(opts.input_dir, "sig_scores_mA_*.npz")))
    if not sig_files:
        print("ERROR: No sig_scores_mA_*.npz files found."); sys.exit(1)

    print(f"\nFound {len(sig_files)} signal score files")
    roc_results = {}
    all_sig_scores = []

    for sf in sig_files:
        mass_str = os.path.basename(sf).replace("sig_scores_mA_", "").replace(".npz", "")
        d = np.load(sf, allow_pickle=True)
        sig_scores = d['scores']
        print(f"\n  mA = {mass_str} MeV: {len(sig_scores)} events")
        print(f"    Mean score={np.mean(sig_scores):.6f}, Median={np.median(sig_scores):.6f}")

        all_sig_scores.append(sig_scores)

        # ── ROC for this mass point ──
        fpr, tpr, thresholds, auc_val = make_roc(sig_scores, bkg_scores)
        roc_results[mass_str] = (fpr, tpr, auc_val)
        print(f"    AUC = {auc_val:.6f}")

        # Save ROC arrays
        np.savez(os.path.join(out_dir, f"roc_mA_{mass_str}.npz"),
                 fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc_val)

        # Plots
        plot_single_roc(fpr, tpr, auc_val, mass_str,
                       os.path.join(out_dir, f"roc_mA_{mass_str}.png"))
        plot_score_distributions(sig_scores, bkg_scores, mass_str,
                                os.path.join(out_dir, f"scores_mA_{mass_str}.png"))

    # ── Combined ROC ──
    if all_sig_scores:
        combined_sig = np.concatenate(all_sig_scores)
        fpr_all, tpr_all, thresh_all, auc_all = make_roc(combined_sig, bkg_scores)
        print(f"\n  Combined AUC = {auc_all:.6f}")
        np.savez(os.path.join(out_dir, "roc_combined.npz"),
                 fpr=fpr_all, tpr=tpr_all, thresholds=thresh_all, auc=auc_all)
        plot_combined_roc(roc_results, os.path.join(out_dir, "roc_all_masses_rejection.png"))
        plot_combined_roc_standard(roc_results, os.path.join(out_dir, "roc_all_masses_standard.png"))

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Merge complete. Outputs in: {out_dir}")
    print(f"{'='*60}")
    print(f"\nBackground: {len(bkg_scores)} events from {len(bkg_files)} job chunks")
    for mass_str in sorted(roc_results.keys()):
        fpr, tpr, auc_val = roc_results[mass_str]
        print(f"  mA = {mass_str} MeV : AUC = {auc_val:.6f}")
    if all_sig_scores:
        print(f"  Combined         : AUC = {auc_all:.6f}")

    print(f"\nFiles produced:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == '__main__':
    main()
