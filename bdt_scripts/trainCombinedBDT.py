"""
trainCombinedBDT.py
====================
Reads LDMX ROOT files, applies pre-discriminator cuts (PreDiscCuts.py logic),
computes the 47 ECal BDT features from raw EcalRecHits + scoring planes
(reimplementing EcalVetoProcessor::buildBDTFeatureVector in Python), computes
the 11 HCal BDT features (reimplementing EaTVisFeatures::analyze), and trains
a single XGBoost BDT on the combined 58-feature input vector.

Pre-discriminator cuts (from PreDiscCuts.py):
    ECal: sum(non-noise energy) < ecal_threshold  (default 3160 MeV)
    HCal: 12 * sum(section-0 energy) > hcal_threshold  (default 4440 MeV)

Usage
-----
    python3 trainCombinedBDT.py \\
        --bkg_dir  /path/to/background/root/files/ \\
        --sig_dir  /path/to/signal/root/files/ \\
        --out_dir  /scratch/<user>/combined_bdt/ \\
        --roc_file /path/to/RoC_v14_8gev.csv
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import argparse
import glob
import math
import os
import sys
import pickle as pkl

import awkward as ak
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb

# ─── Constants ──────────────────────────────────────────────────────────────
N_ECAL_LAYERS = 34
BEAM_ENERGY_MEV = 8000.0
SEG_LAYERS = [0, 6, 17, 32]   # 3 segments: layers 0-5, 6-16, 17-31 (matching C++)
N_SEGMENTS = 3
N_REGIONS = 5
N_ECAL_FEATURES = 47
N_HCAL_FEATURES = 11

# Approximate ECal hex cell center-to-center distance (mm) for v14 geometry.
# Used to approximate summedTightIso which normally requires hex geometry.
ECAL_CELL_PITCH_MM = 10.0

# ─── Feature Names ─────────────────────────────────────────────────────────
ECAL_FEATURE_NAMES = [
    "nReadoutHits", "summedDet", "summedTightIso", "maxCellDep",
    "showerRMS", "xStd", "yStd", "avgLayerHit", "stdLayerHit",
    "deepestLayerHit", "ecalBackEnergy",
    "nStraight_placeholder", "firstNearPHLayer_placeholder",
    "nNearPHHits_placeholder", "photonTerritoryHits_placeholder",
    "epSep", "epDot",
    "energySeg_0", "xMeanSeg_0", "yMeanSeg_0", "layerMeanSeg_0",
    "energySeg_1", "yMeanSeg_2",
    "eleContEnergy_0_0", "eleContEnergy_1_0", "eleContYMean_0_0",
    "eleContEnergy_0_1", "eleContEnergy_1_1", "eleContYMean_0_1",
    "phContNHits_0_0", "phContYMean_0_0", "phContNHits_0_1",
    "outContEnergy_0_0", "outContEnergy_1_0", "outContEnergy_2_0",
    "outContNHits_0_0", "outContXMean_0_0", "outContYMean_0_0",
    "outContYMean_1_0", "outContYStd_0_0",
    "outContEnergy_0_1", "outContEnergy_1_1", "outContEnergy_2_1",
    "outContLayerMean_0_1", "outContLayerStd_0_1",
    "outContEnergy_0_2", "outContLayerMean_0_2",
]

HCAL_FEATURE_NAMES = [
    "hcal_xMean", "hcal_yMean", "hcal_rMean",
    "hcal_xStd", "hcal_yStd", "hcal_zStd",
    "hcal_numHits", "hcal_layersHit",
    "hcal_isoHits", "hcal_isoEnergy", "hcal_Etot",
]


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def ecal_layers_from_ids(raw_ids):
    """Vectorized: extract ECal layer from packed EcalID (bits 17-22)."""
    return (raw_ids.astype(np.int64) >> 17) & 0x3F

def sim_special_plane(raw_id):
    """Extract plane number from SimSpecialID (lower 20 bits of payload)."""
    return int(raw_id) & 0xFFFFF


# ═══════════════════════════════════════════════════════════════════════════
#  RADII OF CONTAINMENT
# ═══════════════════════════════════════════════════════════════════════════

def load_roc_file(filepath):
    """Load RoC CSV. Rows: [theta_min, theta_max, p_min, p_max, r0..r33]."""
    roc_data = []
    with open(filepath) as f:
        f.readline()  # skip header
        for line in f:
            vals = [float(v.strip()) if v.strip() != '' else -1.0
                    for v in line.strip().split(',')]
            roc_data.append(vals)
    return roc_data


def get_containment_radii(roc_data, recoil_theta, recoil_p_mag):
    """Determine electron and photon containment radii per layer."""
    ele_radii = roc_data[0][4:]
    for row in roc_data:
        theta_min, theta_max, p_min, p_max = row[0], row[1], row[2], row[3]
        inrange = True
        if theta_min != -1.0: inrange = inrange and (recoil_theta >= theta_min)
        if theta_max != -1.0: inrange = inrange and (recoil_theta < theta_max)
        if p_min != -1.0:     inrange = inrange and (recoil_p_mag >= p_min)
        if p_max != -1.0:     inrange = inrange and (recoil_p_mag < p_max)
        if inrange:
            ele_radii = row[4:]
    photon_radii = roc_data[0][4:]  # photon always uses default bin
    return np.array(ele_radii, dtype=np.float64), np.array(photon_radii, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER Z-POSITION LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

def build_layer_z_lookup(tree, pass_name, n_events=200):
    """Build dict mapping ECal layer → sensor z position from data."""
    id_key = f"EcalRecHits_{pass_name}.id_"
    z_key  = f"EcalRecHits_{pass_name}.zpos_"
    arrays = tree.arrays([id_key, z_key], library="ak", entry_stop=n_events)
    all_ids = ak.to_numpy(ak.flatten(arrays[id_key])).astype(np.int64)
    all_z   = ak.to_numpy(ak.flatten(arrays[z_key]))
    layers  = (all_ids >> 17) & 0x3F
    layer_z = {}
    for l, z in zip(layers, all_z):
        if l not in layer_z:
            layer_z[int(l)] = float(z)
    if len(layer_z) < N_ECAL_LAYERS:
        print(f"  WARNING: only found {len(layer_z)}/{N_ECAL_LAYERS} ECal layers in z-lookup")
    return layer_z


# ═══════════════════════════════════════════════════════════════════════════
#  TRAJECTORY PROJECTION
# ═══════════════════════════════════════════════════════════════════════════

def project_trajectory(momentum, position, layer_z):
    """Project straight-line trajectory to each ECal layer. Returns dict layer→(x,y)."""
    if momentum[2] <= 0:
        return {}
    traj = {}
    for i_layer in range(N_ECAL_LAYERS):
        if i_layer not in layer_z:
            continue
        z = layer_z[i_layer]
        traj[i_layer] = (
            position[0] + (momentum[0] / momentum[2]) * (z - position[2]),
            position[1] + (momentum[1] / momentum[2]) * (z - position[2]),
        )
    return traj


# ═══════════════════════════════════════════════════════════════════════════
#  RECOIL ELECTRON FINDING
# ═══════════════════════════════════════════════════════════════════════════

def find_recoil_track_id(evt_track_ids, evt_pdg_ids, evt_parents):
    """Find recoil electron: PDG==11 with parent track ID 0."""
    for i in range(len(evt_track_ids)):
        if int(evt_pdg_ids[i]) == 11:
            for p in evt_parents[i]:
                if int(p) == 0:
                    return int(evt_track_ids[i])
    return None


def find_sp_recoil(sp_ids, sp_tids, sp_px, sp_py, sp_pz,
                   sp_x, sp_y, sp_z, recoil_tid, target_plane):
    """Find recoil's scoring-plane hit at target_plane. Returns (p_vec, pos_vec) or (None,None)."""
    best_p, best_pos, pmax = None, None, 0.0
    for i in range(len(sp_ids)):
        if sim_special_plane(int(sp_ids[i])) != target_plane: continue
        pz = float(sp_pz[i])
        if pz <= 0: continue
        if int(sp_tids[i]) != recoil_tid: continue
        px, py = float(sp_px[i]), float(sp_py[i])
        pmag = math.sqrt(px*px + py*py + pz*pz)
        if pmag > pmax:
            pmax = pmag
            best_p   = [px, py, pz]
            best_pos = [float(sp_x[i]), float(sp_y[i]), float(sp_z[i])]
    return best_p, best_pos


# ═══════════════════════════════════════════════════════════════════════════
#  PRE-DISCRIMINATOR CUTS (from PreDiscCuts.py)
# ═══════════════════════════════════════════════════════════════════════════

def apply_pre_discriminator_cuts(tree, ecal_rec_pass, hcal_rec_pass,
                                  is_noise_name, ecal_threshold, hcal_threshold):
    branches = [
        f"EcalRecHits_{ecal_rec_pass}.energy_",
        f"EcalRecHits_{ecal_rec_pass}.{is_noise_name}",
        f"HcalRecHits_{hcal_rec_pass}.energy_",
        f"HcalRecHits_{hcal_rec_pass}.section_",
    ]
    arrays = tree.arrays(branches, library="ak")
    ecal_energy = arrays[f"EcalRecHits_{ecal_rec_pass}.energy_"]
    ecal_noise  = arrays[f"EcalRecHits_{ecal_rec_pass}.{is_noise_name}"]
    ecal_pass = ak.sum(ecal_energy * (~ecal_noise), axis=1) < ecal_threshold
    hcal_energy  = arrays[f"HcalRecHits_{hcal_rec_pass}.energy_"]
    hcal_section = arrays[f"HcalRecHits_{hcal_rec_pass}.section_"]
    hcal_pass = 12 * ak.sum(hcal_energy * (hcal_section == 0), axis=1) > hcal_threshold
    return ecal_pass & hcal_pass


# ═══════════════════════════════════════════════════════════════════════════
#  ECAL BDT FEATURES — per-event computation (47 features)
# ═══════════════════════════════════════════════════════════════════════════

def compute_ecal_features_event(hit_ids, hit_energy, hit_x, hit_y, hit_z, hit_time,
                                 ele_traj, photon_traj,
                                 ele_radii, photon_radii,
                                 ep_sep, ep_dot, layer_z):
    """Compute 47 ECal BDT features for one event. Returns np array."""
    feats = np.zeros(N_ECAL_FEATURES, dtype=np.float64)

    # Filter hits with energy > 0
    good = hit_energy > 0
    e = hit_energy[good]; x = hit_x[good]; y = hit_y[good]
    z = hit_z[good]; ids = hit_ids[good]; t = hit_time[good]
    layers = ecal_layers_from_ids(ids)
    n = len(e)

    if n == 0:
        feats[11:15] = -1.0; feats[15] = ep_sep; feats[16] = ep_dot
        return feats

    summed_det = np.sum(e)
    max_cell_dep = np.max(e)

    # ── Shower centroid & RMS ──
    cx = np.sum(x * e) / summed_det
    cy = np.sum(y * e) / summed_det
    delta_r = np.sqrt((x - cx)**2 + (y - cy)**2)
    shower_rms = np.sum(delta_r * e) / summed_det

    # ── summedTightIso (position-based approximation of hex-grid isolation) ──
    centroid_idx = np.argmin(delta_r)
    centroid_x, centroid_y = x[centroid_idx], y[centroid_idx]
    tight_iso_sum = 0.0
    for i in range(n):
        if math.sqrt((x[i]-centroid_x)**2 + (y[i]-centroid_y)**2) < 1.5 * ECAL_CELL_PITCH_MM:
            continue
        same_layer = (layers == layers[i])
        same_layer[i] = False
        if np.any(same_layer):
            dists = np.sqrt((x[same_layer]-x[i])**2 + (y[same_layer]-y[i])**2)
            if np.min(dists) <= ECAL_CELL_PITCH_MM:
                continue
        tight_iso_sum += e[i]

    # ── Layer statistics ──
    fl = layers.astype(np.float64)
    avg_layer = np.mean(fl)                              # unweighted
    w_avg_layer = np.sum(fl * e) / summed_det            # energy-weighted
    deepest_layer = int(np.max(layers))
    ecal_back_energy = np.sum(e[layers >= 20])

    # ── Energy-weighted position means & stds ──
    x_mean = np.sum(x * e) / summed_det
    y_mean = np.sum(y * e) / summed_det
    x_std = math.sqrt(np.sum(e * (x - x_mean)**2) / summed_det)
    y_std = math.sqrt(np.sum(e * (y - y_mean)**2) / summed_det)
    std_layer = math.sqrt(np.sum(e * (fl - w_avg_layer)**2) / summed_det)

    # ── Trajectory distances per hit ──
    has_traj = len(ele_traj) > 0 and len(photon_traj) > 0
    dist_ele = np.full(n, -1.0)
    dist_ph  = np.full(n, -1.0)
    if has_traj:
        for i in range(n):
            l = int(layers[i])
            if l in ele_traj:
                ex, ey = ele_traj[l]
                dist_ele[i] = math.sqrt((x[i]-ex)**2 + (y[i]-ey)**2)
            if l in photon_traj:
                px_, py_ = photon_traj[l]
                dist_ph[i] = math.sqrt((x[i]-px_)**2 + (y[i]-py_)**2)

    # Per-hit radii
    hit_er = np.array([ele_radii[int(l)] if int(l) < len(ele_radii) else 0.0 for l in layers])
    hit_pr = np.array([photon_radii[int(l)] if int(l) < len(photon_radii) else 0.0 for l in layers])

    # ── Segment variables ──
    energy_seg = np.zeros(N_SEGMENTS)
    x_mean_seg = np.zeros(N_SEGMENTS)
    y_mean_seg = np.zeros(N_SEGMENTS)
    layer_mean_seg = np.zeros(N_SEGMENTS)
    for iseg in range(N_SEGMENTS):
        m = (layers >= SEG_LAYERS[iseg]) & (layers <= SEG_LAYERS[iseg+1] - 1)
        if np.any(m):
            energy_seg[iseg] = np.sum(e[m])
            if energy_seg[iseg] > 0:
                x_mean_seg[iseg] = np.sum(x[m]*e[m]) / energy_seg[iseg]
                y_mean_seg[iseg] = np.sum(y[m]*e[m]) / energy_seg[iseg]
                layer_mean_seg[iseg] = np.sum(fl[m]*e[m]) / energy_seg[iseg]

    # ── Containment variables — first pass (sums and means) ──
    ec_e = np.zeros((N_REGIONS, N_SEGMENTS)); ec_ym = np.zeros((N_REGIONS, N_SEGMENTS))
    gc_e = np.zeros((N_REGIONS, N_SEGMENTS)); gc_n = np.zeros((N_REGIONS, N_SEGMENTS), dtype=np.int32)
    gc_ym = np.zeros((N_REGIONS, N_SEGMENTS))
    oc_e = np.zeros((N_REGIONS, N_SEGMENTS)); oc_n = np.zeros((N_REGIONS, N_SEGMENTS), dtype=np.int32)
    oc_xm = np.zeros((N_REGIONS, N_SEGMENTS)); oc_ym = np.zeros((N_REGIONS, N_SEGMENTS))
    oc_lm = np.zeros((N_REGIONS, N_SEGMENTS))

    for i in range(n):
        l = int(layers[i]); ei = e[i]; xi = x[i]; yi = y[i]
        de = dist_ele[i]; dp = dist_ph[i]; re = hit_er[i]; rp = hit_pr[i]
        for iseg in range(N_SEGMENTS):
            if not (SEG_LAYERS[iseg] <= l <= SEG_LAYERS[iseg+1]-1): continue
            for ireg in range(N_REGIONS):
                if de >= 0 and de >= ireg*re and de < (ireg+1)*re:
                    ec_e[ireg,iseg] += ei
                    ec_ym[ireg,iseg] += yi*ei
                if dp >= 0 and dp >= ireg*rp and dp < (ireg+1)*rp:
                    gc_e[ireg,iseg] += ei; gc_n[ireg,iseg] += 1
                    gc_ym[ireg,iseg] += yi*ei
                if de >= 0 and dp >= 0 and de > (ireg+1)*re and dp > (ireg+1)*rp:
                    oc_e[ireg,iseg] += ei; oc_n[ireg,iseg] += 1
                    oc_xm[ireg,iseg] += xi*ei; oc_ym[ireg,iseg] += yi*ei
                    oc_lm[ireg,iseg] += l*ei

    # Normalize means
    for iseg in range(N_SEGMENTS):
        for ireg in range(N_REGIONS):
            if ec_e[ireg,iseg] > 0: ec_ym[ireg,iseg] /= ec_e[ireg,iseg]
            if gc_e[ireg,iseg] > 0: gc_ym[ireg,iseg] /= gc_e[ireg,iseg]
            if oc_e[ireg,iseg] > 0:
                oc_xm[ireg,iseg] /= oc_e[ireg,iseg]
                oc_ym[ireg,iseg] /= oc_e[ireg,iseg]
                oc_lm[ireg,iseg] /= oc_e[ireg,iseg]

    # ── Second pass: outside containment stds ──
    oc_ys = np.zeros((N_REGIONS, N_SEGMENTS))
    oc_ls = np.zeros((N_REGIONS, N_SEGMENTS))
    for i in range(n):
        l = int(layers[i]); ei = e[i]; xi = x[i]; yi = y[i]
        de = dist_ele[i]; dp = dist_ph[i]; re = hit_er[i]; rp = hit_pr[i]
        for iseg in range(N_SEGMENTS):
            if not (SEG_LAYERS[iseg] <= l <= SEG_LAYERS[iseg+1]-1): continue
            for ireg in range(N_REGIONS):
                if de >= 0 and dp >= 0 and de > (ireg+1)*re and dp > (ireg+1)*rp:
                    oc_ys[ireg,iseg] += (yi - oc_ym[ireg,iseg])**2 * ei
                    oc_ls[ireg,iseg] += (l  - oc_lm[ireg,iseg])**2 * ei

    for iseg in range(N_SEGMENTS):
        for ireg in range(N_REGIONS):
            if oc_e[ireg,iseg] > 0:
                oc_ys[ireg,iseg] = math.sqrt(oc_ys[ireg,iseg] / oc_e[ireg,iseg])
                oc_ls[ireg,iseg] = math.sqrt(oc_ls[ireg,iseg] / oc_e[ireg,iseg])

    # ── Assemble 47-feature vector ──
    feats[0]=n; feats[1]=summed_det; feats[2]=tight_iso_sum; feats[3]=max_cell_dep
    feats[4]=shower_rms; feats[5]=x_std; feats[6]=y_std
    feats[7]=avg_layer; feats[8]=std_layer; feats[9]=deepest_layer; feats[10]=ecal_back_energy
    feats[11:15] = -1.0
    feats[15]=ep_sep; feats[16]=ep_dot
    feats[17]=energy_seg[0]; feats[18]=x_mean_seg[0]; feats[19]=y_mean_seg[0]
    feats[20]=layer_mean_seg[0]; feats[21]=energy_seg[1]; feats[22]=y_mean_seg[2]
    feats[23]=ec_e[0,0]; feats[24]=ec_e[1,0]; feats[25]=ec_ym[0,0]
    feats[26]=ec_e[0,1]; feats[27]=ec_e[1,1]; feats[28]=ec_ym[0,1]
    feats[29]=gc_n[0,0]; feats[30]=gc_ym[0,0]; feats[31]=gc_n[0,1]
    feats[32]=oc_e[0,0]; feats[33]=oc_e[1,0]; feats[34]=oc_e[2,0]
    feats[35]=oc_n[0,0]; feats[36]=oc_xm[0,0]; feats[37]=oc_ym[0,0]
    feats[38]=oc_ym[1,0]; feats[39]=oc_ys[0,0]
    feats[40]=oc_e[0,1]; feats[41]=oc_e[1,1]; feats[42]=oc_e[2,1]
    feats[43]=oc_lm[0,1]; feats[44]=oc_ls[0,1]
    feats[45]=oc_e[0,2]; feats[46]=oc_lm[0,2]
    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  ECAL FEATURES — file-level extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_ecal_features(tree, pass_name, event_mask, roc_data, layer_z):
    """Extract 47 ECal BDT features for all passing events."""
    n_pass = int(ak.sum(event_mask))

    # ── Read all branches at once ──
    ecal_b  = [f"EcalRecHits_{pass_name}.{k}" for k in ["id_","energy_","xpos_","ypos_","zpos_","time_"]]
    esp_b   = [f"EcalScoringPlaneHits_{pass_name}.{k}" for k in ["id_","track_id_","px_","py_","pz_","x_","y_","z_"]]
    tsp_b   = [f"TargetScoringPlaneHits_{pass_name}.{k}" for k in ["id_","track_id_","px_","py_","pz_","x_","y_","z_"]]
    sim_b   = [f"SimParticles_{pass_name}.first",
               f"SimParticles_{pass_name}.second.pdg_id_",
               f"SimParticles_{pass_name}.second.parents_"]

    arrays = tree.arrays(ecal_b + esp_b + tsp_b + sim_b, library="ak")
    pn = pass_name  # shorthand

    # Apply mask
    def m(key): return arrays[key][event_mask]

    ecal_id=m(f"EcalRecHits_{pn}.id_"); ecal_e=m(f"EcalRecHits_{pn}.energy_")
    ecal_x=m(f"EcalRecHits_{pn}.xpos_"); ecal_y=m(f"EcalRecHits_{pn}.ypos_")
    ecal_z=m(f"EcalRecHits_{pn}.zpos_"); ecal_t=m(f"EcalRecHits_{pn}.time_")

    esp_id=m(f"EcalScoringPlaneHits_{pn}.id_"); esp_tid=m(f"EcalScoringPlaneHits_{pn}.track_id_")
    esp_px=m(f"EcalScoringPlaneHits_{pn}.px_"); esp_py=m(f"EcalScoringPlaneHits_{pn}.py_")
    esp_pz=m(f"EcalScoringPlaneHits_{pn}.pz_")
    esp_x=m(f"EcalScoringPlaneHits_{pn}.x_"); esp_y=m(f"EcalScoringPlaneHits_{pn}.y_")
    esp_z=m(f"EcalScoringPlaneHits_{pn}.z_")

    tsp_id=m(f"TargetScoringPlaneHits_{pn}.id_"); tsp_tid=m(f"TargetScoringPlaneHits_{pn}.track_id_")
    tsp_px=m(f"TargetScoringPlaneHits_{pn}.px_"); tsp_py=m(f"TargetScoringPlaneHits_{pn}.py_")
    tsp_pz=m(f"TargetScoringPlaneHits_{pn}.pz_")
    tsp_x=m(f"TargetScoringPlaneHits_{pn}.x_"); tsp_y=m(f"TargetScoringPlaneHits_{pn}.y_")
    tsp_z=m(f"TargetScoringPlaneHits_{pn}.z_")

    sim_tids=m(f"SimParticles_{pn}.first"); sim_pdgs=m(f"SimParticles_{pn}.second.pdg_id_")
    sim_pars=m(f"SimParticles_{pn}.second.parents_")

    features = np.zeros((n_pass, N_ECAL_FEATURES), dtype=np.float32)

    for evt in range(n_pass):
        # ── Numpy-ify event arrays ──
        eid = ak.to_numpy(ecal_id[evt]).astype(np.int64)
        ee  = ak.to_numpy(ecal_e[evt]).astype(np.float64)
        ex  = ak.to_numpy(ecal_x[evt]).astype(np.float64)
        ey  = ak.to_numpy(ecal_y[evt]).astype(np.float64)
        ez  = ak.to_numpy(ecal_z[evt]).astype(np.float64)
        et  = ak.to_numpy(ecal_t[evt]).astype(np.float64)

        # ── Find recoil electron ──
        recoil_tid = find_recoil_track_id(
            ak.to_numpy(sim_tids[evt]),
            ak.to_numpy(sim_pdgs[evt]),
            sim_pars[evt]
        )

        rp = rpos = rp_tgt = rpos_tgt = None
        if recoil_tid is not None:
            rp, rpos = find_sp_recoil(
                ak.to_numpy(esp_id[evt]).astype(np.int64), ak.to_numpy(esp_tid[evt]),
                ak.to_numpy(esp_px[evt]), ak.to_numpy(esp_py[evt]), ak.to_numpy(esp_pz[evt]),
                ak.to_numpy(esp_x[evt]), ak.to_numpy(esp_y[evt]), ak.to_numpy(esp_z[evt]),
                recoil_tid, 31)
            rp_tgt, rpos_tgt = find_sp_recoil(
                ak.to_numpy(tsp_id[evt]).astype(np.int64), ak.to_numpy(tsp_tid[evt]),
                ak.to_numpy(tsp_px[evt]), ak.to_numpy(tsp_py[evt]), ak.to_numpy(tsp_pz[evt]),
                ak.to_numpy(tsp_x[evt]), ak.to_numpy(tsp_y[evt]), ak.to_numpy(tsp_z[evt]),
                recoil_tid, 1)

        # ── Project trajectories ──
        ele_traj = {}; photon_traj = {}
        ep_sep = 999.0; ep_dot = 999.0
        recoil_p_mag = -1.0; recoil_theta = -1.0

        have_both = (rp is not None and rpos is not None and
                     rp_tgt is not None and rpos_tgt is not None and
                     rp[2] > 0 and rp_tgt[2] > 0)
        if have_both:
            ele_traj = project_trajectory(rp, rpos, layer_z)
            photon_mom = [-rp_tgt[0], -rp_tgt[1], BEAM_ENERGY_MEV - rp_tgt[2]]
            photon_traj = project_trajectory(photon_mom, rpos_tgt, layer_z)

            recoil_p_mag = math.sqrt(sum(p*p for p in rp))
            if recoil_p_mag > 0:
                recoil_theta = math.acos(rp[2] / recoil_p_mag) * 180.0 / math.pi

            if ele_traj and photon_traj:
                common = sorted(set(ele_traj.keys()) & set(photon_traj.keys()))
                if len(common) >= 2:
                    fl, ll = common[0], common[-1]
                    es = np.array([ele_traj[fl][0], ele_traj[fl][1], layer_z[fl]])
                    ee_ = np.array([ele_traj[ll][0], ele_traj[ll][1], layer_z[ll]])
                    ps = np.array([photon_traj[fl][0], photon_traj[fl][1], layer_z[fl]])
                    pe = np.array([photon_traj[ll][0], photon_traj[ll][1], layer_z[ll]])
                    ev = ee_ - es; pv = pe - ps
                    en = ev / (np.linalg.norm(ev)+1e-12)
                    pn_ = pv / (np.linalg.norm(pv)+1e-12)
                    ep_dot = float(np.dot(en, pn_))
                    ep_sep = math.sqrt((es[0]-ps[0])**2 + (es[1]-ps[1])**2)

        ele_radii, photon_radii = get_containment_radii(roc_data, recoil_theta, recoil_p_mag)

        features[evt] = compute_ecal_features_event(
            eid, ee, ex, ey, ez, et,
            ele_traj, photon_traj, ele_radii, photon_radii,
            ep_sep, ep_dot, layer_z)

        if (evt+1) % 5000 == 0:
            print(f"    ECal features: {evt+1}/{n_pass} events")

    return features


# ═══════════════════════════════════════════════════════════════════════════
#  HCAL BDT FEATURES (11 features, from EaTVisFeatures::analyze)
# ═══════════════════════════════════════════════════════════════════════════

def extract_hcal_features(tree, hcal_rec_pass, event_mask):
    prefix = f"HcalRecHits_{hcal_rec_pass}"
    hcal_branches = [f"{prefix}.{b}" for b in ["energy_","xpos_","ypos_","zpos_","layer_","section_"]]
    arrays = tree.arrays(hcal_branches, library="ak")
    energy=arrays[f"{prefix}.energy_"][event_mask]; xpos=arrays[f"{prefix}.xpos_"][event_mask]
    ypos=arrays[f"{prefix}.ypos_"][event_mask]; zpos=arrays[f"{prefix}.zpos_"][event_mask]
    layer=arrays[f"{prefix}.layer_"][event_mask]; section=arrays[f"{prefix}.section_"][event_mask]

    n_pass = int(ak.sum(event_mask))
    features = np.zeros((n_pass, N_HCAL_FEATURES), dtype=np.float32)

    for ei in range(n_pass):
        eE=ak.to_numpy(energy[ei]); eX=ak.to_numpy(xpos[ei]); eY=ak.to_numpy(ypos[ei])
        eZ=ak.to_numpy(zpos[ei]); eL=ak.to_numpy(layer[ei]); eS=ak.to_numpy(section[ei])

        good = (eE>0)&(eS==0)&(np.abs(eX)<=1000)&(np.abs(eY)<=1000)
        e_=eE[good]; x_=eX[good]; y_=eY[good]; z_=eZ[good]; l_=eL[good]

        all_back = (eE>0)&(eS==0)
        hcalE = 12.0*np.sum(eE[all_back])
        nh = len(e_)
        if nh == 0:
            features[ei,10] = hcalE; continue

        es = np.sum(e_); r_ = np.sqrt(x_**2+y_**2)
        if es > 0:
            xm=np.sum(x_*e_)/es; ym=np.sum(y_*e_)/es; zm=np.sum(z_*e_)/es; rm=np.sum(r_*e_)/es
            xs=np.sqrt(np.sum(e_*(x_-xm)**2)/es); ys=np.sqrt(np.sum(e_*(y_-ym)**2)/es)
            zs=np.sqrt(np.sum(e_*(z_-zm)**2)/es)
        else:
            xm=ym=zm=rm=xs=ys=zs=0.0

        lh = len(np.unique(l_))
        ih=0; ie=0.0
        for k in range(nh):
            cl=9999.0
            for j in range(nh):
                if l_[j]!=l_[k]: continue
                d=abs(y_[j]-y_[k]) if l_[k]%2==0 else abs(x_[j]-x_[k])
                if d>0 and d<cl: cl=d
            if cl>50.0: ih+=1; ie+=e_[k]

        features[ei]=[xm,ym,rm,xs,ys,zs,nh,lh,ih,ie,hcalE]
        if (ei+1)%5000==0: print(f"    HCal features: {ei+1}/{n_pass} events")

    return features


# ═══════════════════════════════════════════════════════════════════════════
#  FILE / DIRECTORY PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_root_file(filepath, is_signal, ecal_threshold, hcal_threshold, roc_data):
    pass_name = 'eat_vis' if is_signal else 'eat'
    with uproot.open(filepath) as f:
        tree = f["LDMX_Events"]
        total = tree.num_entries
        event_mask = apply_pre_discriminator_cuts(
            tree, pass_name, pass_name, 'is_noise_', ecal_threshold, hcal_threshold)
        n_pass = int(ak.sum(event_mask))
        if n_pass == 0: return None, None
        print(f"  {os.path.basename(filepath)}: {n_pass}/{total} pass cuts ({100.*n_pass/total:.2f}%)")

        layer_z = build_layer_z_lookup(tree, pass_name)
        ecal_feats = extract_ecal_features(tree, pass_name, event_mask, roc_data, layer_z)
        hcal_feats = extract_hcal_features(tree, pass_name, event_mask)
        combined = np.hstack([ecal_feats, hcal_feats])
        labels = np.full(n_pass, int(is_signal), dtype=np.float32)
        return combined, labels


def process_directory(directory, is_signal, ecal_threshold, hcal_threshold, roc_data):
    root_files = sorted(glob.glob(os.path.join(directory, "*.root")))
    if not root_files:
        print(f"WARNING: No .root files found in {directory}"); return None, None
    all_f, all_l = [], []
    label = "signal" if is_signal else "background"
    print(f"\nProcessing {len(root_files)} {label} files from:\n  {directory}")
    for fp in root_files:
        try:
            feats, labs = process_root_file(fp, is_signal, ecal_threshold, hcal_threshold, roc_data)
            if feats is not None: all_f.append(feats); all_l.append(labs)
        except Exception as ex:
            print(f"  ERROR processing {os.path.basename(fp)}: {ex}"); continue
    if not all_f: return None, None
    return np.vstack(all_f), np.concatenate(all_l)


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def train_bdt(sig_f, sig_l, bkg_f, bkg_l, opts):
    tx = np.vstack([sig_f, bkg_f]); ty = np.concatenate([sig_l, bkg_l])
    tx[np.isnan(tx)] = 0.0; ty[np.isnan(ty)] = 0.0
    idx = np.random.default_rng(opts.seed).permutation(len(ty))
    tx = tx[idx]; ty = ty[idx]

    nf = N_ECAL_FEATURES + N_HCAL_FEATURES
    print(f"\n{'='*60}\nTraining combined ECal+HCal BDT\n{'='*60}")
    print(f"  Features: {nf} ({N_ECAL_FEATURES} ECal + {N_HCAL_FEATURES} HCal)")
    print(f"  Signal: {int(np.sum(ty==1))}  Background: {int(np.sum(ty==0))}")
    print(f"  eta={opts.eta}  depth={opts.depth}  trees={opts.tree_number}\n")

    gbm = xgb.XGBClassifier(
        max_depth=opts.depth, learning_rate=opts.eta, n_estimators=opts.tree_number,
        verbosity=1, min_child_weight=5, eval_metric='auc', scale_pos_weight=0.25,
        subsample=0.9, colsample_bytree=0.85, seed=opts.seed, nthread=1)
    gbm.fit(tx, ty)
    print(f"\nDone. Shape={tx.shape}  Labels={np.unique(ty, return_counts=True)}")
    print(f"  Features in trees: {len(gbm.get_booster().get_score())}")
    return gbm


def save_outputs(gbm, out_dir, out_name):
    adds = 0
    while True:
        c = os.path.join(out_dir, f"{out_name}_{adds}")
        if not os.path.exists(c):
            try: os.makedirs(c); break
            except: pass
        adds += 1
    fd = os.path.join(out_dir, f"{out_name}_{adds}")
    pp = os.path.join(fd, f"{out_name}_{adds}_weights.pkl")
    with open(pp, 'wb') as f: pkl.dump(gbm, f)
    fig, ax = plt.subplots(figsize=(12,20))
    xgb.plot_importance(gbm, ax=ax)
    fp = os.path.join(fd, f"{out_name}_{adds}_fimportance.png")
    fig.savefig(fp, dpi=500, bbox_inches='tight', pad_inches=0.5); plt.close(fig)
    np_ = os.path.join(fd, f"{out_name}_{adds}_feature_names.txt")
    with open(np_, 'w') as f:
        for i, nm in enumerate(ECAL_FEATURE_NAMES + HCAL_FEATURE_NAMES):
            f.write(f"{i}: {nm}\n")
    print(f"\nSaved to: {fd}")
    return fd


def convert_to_onnx(pkl_path, onnx_path, n_features):
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        with open(pkl_path,'rb') as f: model=pkl.load(f)
        onnx_model = convert_xgboost(model, initial_types=[('float_input',FloatTensorType([None,n_features]))])
        with open(onnx_path,'wb') as f: f.write(onnx_model.SerializeToString())
        print(f"  ONNX: {onnx_path}")
    except ImportError:
        print("  WARNING: onnxmltools not installed, skipping ONNX conversion.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train combined ECal+HCal BDT.")
    parser.add_argument('--bkg_dir', required=True)
    parser.add_argument('--sig_dir', required=True)
    parser.add_argument('--out_dir', default='/scratch/combined_bdt/')
    parser.add_argument('--out_name', default='combined_bdt')
    parser.add_argument('--roc_file', default='/home/yun5pc/ldmx/ldmx-sw/Ecal/data/RoC_v14_8gev.csv')
    parser.add_argument('--ecal_threshold', type=int, default=3160)
    parser.add_argument('--hcal_threshold', type=int, default=4440)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--eta', type=float, default=0.023)
    parser.add_argument('--tree_number', type=int, default=1000)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--convert_onnx', action='store_true')
    parser.add_argument('--onnx_path', default=None)
    opts = parser.parse_args()
    np.random.seed(opts.seed)

    print(f"Loading RoC: {opts.roc_file}")
    roc = load_roc_file(opts.roc_file)
    print(f"  {len(roc)} bins, {len(roc[0])-4} radii/bin")

    sf, sl = process_directory(opts.sig_dir, True, opts.ecal_threshold, opts.hcal_threshold, roc)
    if sf is None: print("ERROR: No signal events."); sys.exit(1)
    bf, bl = process_directory(opts.bkg_dir, False, opts.ecal_threshold, opts.hcal_threshold, roc)
    if bf is None: print("ERROR: No background events."); sys.exit(1)

    gbm = train_bdt(sf, sl, bf, bl, opts)
    fd = save_outputs(gbm, opts.out_dir, opts.out_name)

    if opts.convert_onnx:
        pp = [f for f in os.listdir(fd) if f.endswith('_weights.pkl')][0]
        convert_to_onnx(os.path.join(fd,pp), opts.onnx_path or os.path.join(fd,"combined_bdt.onnx"),
                        N_ECAL_FEATURES+N_HCAL_FEATURES)

if __name__ == '__main__':
    main()
