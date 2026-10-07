import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import scipy
import json
from thoi.commons import _normalize_input_data
import torch
import h5py

EXCLUDED_SOURCE_SUBJECTS = {
    "DMT": {3},
    "KETA": {14},
    "PSILO": {1, 3},
}

repo_root = r"C:\CAMILO\Brain_Entropy\CODES\PSYCH\high-order-psychedelics"  # os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(repo_root, "data")

DMT_TS_path = os.path.join(data_path, "DMT_TS_FC.mat")
KETA_TS_path = os.path.join(data_path, "KETA_TS_FC.mat")
LSD_TS_path = os.path.join(data_path, "LSD_TS_FC.mat")
MDMA_TS_path = os.path.join(data_path, "MDMA_TS_FC.mat")
PSILO_TS_path = os.path.join(data_path, "PSILO_TS_FC.mat")
DMT_TS_FC = scipy.io.loadmat(DMT_TS_path)
KETA_TS_FC = scipy.io.loadmat(KETA_TS_path)
LSD_TS_FC = scipy.io.loadmat(LSD_TS_path)
MDMA_TS_FC = scipy.io.loadmat(MDMA_TS_path)
PSILO_TS_FC = scipy.io.loadmat(PSILO_TS_path)
DMT_TS_FC.keys()
DMT_TS_FC["__header__"]
DMT_TS_FC["__version__"]
DMT_TS_FC["__globals__"]
DMT_TS_FC["DataCorrelNoRat"]


def _extract_timeseries_length(ts_array, n_regions=90):
    """Infer time dimension from a 2D ROI x time (or time x ROI) matrix."""
    if ts_array.ndim < 2:
        return np.nan

    dim0, dim1 = ts_array.shape[0], ts_array.shape[1]
    if dim0 == n_regions and dim1 != n_regions:
        return dim1
    if dim1 == n_regions and dim0 != n_regions:
        return dim0

    # Fallback when region count is not exactly n_regions.
    return max(dim0, dim1)


def _print_mat_summary(mat_name, mat_obj, data_key, state_index):
    subjects_data = mat_obj[data_key][0]
    n_subjects = subjects_data.shape[0]

    ts_lengths = []
    for s in range(n_subjects):
        ts = subjects_data[s][state_index]
        ts_lengths.append(_extract_timeseries_length(ts))

    ts_lengths = np.array(ts_lengths, dtype=float)
    valid_lengths = ts_lengths[~np.isnan(ts_lengths)]

    if valid_lengths.size == 0:
        length_msg = "N/A"
    else:
        unique_lengths = np.unique(valid_lengths)
        if unique_lengths.size == 1:
            length_msg = str(int(unique_lengths[0]))
        else:
            length_msg = f"{valid_lengths.mean():.2f} (mean, variable lengths)"

    header = mat_obj.get("__header__", b"")
    if isinstance(header, bytes):
        header = header.decode("utf-8", errors="replace")

    print(f"\\n[{mat_name}]")
    print(f"subjects: {n_subjects}")
    print(f"timeseries_length: {length_msg}")
    print(f"__header__: {header}")
_print_mat_summary("DMT_TS_FC.mat", DMT_TS_FC, "DataCorrelNoRat", 1)
_print_mat_summary("KETA_TS_FC.mat", KETA_TS_FC, "DataCorrel", 2)
_print_mat_summary("LSD_TS_FC.mat", LSD_TS_FC, "DataCorrel", 2)
_print_mat_summary("MDMA_TS_FC.mat", MDMA_TS_FC, "DataCorrel", 4)
_print_mat_summary("PSILO_TS_FC.mat", PSILO_TS_FC, "DataCorrel", 2)

def main():


    drugs_dict = {}
    for drug in ["DMT", "LSD", "KETA", "PSILO", "MDMA"]:
        drugs_dict[drug] = {}
        drugs_dict[drug]["covmats"] = []  # first PCB, then drug
        drugs_dict[drug]["samples"] = []  # first PCB, then drug

    for s in range(DMT_TS_FC["DataCorrelNoRat"][0].shape[0]):
        if s in EXCLUDED_SOURCE_SUBJECTS["DMT"]:
            continue
        state_data_d = DMT_TS_FC["DataCorrelNoRat"][0][s][1]  # DMT
        covmats_drug, D, Nvars, T_drug = _normalize_input_data(state_data_d.T)
        state_data_pcb = DMT_TS_FC["DataCorrelNoRat"][0][s][3]  # PCB
        covmats_pcb, D, Nvars, T_pcb = _normalize_input_data(state_data_pcb.T)
        drugs_dict["DMT"]["covmats"].append(
            np.stack([covmats_pcb[0].numpy(), covmats_drug[0].numpy()])
        )
        drugs_dict["DMT"]["samples"].append([T_pcb[0], T_drug[0]])

    for s in range(KETA_TS_FC["DataCorrel"][0].shape[0]):
        if s in EXCLUDED_SOURCE_SUBJECTS["KETA"]:
            continue
        state_data_d = KETA_TS_FC["DataCorrel"][0][s][2]  # KET
        covmats_drug, D, Nvars, T_drug = _normalize_input_data(state_data_d.T)
        state_data_pcb = KETA_TS_FC["DataCorrel"][0][s][5]  # PCB
        covmats_pcb, D, Nvars, T_pcb = _normalize_input_data(state_data_pcb.T)
        drugs_dict["KETA"]["covmats"].append(
            np.stack([covmats_pcb[0].numpy(), covmats_drug[0].numpy()])
        )
        drugs_dict["KETA"]["samples"].append([T_pcb[0], T_drug[0]])

    for s in range(LSD_TS_FC["DataCorrel"][0].shape[0]):
        state_data_d = LSD_TS_FC["DataCorrel"][0][s][2]  # LSD
        covmats_drug, D, Nvars, T_drug = _normalize_input_data(state_data_d.T)
        state_data_pcb = LSD_TS_FC["DataCorrel"][0][s][5]  # PCB
        covmats_pcb, D, Nvars, T_pcb = _normalize_input_data(state_data_pcb.T)
        drugs_dict["LSD"]["covmats"].append(
            np.stack([covmats_pcb[0].numpy(), covmats_drug[0].numpy()])
        )
        drugs_dict["LSD"]["samples"].append([T_pcb[0], T_drug[0]])

    for s in range(MDMA_TS_FC["DataCorrel"][0].shape[0]):
        state_data_d = MDMA_TS_FC["DataCorrel"][0][s][4]  # MDMA
        covmats_drug, D, Nvars, T_drug = _normalize_input_data(state_data_d.T)
        state_data_pcb = MDMA_TS_FC["DataCorrel"][0][s][9]  # PCB
        covmats_pcb, D, Nvars, T_pcb = _normalize_input_data(state_data_pcb.T)
        drugs_dict["MDMA"]["covmats"].append(
            np.stack([covmats_pcb[0].numpy(), covmats_drug[0].numpy()])
        )
        drugs_dict["MDMA"]["samples"].append([T_pcb[0], T_drug[0]])

    for s in range(PSILO_TS_FC["DataCorrel"][0].shape[0]):
        if s in EXCLUDED_SOURCE_SUBJECTS["PSILO"]:
            continue
        state_data_d = PSILO_TS_FC["DataCorrel"][0][s][2]  # PSILO
        if state_data_d.shape[0] < 90:
            continue
        covmats_drug, D, Nvars, T_drug = _normalize_input_data(state_data_d.T)
        state_data_pcb = PSILO_TS_FC["DataCorrel"][0][s][5]  # PCB
        covmats_pcb, D, Nvars, T_pcb = _normalize_input_data(state_data_pcb.T)
        drugs_dict["PSILO"]["covmats"].append(
            np.stack([covmats_pcb[0].numpy(), covmats_drug[0].numpy()])
        )
        drugs_dict["PSILO"]["samples"].append([T_pcb[0], T_drug[0]])

    # for drug, drug_d in drugs_dict.items():
    #     print(drug, drug_d["samples"])

    results_path = data_path
    covmat_name = "PSYCHEDELIC_covmats"

    with h5py.File(f"{results_path}/{covmat_name}.h5", "w") as h5f:
        h5f.attrs["excluded_source_subjects"] = json.dumps(
            {
                drug: sorted(list(indices))
                for drug, indices in EXCLUDED_SOURCE_SUBJECTS.items()
            }
        )
        for drug_name, drug_d in drugs_dict.items():
            print(drug_name)
            grp = h5f.create_group(drug_name)
            grp.create_dataset(
                "covmats",
                data=np.array(drug_d["covmats"]),
                compression="gzip",
                chunks=True,
            )  # (N, 2, 90, 90)
            grp.create_dataset(
                "samples",
                data=np.array(drug_d["samples"]),
                compression="gzip",
                chunks=True,
            )  # (N, 2)


if __name__ == "__main__":

    main()
