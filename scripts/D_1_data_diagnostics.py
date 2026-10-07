from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import scipy.io
from thoi.commons import _normalize_input_data

CSV_SAVE_KWARGS = {
    "index": False,
    "encoding": "utf-8-sig",
    "sep": ";",
    "decimal": ",",
}

EXCLUDED_SOURCE_SUBJECTS = {
    "DMT": {3},
    "KETA": {14},
    "PSILO": {1, 3},
}

DRUG_SPECS = {
    "DMT": {
        "mat_file": "DMT_TS_FC.mat",
        "root_key": "DataCorrelNoRat",
        "drug_idx": 1,
        "placebo_idx": 3,
        "min_rows": None,
    },
    "KETA": {
        "mat_file": "KETA_TS_FC.mat",
        "root_key": "DataCorrel",
        "drug_idx": 2,
        "placebo_idx": 5,
        "min_rows": None,
    },
    "LSD": {
        "mat_file": "LSD_TS_FC.mat",
        "root_key": "DataCorrel",
        "drug_idx": 2,
        "placebo_idx": 5,
        "min_rows": None,
    },
    "MDMA": {
        "mat_file": "MDMA_TS_FC.mat",
        "root_key": "DataCorrel",
        "drug_idx": 4,
        "placebo_idx": 9,
        "min_rows": None,
    },
    "PSILO": {
        "mat_file": "PSILO_TS_FC.mat",
        "root_key": "DataCorrel",
        "drug_idx": 2,
        "placebo_idx": 5,
        "min_rows": 90,
    },
}

STATE_SPECS = [("placebo", "placebo_idx"), ("drug", "drug_idx")]


def safe_condition_number(matrix):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] == 0:
        return np.inf
    return float(singular_values[0] / singular_values[-1])


def matrix_stats(matrix):
    eigvals = np.linalg.eigvalsh(matrix)
    diag = np.diag(matrix)
    return {
        "is_finite": bool(np.isfinite(matrix).all()),
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "shape_0": int(matrix.shape[0]),
        "shape_1": int(matrix.shape[1]),
        "symmetry_max_abs_err": float(np.max(np.abs(matrix - matrix.T))),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "diag_nonpositive_count": int(np.sum(diag <= 0)),
        "eig_min": float(np.min(eigvals)),
        "eig_max": float(np.max(eigvals)),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": safe_condition_number(matrix),
        "abs_max": float(np.max(np.abs(matrix))),
    }


def timeseries_stats(timeseries):
    row_var = np.var(timeseries, axis=0)
    col_var = np.var(timeseries, axis=1)
    return {
        "is_finite": bool(np.isfinite(timeseries).all()),
        "nan_count": int(np.isnan(timeseries).sum()),
        "inf_count": int(np.isinf(timeseries).sum()),
        "n_regions": int(timeseries.shape[0]),
        "n_timepoints": int(timeseries.shape[1]),
        "abs_max": float(np.max(np.abs(timeseries))),
        "global_mean": float(np.mean(timeseries)),
        "global_std": float(np.std(timeseries)),
        "zero_var_region_count": int(np.sum(np.isclose(row_var, 0.0))),
        "zero_var_timepoint_count": int(np.sum(np.isclose(col_var, 0.0))),
    }


def load_raw_pairs(data_dir):
    pairs = {}
    for drug_name, spec in DRUG_SPECS.items():
        mat_path = data_dir / spec["mat_file"]
        mat_data = scipy.io.loadmat(mat_path)
        root = mat_data[spec["root_key"]][0]
        entries = []
        for subject_idx in range(root.shape[0]):
            if subject_idx in EXCLUDED_SOURCE_SUBJECTS.get(drug_name, set()):
                continue
            drug_ts = root[subject_idx][spec["drug_idx"]]
            if spec["min_rows"] is not None and drug_ts.shape[0] < spec["min_rows"]:
                continue
            placebo_ts = root[subject_idx][spec["placebo_idx"]]
            entries.append(
                {
                    "subject_idx": len(entries),
                    "source_subject_idx": int(subject_idx),
                    "placebo": placebo_ts,
                    "drug": drug_ts,
                }
            )
        pairs[drug_name] = entries
    return pairs


def recompute_covariance(timeseries):
    covmats, _, _, samples = _normalize_input_data(timeseries.T)
    return covmats[0].numpy(), int(samples[0])


def run_diagnostics(data_dir, output_dir):
    raw_pairs = load_raw_pairs(data_dir)
    cov_path = data_dir / "PSYCHEDELIC_covmats.h5"

    time_series_rows = []
    covariance_rows = []
    comparison_rows = []
    summary_rows = []

    with h5py.File(cov_path, "r") as h5f:
        for drug_name, entries in raw_pairs.items():
            stored_covmats = np.array(h5f[drug_name]["covmats"])
            stored_samples = np.array(h5f[drug_name]["samples"])

            if stored_covmats.shape[0] != len(entries):
                raise ValueError(
                    f"Stored {drug_name} subject count {stored_covmats.shape[0]} does not match raw count {len(entries)}."
                )

            for entry in entries:
                subject_idx = entry["subject_idx"]
                source_subject_idx = entry["source_subject_idx"]
                for state_idx, (state_name, state_key) in enumerate(STATE_SPECS):
                    timeseries = entry[state_name]
                    ts_stats = timeseries_stats(timeseries)
                    time_series_rows.append(
                        {
                            "drug": drug_name,
                            "subject_idx": subject_idx,
                            "source_subject_idx": source_subject_idx,
                            "state": state_name,
                            **ts_stats,
                        }
                    )

                    recomputed_cov, recomputed_samples = recompute_covariance(
                        timeseries
                    )
                    stored_cov = stored_covmats[subject_idx, state_idx]
                    stored_sample = int(stored_samples[subject_idx, state_idx])

                    cov_stats = matrix_stats(stored_cov)
                    covariance_rows.append(
                        {
                            "drug": drug_name,
                            "subject_idx": subject_idx,
                            "source_subject_idx": source_subject_idx,
                            "state": state_name,
                            "stored_sample": stored_sample,
                            **cov_stats,
                        }
                    )

                    diff = stored_cov - recomputed_cov
                    comparison_rows.append(
                        {
                            "drug": drug_name,
                            "subject_idx": subject_idx,
                            "source_subject_idx": source_subject_idx,
                            "state": state_name,
                            "stored_sample": stored_sample,
                            "recomputed_sample": recomputed_samples,
                            "sample_match": bool(stored_sample == recomputed_samples),
                            "cov_allclose": bool(
                                np.allclose(
                                    stored_cov, recomputed_cov, atol=1e-10, rtol=1e-8
                                )
                            ),
                            "cov_max_abs_diff": float(np.max(np.abs(diff))),
                            "cov_mean_abs_diff": float(np.mean(np.abs(diff))),
                            "cov_fro_abs_diff": float(np.linalg.norm(diff, ord="fro")),
                            "recomputed_is_finite": bool(
                                np.isfinite(recomputed_cov).all()
                            ),
                            "stored_is_finite": bool(np.isfinite(stored_cov).all()),
                        }
                    )

            ts_df_drug = pd.DataFrame(
                [r for r in time_series_rows if r["drug"] == drug_name]
            )
            cov_df_drug = pd.DataFrame(
                [r for r in covariance_rows if r["drug"] == drug_name]
            )
            cmp_df_drug = pd.DataFrame(
                [r for r in comparison_rows if r["drug"] == drug_name]
            )

            summary_rows.append(
                {
                    "drug": drug_name,
                    "n_subjects": int(len(entries)),
                    "raw_nonfinite_states": int((~ts_df_drug["is_finite"]).sum()),
                    "raw_zero_var_region_states": int(
                        (ts_df_drug["zero_var_region_count"] > 0).sum()
                    ),
                    "stored_nonfinite_cov_states": int(
                        (~cov_df_drug["is_finite"]).sum()
                    ),
                    "stored_nonpositive_diag_states": int(
                        (cov_df_drug["diag_nonpositive_count"] > 0).sum()
                    ),
                    "stored_nonspd_states": int((cov_df_drug["eig_min"] <= 0).sum()),
                    "comparison_sample_mismatch_states": int(
                        (~cmp_df_drug["sample_match"]).sum()
                    ),
                    "comparison_cov_mismatch_states": int(
                        (~cmp_df_drug["cov_allclose"]).sum()
                    ),
                    "comparison_max_abs_diff": float(
                        cmp_df_drug["cov_max_abs_diff"].max()
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    time_series_df = pd.DataFrame(time_series_rows)
    covariance_df = pd.DataFrame(covariance_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    summary_df = pd.DataFrame(summary_rows)

    time_series_df.to_csv(output_dir / "time_series_diagnostics.csv", **CSV_SAVE_KWARGS)
    covariance_df.to_csv(output_dir / "covariance_diagnostics.csv", **CSV_SAVE_KWARGS)
    comparison_df.to_csv(
        output_dir / "covariance_rebuild_comparison.csv", **CSV_SAVE_KWARGS
    )
    summary_df.to_csv(output_dir / "diagnostics_summary.csv", **CSV_SAVE_KWARGS)

    suspicious = {
        "time_series": time_series_df[
            (~time_series_df["is_finite"])
            | (time_series_df["zero_var_region_count"] > 0)
        ],
        "covariance": covariance_df[
            (~covariance_df["is_finite"])
            | (covariance_df["diag_nonpositive_count"] > 0)
            | (covariance_df["eig_min"] <= 0)
        ],
        "comparison": comparison_df[
            (~comparison_df["sample_match"]) | (~comparison_df["cov_allclose"])
        ],
    }

    for name, df in suspicious.items():
        df.to_csv(output_dir / f"suspicious_{name}.csv", **CSV_SAVE_KWARGS)

    run_info = {
        "data_dir": str(data_dir),
        "covariance_file": str(cov_path),
        "n_drugs": int(len(summary_rows)),
        "excluded_source_subjects": {
            drug: sorted(list(indices))
            for drug, indices in EXCLUDED_SOURCE_SUBJECTS.items()
        },
        "checks": [
            "raw finite values",
            "raw zero-variance regions",
            "stored covariance finite values",
            "stored covariance symmetry and eigenvalues",
            "stored samples versus recomputed samples",
            "stored covariance versus recomputed covariance",
        ],
    }
    with open(output_dir / "run_info.json", "w", encoding="ascii") as f:
        json.dump(run_info, f, indent=2)

    print(f"Diagnostics saved to: {output_dir}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    output_dir = repo_root / "results" / "D_1_data_diagnostics"
    run_diagnostics(data_dir, output_dir)


if __name__ == "__main__":
    main()
