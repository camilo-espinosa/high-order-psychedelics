from pathlib import Path
import json
import logging

import h5py
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
from thoi.measures.gaussian_copula import nplets_measures

STATE_NAMES = ["placebo", "drug"]
METRIC_NAMES = [
    "o_information",
    "mean_fisher_z",
]
CSV_SAVE_KWARGS = {
    "index": False,
    "encoding": "utf-8-sig",
    "sep": ";",
    "decimal": ",",
}


def cov_to_corr(cov, eps=1e-12):
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    denom = np.outer(std, std)
    corr = cov / np.clip(denom, eps, None)
    return np.clip(corr, -0.999999, 0.999999)


def corr_pair_values(corr):
    tri = np.triu_indices_from(corr, k=1)
    return corr[tri]


def full_region_o_information(cov, device):
    n_regions = cov.shape[0]
    all_regions_nplet = [list(range(n_regions))]
    cov_t = torch.as_tensor(cov, dtype=torch.float64, device=device)
    measures = nplets_measures(
        cov_t,
        nplets=all_regions_nplet,
        covmat_precomputed=True,
        device=device,
        verbose=logging.WARNING,
    )
    return float(measures[0, 0, 2].detach().cpu().item())


def subject_metrics(cov, device):
    corr = cov_to_corr(cov)
    vals = corr_pair_values(corr)
    vals_z = np.arctanh(vals)
    return {
        "o_information": full_region_o_information(cov, device=device),
        "mean_fisher_z": float(np.mean(vals_z)),
    }


def sign_flip_permutation_pvalue(diff, n_perm=20000, seed=42):
    rng = np.random.default_rng(seed)
    observed = abs(np.mean(diff))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    perm_means = np.abs(np.mean(signs * diff[None, :], axis=1))
    return float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))


def paired_stats(placebo_vals, drug_vals):
    diff = drug_vals - placebo_vals
    n = diff.size
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    dz = (
        float(mean_diff / sd_diff)
        if sd_diff and not np.isclose(sd_diff, 0.0)
        else np.nan
    )

    t_stat, p_t = stats.ttest_rel(drug_vals, placebo_vals, nan_policy="omit")

    if n >= 5 and np.any(diff != 0):
        try:
            w_stat, p_w = stats.wilcoxon(
                diff, zero_method="wilcox", alternative="two-sided"
            )
        except ValueError:
            w_stat, p_w = np.nan, np.nan
    else:
        w_stat, p_w = np.nan, np.nan

    p_perm = sign_flip_permutation_pvalue(diff) if n > 1 else np.nan

    return {
        "n_subjects": int(n),
        "mean_placebo": float(np.mean(placebo_vals)),
        "mean_drug": float(np.mean(drug_vals)),
        "mean_diff_drug_minus_placebo": mean_diff,
        "cohens_dz": dz,
        "ttest_t": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "ttest_p": float(p_t) if np.isfinite(p_t) else np.nan,
        "wilcoxon_w": float(w_stat) if np.isfinite(w_stat) else np.nan,
        "wilcoxon_p": float(p_w) if np.isfinite(p_w) else np.nan,
        "signflip_p": p_perm,
    }


def save_paired_plot(df_pairs, metric, drug, out_path):
    plt.figure(figsize=(6, 5))
    x = np.array([0, 1])
    for _, row in df_pairs.iterrows():
        y = [row[f"{metric}_placebo"], row[f"{metric}_drug"]]
        plt.plot(x, y, color="#8c8c8c", alpha=0.45, linewidth=1)
    plt.scatter(
        np.zeros(df_pairs.shape[0]),
        df_pairs[f"{metric}_placebo"],
        color="#2563eb",
        label="Placebo",
        s=32,
        alpha=0.9,
    )
    plt.scatter(
        np.ones(df_pairs.shape[0]),
        df_pairs[f"{metric}_drug"],
        color="#dc2626",
        label="Drug",
        s=32,
        alpha=0.9,
    )
    plt.xticks([0, 1], ["Placebo", "Drug"])
    plt.title(f"{drug}: paired {metric} by subject")
    plt.ylabel(metric)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_eda(data_file, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(data_file, "r") as h5f:
        for drug in h5f.keys():
            covmats = np.array(h5f[drug]["covmats"])
            samples = np.array(h5f[drug]["samples"])

            # covmats shape: (subjects, 2, R, R), state order [placebo, drug]
            n_subjects = covmats.shape[0]
            drug_dir = out_dir / drug
            drug_dir.mkdir(parents=True, exist_ok=True)

            long_rows = []
            for subj_idx in range(n_subjects):
                for state_idx, state_name in enumerate(STATE_NAMES):
                    m = subject_metrics(covmats[subj_idx, state_idx], device=device)
                    long_rows.append(
                        {
                            "drug": drug,
                            "subject_idx": subj_idx,
                            "state": state_name,
                            "samples": int(samples[subj_idx, state_idx]),
                            **m,
                        }
                    )

            df_long = pd.DataFrame(long_rows)
            df_long.to_csv(drug_dir / "subject_metrics_long.csv", **CSV_SAVE_KWARGS)

            df_pivot = df_long.pivot(index="subject_idx", columns="state")
            df_pivot.columns = [f"{c[0]}_{c[1]}" for c in df_pivot.columns]
            df_pairs = df_pivot.reset_index()

            stats_rows = []
            for metric in METRIC_NAMES:
                placebo_vals = df_pairs[f"{metric}_placebo"].to_numpy()
                drug_vals = df_pairs[f"{metric}_drug"].to_numpy()
                s = paired_stats(placebo_vals, drug_vals)
                s["drug"] = drug
                s["metric"] = metric
                stats_rows.append(s)
                summary_rows.append(s)

                save_paired_plot(
                    df_pairs,
                    metric,
                    drug,
                    drug_dir / f"paired_{metric}.png",
                )

            pd.DataFrame(stats_rows).to_csv(
                drug_dir / "paired_stats.csv", **CSV_SAVE_KWARGS
            )
            df_pairs.to_csv(drug_dir / "subject_metrics_paired.csv", **CSV_SAVE_KWARGS)

            with open(drug_dir / "run_info.json", "w", encoding="ascii") as f:
                json.dump(
                    {
                        "drug": drug,
                        "n_subjects": int(n_subjects),
                        "n_regions": int(covmats.shape[-1]),
                        "state_order": STATE_NAMES,
                        "metrics": METRIC_NAMES,
                        "o_information": "computed with all regions as one n-plet",
                        "device": str(device),
                    },
                    f,
                    indent=2,
                )

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "paired_stats_all_drugs.csv", **CSV_SAVE_KWARGS
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_file = repo_root / "data" / "PSYCHEDELIC_covmats.h5"
    out_dir = repo_root / "results" / "EDA_01_paired"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Expected covariance file at: {data_file}. Run scripts/A_1_data.py first."
        )

    run_eda(data_file, out_dir)
    print(f"EDA complete. Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
