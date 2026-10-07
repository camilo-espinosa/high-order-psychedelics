from pathlib import Path
from functools import partial
import argparse
import json
import logging

import h5py
import numpy as np
import pandas as pd
import torch
import sys

sys.path.append(r"C:\CAMILO\Brain_Entropy\CODES\PSYCH\high-order-psychedelics")
from utils.thoi_utils import simulated_annealing_parallel

CSV_SAVE_KWARGS = {
    "index": False,
    "encoding": "utf-8-sig",
    "sep": ";",
    "decimal": ",",
}


def paired_delta_O_metric_per_subject_p1(batched_measures):
    """
    Shape-compatible paired objective for simulated_annealing_parallel.

    Returns (R, P), one paired delta-O score per repeat and subject:
      polarity * (Omega_drug - Omega_placebo)

    This keeps subject-level n-plet outputs explicit while remaining the
    paired analogue of max_difference_pairs.
    """
    O = batched_measures[..., 2]  # (R, P, 2)
    delta_per_subj = O[..., 0] - O[..., 1]  # (R, P)
    return delta_per_subj


def paired_delta_O_metric_per_subject_p2(batched_measures):
    """
    Shape-compatible paired objective for simulated_annealing_parallel.

    Returns (R, P), one paired delta-O score per repeat and subject:
      polarity * (Omega_drug - Omega_placebo)

    This keeps subject-level n-plet outputs explicit while remaining the
    paired analogue of max_difference_pairs.
    """
    O = batched_measures[..., 2]  # (R, P, 2)
    delta_per_subj = O[..., 0] - O[..., 1]  # (R, P)
    return -delta_per_subj


def polarity_label(polarity):
    return "drug_gt_placebo" if polarity == 1 else "placebo_gt_drug"


def run_for_drug(
    drug_name,
    covmats,
    out_dir,
    device,
    order_min,
    order_max,
    repeat,
    max_iterations,
    early_stop,
    initial_temp,
    cooling_rate,
    batch_size,
):
    # Input from A_1_data.py is [placebo, drug]; reorder to [drug, placebo].
    covmats_pairs = covmats[:, [1, 0], :, :]

    n_subjects = int(covmats_pairs.shape[0])
    n_regions = int(covmats_pairs.shape[-1])
    X = torch.as_tensor(covmats_pairs, dtype=torch.float64, device=device)

    rows = []

    for order in range(order_min, order_max + 1):
        for polarity in (1, -1):
            print(
                f"Running drug: {drug_name}, order {order}, polarity {polarity_label(polarity)}"
            )
            metric = (
                paired_delta_O_metric_per_subject_p1
                if polarity == 1
                else paired_delta_O_metric_per_subject_p2
            )

            best_solution, best_energy = simulated_annealing_parallel(
                X=X,
                order=order,
                covmat_precomputed=True,
                repeat=repeat,
                batch_size=batch_size,
                device=device,
                max_iterations=max_iterations,
                early_stop=early_stop,
                initial_temp=initial_temp,
                cooling_rate=cooling_rate,
                metric=metric,
                largest=True,
                verbose=logging.WARNING,
            )

            # Shapes: best_solution -> (repeat, n_subjects, order)
            #         best_energy   -> (repeat, n_subjects), objective values
            best_solution_np = best_solution.detach().cpu().numpy()
            best_energy_np = best_energy.detach().cpu().numpy()

            for rep_idx in range(repeat):
                for subj_idx in range(n_subjects):
                    objective_score = float(best_energy_np[rep_idx, subj_idx])
                    delta_o = float(objective_score * polarity)
                    nplet = best_solution_np[rep_idx, subj_idx].tolist()
                    rows.append(
                        {
                            "drug": drug_name,
                            "order": int(order),
                            "polarity": int(polarity),
                            "polarity_name": polarity_label(polarity),
                            "repeat_idx": int(rep_idx),
                            "subject_idx": int(subj_idx),
                            "objective_score": objective_score,
                            "delta_o_drug_minus_placebo": delta_o,
                            "nplet": nplet,
                            "n_regions": n_regions,
                            "state_axis_order": "[drug, placebo]",
                        }
                    )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"paired_optimal_nplets_{drug_name}.parquet"
    pkl_path = out_dir / f"paired_optimal_nplets_{drug_name}.pkl.gz"
    csv_path = out_dir / f"paired_optimal_nplets_{drug_name}.csv"

    # Efficient binary dataframe first; keep CSV for easy inspection.
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        df.to_pickle(pkl_path, compression="gzip")

    # Store n-plets as compact JSON strings in CSV output.
    df_csv = df.copy()
    df_csv["nplet"] = df_csv["nplet"].apply(
        lambda x: json.dumps(x, separators=(",", ":"))
    )
    df_csv.to_csv(csv_path, **CSV_SAVE_KWARGS)

    summary = (
        df.groupby(["drug", "order", "polarity", "polarity_name"], as_index=False)
        .agg(
            mean_delta_o=("delta_o_drug_minus_placebo", "mean"),
            std_delta_o=("delta_o_drug_minus_placebo", "std"),
            n_rows=("delta_o_drug_minus_placebo", "size"),
        )
        .sort_values(["order", "polarity"], ascending=[True, False])
    )
    summary.to_csv(
        out_dir / f"paired_optimal_nplets_{drug_name}_summary.csv",
        **CSV_SAVE_KWARGS,
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Script 3: paired SA search for optimal n-plets (orders 3-12) per drug"
    )
    parser.add_argument("--order-min", type=int, default=3)
    parser.add_argument("--order-max", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--early-stop", type=int, default=1000)
    parser.add_argument("--initial-temp", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=2000000)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_file = repo_root / "data" / "PSYCHEDELIC_covmats.h5"
    out_dir = repo_root / "results" / "C_1_paired_optimal_nplets"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Expected covariance file at: {data_file}. Run scripts/A_1_data.py first."
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    all_frames = []
    with h5py.File(data_file, "r") as h5f:
        for drug_name in sorted(h5f.keys()):
            covmats = np.array(h5f[drug_name]["covmats"])
            df = run_for_drug(
                drug_name=drug_name,
                covmats=covmats,
                out_dir=out_dir,
                device=device,
                order_min=args.order_min,
                order_max=args.order_max,
                repeat=args.repeat,
                max_iterations=args.max_iterations,
                early_stop=args.early_stop,
                initial_temp=args.initial_temp,
                cooling_rate=args.cooling_rate,
                batch_size=args.batch_size,
            )
            all_frames.append(df)

    all_df = pd.concat(all_frames, axis=0, ignore_index=True)

    try:
        all_df.to_parquet(
            out_dir / "paired_optimal_nplets_all_drugs.parquet", index=False
        )
    except Exception:
        all_df.to_pickle(
            out_dir / "paired_optimal_nplets_all_drugs.pkl.gz", compression="gzip"
        )

    all_csv = all_df.copy()
    all_csv["nplet"] = all_csv["nplet"].apply(
        lambda x: json.dumps(x, separators=(",", ":"))
    )
    all_csv.to_csv(out_dir / "paired_optimal_nplets_all_drugs.csv", **CSV_SAVE_KWARGS)

    run_info = {
        "data_file": str(data_file),
        "output_dir": str(out_dir),
        "orders": [args.order_min, args.order_max],
        "repeat": args.repeat,
        "max_iterations": args.max_iterations,
        "early_stop": args.early_stop,
        "initial_temp": args.initial_temp,
        "cooling_rate": args.cooling_rate,
        "batch_size": args.batch_size,
        "device": str(device),
        "state_axis_order_used": "[drug, placebo]",
        "objective": "paired delta O",
        "polarities": {
            "+1": "Omega(drug) > Omega(placebo)",
            "-1": "Omega(placebo) > Omega(drug)",
        },
    }
    with open(out_dir / "run_info.json", "w", encoding="ascii") as f:
        json.dump(run_info, f, indent=2)

    print(f"Done. Saved paired optimal n-plets to: {out_dir}")


if __name__ == "__main__":
    main()
