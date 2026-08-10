import argparse
import os
import platform
import random
import socket
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from sklearn.utils import resample

from utils.util_v2 import (
    evaluate_classifiers,
    extract_activations_cached,
    load_data_final,
    load_model,
    prepare_data,
    train_classifiers,
)


def print_system_info():
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    print(f"Hostname: {socket.gethostname()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")

    print(f"\nCPU: {platform.processor()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical cores: {psutil.cpu_count(logical=True)}")

    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"RAM: {ram_gb:.1f} GB")

    print(f"\nPyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                f"GPU {i}: {props.name} "
                f"({props.total_memory / (1024**3):.1f} GB)"
            )
    else:
        print("CUDA: Not available")

    print("=" * 60 + "\n")


def get_name(path):
    """Extract a clean dataset name (without extension or path)."""
    return os.path.splitext(os.path.basename(path))[0]


def make_dataset_id(lie_path, truth_path, seed):
    """
    Build a stable ID for (lie, truth) dataset pair + seed.
    Used to cache activations across experiments.
    """
    lie_name = get_name(lie_path)
    truth_name = get_name(truth_path)
    return f"{lie_name}__{truth_name}__seed{seed}"


def load_config(path="config.yml"):
    """Load YAML configuration file."""
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, path), "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "extract_only", "train_only"],
        help="Pipeline mode: full = extract+train+test, extract_only = GPU forward passes only, train_only = CPU probe training & evaluation only",
    )
    return parser.parse_args()


def main():
    start_time = time.time()
    start_dt = datetime.now()

    print(f"Execution started: {start_dt:%Y-%m-%d %H:%M:%S}")

    print_system_info()

    args = parse_args()
    # --- Load config and set seed ---
    cfg = load_config("configs/config.yml")
    seed = cfg["experiment"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"\nLoaded config and initialized with seed {seed}\n")

    # Load model only in modes that require forward passes
    if args.mode in ("full", "extract_only"):
        model, tokenizer = load_model(cfg["model"]["name"])
    else:
        print("Train-only mode: loading tokenizer only")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        model = None

    for experiment in cfg["experiment_data"]["experiments"]:
        print(f"Running experiment: {experiment}")

        cfg_experiment = load_config(f"configs/experiments/{experiment}")

        train_lie_name = get_name(
            cfg_experiment["experiment_data"]["training_data"]["lie"]
        )
        train_truth_name = get_name(
            cfg_experiment["experiment_data"]["training_data"]["truth"]
        )
        train_base_name = cfg_experiment["experiment_data"]["name"]  # use truth name as reference
        print(f"Training on: {train_lie_name} / {train_truth_name}")

        print(f"Loaded model: {cfg['model']['name']}\n")

        # --- Load and prepare TRAINING data ---
        deceptive_data, honest_data = load_data_final(
            cfg_experiment["experiment_data"]["training_data"]["lie"],
            cfg_experiment["experiment_data"]["training_data"]["truth"],
        )

        # Balance training set
        min_len = min(len(deceptive_data), len(honest_data))
        if min_len != 0:
            deceptive_data = resample(
                deceptive_data, replace=False, n_samples=min_len, random_state=seed
            )
            honest_data = resample(
                honest_data, replace=False, n_samples=min_len, random_state=seed
            )

        texts_train, labels_train = prepare_data(
            deceptive_data, honest_data, tokenizer, crop=0
        )

        # --- Extract layer activations (training set) with caching ---
        train_lie_path = cfg_experiment["experiment_data"]["training_data"]["lie"]
        train_truth_path = cfg_experiment["experiment_data"]["training_data"]["truth"]
        train_dataset_id = make_dataset_id(train_lie_path, train_truth_path, seed)

        if args.mode in ("full", "extract_only"):
            # With forward pass
            all_layers_activations_train = extract_activations_cached(
                texts_train,
                labels_train,
                tokenizer,
                model,
                batch_size=cfg["activations"]["batch_size"],
                use_mean=cfg["experiment"]["mean"],
                save_both=True,
                model_name=cfg["model"]["name"],
                dataset_id=train_dataset_id,
                force_load=False,
            )
        else:
            # CPU-only mode → do NOT run forward pass
            all_layers_activations_train = extract_activations_cached(
                texts_train,
                labels_train,
                tokenizer,
                model=None,
                batch_size=None,
                use_mean=cfg["experiment"]["mean"],
                model_name=cfg["model"]["name"],
                dataset_id=train_dataset_id,
                force_load=True,
            )

        # --- Train classifiers ---
        if args.mode == "extract_only":
            print("Mode=extract_only → Skipping probe training.")
        else:
            print("\n=== TRAINING CLASSIFIERS ===\n")
            train_results = train_classifiers(
                all_layers_activations_train,
                labels_train,
                cfg,
                random_seed=seed,
                train_dataset_id=train_dataset_id,
            )
            trained_models = train_results["models"]

        base_output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        model_output_dir = os.path.join(base_output_dir, cfg["model"]["name"])
        os.makedirs(model_output_dir, exist_ok=True)

        pooling_suffix = "mean_tokens" if cfg["experiment"]["mean"] else "last_token"
        train_output_dir = os.path.join(
            model_output_dir, f"{train_base_name}__{pooling_suffix}"
        )
        os.makedirs(train_output_dir, exist_ok=True)

        # --- Loop across test datasets ---
        for test_cfg in cfg_experiment["experiment_data"]["test_data"]:
            test_name = test_cfg.get("name", get_name(test_cfg["truth"]))
            print(f"\n=== TESTING ON TARGET DATASET: {test_name.upper()} ===\n")

            # --- Load and prepare test data ---
            deceptive_data_target, honest_data_target = load_data_final(
                test_cfg["lie"], test_cfg["truth"]
            )

            # Balance test data
            min_len = min(len(deceptive_data_target), len(honest_data_target))
            if min_len != 0:
                deceptive_data_target = resample(
                    deceptive_data_target,
                    replace=False,
                    n_samples=min_len,
                    random_state=seed,
                )
                honest_data_target = resample(
                    honest_data_target,
                    replace=False,
                    n_samples=min_len,
                    random_state=seed,
                )

            texts_test, labels_test = prepare_data(
                deceptive_data_target, honest_data_target, tokenizer, crop=0
            )

            # --- Extract activations for the test data (with caching) ---
            test_lie_path = test_cfg["lie"]
            test_truth_path = test_cfg["truth"]
            test_dataset_id = make_dataset_id(test_lie_path, test_truth_path, seed)

            if args.mode in ("full", "extract_only"):
                all_layers_activations_test = extract_activations_cached(
                    texts_test,
                    labels_test,
                    tokenizer,
                    model,
                    batch_size=cfg["activations"]["batch_size"],
                    use_mean=cfg["experiment"]["mean"],
                    model_name=cfg["model"]["name"],
                    dataset_id=test_dataset_id,
                    force_load=False,
                )
            else:
                all_layers_activations_test = extract_activations_cached(
                    texts_test,
                    labels_test,
                    tokenizer,
                    model=None,
                    batch_size=None,
                    use_mean=cfg["experiment"]["mean"],
                    model_name=cfg["model"]["name"],
                    dataset_id=test_dataset_id,
                    force_load=True,
                )

            if args.mode == "extract_only":
                print(
                    "Mode=extract_only → Computed and cached test activations. Skipping evaluation.\n"
                )
                continue

            # --- Single test evaluation ---
            results_test = evaluate_classifiers(
                trained_models,
                all_layers_activations_test,
                np.array(labels_test),
            )

            print(f"\nEvaluated on test dataset: {test_name}")
            for clf_name in results_test:
                print(
                    f"{clf_name} layer 0 test ACC: {results_test[clf_name]['acc'][0]:.4f}"
                )

            # --- Save results summary (CSV) ---
            df_train_acc = pd.DataFrame(
                {f"{name}_acc": res["acc"] for name, res in train_results["results"].items()}
            )
            df_train_auc = pd.DataFrame(
                {f"{name}_auc": res["auc"] for name, res in train_results["results"].items()}
            )

            df_test_acc = pd.DataFrame(
                {f"{name}_acc_test": res["acc"] for name, res in results_test.items()}
            )
            df_test_auc = pd.DataFrame(
                {f"{name}_auc_test": res["auc"] for name, res in results_test.items()}
            )
            df_test_fpr = pd.DataFrame(
                {f"{name}_fpr_test": res["fpr"] for name, res in results_test.items()}
            )
            df_test_fnr = pd.DataFrame(
                {f"{name}_fnr_test": res["fnr"] for name, res in results_test.items()}
            )

            df_summary = pd.concat(
                [
                    df_train_acc,
                    df_train_auc,
                    df_test_acc,
                    df_test_auc,
                    df_test_fpr,
                    df_test_fnr,
                ],
                axis=1,
            )

            safe_output_dir = f"{train_output_dir}_with_rates_corrected"
            os.makedirs(safe_output_dir, exist_ok=True)

            csv_path = os.path.join(
                safe_output_dir, f"{train_base_name}__{test_name}.csv"
            )
            df_summary.to_csv(csv_path, index=False)
            print(f"\nResults summary saved to: {csv_path}")

            # --- Save raw sample outputs (NPZ) ---
            npz_payload = {"y_true": np.array(labels_test)}
            for clf_name in results_test:
                preds_2d = np.vstack(results_test[clf_name]["preds_raw"])
                probs_2d = np.vstack(results_test[clf_name]["probs_raw"])

                npz_payload[f"{clf_name}_preds"] = preds_2d
                npz_payload[f"{clf_name}_probs"] = probs_2d

            npz_path = os.path.join(
                safe_output_dir, f"{train_base_name}__{test_name}_raw_outputs.npz"
            )
            np.savez_compressed(npz_path, **npz_payload)
            print(
                f"Raw predictions and continuous scores saved to: {npz_path}\n"
            )

            # --- Plot Accuracy, AUC, FPR, and FNR per layer ---
            layers = np.arange(
                len(next(iter(train_results["results"].values()))["acc"])
            )
            plt.figure(figsize=(12, 15))

            # 1. Accuracy comparison
            plt.subplot(3, 1, 1)
            for name in train_results["results"]:
                plt.plot(
                    layers,
                    train_results["results"][name]["acc"],
                    marker="o",
                    label=f"{name} (Val)",
                )
                plt.plot(
                    layers,
                    results_test[name]["acc"],
                    marker="x",
                    linestyle="--",
                    label=f"{name} (Test)",
                )
            plt.title("Layer-wise Accuracy")
            plt.xlabel("Layer Index")
            plt.ylabel("Accuracy")
            plt.legend(loc="best", fontsize=9)
            plt.grid(True, linestyle="--", alpha=0.6)

            # 2. AUC comparison
            plt.subplot(3, 1, 2)
            for name in train_results["results"]:
                plt.plot(
                    layers,
                    train_results["results"][name]["auc"],
                    marker="o",
                    label=f"{name} (Val)",
                )
                plt.plot(
                    layers,
                    results_test[name]["auc"],
                    marker="x",
                    linestyle="--",
                    label=f"{name} (Test)",
                )
            plt.title("Layer-wise AUC")
            plt.xlabel("Layer Index")
            plt.ylabel("ROC AUC")
            plt.legend(loc="best", fontsize=9)
            plt.grid(True, linestyle="--", alpha=0.6)

            # 3. FPR & FNR comparison
            plt.subplot(3, 1, 3)
            for name in train_results["results"]:
                if "fpr" in train_results["results"][name]:
                    plt.plot(
                        layers,
                        train_results["results"][name]["fpr"],
                        marker="o",
                        color="crimson",
                        label=f"{name} FPR (Val)",
                    )
                plt.plot(
                    layers,
                    results_test[name]["fpr"],
                    marker="x",
                    linestyle="--",
                    color="firebrick",
                    label=f"{name} FPR (Test)",
                )

                if "fnr" in train_results["results"][name]:
                    plt.plot(
                        layers,
                        train_results["results"][name]["fnr"],
                        marker="s",
                        color="darkblue",
                        label=f"{name} FNR (Val)",
                    )
                plt.plot(
                    layers,
                    results_test[name]["fnr"],
                    marker="^",
                    linestyle="--",
                    color="dodgerblue",
                    label=f"{name} FNR (Test)",
                )

            plt.title("Layer-wise Error Rates (FPR & FNR)")
            plt.xlabel("Layer Index")
            plt.ylabel("Rate (Lower is better)")
            plt.legend(loc="best", fontsize=9)
            plt.grid(True, linestyle="--", alpha=0.6)

            plt.suptitle(
                f"Validation vs Test Performance\n"
                f"{train_base_name.replace('_', ' ')} → {test_name.replace('_', ' ')}",
                fontsize=14,
                fontweight="bold",
                y=1.01,
            )

            plt.tight_layout()

            plot_filename = f"{train_base_name}__{test_name}.png"
            plot_path = os.path.join(safe_output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Performance comparison plot saved to: {plot_path}\n")

    end_time = time.time()
    end_dt = datetime.now()
    elapsed = end_time - start_time

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "=" * 60)
    print(f"Execution started : {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Execution finished: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(
        f"Total runtime      : "
        f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
