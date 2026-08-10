import hashlib
import json
import os
import matplotlib.pyplot as plt
from statistics import mean
import torch
import pickle
import torch.nn as nn
from einops import einsum
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import yaml
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


from utils.classifiers import CenteredLogisticRegression

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chi2_contingency

from sklearn.base import clone
import time
import logging

force_retrain_probes = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def _get_sklearn_scores(clf, X):
    try:
        probs = clf.predict_proba(X)
                                                              
        if probs.shape[1] == 2:
            return probs[:, 1]
                            
    except Exception:
        pass

    try:
                                                                                
        df = clf.decision_function(X)
                                                                         
        if df.ndim == 1:
            return df
        else:
                                                 
            if df.shape[1] >= 2:
                return df[:, 1]
            else:
                return df.ravel()
    except Exception:
        pass

                                                                                 
    return clf.predict(X)
                            

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = {i: "40GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "160GiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.eval()
    return model, tokenizer
   

def load_data_final(deceptive_path, honest_path):
    base_dir = os.path.dirname(os.path.dirname(__file__))                     
    if deceptive_path == honest_path:
        files = [deceptive_path]
    else:
        files = [deceptive_path, honest_path]

    deceptive_data = []
    honest_data = []

    for path in files:
        if not os.path.exists(path):
            print(f"Warning: file not found: {path}")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON line in {path}")
                    continue

                                                            
                if "label" not in entry:
                    print(f"⚠️ Missing label in {path}, skipping entry.")
                    continue

                                                   
                if entry["label"]:
                    honest_data.append(entry)
                else:
                    deceptive_data.append(entry)

    print(f"Loaded {len(deceptive_data)} deceptive and {len(honest_data)} honest entries.")
    return deceptive_data, honest_data


def prepare_data(deceptive_sampled, honest_data, tokenizer, crop=0):
    """
    Prepares the text inputs for activation extraction.

    Supports two formats:
      1) {"final": "...", "label": 0/1}
      2) {"prompt": "...", "output": "...", "label": 0/1}

    For (prompt, output), we build a Qwen-compatible chat transcript.
    """

    all_data = deceptive_sampled + honest_data

    filtered_data = [
        e for e in all_data
        if (
            ("final" in e and isinstance(e["final"], str) and e["final"].strip()) or
            ("prompt" in e and "output" in e
             and isinstance(e["prompt"], str)
             and isinstance(e["output"], str))
        )
    ]

    texts = []
    labels = []

    for e in filtered_data:

                                         
        if "prompt" in e and "output" in e:

            chat_str = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": e["prompt"]},
                    {"role": "assistant", "content": e["output"]},
                ],
                tokenize=False
            )

            texts.append(chat_str)
            labels.append(e["label"])
            continue

                               
        if "final" in e and isinstance(e["final"], str):
            texts.append(e["final"])
            labels.append(e["label"])
            continue

    print(f"Total usable texts: {len(texts)}")
    return texts, labels



def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(
            f"{prefix} | GPU mem allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB"
        )


def extract_activations(
    texts,
    tokenizer,
    model,
    batch_size=8,
    max_length=512,
    use_mean=False,
    return_both=False,
):
    model.eval()
    model.config.output_hidden_states = True

    total_batches = (len(texts) + batch_size - 1) // batch_size
    start_time = time.time()

    logging.info(f"[ACT] Starting activation extraction")
    logging.info(f"[ACT] Total samples: {len(texts)} | Batch size: {batch_size} | Total batches: {total_batches}")

    all_layers_activations = []
    all_layers_last = []
    all_layers_mean = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
        batch_start = time.time()
        batch = texts[i:i + batch_size]

        logging.info(f"[ACT] Batch {batch_idx+1}/{total_batches} — Tokenizing")

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        logging.info(
            f"[ACT] Batch {batch_idx+1} — input shape: {inputs['input_ids'].shape}"
        )

        logging.info(f"[ACT] Batch {batch_idx+1} — Forward pass")
        with torch.no_grad():
            outputs = model(**inputs)

        logging.info(
            f"[ACT] Batch {batch_idx+1} — Forward done | "
            f"{len(outputs.hidden_states)-1} layers"
        )

        log_gpu_memory(f"[ACT] Batch {batch_idx+1}")

        hidden_states = outputs.hidden_states[1:]                        

                              
        last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
        batch_last = torch.stack(
            [h[torch.arange(h.size(0)), last_token_indices] for h in hidden_states],
            dim=1
        )

                        
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        batch_mean = []
        for h in hidden_states:
            masked_sum = (h * attention_mask).sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            batch_mean.append(masked_sum / lengths)
        batch_mean = torch.stack(batch_mean, dim=1)

        if return_both:
            all_layers_last.append(batch_last.cpu().float())
            all_layers_mean.append(batch_mean.cpu().float())
        else:
            batch_processed = batch_mean if use_mean else batch_last
            all_layers_activations.append(batch_processed.cpu().float())

        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time

        logging.info(
            f"[ACT] Batch {batch_idx+1} complete | "
            f"{batch_time:.2f}s batch | {elapsed/60:.2f} min total"
        )

    if return_both:
        return {
            "last_token": torch.cat(all_layers_last, dim=0).numpy(),
            "mean": torch.cat(all_layers_mean, dim=0).numpy(),
        }

    return torch.cat(all_layers_activations, dim=0).numpy()


def _get_activation_cache_paths(model_name, dataset_id, use_mean, base_cache_dir="cache/activations"):
    """
    Build paths for cached activations and labels.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))             
    safe_model = model_name.replace("/", "_")
    mode = "mean" if use_mean else "last_token"

    cache_dir = os.path.join(base_dir, base_cache_dir, safe_model, dataset_id, mode)
    os.makedirs(cache_dir, exist_ok=True)

    activations_path = os.path.join(cache_dir, "activations.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")
    return activations_path, labels_path


def extract_activations_cached(
    texts,
    labels,
    tokenizer,
    model,
    batch_size=8,
    use_mean=False,
    save_both=True,
    model_name="model",
    dataset_id="default",
    force_load=False,
):
    activations_path, labels_path = _get_activation_cache_paths(
        model_name=model_name,
        dataset_id=dataset_id,
        use_mean=use_mean,
    )

    mean_path, mean_labels_path = _get_activation_cache_paths(
        model_name=model_name,
        dataset_id=dataset_id,
        use_mean=True,
    )
    last_path, last_labels_path = _get_activation_cache_paths(
        model_name=model_name,
        dataset_id=dataset_id,
        use_mean=False,
    )

    if save_both:
        if os.path.exists(mean_path) and os.path.exists(last_path):
            print(f"[Cache] Loaded cached activations (both modes) for '{dataset_id}'")
            return np.load(mean_path if use_mean else last_path)

    if force_load:
        if not os.path.exists(activations_path):
            raise FileNotFoundError(
                f"[Cache] force_load=True, but cached activations missing:\n{activations_path}"
            )
        return np.load(activations_path)

    if os.path.exists(activations_path) and os.path.exists(labels_path):
        acts = np.load(activations_path)
        cached_labels = np.load(labels_path)
        if acts.shape[0] == len(texts) and len(cached_labels) == len(labels):
            print(f"[Cache] Loaded activations for '{dataset_id}'")
            return acts

    if model is None:
        raise RuntimeError("Cache missing and model=None (train_only mode).")

                       
    if save_both:
        acts = extract_activations(
            texts,
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=512,
            return_both=True,
        )

        np.save(mean_path, acts["mean"])
        np.save(last_path, acts["last_token"])
        np.save(mean_labels_path, np.array(labels))
        np.save(last_labels_path, np.array(labels))

        return acts["mean"] if use_mean else acts["last_token"]

    else:
        acts = extract_activations(
            texts,
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=512,
            use_mean=use_mean,
        )
        np.save(activations_path, acts)
        np.save(labels_path, np.array(labels))
        return acts

def get_probe_cache_status(
    model_name,
    train_dataset_id,
    use_mean,
    classifier_names,
    num_layers,
    base_cache_dir="cache/probes"
):
    base_dir = os.path.dirname(os.path.dirname(__file__))             
    safe_model = model_name.replace("/", "_")
    mode = "mean" if use_mean else "last_token"

    cache_dir = os.path.join(base_dir, base_cache_dir, safe_model, train_dataset_id, mode)
    os.makedirs(cache_dir, exist_ok=True)

    meta_path = os.path.join(cache_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return cache_dir, False

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except:
        return cache_dir, False

    print("DEBUG → Expected classifiers:", classifier_names)
    print("DEBUG → Metadata classifiers:", meta.get("classifiers", []))
    print("DEBUG → Expected num_layers:", num_layers)
    print("DEBUG → Metadata num_layers:", meta.get("num_layers"))

                                                          
    if meta.get("num_layers") != num_layers:
        return cache_dir, False
    if not set(classifier_names).issubset(set(meta.get("classifiers", []))):
        return cache_dir, False

                                  
    for clf in classifier_names:
        for layer in range(num_layers):
            pkl_path = os.path.join(cache_dir, f"{clf}_layer{layer}.pkl")
            if not os.path.exists(pkl_path):
                return cache_dir, False

    return cache_dir, True

def _save_probes(cache_dir, classifier_names, trained_models, val_results):
                   
    meta = {
        "classifiers": classifier_names,
        "num_layers": len(next(iter(trained_models.values()))),
        "val_results": val_results
    }
    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

                                    
    for clf_name, models in trained_models.items():
        for layer_idx, model in enumerate(models):

                                
            if hasattr(model, "predict"):
                path = os.path.join(cache_dir, f"{clf_name}_layer{layer_idx}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(model, f)

def _load_probes(cache_dir, classifier_names):
    trained_models = {}

    with open(os.path.join(cache_dir, "metadata.json"), "r") as f:
        meta = json.load(f)

    num_layers = meta["num_layers"]
    val_results = meta.get("val_results", None)

    if val_results is not None:
        val_results = {
            k: v for k, v in val_results.items()
            if k in classifier_names
        }

    for clf_name in classifier_names:
        models_per_layer = []
        for layer_idx in range(num_layers):
            pkl_path = os.path.join(cache_dir, f"{clf_name}_layer{layer_idx}.pkl")
            with open(pkl_path, "rb") as f:
                models_per_layer.append(pickle.load(f))

        trained_models[clf_name] = models_per_layer

    return trained_models, val_results


def train_classifiers(all_layers_activations, labels, cfg, random_seed, train_dataset_id):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    X = all_layers_activations
    y = np.array(labels)
    num_layers = X.shape[1]

    idx_train, idx_val, y_train, y_val = train_test_split(
        np.arange(len(y)),
        y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y,
    )

    sklearn_models = {
        "lr": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "lr_centered": CenteredLogisticRegression(C=1.0),
        "rf": RandomForestClassifier(
            n_estimators=1000,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=42,
            verbose=0
        ),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=300, random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "svm": SVC(kernel="rbf", C=2, gamma="scale", probability=True),
    }

    selected = [c.lower() for c in cfg["classifiers"]]
    print(f"\nTraining {len(selected)} classifiers across {num_layers} layers...\n")

    model_name = cfg["model"]["name"]
    use_mean = cfg["experiment"]["mean"]
    classifier_names = [c.upper() for c in selected]

    # --- Probe caching: check if we already trained these probes ---
    cache_dir, cache_exists = get_probe_cache_status(
        model_name=model_name,
        train_dataset_id=train_dataset_id,
        use_mean=use_mean,
        classifier_names=classifier_names,
        num_layers=num_layers,
    )
        
    if force_retrain_probes:
        cache_exists = False

    if cache_exists:
        print(f"[Cache] Loaded cached probes from {cache_dir}")
        trained_models, val_results = _load_probes(cache_dir, classifier_names)
        results = val_results
        return {"results": results, "models": trained_models}
    
    else:
        print(f"Probes for '{train_dataset_id}' not found → training new probes...")


    results = {name.upper(): {"acc": [], "auc": []} for name in selected}
    trained_models = {name.upper(): [] for name in selected}

    for layer_id in range(num_layers):
        X_layer = X[:, layer_id, :]
        X_train, X_val = X_layer[idx_train], X_layer[idx_val]

        for name, model in sklearn_models.items():
            if name in selected:
                try:
                    clf = clone(model)
                    clf.fit(X_train, y_train)
                    y_pred_val = clf.predict(X_val)
                    acc_val = accuracy_score(y_val, y_pred_val)
                    auc_val = (
                        roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
                        if len(np.unique(y_val)) >= 2 else np.nan
                    )
                except Exception as e:
                    print(f"⚠️ {name.upper()} failed on layer {layer_id}: {e}")
                    acc_val, auc_val = np.nan, np.nan
    
                trained_models[name.upper()].append(clf)
    
                results[name.upper()]["acc"].append(acc_val)
                results[name.upper()]["auc"].append(auc_val)

        summary = [
            f"Layer {layer_id:2d}: {n}: ACC={results[n]['acc'][-1]:.3f}, AUC={np.nan_to_num(results[n]['auc'][-1], nan=0):.3f}"
            for n in results
        ]
        print(f"{' | '.join(summary)}")

    layers = np.arange(num_layers)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for name in results:
        plt.plot(layers, results[name]["acc"], marker="o", label=name)
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Layer")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for name in results:
        plt.plot(layers, results[name]["auc"], marker="o", label=name)
    plt.xlabel("Layer Index")
    plt.ylabel("ROC AUC")
    plt.title("Validation AUC per Layer")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.join(base_dir, "output_plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "output_plots", "layerwise_val_performance.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nValidation performance plot saved to: {out_path}\n")

    _save_probes(cache_dir, classifier_names, trained_models, results)
    print(f"[Cache] Saved trained probes + val results to: {cache_dir}")

    return {"results": results, "models": trained_models}


def evaluate_classifiers(trained_models, all_layers_activations_test, labels_test):
    num_layers = all_layers_activations_test.shape[1]
    num_samples = len(labels_test)                                               
    
    results_test = {
        name: {"acc": [], "auc": [], "fpr": [], "fnr": [], "preds_raw": [], "probs_raw": []} 
        for name in trained_models
    }

    print(f"\nEvaluating {len(trained_models)} classifier types across {num_layers} layers...\n")

    for name, model_list in trained_models.items():
        print(f"Classifier: {name}")
        for layer_id in range(num_layers):
            clf = model_list[layer_id]
            X_test = all_layers_activations_test[:, layer_id, :]

                                         
            if hasattr(clf, "predict"):
                try:
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(labels_test, y_pred)
                    
                    tn, fp, fn, tp = confusion_matrix(labels_test, y_pred, labels=[1, 0]).ravel()         
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                    if hasattr(clf, "predict_proba"):
                        scores = clf.predict_proba(X_test)[:, 1]
                        auc = float(roc_auc_score(labels_test, scores)) if len(np.unique(labels_test)) >= 2 else np.nan
                    else:
                        scores = np.full(num_samples, np.nan)
                        auc = np.nan
                except Exception as e:
                    print(f"  Layer {layer_id}: sklearn model failed ({e})")
                    acc, auc, fpr, fnr = np.nan, np.nan, np.nan, np.nan
                    y_pred = np.full(num_samples, np.nan)
                    scores = np.full(num_samples, np.nan)
            else:
                print(f"  Layer {layer_id}: Unrecognized model type")
                acc, auc, fpr, fnr = np.nan, np.nan, np.nan, np.nan
                y_pred = np.full(num_samples, np.nan)
                scores = np.full(num_samples, np.nan)

            results_test[name]["acc"].append(acc)
            results_test[name]["auc"].append(auc)
            results_test[name]["fpr"].append(fpr)
            results_test[name]["fnr"].append(fnr)
            results_test[name]["preds_raw"].append(y_pred)
            results_test[name]["probs_raw"].append(scores)
            
            print(f"  Layer {layer_id:2d} | ACC={acc:.3f} | AUC={auc:.3f} | FPR={fpr:.3f} | FNR={fnr:.3f}")

        acc_mean = np.nanmean(results_test[name]["acc"])
        auc_mean = np.nanmean(results_test[name]["auc"])
        print(f"  Summary → ACC={acc_mean:.3f} | AUC={auc_mean:.3f}\n")

    print("Testing complete.\n")
    return results_test
    


def analyze_layers(all_layers_activations, labels, all_layers_activations_target, labels_target, cfg, random_seed):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    X = all_layers_activations
    y = np.array(labels)
    X_target = all_layers_activations_target
    y_target = np.array(labels_target)

    num_layers = X.shape[1]

                                                          
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)),
        y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y
    )

    sklearn_models = {
        "lr": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=2000, random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,128), max_iter=300, random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=200, random_state=42),
                                                                                                    
        "svm": SVC(kernel="rbf", C=2, gamma="scale", probability=True),
    }

    selected = [c.lower() for c in cfg["classifiers"]]
                                                                                             
    results = {name.upper(): {'acc': [], 'auc': []} for name in selected}
    results_target = {name.upper(): {'acc': [], 'auc': []} for name in selected}

    print(f"Training {len(selected)} classifiers on {num_layers} layers...\n")

    for layer_id in range(num_layers):
        X_layer = X[:, layer_id, :]
        X_train, X_test = X_layer[X_train_idx], X_layer[X_test_idx]
        X_layer_target = X_target[:, layer_id, :]

                                
        for name, model in sklearn_models.items():
            if name in selected:
                model.fit(X_train, y_train)

                                                                             
                y_pred_val = model.predict(X_test)
                acc_val = accuracy_score(y_test, y_pred_val)

                                                                                             
                try:
                    y_pred_tgt = model.predict(X_layer_target)
                    acc_tgt = accuracy_score(y_target, y_pred_tgt)
                except Exception:
                    acc_tgt = np.nan

                                                                      
                if len(np.unique(y_test)) >= 2:
                    scores_val = _get_sklearn_scores(model, X_test)
                    try:
                        auc_val = float(roc_auc_score(y_test, scores_val))
                    except Exception:
                        auc_val = np.nan
                else:
                    auc_val = np.nan

                if len(np.unique(y_target)) >= 2:
                    try:
                        scores_tgt = _get_sklearn_scores(model, X_layer_target)
                        auc_tgt = float(roc_auc_score(y_target, scores_tgt))
                    except Exception:
                        auc_tgt = np.nan
                else:
                    auc_tgt = np.nan

                results[name.upper()]['acc'].append(acc_val)
                results[name.upper()]['auc'].append(auc_val)
                results_target[name.upper()]['acc'].append(acc_tgt)
                results_target[name.upper()]['auc'].append(auc_tgt)


                                              
        summary = []
        for n in results:
            val_acc = results[n]['acc'][-1]
            val_auc = results[n]['auc'][-1]
            tgt_acc = results_target[n]['acc'][-1]
            tgt_auc = results_target[n]['auc'][-1]
            summary.append(f"{n}: acc {val_acc:.3f} (tgt {tgt_acc:.3f}) | auc {np.nan_to_num(val_auc, nan=0):.3f} (tgt {np.nan_to_num(tgt_auc, nan=0):.3f})")
        print(f"Layer {layer_id:2d} → " + " | ".join(summary))

                                                           
    plt.figure(figsize=(12, 8))
    layers = range(num_layers)

                      
    plt.subplot(2, 1, 1)
    for name in results:
        plt.plot(layers, results[name]['acc'], marker='o', label=f"{name} (val)")
        plt.plot(layers, results_target[name]['acc'], marker='x', linestyle='--', label=f"{name} (tgt)")
    plt.ylabel("Accuracy")
    plt.title("Validation vs Target Accuracy by Layer")
    plt.legend()
    plt.grid(True)

                 
    plt.subplot(2, 1, 2)
    for name in results:
        plt.plot(layers, results[name]['auc'], marker='o', label=f"{name} (val)")
        plt.plot(layers, results_target[name]['auc'], marker='x', linestyle='--', label=f"{name} (tgt)")
    plt.xlabel("Layer index")
    plt.ylabel("ROC AUC")
    plt.title("Validation vs Target ROC AUC by Layer")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(os.path.join(base_dir, "output_plots"), exist_ok=True)
    plot_file = os.path.join(base_dir, "output_plots", "layerwise_probe_acc_auc_transfer.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print(f"\nSaved plot to {plot_file}\n")

                         
    return {"val": results, "target": results_target}


from sklearn.decomposition import PCA
