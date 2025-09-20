# pdiagnosis_parallel.py
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import networkx as nx
import os
import re
import json
import ast
import time
import warnings
import concurrent.futures
from utils.CSVFormat import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import glob

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("mode.chained_assignment", None)  # None is equivalent to 'ignore'

# Global hyperparameter variables
T_WINDOW = 30  # Sliding window size
W1_WEIGHT = 0.5  # KDE weight
W2_WEIGHT = 0.3  # WMA weight
W3_WEIGHT = 0.2  # delta_R weight
BANDWIDTH = 1.0  # KDE bandwidth parameter


def fit_kde_model_for_span(df, span_name, bandwidth=None):
    """Fit KDE model for specified span"""
    if bandwidth is None:
        bandwidth = BANDWIDTH
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    span_df = df[df["name"] == span_name]
    kpi_train_values = span_df.filter(regex="^metric@").values
    kde.fit(kpi_train_values)
    return kde


def compute_kpi_anomalies_by_span(train_df, predict_df, nodes, T=None, w1=None, w2=None, w3=None):
    """Calculate KPI anomaly scores for each span - fully vectorized version"""
    if T is None:
        T = T_WINDOW
    if w1 is None:
        w1 = W1_WEIGHT
    if w2 is None:
        w2 = W2_WEIGHT
    if w3 is None:
        w3 = W3_WEIGHT

    metric_columns = [col for col in train_df.columns if col.startswith("metric@")]

    # Optimization: pre-group all data to avoid repeated filtering
    train_groups = {node: train_df[train_df["name"] == node] for node in nodes}
    predict_groups = {node: predict_df[predict_df["name"] == node] for node in nodes}

    # Pre-train all KDE models
    kde_models = {}
    for node in nodes:
        if len(train_groups[node]) > 0:
            kde = KernelDensity(kernel="gaussian", bandwidth=BANDWIDTH)
            kpi_train_values = train_groups[node][metric_columns].values
            kde.fit(kpi_train_values)
            kde_models[node] = kde

    for node in nodes:
        if node not in kde_models:
            continue

        kde = kde_models[node]
        span_train_df = train_groups[node]
        span_predict_df = predict_groups[node]

        # Vectorized processing of training set
        if len(span_train_df) > 0:
            anomalies_train = []
            anomaly_metrics_train = []

            kpi_data = span_train_df[metric_columns].values
            # Vectorized calculation of basic statistics for all rows
            zero_mask = np.all(kpi_data == 0, axis=1)

            for i, (idx, row) in enumerate(span_train_df.iterrows()):
                kpi_values = kpi_data[i]

                if zero_mask[i]:
                    anomaly_score = 0
                    sorted_metrics = metric_columns[:]
                else:
                    wma = np.mean(kpi_values[-T:]) if len(kpi_values) >= T else np.mean(kpi_values)
                    abs_diff = np.abs(kpi_values[-1] - wma)

                    mean_kpi = np.mean(kpi_values)
                    if mean_kpi == 0:
                        delta_R = np.max(np.diff(kpi_values)) if len(kpi_values) > 1 else 0
                    else:
                        delta_R = (
                            np.max(np.diff(kpi_values)) / mean_kpi if len(kpi_values) > 1 else 0
                        )

                    p_value = kde.score_samples(kpi_values.reshape(1, -1))[0]
                    anomaly_score = w1 * p_value + w2 * abs_diff + w3 * delta_R

                    # Fast sorting
                    sorted_indices = np.argsort(np.abs(kpi_values))[::-1]
                    sorted_metrics = [metric_columns[j] for j in sorted_indices]

                anomalies_train.append(anomaly_score)
                anomaly_metrics_train.append(str(sorted_metrics))

            # Batch update to avoid row-by-row .loc operations
            train_df.loc[span_train_df.index, "anomaly_score"] = anomalies_train
            train_df.loc[span_train_df.index, "anomaly_metric"] = anomaly_metrics_train

        # Vectorized processing of prediction set - similar logic
        if len(span_predict_df) > 0:
            anomalies_predict = []
            anomaly_metrics_predict = []

            kpi_data = span_predict_df[metric_columns].values
            zero_mask = np.all(kpi_data == 0, axis=1)

            for i, (idx, row) in enumerate(span_predict_df.iterrows()):
                kpi_values = kpi_data[i]

                if zero_mask[i]:
                    anomaly_score = 0
                    sorted_metrics = metric_columns[:]
                else:
                    wma = np.mean(kpi_values[-T:]) if len(kpi_values) >= T else np.mean(kpi_values)
                    abs_diff = np.abs(kpi_values[-1] - wma)

                    mean_kpi = np.mean(kpi_values)
                    if mean_kpi == 0:
                        delta_R = np.max(np.diff(kpi_values)) if len(kpi_values) > 1 else 0
                    else:
                        delta_R = (
                            np.max(np.diff(kpi_values)) / mean_kpi if len(kpi_values) > 1 else 0
                        )

                    p_value = kde.score_samples(kpi_values.reshape(1, -1))[0]
                    anomaly_score = w1 * p_value + w2 * abs_diff + w3 * delta_R

                    sorted_indices = np.argsort(np.abs(kpi_values))[::-1]
                    sorted_metrics = [metric_columns[j] for j in sorted_indices]

                anomalies_predict.append(anomaly_score)
                anomaly_metrics_predict.append(str(sorted_metrics))

            predict_df.loc[span_predict_df.index, "anomaly_score"] = anomalies_predict
            predict_df.loc[span_predict_df.index, "anomaly_metric"] = anomaly_metrics_predict

    return train_df, predict_df


def apply_3sigma(series, mean=None, std_dev=None):
    """3-sigma anomaly detection - vectorized version"""
    if mean is None or std_dev is None:
        mean = series.mean()
        std_dev = series.std()
    if std_dev == 0:
        return pd.Series(0, index=series.index), mean, std_dev

    # Vectorized calculation, avoiding apply
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    anomaly_flags = ((series < lower_bound) | (series > upper_bound)).astype(int)
    return anomaly_flags, mean, std_dev


def fill_zero_with_neighbors(df, metric_columns):
    """Zero value filling - optimized version"""
    df = df.copy()  # Avoid modifying original data

    for col in metric_columns:
        # Optimization: use pandas interpolate method, much faster than nested loops
        mask = (df[col] == 0) | df[col].isna()
        if mask.any():
            # First interpolate with neighboring values
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            # If NaN still exists, use forward and backward fill
            df[col] = df[col].fillna(method="bfill").fillna(method="ffill")
            # If entire column is 0/NaN, fill with 0
            df[col] = df[col].fillna(0)

    return df


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate HR@k"""
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        hit = any(root in predicted_roots[:k] for root in actual_roots)
        hr_k.append(hit)
    return np.mean(hr_k)


def ndcg_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate NDCG@k"""
    ndcg_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        dcg = 0
        for i, predicted_root in enumerate(predicted_roots[:k]):
            rel = 1 if predicted_root in actual_roots else 0
            dcg += (2**rel - 1) / np.log2(i + 2)

        idcg = 0
        ideal_relevance = [1 if root in actual_roots else 0 for root in predicted_roots[:k]]
        ideal_relevance.sort(reverse=True)
        for i, rel in enumerate(ideal_relevance):
            idcg += (2**rel - 1) / np.log2(i + 2)

        ndcg_k.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcg_k)


def write_results_to_csv(
    service_labels,
    service_lists,
    nonservice_labels,
    nonservice_lists,
    ks=[1, 3, 5],
    filename="evaluation_results.csv",
):
    """Output results to CSV file"""
    results = []
    for k in ks:
        hr_k = hit_ratio_at_k(service_labels, service_lists, k)
        ndcg_k = ndcg_at_k(service_labels, service_lists, k)
        nonservicehr_k = hit_ratio_at_k(nonservice_labels, nonservice_lists, k)
        nonservicendcg_k = ndcg_at_k(nonservice_labels, nonservice_lists, k)
        # Save results
        results.append([k, hr_k, ndcg_k, nonservicehr_k, nonservicendcg_k])

    # Use pandas DataFrame to store results
    df = pd.DataFrame(
        results, columns=["k", "HR@k", "NDCG@k", "nonserviceHR@k", "nonserviceNDCG@k"]
    )
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

    # Add this line to print results
    print("\nEvaluation Results:")
    print(df)

    return df


def safe_literal_eval(s):
    """Safely parse string to Python object"""
    try:
        if isinstance(s, str):
            # Handle NumPy data types
            if "np.int64" in s:
                # Convert np.int64(x) in string to x
                s = s.replace("np.int64(", "").replace(")", "")
            return ast.literal_eval(s)
        return s
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Unable to parse string: {s}")
        print(f"Error details: {str(e)}")
        return []


def sub_main(train_path, predict_path, failure_label_path, nodes):
    """Main diagnosis process - using improved detection algorithm"""
    print(f"\n=== Starting folder processing ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)

    try:
        # Load data
        print("Loading data...", flush=True)
        stage1_start = time.time()

        # Read data with specified amount
        train_df = pd.read_csv(train_path + "/data_final.csv")
        predict_df = pd.read_csv(predict_path + "/data_final.csv")

        # Data volume control
        if len(train_df) > 100000:
            print("Training set rows > 100k, switching to data_final_sampled.csv")
            train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
            if len(train_df) > 30000:
                print("Training set rows > 30k, switching to data_final_sampled_005.csv")
                train_df = pd.read_csv(train_path + "/data_final_sampled_005.csv")
        if len(predict_df) > 100000:
            print("Prediction set rows > 100k, switching to data_final_sampled.csv")
            predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")
            if len(predict_df) > 30000:
                print("Prediction set rows > 30k, switching to data_final_sampled_005.csv")
                predict_df = pd.read_csv(predict_path + "/data_final_sampled_005.csv")

        def sample_data(df, size, select_outstanding=False, random_state=42):
            """Sample data: select_outstanding=True selects outstanding data (high variance), False selects stable data (low variance)"""
            if len(df) <= size:
                return df

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                variances = df[numeric_cols].var(axis=1)
                indices = (
                    variances.nlargest(size).index
                    if select_outstanding
                    else variances.nsmallest(size).index
                )
                return df.loc[indices]
            else:
                return df.sample(n=size, random_state=random_state)

        TRAIN_SIZE = 1500
        PREDICT_SIZE = 100

        # Modify sampling part
        train_df = sample_data(train_df, TRAIN_SIZE, select_outstanding=False)  # Select stable data
        predict_df = sample_data(predict_df, PREDICT_SIZE, select_outstanding=True)  # Select outstanding data

        load_end_time = time.time()
        print(f"Training data size: {train_df.shape}", flush=True)
        print(f"Prediction data size: {predict_df.shape}", flush=True)

    except:
        print("Failed to load data")
        return None, None, None, None

    stage1_end = time.time()
    print(f"Stage 1 time: {stage1_end - stage1_start:.2f} seconds")

    print("Processing metric data...")
    stage2_start = time.time()

    metric_columns = [col for col in train_df.columns if col.startswith("metric@")]
    train_df = fill_zero_with_neighbors(train_df, metric_columns)
    predict_df = fill_zero_with_neighbors(predict_df, metric_columns)

    # Keep common columns
    common_columns = train_df.columns.intersection(predict_df.columns)
    train_df = train_df[common_columns]
    predict_df = predict_df[common_columns]
    print(f"Number of common columns: {len(common_columns)}")

    stage2_end = time.time()
    print(f"Stage 2 time: {stage2_end - stage2_start:.2f} seconds")

    # Get all unique name values in data
    data_names = set(train_df["name"].unique()) | set(predict_df["name"].unique())
    # Take intersection of nodes and data names
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    print("Calculating KPI anomaly scores...")
    stage4_start = time.time()
    # Use improved vectorized algorithm
    train_df, predict_df = compute_kpi_anomalies_by_span(train_df, predict_df, nodes)
    stage4_end = time.time()
    print(f"Stage 4 time: {stage4_end - stage4_start:.2f} seconds")

    print(f"Processing node data ({len(nodes)} nodes)...")
    stage5_start = time.time()
    # Define log columns
    log_columns = [col for col in train_df.columns if col.startswith("log@")]

    # Pre-group all data
    train_groups = {node: train_df[train_df["name"] == node].copy() for node in nodes}
    predict_groups = {node: predict_df[predict_df["name"] == node].copy() for node in nodes}

    # Pre-calculate all log baseline data
    log_baselines = {}
    for node in nodes:
        if len(train_groups[node]) > 0:
            log_baselines[node] = train_groups[node].iloc[0][log_columns].values
        else:
            log_baselines[node] = np.zeros(len(log_columns))

    # Batch process all nodes
    train_all_dfs = {}
    predict_all_dfs = {}

    for name in nodes:
        train_group = train_groups[name]
        predict_group = predict_groups[name]

        if len(train_group) == 0:
            continue

        # Vectorized 3-sigma anomaly detection
        try:
            train_group["anomaly_detection"], mean_train, std_dev_train = apply_3sigma(
                train_group["duration"]
            )
            if len(predict_group) > 0:
                predict_group["anomaly_detection"], _, _ = apply_3sigma(
                    predict_group["duration"], mean_train, std_dev_train
                )
        except:
            print("Using elapsedTime")
            train_group["anomaly_detection"], mean_train, std_dev_train = apply_3sigma(
                train_group["elapsedTime"]
            )
            if len(predict_group) > 0:
                predict_group["anomaly_detection"], _, _ = apply_3sigma(
                    predict_group["elapsedTime"], mean_train, std_dev_train
                )

        # Vectorized log anomaly detection
        baseline = log_baselines[name]

        # Training set log anomaly detection
        if len(train_group) > 0:
            train_log_data = train_group[log_columns].values
            # Vectorized comparison
            anomaly_log_matrix = (train_log_data != baseline).astype(int)
            train_group["anomaly_log"] = [str(row.tolist()) for row in anomaly_log_matrix]

        # Prediction set log anomaly detection
        if len(predict_group) > 0:
            predict_log_data = predict_group[log_columns].values
            anomaly_log_matrix = (predict_log_data != baseline).astype(int)
            predict_group["anomaly_log"] = [str(row.tolist()) for row in anomaly_log_matrix]

        # Update dictionary
        train_all_dfs[f"train_{name}_df"] = train_group
        predict_all_dfs[f"predict_{name}_df"] = predict_group

    stage5_end = time.time()
    print(f"Stage 5 time: {stage5_end - stage5_start:.2f} seconds", flush=True)

    print("Merging data and processing root causes...")
    stage6_start = time.time()

    # Merge all predict_{name}_df data
    combined_df = pd.concat(
        [
            predict_all_dfs[f"predict_{name}_df"]
            for name in nodes
            if f"predict_{name}_df" in predict_all_dfs
        ]
    )
    # Group by trace_rowid
    grouped = combined_df.groupby("trace_rowid")

    combined_df["anomaly_log_count"] = combined_df["anomaly_log"].apply(
        lambda x: sum(safe_literal_eval(x)) if isinstance(x, str) else 0
    )
    global_sorted_df = combined_df.sort_values(
        by=["anomaly_detection", "anomaly_log_count", "anomaly_score"],
        ascending=[False, False, False],
    )
    global_sorted_names = global_sorted_df["name"].tolist()

    # Get labels
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]
    nonservice_label = failure_label[label_str]["nonservice"]
    print(f"Service root cause labels: {service_label}")
    print(f"Non-service root cause labels: {nonservice_label}")

    service_labels_sub = []
    nonservice_labels_sub = []
    service_lists_sub = []
    nonservice_lists_sub = []

    # Process each trace_rowid
    for i, (trace_rowid, group) in enumerate(grouped):
        # Calculate anomaly_log_count
        group["anomaly_log_count"] = group["anomaly_log"].apply(
            lambda x: sum(safe_literal_eval(x)) if isinstance(x, str) else 0
        )

        # First level selection: rows with anomaly_detection=1
        anomaly_detection_group = group[group["anomaly_detection"] == 1]
        anomaly_detection_group = anomaly_detection_group.sort_values(
            by=["anomaly_log_count", "anomaly_score"], ascending=[False, False]
        )

        # Second level selection: rows with anomaly_detection=0
        non_anomaly_detection_group = group[group["anomaly_detection"] == 0]
        non_anomaly_detection_group = non_anomaly_detection_group.sort_values(
            by=["anomaly_log_count", "anomaly_score"], ascending=[False, False]
        )

        # Merge results
        final_group = pd.concat([anomaly_detection_group, non_anomaly_detection_group])
        sorted_names = final_group["name"].tolist()

        # Build nonservice sorted list
        new_sorted_list = []
        for name in sorted_names:
            current_row = final_group[final_group["name"] == name].iloc[0]
            anomaly_log_list = safe_literal_eval(current_row["anomaly_log"])
            anomaly_metric_list = safe_literal_eval(current_row["anomaly_metric"])

            if 1 in anomaly_log_list:
                changed_logs = [
                    log_name
                    for log_name, log_value in zip(current_row.index, anomaly_log_list)
                    if log_value == 1 and log_name.startswith("log@")
                ]
                for log_name in changed_logs:
                    new_sorted_list.append(f"{name}>{log_name}")

            if name != "query":
                for metric in anomaly_metric_list:
                    new_sorted_list.append(f"{name}>{metric}")

        service_lists_sub.append(sorted_names)
        nonservice_lists_sub.append(new_sorted_list)
        service_labels_sub.append(service_label)
        nonservice_labels_sub.append(nonservice_label)

    # Service only keep top results
    from collections import Counter

    merged_service = []
    for lst in service_lists_sub:
        merged_service.extend(lst)
    counter_service = Counter(merged_service)

    top_service = [item for item, _ in counter_service.most_common()]
    # Fallback: if top results are empty, use global sorting to fill
    if not top_service:
        top_service = [n for n in global_sorted_names if n != "query"]

    service_lists_sub = top_service
    service_labels_sub = (
        service_labels_sub[0] if service_labels_sub else service_label
    )  # Keep only one label

    # Nonservice also only keep top results
    merged_nonservice = []
    for lst in nonservice_lists_sub:
        merged_nonservice.extend(lst)
    counter_nonservice = Counter(merged_nonservice)
    top_nonservice = [item for item, _ in counter_nonservice.most_common()]
    # Fallback
    if not top_nonservice:
        fallback_nonservice = []
        for name in global_sorted_names:
            row = global_sorted_df[global_sorted_df["name"] == name].iloc[0]
            anomaly_log_list = safe_literal_eval(row["anomaly_log"])
            anomaly_metric_list = safe_literal_eval(row["anomaly_metric"])
            changed_logs = [
                log_name
                for log_name, log_value in zip(row.index, anomaly_log_list)
                if log_value == 1 and log_name.startswith("log@")
            ]
            for log_name in changed_logs:
                fallback_nonservice.append(f"{name}>{log_name}")
            if name != "query":
                for metric in anomaly_metric_list:
                    fallback_nonservice.append(f"{name}>{metric}")
            if len(fallback_nonservice) >= 3:
                break
        top_nonservice = fallback_nonservice

    nonservice_lists_sub = top_nonservice
    nonservice_labels_sub = (
        nonservice_labels_sub[0] if nonservice_labels_sub else nonservice_label
    )  # Keep only one label

    stage6_end = time.time()
    print(f"Stage 6 time: {stage6_end - stage6_start:.2f} seconds")

    total_time = stage6_end - stage1_start
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Time proportion by stage:")
    print(f"Stage 1: {(stage1_end - stage1_start)/total_time*100:.1f}%")
    print(f"Stage 2: {(stage2_end - stage2_start)/total_time*100:.1f}%")
    print(f"Stage 4: {(stage4_end - stage4_start)/total_time*100:.1f}%")
    print(f"Stage 5: {(stage5_end - stage5_start)/total_time*100:.1f}%")
    print(f"Stage 6: {(stage6_end - stage6_start)/total_time*100:.1f}%")

    return service_lists_sub, nonservice_lists_sub, service_labels_sub, nonservice_labels_sub


def run_single_params(params):
    """Run single hyperparameter configuration"""
    T, w1, w2, w3, bandwidth = params

    # Set global parameters
    global T_WINDOW, W1_WEIGHT, W2_WEIGHT, W3_WEIGHT, BANDWIDTH
    T_WINDOW = T
    W1_WEIGHT = w1
    W2_WEIGHT = w2
    W3_WEIGHT = w3
    BANDWIDTH = bandwidth

    print(f"Running parameters: T={T}, w=({w1},{w2},{w3}), bandwidth={bandwidth}")

    try:
        # Read configuration file
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        dependency_graph = config["dependency_graph"]
        jump_folder = config["jump_folder"]
        data_final_dir = config["data_final_dir"]
        normal_data_dir = config["normal_data_dir"]
        output_dir = config["output_dir"]
        failure_label_path = config["failure_label_path"]
        nodes = config["nodes"]

        # Create output directory named by parameters
        param_folder = f"T{T}_w{w1:.1f}_{w2:.1f}_{w3:.1f}_bw{bandwidth:.1f}"
        param_output_dir = os.path.join(output_dir, param_folder)
        os.makedirs(param_output_dir, exist_ok=True)

        # Get all data folders
        folder_names = [
            f
            for f in os.listdir(data_final_dir)
            if os.path.isdir(os.path.join(data_final_dir, f)) and f != "00"
        ]
        folder_names.sort()

        train_path = normal_data_dir
        service_lists = []
        nonservice_lists = []
        service_labels = []
        nonservice_labels = []

        # Store detailed results
        detailed_results = []

        # Process each folder
        for folder_name in folder_names:
            predict_path = data_final_dir + "/" + folder_name
            label_match = re.search(r"(\d+)_", predict_path)
            if not label_match:
                continue

            label_str = label_match.group(1)
            if int(label_str) in jump_folder:
                continue

            try:
                case_start_time = time.time()
                result = sub_main(train_path, predict_path, failure_label_path, nodes)
                if result is None:
                    continue
                service_list, nonservice_list, service_label, nonservice_label = result
                case_end_time = time.time()
                case_runtime = case_end_time - case_start_time

                # Filter query nodes
                service_list = [item for item in service_list if item != "query"]

                service_lists.append(service_list)
                nonservice_lists.append(nonservice_list)
                service_labels.append(service_label)
                nonservice_labels.append(nonservice_label)

                # Save detailed results
                detailed_results.append(
                    {
                        "task_id": label_str,
                        "service_list": service_list,
                        "nonservice_list": nonservice_list,
                        "service_label": service_label,
                        "nonservice_label": nonservice_label,
                        "runtime_seconds": case_runtime,
                    }
                )

            except Exception as e:
                print(f"Error processing folder {folder_name}: {str(e)}")
                continue

        # Save hyperparameter experiment results
        if service_lists:
            # Save evaluation results
            result_path = os.path.join(param_output_dir, "evaluation_results.csv")
            write_results_to_csv(
                service_labels,
                service_lists,
                nonservice_labels,
                nonservice_lists,
                filename=result_path,
            )

            # Save detailed results
            detailed_path = os.path.join(param_output_dir, "detailed_results.csv")
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.to_csv(detailed_path, index=False)

            print(f"Completed parameter combination: T={T}, w=({w1},{w2},{w3}), bandwidth={bandwidth}")
            return param_folder

        return None

    except Exception as e:
        print(f"Parameter combination failed: T={T}, w=({w1},{w2},{w3}), bandwidth={bandwidth}, error: {e}")
        return None


def write_hyperparameter_summary(all_results, output_dir):
    """Summarize all hyperparameter experiment results"""
    if not all_results:
        print("No hyperparameter experiment results to summarize")
        return

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)

    # Save summary results
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "pdiagnose_hyperparameter_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n=== Hyperparameter Tuning Summary Results ===")
    print(f"Total experiments: {len(all_results)}")
    print(f"Summary results saved to: {summary_path}")

    # Sort by HR@1 to show best results
    if len(all_results) > 0:
        best_hr1 = summary_df.loc[summary_df["HR@1"].idxmax()]
        print(f"\nBest HR@1: {best_hr1['HR@1']:.4f}")
        print(
            f"Parameters: T={best_hr1['T']}, w=({best_hr1['w1']},{best_hr1['w2']},{best_hr1['w3']}), bandwidth={best_hr1['bandwidth']}"
        )

    return summary_df


if __name__ == "__main__":
    """Hyperparameter search with multi-threading execution"""
    # Start timing
    start_time = time.time()

    # Read configuration
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    output_dir = config["output_dir"]
    jump_folder = config["jump_folder"]
    data_final_dir = config["data_final_dir"]
    normal_data_dir = config["normal_data_dir"]
    failure_label_path = config["failure_label_path"]
    nodes = config["nodes"]

    # (w1, w2, w3): Control weight allocation of three anomaly detection methods. p_value: KDE probability density estimation (distribution anomaly), abs_diff: sliding window average deviation (trend anomaly), delta_R: relative change rate (sudden change anomaly)
    # BANDWIDTH: Control smoothness of KDE kernel density estimation. Larger values are smoother
    # Hyperparameter combinations
    param_combinations = [
        (T, w1, w2, w3, bw)
        for T in [30, 100, 500]
        for w1, w2, w3 in [(0.7, 0.15, 0.15), (0.15, 0.7, 0.15), (0.15, 0.15, 0.7)]
        for bw in [0.5, 1.5, 3]
    ]

    print(f"Starting pdiagnose hyperparameter search, total {len(param_combinations)} parameter sets")

    # Check completed experiments to avoid duplicate runs
    pending = []
    for p in param_combinations:
        T, w1, w2, w3, bw = p
        param_folder = f"T{T}_w{w1:.1f}_{w2:.1f}_{w3:.1f}_bw{bw:.1f}"
        result_file = os.path.join(output_dir, param_folder, "evaluation_results.csv")
        if not os.path.exists(result_file):
            pending.append(p)

    print(f"Remaining {len(pending)} parameter sets to run")

    if len(pending) > 0:
        # Parallel execution of hyperparameter tuning
        max_workers = max(int(os.cpu_count() / 10 - 1), 1)
        all_results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_params, p): p for p in pending}
            completed = 0

            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()  # 10 minute timeout
                    if result:
                        all_results.append(result)
                        print(f"Progress [{completed}/{len(pending)}]: Completed {result}")
                    else:
                        print(f"Progress [{completed}/{len(pending)}]: Skipped")
                except Exception as e:
                    params = futures[future]
                    print(f"Progress [{completed}/{len(pending)}]: Failed - parameters{params}, error: {str(e)}")

        print(f"Hyperparameter tuning completed, successfully ran {len(all_results)} parameter sets")

    end_time = time.time()  # End timing
    run_time = end_time - start_time
    print(f"\nTotal runtime: {run_time:.4f} seconds")
    print("=== Program execution completed ===")

    # Summarize HR@1 results from all parameter folders and output sorted results
    result_rows = []
    hr_cols, ndcg_cols = set(), set()

    for param_folder in os.listdir(output_dir):
        param_path = os.path.join(output_dir, param_folder)
        eval_csv = os.path.join(param_path, "evaluation_results.csv")
        if os.path.isdir(param_path) and os.path.exists(eval_csv):
            try:
                df = pd.read_csv(eval_csv)
                row = {"param_folder": param_folder}
                for _, r in df.iterrows():
                    k = int(r["k"])
                    row[f"HR@{k}"] = r["HR@k"]
                    row[f"NDCG@{k}"] = r["NDCG@k"]
                    hr_cols.add(f"HR@{k}")
                    ndcg_cols.add(f"NDCG@{k}")
                    # If there are nonservice metrics, add them too
                    if "nonserviceHR@k" in r.index:
                        row[f"nonserviceHR@{k}"] = r["nonserviceHR@k"]
                    if "nonserviceNDCG@k" in r.index:
                        row[f"nonserviceNDCG@{k}"] = r["nonserviceNDCG@k"]
                result_rows.append(row)
            except Exception as e:
                print(f"Failed to read {eval_csv}: {e}")

    if result_rows:
        # Ensure metrics and k are ordered
        def colkey(x):
            if "@" in x:
                return (int(x.split("@")[1]), x.split("@")[0])
            return (-1, x)

        extra_cols = set()
        for r in result_rows:
            extra_cols.update(r.keys())
        extra_cols.discard("param_folder")
        columns = ["param_folder"] + sorted(extra_cols, key=colkey)
        df = pd.DataFrame(result_rows)[columns]
        if "HR@1" in df.columns:
            df = df.sort_values("HR@1", ascending=False)
        df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
        print(f"Output {len(df)} rows to all_results.csv")
    else:
        print("No results")
