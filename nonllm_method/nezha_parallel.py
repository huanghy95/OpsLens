# nezha_parallel.py
import os
import json
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import time
import concurrent.futures
from utils.CSVFormat import *
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Global hyperparameter variables
SIGMA_THRESHOLD = 3  # 3-sigma threshold
MIN_SUPPORT = 5  # Minimum support
MIN_SCORE = 0.67  # Minimum score threshold


class AnomalyDetector:
    """Application-level metric anomaly detector using 3-sigma rule - optimized version"""

    def __init__(self, sigma_threshold=None):
        self.sigma_threshold = sigma_threshold if sigma_threshold is not None else SIGMA_THRESHOLD
        self.baseline_stats = {}

    def fit_baseline(self, df, metric_columns):
        """Calculate mean and standard deviation for normal phase"""
        for col in metric_columns:
            if col in df.columns:
                self.baseline_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

    def detect_anomaly(self, series, col_name):
        """Detect if metric is anomalous (violates 3-sigma rule) - vectorized version"""
        if col_name not in self.baseline_stats:
            return pd.Series(0, index=series.index)

        mean = self.baseline_stats[col_name]["mean"]
        std = self.baseline_stats[col_name]["std"]

        if std == 0:
            return pd.Series(0, index=series.index)

        # Vectorized calculation, avoid apply
        if "duration" in col_name or "latency" in col_name:
            upper_bound = mean + self.sigma_threshold * std
            return (series > upper_bound).astype(int)
        elif "success_rate" in col_name or "availability" in col_name:
            lower_bound = mean - self.sigma_threshold * std
            return (series < lower_bound).astype(int)
        else:
            lower_bound = mean - self.sigma_threshold * std
            upper_bound = mean + self.sigma_threshold * std
            return ((series < lower_bound) | (series > upper_bound)).astype(int)


class DataIntegrator:
    """Data integrator - batch processing version"""

    def __init__(self):
        self.log_templates = {}

    def extract_log_templates(self, log_series):
        """Extract log templates, remove dynamic parameters"""
        first_value = log_series.iloc[0] if len(log_series) > 0 else ""
        return first_value

    def generate_metric_events(self, df, anomaly_detector):
        """Generate metric anomaly events - batch processing version"""
        metric_columns = [col for col in df.columns if col.startswith("metric@")]

        # Batch process all metric columns
        for col in metric_columns:
            anomaly_col = f"{col}_anomaly"
            df[anomaly_col] = anomaly_detector.detect_anomaly(df[col], col)

        return df

    def generate_log_events(self, df):
        """Generate log events - vectorized version"""
        log_columns = [col for col in df.columns if col.startswith("log@")]

        for col in log_columns:
            if col not in self.log_templates:
                self.log_templates[col] = self.extract_log_templates(df[col])

            template = self.log_templates[col]
            change_col = f"{col}_change"
            # Vectorized comparison, avoid apply
            df[change_col] = (df[col] != template).astype(int)

        return df


class PatternMiner:
    """Pattern miner - batch calculation version"""

    def __init__(self, min_support=None):
        self.min_support = min_support if min_support is not None else MIN_SUPPORT

    def calculate_support(self, df, event_columns):
        """Calculate pattern support - vectorized version"""
        # Only select existing columns
        existing_columns = [col for col in event_columns if col in df.columns]

        if not existing_columns:
            return {}

        # Batch calculate support for all columns
        support_series = df[existing_columns].sum()

        # Filter patterns that meet minimum support
        support_dict = support_series[support_series >= self.min_support].to_dict()

        return support_dict


class PatternRanker:
    """Pattern ranker, calculate expected pattern and actual pattern scores"""

    def __init__(self, min_score_threshold=None):
        self.min_score_threshold = (
            min_score_threshold if min_score_threshold is not None else MIN_SCORE
        )

    def rank_expected_patterns(self, normal_support, fault_support):
        """Calculate expected pattern score E(ρ) = C(ρ)/(P(ρ) + C(ρ))"""
        expected_scores = {}

        for pattern in normal_support:
            c_support = normal_support[pattern]
            p_support = fault_support.get(pattern, 0)

            if c_support + p_support > 0:
                score = c_support / (p_support + c_support)
                if score >= self.min_score_threshold:
                    expected_scores[pattern] = score

        return expected_scores

    def rank_actual_patterns(self, normal_support, fault_support):
        """Calculate actual pattern score A(ρ) = P(ρ)/(C(ρ) + P(ρ))"""
        actual_scores = {}

        for pattern in fault_support:
            p_support = fault_support[pattern]
            c_support = normal_support.get(pattern, 0)

            if c_support + p_support > 0:
                score = p_support / (c_support + p_support)
                if score >= self.min_score_threshold:
                    actual_scores[pattern] = score

        return actual_scores


class PatternAggregator:
    """Pattern aggregator, aggregate expected and actual patterns, output root cause candidates"""

    def __init__(self, dependency_graph=None):
        self.dependency_graph = dependency_graph or {}

    def filter_redundant_patterns(self, expected_scores, actual_scores):
        """Filter redundant patterns, keep root patterns"""
        all_patterns = {}

        for pattern, score in expected_scores.items():
            all_patterns[pattern] = max(all_patterns.get(pattern, 0), score)

        for pattern, score in actual_scores.items():
            all_patterns[pattern] = max(all_patterns.get(pattern, 0), score)

        return all_patterns

    def aggregate_patterns(self, expected_scores, actual_scores):
        """Aggregate patterns and sort"""
        filtered_patterns = self.filter_redundant_patterns(expected_scores, actual_scores)

        sorted_patterns = sorted(
            filtered_patterns.items(),
            key=lambda x: (1 if "duration" in x[0] else 0, x[1]),
            reverse=True,
        )

        return sorted_patterns


class RootCauseAnalyzer:
    """Root cause analysis main class - optimized version"""

    def __init__(self, min_support=None, min_score=None, sigma_threshold=None):
        self.anomaly_detector = AnomalyDetector(sigma_threshold)
        self.data_integrator = DataIntegrator()
        self.pattern_miner = PatternMiner(min_support)
        self.pattern_ranker = PatternRanker(min_score)
        self.pattern_aggregator = PatternAggregator()

    def analyze_node_patterns(self, train_df, fault_df, node_name):
        """Analyze anomaly patterns for a single node"""
        train_node = train_df[train_df["name"] == node_name].copy()
        fault_node = fault_df[fault_df["name"] == node_name].copy()

        if len(train_node) == 0 or len(fault_node) == 0:
            return []

        metric_columns = [col for col in train_node.columns if col.startswith("metric@")]
        metric_columns.append("duration")

        self.anomaly_detector.fit_baseline(train_node, metric_columns)

        train_processed = self.data_integrator.generate_metric_events(
            train_node, self.anomaly_detector
        )
        train_processed = self.data_integrator.generate_log_events(train_processed)

        fault_processed = self.data_integrator.generate_metric_events(
            fault_node, self.anomaly_detector
        )
        fault_processed = self.data_integrator.generate_log_events(fault_processed)

        event_columns = [
            col
            for col in train_processed.columns
            if col.endswith("_anomaly") or col.endswith("_change")
        ]

        normal_support = self.pattern_miner.calculate_support(train_processed, event_columns)
        fault_support = self.pattern_miner.calculate_support(fault_processed, event_columns)

        expected_scores = self.pattern_ranker.rank_expected_patterns(normal_support, fault_support)
        actual_scores = self.pattern_ranker.rank_actual_patterns(normal_support, fault_support)

        patterns = self.pattern_aggregator.aggregate_patterns(expected_scores, actual_scores)

        return patterns

    def analyze_all_nodes(self, train_df, fault_df, nodes):
        """Analyze all nodes - pre-grouped optimization version"""
        # Pre-group by node to avoid repeated filtering
        train_groups = {node: train_df[train_df["name"] == node].copy() for node in nodes}
        fault_groups = {node: fault_df[fault_df["name"] == node].copy() for node in nodes}

        all_node_patterns = {}

        for node in nodes:
            train_node = train_groups[node]
            fault_node = fault_groups[node]

            if len(train_node) == 0 or len(fault_node) == 0:
                continue

            metric_columns = [col for col in train_node.columns if col.startswith("metric@")]
            metric_columns.append("duration")

            # Each node uses independent detector
            node_detector = AnomalyDetector(self.anomaly_detector.sigma_threshold)
            node_detector.fit_baseline(train_node, metric_columns)

            train_processed = self.data_integrator.generate_metric_events(train_node, node_detector)
            train_processed = self.data_integrator.generate_log_events(train_processed)

            fault_processed = self.data_integrator.generate_metric_events(fault_node, node_detector)
            fault_processed = self.data_integrator.generate_log_events(fault_processed)

            event_columns = [
                col
                for col in train_processed.columns
                if col.endswith("_anomaly") or col.endswith("_change")
            ]

            normal_support = self.pattern_miner.calculate_support(train_processed, event_columns)
            fault_support = self.pattern_miner.calculate_support(fault_processed, event_columns)

            expected_scores = self.pattern_ranker.rank_expected_patterns(
                normal_support, fault_support
            )
            actual_scores = self.pattern_ranker.rank_actual_patterns(normal_support, fault_support)

            patterns = self.pattern_aggregator.aggregate_patterns(expected_scores, actual_scores)

            if patterns:
                all_node_patterns[node] = patterns

        # Calculate service scores
        service_scores = {}
        for node, patterns in all_node_patterns.items():
            if patterns:
                max_score = max([score for _, score in patterns])
                service_scores[node] = max_score

        service_list = sorted(service_scores.keys(), key=lambda x: service_scores[x], reverse=True)

        # Generate nonservice list
        nonservice_patterns = []
        for node, patterns in all_node_patterns.items():
            for pattern, score in patterns:
                clean_pattern = re.sub(r"(_anomaly|_change)$", "", pattern)
                nonservice_patterns.append((f"{node}>{clean_pattern}", score))

        nonservice_list = [
            pattern for pattern, _ in sorted(nonservice_patterns, key=lambda x: x[1], reverse=True)
        ]

        # Fallback sorting: if no pattern results, sort nodes by fault phase sample size in descending order, ensure not empty
        if not service_list:
            node_sizes = {n: len(fault_groups[n]) for n in nodes}
            service_list = sorted(
                [n for n in nodes if node_sizes.get(n, 0) > 0],
                key=lambda x: node_sizes[x],
                reverse=True,
            )
            if not service_list:
                service_list = nodes[:]

        if not nonservice_list:
            nonservice_list = [f"{n}>duration" for n in service_list]

        return service_list, nonservice_list


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


def sub_main(train_path, predict_path, failure_label_path, nodes):
    """Main diagnosis process - using nezha optimization algorithm"""
    print(f"\n=== Starting folder processing ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path + "/data_final.csv")
    predict_df = pd.read_csv(predict_path + "/data_final.csv")
    if len(train_df) > 100000:
        print("Training set rows > 100k, switching to data_final_sampled.csv")
        train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
    if len(predict_df) > 100000:
        print("Prediction set rows > 100k, switching to data_final_sampled.csv")
        predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")

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

    # Modify sampling section
    train_df = sample_data(train_df, TRAIN_SIZE, select_outstanding=False)  # Select stable data
    predict_df = sample_data(predict_df, PREDICT_SIZE, select_outstanding=True)  # Select outstanding data



    # Keep common columns
    print("Processing common columns...")
    common_columns = train_df.columns.intersection(predict_df.columns)
    train_df = train_df[common_columns]
    predict_df = predict_df[common_columns]
    print(f"Number of common columns: {len(common_columns)}")

    # Get node intersection
    data_names = set(train_df["name"].unique()) | set(predict_df["name"].unique())
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    # Execute root cause analysis
    print("Using nezha algorithm for root cause analysis...")
    analyzer = RootCauseAnalyzer()
    service_list, nonservice_list = analyzer.analyze_all_nodes(train_df, predict_df, nodes)

    # Get labels
    try:
        label_match = re.search(r"(\d+)_", predict_path)
        label_str = label_match.group(1)
        with open(failure_label_path, "r") as f:
            failure_label = json.load(f)
        service_label = failure_label[label_str]["service"]
        nonservice_label = failure_label[label_str]["nonservice"]
        print(f"Detected service root causes: {service_list}")
        print(f"Detected non-service root causes: {nonservice_list}")
        print(f"Actual service root causes: {service_label}")
        print(f"Actual non-service root causes: {nonservice_label}")
    except Exception as e:
        print(f"Failed to read labels: {str(e)}")
        return None, None, None, None

    return service_list, nonservice_list, service_label, nonservice_label


def write_results_to_csv(
    service_labels,
    service_lists,
    nonservice_labels,
    nonservice_lists,
    ks=[1, 3, 5],
    filename="nezha_evaluation_results.csv",
):
    """Save evaluation results to CSV file"""
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


def run_single_params(params):
    """Run single hyperparameter set"""
    sigma_threshold, min_support, min_score = params

    # Set global parameters
    global SIGMA_THRESHOLD, MIN_SUPPORT, MIN_SCORE
    SIGMA_THRESHOLD = sigma_threshold
    MIN_SUPPORT = min_support
    MIN_SCORE = min_score

    print(f"Running parameters: sigma={sigma_threshold}, support={min_support}, score={min_score}")

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

        param_folder = f"sigma{sigma_threshold}_support{min_support}_score{min_score:.2f}"
        param_output_dir = os.path.join(output_dir, param_folder)
        os.makedirs(param_output_dir, exist_ok=True)

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
                result = sub_main(train_path, predict_path, failure_label_path, nodes)
                if result is None:
                    continue
                service_list, nonservice_list, service_label, nonservice_label = result

                # Filter query nodes
                service_list = [item for item in service_list if item != "query"]

                service_lists.append(service_list)
                nonservice_lists.append(nonservice_list)
                service_labels.append(service_label)
                nonservice_labels.append(nonservice_label)

                detailed_results.append(
                    {
                        "task_id": label_str,
                        "service_list": service_list,
                        "nonservice_list": nonservice_list,
                        "service_label": service_label,
                        "nonservice_label": nonservice_label,
                    }
                )

            except Exception as e:
                print(f"Error processing folder {folder_name}: {str(e)}")
                continue

        # Save hyperparameter experiment results
        if service_lists:
            result_path = os.path.join(param_output_dir, "evaluation_results.csv")

            # Calculate and save evaluation metrics
            write_results_to_csv(
                service_labels,
                service_lists,
                nonservice_labels,
                nonservice_lists,
                filename=result_path,
            )

            detailed_path = os.path.join(param_output_dir, "detailed_results.csv")
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.to_csv(detailed_path, index=False)

            print(
                f"Completed parameter combination: sigma={sigma_threshold}, support={min_support}, score={min_score}"
            )
            return param_folder

        return None

    except Exception as e:
        print(
            f"Parameter combination failed: sigma={sigma_threshold}, support={min_support}, score={min_score}, error: {e}"
        )
        return None


if __name__ == "__main__":
    # Start timing
    start_time = time.time()

    # Read configuration
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    dependency_graph = config["dependency_graph"]
    jump_folder = config["jump_folder"]
    data_final_dir = config["data_final_dir"]
    normal_data_dir = config["normal_data_dir"]
    output_dir = config["output_dir"]
    failure_label_path = config["failure_label_path"]
    nodes = config["nodes"]

    # Hyperparameter search configuration
    # sigma: 3-sigma threshold, support: minimum support, score: minimum score threshold
    # MIN_SCORE: controls the minimum threshold for pattern scoring, determines which patterns are considered valid root cause candidates, lower means more candidates
    # MIN_SUPPORT: controls the minimum frequency threshold for pattern occurrence, filters low-frequency noise patterns. Lower includes more low-frequency noise patterns.
    param_combinations = [
        (sigma, support, score)
        for sigma in [2, 3, 4]
        for support in [5, 20, 40, 80]
        for score in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]



    print(f"Starting nezha hyperparameter search, total {len(param_combinations)} parameter sets")

    # Check completed experiments to avoid duplicate runs
    pending = []
    for p in param_combinations:
        sigma, support, score = p
        param_folder = f"sigma{sigma}_support{support}_score{score:.2f}"
        result_file = os.path.join(output_dir, param_folder, "evaluation_results.csv")
        if not os.path.exists(result_file):
            pending.append(p)

    print(f"Remaining {len(pending)} parameter sets to run")

    if len(pending) > 0:
        # Parallel execution of hyperparameter tuning
        max_workers = max(int(os.cpu_count() / 10 - 1), 1)
        all_results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_params, p): p for p in pending}
            completed = 0

            for future in concurrent.futures.as_completed(futures):
                completed += 1
                try:
                    result = future.result()
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

    # Summarize HR@1 results from all parameter folders, sort and output
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
