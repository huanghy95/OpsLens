# microscope_parallel.py
import pandas as pd
import numpy as np
from scipy.stats import chi2
from itertools import combinations
import networkx as nx
import os
import re
import json
import time
import concurrent.futures
from utils.CSVFormat import *
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Global hyperparameter variables
MAX_Z_CARDINALITY = 3  # PC algorithm maximum conditional set size
SIGNIFICANCE_LEVEL = 0.02  # G2 independence test significance level
BINS = 3  # Number of bins for continuous data discretization
SIGMA_THRESHOLD = 3  # 3-sigma anomaly detection threshold
PC_DATA_SIZE = 200  # PC algorithm data size
METRIC_FIELD = "duration"  # Metric field


def detect_metric_columns(df):
    """Detect metric-type columns"""
    metric_columns = []
    for col in df.columns:
        if col.startswith("metric@") or "latency" in col.lower() or "duration" in col.lower():
            metric_columns.append(col)
    return metric_columns


def detect_log_columns(df):
    """Detect log-type columns"""
    log_columns = []
    for col in df.columns:
        if col.startswith("log@"):
            log_columns.append(col)
    return log_columns


def detect_trace_columns(df):
    """Detect trace-type columns"""
    trace_columns = []
    for col in df.columns:
        if col.startswith("trace@") or "span" in col.lower():
            trace_columns.append(col)
    return trace_columns


def apply_3sigma_anomaly_detection(series, reference_mean=None, reference_std=None):
    """Use 3-sigma rule for anomaly detection"""
    if reference_mean is None or reference_std is None:
        reference_mean = series.mean()
        reference_std = series.std()

    if reference_std == 0:
        return series.apply(lambda x: 0), reference_mean, reference_std

    lower_bound = reference_mean - SIGMA_THRESHOLD * reference_std
    upper_bound = reference_mean + SIGMA_THRESHOLD * reference_std
    anomaly_flags = series.apply(lambda x: 1 if x < lower_bound or x > upper_bound else 0)

    return anomaly_flags, reference_mean, reference_std


def discretize_continuous_data(series, bins=None):
    """Fixed discretization function"""
    if bins is None:
        bins = BINS

    # Check if series is empty or all NaN
    if len(series) == 0 or series.isna().all():
        return pd.Series([0] * len(series), index=series.index)

    # Check if variance is 0 or series length is insufficient
    valid_data = series.dropna()
    if len(valid_data) == 0 or valid_data.std() == 0 or len(valid_data) < bins:
        return pd.Series([0] * len(series), index=series.index)

    try:
        # Use qcut instead of cut to avoid duplicate value issues
        result = pd.qcut(series, q=bins, labels=False, duplicates="drop")

        # If qcut fails, use simple splitting
        if result is None or result.isna().all():
            # Manual splitting
            min_val, max_val = series.min(), series.max()
            if min_val == max_val:
                return pd.Series([0] * len(series), index=series.index)

            # Create equal-interval bins
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            bin_edges[-1] += 1e-6  # Avoid boundary issues
            result = pd.cut(series, bins=bin_edges, labels=False, include_lowest=True)

        # Ensure returned Series has correct index and length
        if result is None or len(result) != len(series):
            return pd.Series([0] * len(series), index=series.index)

        return result.fillna(0).astype(int)

    except Exception as e:
        print(f"Discretization error: {e}")
        return pd.Series([0] * len(series), index=series.index)


def g2_independence_test(x_data, y_data, z_data=None, significance_level=None):
    """Fixed G2 independence test"""
    if significance_level is None:
        significance_level = SIGNIFICANCE_LEVEL

    # Check if data is empty
    if len(x_data) == 0 or len(y_data) == 0:
        return True

    # Ensure all data has consistent length
    min_length = min(len(x_data), len(y_data))
    if z_data is not None and len(z_data) > 0:
        min_length = min(min_length, min(len(z) for z in z_data))

    # Truncate to minimum length and reset index
    x_data = x_data.iloc[:min_length].reset_index(drop=True)
    y_data = y_data.iloc[:min_length].reset_index(drop=True)

    if z_data is not None and len(z_data) > 0:
        z_data = [z.iloc[:min_length].reset_index(drop=True) for z in z_data]

    # Discretize data
    x_discrete = discretize_continuous_data(x_data)
    y_discrete = discretize_continuous_data(y_data)

    # Check discretized data
    if len(x_discrete.dropna()) == 0 or len(y_discrete.dropna()) == 0:
        return True

    try:
        if z_data is not None and len(z_data) > 0:
            # Conditional independence test
            z_discrete_list = []
            for z in z_data:
                z_disc = discretize_continuous_data(z)
                if len(z_disc) != min_length:
                    z_disc = pd.Series([0] * min_length, index=range(min_length))
                z_discrete_list.append(z_disc)

            # Reset all indices to ensure consistency
            x_discrete = x_discrete.reset_index(drop=True)
            y_discrete = y_discrete.reset_index(drop=True)
            z_discrete_list = [z.reset_index(drop=True) for z in z_discrete_list]

            # Create joint variable
            z_combined = pd.DataFrame(z_discrete_list).T
            z_combined.columns = [f"z_{i}" for i in range(len(z_discrete_list))]

            # Fix: use list instead of DataFrame columns
            joint_counts = pd.crosstab(
                [x_discrete, y_discrete],
                [z_combined.iloc[:, i] for i in range(z_combined.shape[1])],
                margins=False,
            )
        else:
            # Marginal independence test
            joint_counts = pd.crosstab(x_discrete, y_discrete, margins=False)

        # Check if joint_counts is empty or invalid
        if joint_counts.empty:
            return True

        # Fix: use values.sum() instead of sum(axis=None)
        total_count = joint_counts.values.sum()
        if total_count == 0:
            return True

        # Calculate expected frequencies
        if len(joint_counts.shape) == 2:
            # 2D case (marginal independence)
            row_sums = joint_counts.sum(axis=1)
            col_sums = joint_counts.sum(axis=0)
            expected = np.outer(row_sums, col_sums) / total_count
        else:
            # Multi-dimensional case (conditional independence) - simplified handling
            expected = np.ones_like(joint_counts.values) * (total_count / joint_counts.size)

        # Fix: avoid division by zero and log(0)
        expected = np.maximum(expected, 1e-10)
        observed = joint_counts.values
        observed = np.maximum(observed, 1e-10)

        # Calculate G2 statistic - only when observed > 0
        mask = observed > 0
        g2_statistic = 0
        if mask.any():
            g2_statistic = 2 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))

        # Calculate degrees of freedom
        if len(joint_counts.shape) == 2:
            degrees_of_freedom = (joint_counts.shape[0] - 1) * (joint_counts.shape[1] - 1)
        else:
            degrees_of_freedom = max(1, joint_counts.size - 1)

        # Calculate p-value
        if degrees_of_freedom > 0 and g2_statistic >= 0:
            p_value = 1 - chi2.cdf(g2_statistic, degrees_of_freedom)
            return p_value > significance_level
        else:
            return True

    except Exception as e:
        print(f"G2 test error: {e}")
        return True


def build_pc_algorithm_graph(data_df, max_z_cardinality=None, significance_level=None):
    """Enhanced PC algorithm - borrowing correlation ideas from reference code"""
    if max_z_cardinality is None:
        max_z_cardinality = MAX_Z_CARDINALITY
    if significance_level is None:
        significance_level = SIGNIFICANCE_LEVEL

    # Only select numeric columns for analysis
    numeric_columns = data_df.select_dtypes(include=[np.number]).columns.tolist()
    if "name" in numeric_columns:
        numeric_columns.remove("name")
    nodes = numeric_columns

    # Check if data is sufficient
    if len(data_df) < 10 or len(nodes) < 2:
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(nodes)
        return directed_graph

    # Build complete undirected graph
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    # Use correlation threshold instead of relying entirely on G2 test
    correlation_threshold = 0.3  # Borrowing idea from reference code

    for i, j in combinations(nodes, 2):
        try:
            # Calculate simple correlation as initial screening
            x_data = data_df[i].dropna()
            y_data = data_df[j].dropna()

            if len(x_data) > 5 and len(y_data) > 5:
                min_len = min(len(x_data), len(y_data))
                corr = np.corrcoef(x_data[:min_len], y_data[:min_len])[0, 1]

                # If correlation is high enough, keep the edge
                if not np.isnan(corr) and abs(corr) > correlation_threshold:
                    graph.add_edge(i, j)
                else:
                    # Only perform G2 test when correlation is low
                    if not g2_independence_test(
                        x_data[:min_len],
                        y_data[:min_len],
                        significance_level=significance_level * 2,
                    ):
                        graph.add_edge(i, j)
        except Exception:
            continue

    # Convert to directed graph
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(graph.nodes())

    # Extract communication edges from dependency graph
    communication_edges = []
    if dependency_graph:
        for source, targets in dependency_graph.items():
            for target in targets:
                communication_edges.append((source, target))

    for edge in graph.edges():
        x, y = edge
        # Prioritize known communication dependencies
        if (x, y) in communication_edges:
            directed_graph.add_edge(x, y)
        elif (y, x) in communication_edges:
            directed_graph.add_edge(y, x)
        else:
            # Use simple causal direction judgment
            if x < y:  # Lexicographic order
                directed_graph.add_edge(x, y)
            else:
                directed_graph.add_edge(y, x)

    print(f"PC algorithm built graph: {len(directed_graph.nodes())} nodes, {len(directed_graph.edges())} edges")
    return directed_graph


def merge_dependency_graphs(communication_graph, non_communication_graph):
    """Merge communication and non-communication dependency graphs, prioritizing network connection results"""
    merged_graph = communication_graph.copy()

    for source in non_communication_graph.nodes():
        if source not in merged_graph:
            merged_graph[source] = []
        for target in non_communication_graph.successors(source):
            conflict = False
            for comm_source, comm_targets in communication_graph.items():
                if target == comm_source and source in comm_targets:
                    conflict = True
                    break

            if not conflict and target not in merged_graph[source]:
                merged_graph[source].append(target)

    return merged_graph


def calculate_neighbor_anomalies_simple(service_name, dependency_graph, predict_all_dfs, row_idx):
    """Simplified version for calculating neighbor anomaly count"""
    # Get neighbor nodes
    connected_neighbours = dependency_graph.get(service_name, [])
    reverse_neighbours = [key for key, values in dependency_graph.items() if service_name in values]
    neighbours = set(connected_neighbours + reverse_neighbours)

    neighbor_anomalies = 0
    for neighbour in neighbours:
        df_key = f"predict_{neighbour}_df"
        if df_key in predict_all_dfs and row_idx < len(predict_all_dfs[df_key]):
            anomaly = predict_all_dfs[df_key].iloc[row_idx]["anomaly_detection"]
            if anomaly == 1:
                neighbor_anomalies += 1

    return neighbor_anomalies


def root_cause_inference(predict_all_dfs, dependency_graph, slo_metric_column, nodes):
    """Enhanced root cause inference - combining scoring ideas from reference code"""
    service_lists = []

    # Get data length
    df_len = None
    for name in nodes:
        df_key = f"predict_{name}_df"
        if df_key in predict_all_dfs and len(predict_all_dfs[df_key]) > 0:
            df_len = len(predict_all_dfs[df_key])
            break

    if df_len is None:
        print("Warning: No valid prediction data found")
        return [[name for name in nodes if name != "query"]]

    for row_idx in range(df_len):
        # Collect comprehensive scores for all services
        service_scores = []

        for service_name in nodes:
            if service_name == "query":  # Skip frontend service
                continue

            df_key = f"predict_{service_name}_df"
            if df_key not in predict_all_dfs or row_idx >= len(predict_all_dfs[df_key]):
                continue

            # 1. Anomaly detection score
            anomaly_score = predict_all_dfs[df_key].iloc[row_idx]["anomaly_detection"]

            # 2. Correlation score with frontend (borrowing from reference code)
            correlation_score = 0
            try:
                if f"predict_query_df" in predict_all_dfs:
                    frontend_slo = predict_all_dfs[f"predict_query_df"][slo_metric_column]
                    service_slo = predict_all_dfs[df_key][slo_metric_column]

                    min_len = min(len(frontend_slo), len(service_slo))
                    if min_len > 1:
                        corr = np.corrcoef(frontend_slo.iloc[:min_len], service_slo.iloc[:min_len])[
                            0, 1
                        ]
                        if not np.isnan(corr):
                            correlation_score = abs(corr)
            except Exception:
                correlation_score = 0

            # 3. Graph structure score (retaining PC algorithm ideas)
            if hasattr(dependency_graph, "has_node") and dependency_graph.has_node(service_name):
                # Calculate importance in graph
                in_degree = (
                    dependency_graph.in_degree(service_name)
                    if hasattr(dependency_graph, "in_degree")
                    else 0
                )
                out_degree = (
                    dependency_graph.out_degree(service_name)
                    if hasattr(dependency_graph, "out_degree")
                    else 0
                )
                graph_score = (in_degree + out_degree) / max(1, len(dependency_graph.nodes()))
            else:
                graph_score = 0

            # 4. Neighbor anomaly score
            neighbor_score = calculate_neighbor_anomalies_simple(
                service_name,
                dependency_graph,
                predict_all_dfs,
                row_idx,
            )

            # Comprehensive scoring: weight adjustment borrowing from reference code
            total_score = (
                anomaly_score * 2.0  # Anomaly detection most important
                + correlation_score * 1.5  # Correlation second
                + graph_score * 1.0  # Graph structure
                + neighbor_score * 0.5  # Neighbor anomalies
            )

            service_scores.append((service_name, total_score))

        # Sort and output
        service_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_services = [name for name, score in service_scores]

        # Ensure at least some candidates
        if not ranked_services:
            ranked_services = [name for name in nodes if name != "query"]

        service_lists.append(ranked_services)

        # Key debugging information
        if row_idx == 0:
            print(f"Service score ranking: {[(name, f'{score:.3f}') for name, score in service_scores]}")

    return service_lists


def simple_anomaly_ranking(predict_all_dfs, nodes):
    """Simple anomaly ranking as fallback solution"""
    service_lists = []

    # Get data length
    df_len = None
    for name in nodes:
        df_key = f"predict_{name}_df"
        if df_key in predict_all_dfs and len(predict_all_dfs[df_key]) > 0:
            df_len = len(predict_all_dfs[df_key])
            break

    if df_len is None:
        return [[name for name in nodes if name != "query"]]

    for row_idx in range(df_len):
        anomalous_services = []
        normal_services = []

        for service_name in nodes:
            if service_name == "query":
                continue

            df_key = f"predict_{service_name}_df"
            if df_key in predict_all_dfs and row_idx < len(predict_all_dfs[df_key]):
                anomaly = predict_all_dfs[df_key].iloc[row_idx]["anomaly_detection"]
                if anomaly == 1:
                    anomalous_services.append(service_name)
                else:
                    normal_services.append(service_name)

        # Anomalous services first, normal services second
        ranked_services = anomalous_services + normal_services
        if not ranked_services:
            ranked_services = [name for name in nodes if name != "query"]

        service_lists.append(ranked_services)

    return service_lists


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate HR@k"""
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # Check if predicted top k root causes contain actual root causes
        hit = any(root in predicted_roots[:k] for root in actual_roots)
        hr_k.append(hit)
    return np.mean(hr_k)


def ndcg_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate NDCG@k"""
    ndcg_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # Calculate DCG@k
        dcg = 0
        for i, predicted_root in enumerate(predicted_roots[:k]):
            rel = 1 if predicted_root in actual_roots else 0
            dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 for 1-based index

        # Calculate IDCG@k (ideal DCG@k)
        idcg = 0
        ideal_relevance = [1 if root in actual_roots else 0 for root in predicted_roots[:k]]
        ideal_relevance.sort(reverse=True)
        for i, rel in enumerate(ideal_relevance):
            idcg += (2**rel - 1) / np.log2(i + 2)  # i+2 for 1-based index

        # Calculate NDCG@k
        ndcg_k.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcg_k)


def sub_main(train_path, predict_path, failure_label_path, nodes, dependency_graph):
    """Main processing function: using microscope enhanced algorithm for root cause analysis"""
    print(f"\n=== Starting to process folder ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    # Load data
    print("Loading data...")
    load_start_time = time.time()
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

    # train_df = (
    #     train_df.sample(n=TRAIN_SIZE, random_state=42)
    #     if len(train_df) > TRAIN_SIZE
    #     else train_df.head(TRAIN_SIZE)
    # )
    # predict_df = (
    #     predict_df.sample(n=PREDICT_SIZE, random_state=42)
    #     if len(predict_df) > PREDICT_SIZE
    #     else predict_df.head(PREDICT_SIZE)
    # )

    # Keep common columns
    print("Processing common columns...")
    common_columns = train_df.columns.intersection(predict_df.columns)
    train_df = train_df[common_columns]
    predict_df = predict_df[common_columns]
    print(f"Number of common columns: {len(common_columns)}")

    # Get all unique name values in data
    data_names = set(train_df["name"].unique()) | set(predict_df["name"].unique())
    # Take intersection of nodes and names in data
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    print("Using microscope enhanced algorithm for anomaly detection...")

    # Group data by service and perform anomaly detection
    train_all_dfs = {}
    predict_all_dfs = {}

    slo_metric_column = METRIC_FIELD

    for name in nodes:
        train_service_df = train_df[train_df["name"] == name].copy()
        predict_service_df = predict_df[predict_df["name"] == name].copy()

        if len(train_service_df) == 0 or len(predict_service_df) == 0:
            print(f"Warning: Service {name} has insufficient data")
            continue

        # Use improved anomaly detection algorithm
        try:
            # Try using duration field
            if METRIC_FIELD in train_service_df.columns:
                train_anomaly, mean_train, std_train = apply_3sigma_anomaly_detection(
                    train_service_df[METRIC_FIELD]
                )
                predict_anomaly, _, _ = apply_3sigma_anomaly_detection(
                    predict_service_df[METRIC_FIELD], mean_train, std_train
                )
                train_service_df["anomaly_detection"] = train_anomaly
                predict_service_df["anomaly_detection"] = predict_anomaly
            elif "elapsedTime" in train_service_df.columns:
                # Backup field
                train_anomaly, mean_train, std_train = apply_3sigma_anomaly_detection(
                    train_service_df["elapsedTime"]
                )
                predict_anomaly, _, _ = apply_3sigma_anomaly_detection(
                    predict_service_df["elapsedTime"], mean_train, std_train
                )
                train_service_df["anomaly_detection"] = train_anomaly
                predict_service_df["anomaly_detection"] = predict_anomaly
                slo_metric_column = "elapsedTime"
            else:
                print(f"Warning: Service {name} missing duration field")
                continue

        except Exception as e:
            print(f"Error occurred while processing anomaly detection for node {name}: {str(e)}")
            continue

        train_all_dfs[f"train_{name}_df"] = train_service_df
        predict_all_dfs[f"predict_{name}_df"] = predict_service_df

    # Check data integrity
    if len(predict_all_dfs) < 2:
        print("Insufficient data, using communication dependency sorting")
        # Use all available services as backup sorting, not hardcoded list
        available_services = [
            name for name in nodes if f"predict_{name}_df" in predict_all_dfs and name != "query"
        ]
        service_lists_sub = (
            available_services
            if available_services
            else [name for name in nodes if name != "query"]
        )
    else:
        # Build causal graph and perform root cause inference
        try:
            # First use communication dependencies
            communication_graph = dependency_graph

            # Try PC algorithm enhancement
            pc_data = train_df.tail(min(PC_DATA_SIZE, len(train_df)))
            pc_graph = build_pc_algorithm_graph(pc_data)

            # Merge graphs
            final_dependency_graph = merge_dependency_graphs(communication_graph, pc_graph)

            # Root cause inference
            service_lists = root_cause_inference(
                predict_all_dfs, final_dependency_graph, slo_metric_column, nodes
            )

            print(f"Root cause inference completed, candidate count: {len(service_lists[0]) if service_lists else 0}")

            # Take top results
            from collections import Counter

            merged_service = []
            for lst in service_lists:
                merged_service.extend(lst)
            counter_service = Counter(merged_service)
            service_lists_sub = [item for item, _ in counter_service.most_common()]

        except Exception as e:
            print(f"Algorithm execution failed, using backup solution: {e}")
            # Backup solution: simple anomaly sorting
            service_lists = simple_anomaly_ranking(predict_all_dfs, nodes)
            if service_lists:
                from collections import Counter

                merged_service = []
                for lst in service_lists:
                    merged_service.extend(lst)
                counter_service = Counter(merged_service)
                service_lists_sub = [item for item, _ in counter_service.most_common()]
            else:
                service_lists_sub = [name for name in nodes if name != "query"]

    # Get labels
    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]

    print(f"Detected service root causes: {service_lists_sub}")
    print(f"Actual service root causes: {service_label}")

    return service_lists_sub, service_label


def write_results_to_csv(all_results, filename="microscope_results.csv"):
    """Save evaluation results to CSV file"""
    detailed_results = []
    for result in all_results:
        actual_roots, predicted_roots = result["actual_roots"], result["predicted_roots"]
        case_id = result["case_id"]
        runtime = result["runtime_seconds"]

        for k in [1, 3, 5]:
            hit = any(root in predicted_roots[:k] for root in actual_roots)
            hr_k = 1 if hit else 0

            dcg = 0
            for i, predicted_root in enumerate(predicted_roots[:k]):
                rel = 1 if predicted_root in actual_roots else 0
                dcg += (2**rel - 1) / np.log2(i + 2)

            idcg = 0
            ideal_relevance = [1 if root in actual_roots else 0 for root in predicted_roots[:k]]
            ideal_relevance.sort(reverse=True)
            for i, rel in enumerate(ideal_relevance):
                idcg += (2**rel - 1) / np.log2(i + 2)

            ndcg_k = dcg / idcg if idcg > 0 else 0

            detailed_results.append(
                {
                    "case_id": case_id,
                    "k": k,
                    "HR@k": hr_k,
                    "NDCG@k": ndcg_k,
                    "actual_roots": str(actual_roots),
                    "predicted_roots": str(predicted_roots),
                    "runtime_seconds": runtime,
                }
            )

    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(filename, index=False)

    # Calculate and save summary results
    summary_results = []
    for k in [1, 3, 5]:
        k_df = detailed_df[detailed_df["k"] == k]
        summary_results.append(
            {
                "k": k,
                "HR@k": k_df["HR@k"].mean(),
                "NDCG@k": k_df["NDCG@k"].mean(),
                "case_count": len(k_df),
                "avg_runtime_seconds": k_df["runtime_seconds"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_results)
    summary_path = filename.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Detailed results saved: {filename}")
    print(f"Summary results saved: {summary_path}")

    # Print results
    print("\nEvaluation Results:")
    print(summary_df)

    return summary_df


def run_single_params(params):
    """Run single hyperparameter configuration"""
    mzc, sl, bins, st, pds = params

    # Set global hyperparameters
    global MAX_Z_CARDINALITY, SIGNIFICANCE_LEVEL, BINS, SIGMA_THRESHOLD, PC_DATA_SIZE
    MAX_Z_CARDINALITY, SIGNIFICANCE_LEVEL, BINS, SIGMA_THRESHOLD, PC_DATA_SIZE = (
        mzc,
        sl,
        bins,
        st,
        pds,
    )

    print(f"Running parameters: mzc={mzc}, sl={sl}, bins={bins}, st={st}, pds={pds}")

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

        # Create output directory named with parameters
        param_folder = f"mzc{mzc}_sl{sl:.2f}_bins{bins}_st{st}_pds{pds}"
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
        service_labels = []
        nonservice_lists = []  # Add empty nonservice list
        nonservice_labels = []  # Add empty nonservice labels

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
                service_list, service_label = sub_main(
                    train_path, predict_path, failure_label_path, nodes, dependency_graph
                )
                case_end_time = time.time()
                case_runtime = case_end_time - case_start_time

                # Filter query nodes
                service_list = [item for item in service_list if item != "query"]

                service_lists.append(service_list)
                service_labels.append(service_label)
                nonservice_list = []  # Empty nonservice list
                nonservice_label = []  # Empty nonservice labels
                nonservice_lists.append(nonservice_list)
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
                print(f"Error occurred while processing folder {folder_name}: {str(e)}")
                continue

        # Save hyperparameter experiment results
        if service_lists:
            # Save evaluation results
            result_path = os.path.join(param_output_dir, "evaluation_results.csv")

            # Use same function format as nezha to calculate and save evaluation metrics
            results = []
            for k in [1, 3, 5]:
                hr_k = hit_ratio_at_k(service_labels, service_lists, k)
                ndcg_k = ndcg_at_k(service_labels, service_lists, k)
                nonservicehr_k = hit_ratio_at_k(nonservice_labels, nonservice_lists, k)
                nonservicendcg_k = ndcg_at_k(nonservice_labels, nonservice_lists, k)
                # Save results
                results.append([k, hr_k, ndcg_k, nonservicehr_k, nonservicendcg_k])

            # Save evaluation results
            df = pd.DataFrame(
                results, columns=["k", "HR@k", "NDCG@k", "nonserviceHR@k", "nonserviceNDCG@k"]
            )
            df.to_csv(result_path, index=False)

            # Save detailed results
            detailed_path = os.path.join(param_output_dir, "detailed_results.csv")
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.to_csv(detailed_path, index=False)

            print(f"Completed parameter combination: mzc={mzc}, sl={sl}, bins={bins}, st={st}, pds={pds}")
            return param_folder

        return None

    except Exception as e:
        print(f"Parameter combination failed: mzc={mzc}, sl={sl}, bins={bins}, st={st}, pds={pds}, error: {e}")
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
    # mzc: PC algorithm maximum conditional set size, sl: G2 independence test significance level, bins: continuous data discretization bins, st: 3-sigma anomaly detection threshold, pds: PC algorithm data size
    # 3-sigma threshold
    # sl: G2 independence test significance level. Controls the strictness of causal graph construction, affecting dependency relationship identification. Larger: stricter causal relationships, sparser graph
    param_combinations = [
        (mzc, sl, bins, st, pds)
        for mzc in [3]
        for sl in [0.02, 0.1, 0.5]
        for bins in [1, 2, 3]
        for st in [0.1, 0.5, 1, 2]
        for pds in [200]
    ]

    print(f"Starting microscope hyperparameter search, total {len(param_combinations)} parameter combinations")

    # Check completed experiments to avoid duplicate runs
    pending = []
    for p in param_combinations:
        mzc, sl, bins, st, pds = p
        param_folder = f"mzc{mzc}_sl{sl:.2f}_bins{bins}_st{st}_pds{pds}"
        result_file = os.path.join(output_dir, param_folder, "evaluation_results.csv")
        if not os.path.exists(result_file):
            pending.append(p)

    print(f"Remaining {len(pending)} parameter combinations to run")

    if len(pending) > 0:
        # Execute hyperparameter tuning in parallel
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
                    print(f"Progress [{completed}/{len(pending)}]: Failed - params{params}, error: {str(e)}")

        print(f"Hyperparameter tuning completed, successfully ran {len(all_results)} parameter combinations")

    end_time = time.time()  # End timing
    run_time = end_time - start_time
    print(f"\nTotal runtime: {run_time:.4f} seconds")
    print("=== Program execution completed ===")

    # Summarize HR@1 results from all parameter folders, sorted output
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
                    # Add nonservice metrics if available
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
