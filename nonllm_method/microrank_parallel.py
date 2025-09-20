# microrank_parallel.py
import networkx as nx
import pandas as pd
import os
import numpy as np
import re
import json
import warnings
import concurrent.futures
from sklearn.preprocessing import normalize
import time
from collections import defaultdict
import glob
from utils.CSVFormat import *
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Global hyperparameter variables
N_MULTIPLIER = 1.5  
OMEGA = 0.005  # call graph weight
PHI = 0.5  # trace span range and type balancing factor
D = 0.85  # PageRank damping factor
METRIC_FIELD = "duration"  # metric field


class MicroRankDetector:
    """MicroRank root cause analysis detector"""

    def __init__(self, n=None, omega=None, phi=None, d=None):
        self.n = n if n is not None else N_MULTIPLIER
        self.omega = omega if omega is not None else OMEGA
        self.phi = phi if phi is not None else PHI
        self.d = d if d is not None else D
        self.operation_stats = {}

    def train_operation_stats(self, train_dfs):
        """Training phase: calculate mean and standard deviation for each operation"""
        for name, df in train_dfs.items():
            if not df.empty:
                # Try to use duration field, if not available use elapsedTime
                if METRIC_FIELD in df.columns:
                    durations = df[METRIC_FIELD]
                elif "elapsedTime" in df.columns:
                    durations = df["elapsedTime"]
                else:
                    continue

                operation = df["name"].iloc[0] if len(df) > 0 else name
                self.operation_stats[operation] = {
                    "mean": durations.mean(),
                    "std": durations.std() if durations.std() > 0 else 0.1,
                }

    def anomaly_detection(self, trace_spans):
        """Anomaly detection: calculate expected latency based on formula (1) and determine anomalies"""
        operation_counts = {}
        actual_latency = 0

        for span in trace_spans:
            operation = span["name"]
            duration = span.get(METRIC_FIELD, span.get("elapsedTime", 0))
            actual_latency += duration
            operation_counts[operation] = operation_counts.get(operation, 0) + 1

        expected_latency = 0
        for operation, count in operation_counts.items():
            if operation in self.operation_stats:
                stats = self.operation_stats[operation]
                expected_latency += count * (stats["mean"] + self.n * stats["std"])
            else:
                expected_latency += count * duration

        return actual_latency > expected_latency

    def prepare_data(self, predict_dfs, dependency_graph):
        """Data preprocessing: build normal/abnormal trace sets and association graph"""
        normal_traces = {}
        abnormal_traces = {}
        operation_trace_graph = nx.Graph()
        call_graph = nx.DiGraph()

        all_traces = defaultdict(list)

        # Build trace data - use existing data structure
        for name, df in predict_dfs.items():
            if not df.empty:
                for _, row in df.iterrows():
                    # Try to get trace_id, if not available generate one
                    trace_id = row.get("trace_rowid", row.get("traceId", f"trace_{row.name}"))
                    all_traces[trace_id].append(row.to_dict())

        # Classify normal and abnormal traces
        for trace_id, spans in all_traces.items():
            is_abnormal = self.anomaly_detection(spans)

            if is_abnormal:
                abnormal_traces[trace_id] = spans
            else:
                normal_traces[trace_id] = spans

            # Build operation-trace association graph
            trace_operations = set()
            for span in spans:
                operation = span["name"]
                trace_operations.add(operation)
                operation_trace_graph.add_node(operation, type="operation")

            operation_trace_graph.add_node(trace_id, type="trace")
            for operation in trace_operations:
                operation_trace_graph.add_edge(operation, trace_id)

        # Build call graph
        for parent, children in dependency_graph.items():
            call_graph.add_node(parent)
            for child in children:
                call_graph.add_node(child)
                call_graph.add_edge(parent, child)

        return normal_traces, abnormal_traces, call_graph, operation_trace_graph

    def pagerank_scoring(self, normal_traces, abnormal_traces, call_graph, operation_trace_graph):
        """PageRank scoring: implement personalized PageRank algorithm"""
        operations = [
            node
            for node in operation_trace_graph.nodes()
            if operation_trace_graph.nodes[node].get("type") == "operation"
        ]
        traces = [
            node
            for node in operation_trace_graph.nodes()
            if operation_trace_graph.nodes[node].get("type") == "trace"
        ]

        if not operations or not traces:
            return {}, {}

        all_nodes = operations + traces
        n_ops = len(operations)
        n_traces = len(traces)
        n_total = n_ops + n_traces

        # Build transition matrix
        A = np.zeros((n_total, n_total))

        # Call graph part
        if call_graph.number_of_edges() > 0:
            for i, op1 in enumerate(operations):
                out_degree = call_graph.out_degree(op1)
                if out_degree > 0:
                    for op2 in call_graph.successors(op1):
                        if op2 in operations:
                            j = operations.index(op2)
                            A[i, j] = self.omega / out_degree

        # Operation-trace connections
        for i, operation in enumerate(operations):
            connected_traces = [t for t in traces if operation_trace_graph.has_edge(operation, t)]
            if connected_traces:
                for trace in connected_traces:
                    j = n_ops + traces.index(trace)
                    A[i, j] = 1.0 / len(connected_traces)

            for k, trace in enumerate(traces):
                if operation_trace_graph.has_edge(operation, trace):
                    trace_ops = [
                        op for op in operations if operation_trace_graph.has_edge(op, trace)
                    ]
                    if trace_ops:
                        A[n_ops + k, i] = 1.0 / len(trace_ops)

        # Build personalization vector
        u = np.zeros(n_total)
        abnormal_trace_ids = list(abnormal_traces.keys())
        if abnormal_trace_ids:
            n_i_inv_sum = 0
            k_i_inv_sum = 0
            trace_weights = {}

            for trace_id in abnormal_trace_ids:
                if trace_id in traces:
                    trace_ops = [
                        op for op in operations if operation_trace_graph.has_edge(op, trace_id)
                    ]
                    n_i = len(trace_ops) if trace_ops else 1
                    k_i = 1

                    n_i_inv = 1.0 / n_i
                    k_i_inv = 1.0 / k_i

                    n_i_inv_sum += n_i_inv
                    k_i_inv_sum += k_i_inv
                    trace_weights[trace_id] = (n_i_inv, k_i_inv)

            for trace_id, (n_i_inv, k_i_inv) in trace_weights.items():
                if trace_id in traces:
                    idx = n_ops + traces.index(trace_id)
                    if n_i_inv_sum > 0 and k_i_inv_sum > 0:
                        theta_i = (
                            self.phi * n_i_inv / n_i_inv_sum
                            + (1 - self.phi) * k_i_inv / k_i_inv_sum
                        )
                        u[idx] = theta_i

        # PageRank iteration
        v = np.ones(n_total) / n_total
        max_iterations = 50
        tolerance = 1e-6

        for iteration in range(max_iterations):
            v_new = self.d * A.T @ v + (1 - self.d) * u
            if np.linalg.norm(v_new - v) < tolerance:
                break
            v = v_new

        # Extract abnormal scores
        abnormal_scores = {operations[i]: v[i] for i in range(n_ops)}

        # Calculate normal scores
        normal_scores = {}
        for op in operations:
            normal_count = sum(
                1
                for trace_spans in normal_traces.values()
                if any(span["name"] == op for span in trace_spans)
            )
            total_normal = len(normal_traces) if normal_traces else 1
            normal_scores[op] = normal_count / total_normal

        return abnormal_scores, normal_scores

    def weighted_spectrum_ranker(
        self, abnormal_scores, normal_scores, normal_traces, abnormal_traces
    ):
        """Weighted spectrum ranking: calculate suspicious scores based on Ochiai formula"""
        operations = set(abnormal_scores.keys()) | set(normal_scores.keys())
        spectrum_scores = {}

        N_f = len(abnormal_traces)
        N_p = len(normal_traces)

        for operation in operations:
            F = abnormal_scores.get(operation, 0)
            P = normal_scores.get(operation, 0)

            N_ef = sum(
                1
                for trace_spans in abnormal_traces.values()
                if any(span["name"] == operation for span in trace_spans)
            )
            N_ep = sum(
                1
                for trace_spans in normal_traces.values()
                if any(span["name"] == operation for span in trace_spans)
            )

            O_ef = F * N_ef
            O_nf = F * (N_f - N_ef) if N_f > N_ef else 0
            O_ep = P * N_ep
            O_np = P * (N_p - N_ep) if N_p > N_ep else 0

            denominator = np.sqrt((O_ef + O_ep) * (O_ef + O_nf))
            if denominator > 0:
                ochiai_score = O_ef / denominator
            else:
                ochiai_score = 0

            spectrum_scores[operation] = ochiai_score

        sorted_operations = sorted(spectrum_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_operations


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate Hit Ratio@k evaluation metric"""
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # Check if the predicted top k root causes contain actual root causes
        hit = any(root in predicted_roots[:k] for root in actual_roots)
        hr_k.append(hit)
    return np.mean(hr_k)


def ndcg_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate NDCG@k evaluation metric"""
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
    """Main processing function: use MicroRank algorithm for root cause analysis"""
    print(f"\n=== Starting folder processing ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    # Load data
    print("Loading data...")
    load_start_time = time.time()
    train_df = pd.read_csv(train_path + "/data_final.csv")
    predict_df = pd.read_csv(predict_path + "/data_final.csv")
    if len(train_df) > 100000:
        print("Training set has more than 100k rows, switching to data_final_sampled.csv")
        train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
    if len(predict_df) > 100000:
        print("Prediction set has more than 100k rows, switching to data_final_sampled.csv")
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

    # Modify sampling part
    train_df = sample_data(train_df, TRAIN_SIZE, select_outstanding=False)  # Select stable data
    predict_df = sample_data(predict_df, PREDICT_SIZE, select_outstanding=True)  # Select outstanding data


    load_end_time = time.time()
    print(f"Training data size: {train_df.shape}", flush=True)
    print(f"Prediction data size: {predict_df.shape}", flush=True)
    print(f"Data loading time: {load_end_time - load_start_time:.2f} seconds", flush=True)

    # Convert time columns
    print("Processing time data...")
    time_process_start = time.time()
    try:
        for df in [train_df, predict_df]:
            try:
                df["start_time"] = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S.%f%z")
            except:
                try:
                    df["start_time"] = pd.to_datetime(
                        df["start_time"], format="%Y-%m-%d %H:%M:%S%z"
                    )
                except:
                    try:
                        df["start_time"] = pd.to_datetime(df["start_time"], format="mixed")
                    except:
                        df["start_time"] = pd.to_datetime(df["start_time"], format="ISO8601")
    except Exception as e:
        print(f"Warning: Time conversion error: {str(e)}")
        print("Trying to use default format...")
        train_df["start_time"] = pd.to_datetime(train_df["start_time"])
        predict_df["start_time"] = pd.to_datetime(predict_df["start_time"])
    time_process_end = time.time()
    print(f"Time data processing time: {time_process_end - time_process_start:.2f} seconds", flush=True)

    # Get node intersection
    print("Preparing MicroRank algorithm...")
    data_names = set(train_df["name"].unique()) | set(predict_df["name"].unique())
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    # Use MicroRank algorithm
    microrank_start_time = time.time()

    # Initialize detector
    detector = MicroRankDetector()

    # Group data by operation
    train_all_dfs = {}
    predict_all_dfs = {}
    for name in nodes:
        train_all_dfs[name] = train_df[train_df["name"] == name]
        predict_all_dfs[name] = predict_df[predict_df["name"] == name]

    # Train operation statistics
    detector.train_operation_stats(train_all_dfs)

    # Data preprocessing
    normal_traces, abnormal_traces, call_graph, operation_trace_graph = detector.prepare_data(
        predict_all_dfs, dependency_graph
    )
    print(f"Normal traces: {len(normal_traces)}, Abnormal traces: {len(abnormal_traces)}")

    # PageRank scoring
    abnormal_scores, normal_scores = detector.pagerank_scoring(
        normal_traces, abnormal_traces, call_graph, operation_trace_graph
    )

    # Weighted spectrum ranking
    sorted_operations = detector.weighted_spectrum_ranker(
        abnormal_scores, normal_scores, normal_traces, abnormal_traces
    )

    # Get service list
    service_list = [operation[0] for operation in sorted_operations]

    microrank_end_time = time.time()

    print("MicroRank algorithm execution completed")
    print(
        f"MicroRank algorithm time: {microrank_end_time - microrank_start_time:.2f} seconds",
        flush=True,
    )

    print("Detected service root causes:")
    for i, (operation, score) in enumerate(sorted_operations):
        print(f"{operation}: {score:.4f} -> Rank {i+1}", flush=True)

    # Get labels
    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]
    print(f"Actual service root cause: {service_label}", flush=True)

    return service_list, service_label


def write_results_to_csv(all_results, filename="microrank_results.csv"):
    """Save evaluation results to CSV file"""
    detailed_results = []
    for result in all_results:
        actual_roots, predicted_roots = (
            result["actual_roots"],
            result["predicted_roots"],
        )
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
    n, omega, phi, d = params

    # Set global parameters
    global N_MULTIPLIER, OMEGA, PHI, D
    N_MULTIPLIER, OMEGA, PHI, D = n, omega, phi, d

    print(f"Running parameters: n={n}, ω={omega}, φ={phi}, d={d}")

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
        param_folder = f"n{n:.1f}_omega{omega:.3f}_phi{phi:.1f}_d{d:.2f}"
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
        nonservice_lists = []
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
                service_list, service_label = sub_main(
                    train_path, predict_path, failure_label_path, nodes, dependency_graph
                )
                case_end_time = time.time()
                case_runtime = case_end_time - case_start_time

                # Filter query operations
                service_list = [item for item in service_list if item != "query"]

                service_lists.append(service_list)
                service_labels.append(service_label)

                # Construct nonservice list (microservice code itself doesn't generate, add empty list here)
                nonservice_list = []
                nonservice_label = []
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
                print(f"Error processing folder {folder_name}: {str(e)}")
                continue

        # Save hyperparameter experiment results
        if service_lists:
            # Save evaluation results
            result_path = os.path.join(param_output_dir, "evaluation_results.csv")

            # Use the same function as nezha to calculate and save evaluation metrics
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

            print(f"Completed parameter combination: n={n}, ω={omega}, φ={phi}, d={d}")
            return param_folder

        return None

    except Exception as e:
        print(f"Parameter combination failed: n={n}, ω={omega}, φ={phi}, d={d}, error: {e}")
        return None


def write_hyperparameter_summary(all_results, output_dir):
    """Summarize all hyperparameter experiment results"""
    if not all_results:
        print("No hyperparameter experiment results to summarize")
        return

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)

    # Save summary results
    hyperparams_dir = os.path.join(output_dir, "hyperparameter_tuning")
    os.makedirs(hyperparams_dir, exist_ok=True)
    summary_path = os.path.join(hyperparams_dir, "hyperparameter_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n=== Hyperparameter Tuning Summary Results ===")
    print(f"Total experiments: {len(all_results)}")
    print(f"Summary results saved to: {summary_path}")

    # Sort by HR@1 to show best results
    best_hr1 = summary_df.loc[summary_df["HR@1"].idxmax()]
    best_hr3 = summary_df.loc[summary_df["HR@3"].idxmax()]
    best_hr5 = summary_df.loc[summary_df["HR@5"].idxmax()]

    print(f"\nBest HR@1: {best_hr1['HR@1']:.4f}")
    print(
        f"Parameters: n={best_hr1['n_multiplier']}, ω={best_hr1['omega']}, φ={best_hr1['phi']}, d={best_hr1['d']}"
    )

    print(f"\nBest HR@3: {best_hr3['HR@3']:.4f}")
    print(
        f"Parameters: n={best_hr3['n_multiplier']}, ω={best_hr3['omega']}, φ={best_hr3['phi']}, d={best_hr3['d']}"
    )

    print(f"\nBest HR@5: {best_hr5['HR@5']:.4f}")
    print(
        f"Parameters: n={best_hr5['n_multiplier']}, ω={best_hr5['omega']}, φ={best_hr5['phi']}, d={best_hr5['d']}"
    )

    # Sort by NDCG@1 to show best results
    best_ndcg1 = summary_df.loc[summary_df["NDCG@1"].idxmax()]
    best_ndcg3 = summary_df.loc[summary_df["NDCG@3"].idxmax()]
    best_ndcg5 = summary_df.loc[summary_df["NDCG@5"].idxmax()]

    print(f"\nBest NDCG@1: {best_ndcg1['NDCG@1']:.4f}")
    print(
        f"Parameters: n={best_ndcg1['n_multiplier']}, ω={best_ndcg1['omega']}, φ={best_ndcg1['phi']}, d={best_ndcg1['d']}"
    )

    print(f"\nBest NDCG@3: {best_ndcg3['NDCG@3']:.4f}")
    print(
        f"Parameters: n={best_ndcg3['n_multiplier']}, ω={best_ndcg3['omega']}, φ={best_ndcg3['phi']}, d={best_ndcg3['d']}"
    )

    print(f"\nBest NDCG@5: {best_ndcg5['NDCG@5']:.4f}")
    print(
        f"Parameters: n={best_ndcg5['n_multiplier']}, ω={best_ndcg5['omega']}, φ={best_ndcg5['phi']}, d={best_ndcg5['d']}"
    )

    return summary_df


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

    # n: standard deviation multiplier, omega: call graph weight, phi: trace range and type balancing factor, d: PageRank damping factor
    # phi: balance the weight of trace range and trace type factors. Lower values focus more on trace diversity, less on trace operation count.
    # d: control random jump probability in PageRank algorithm. Lower values focus more on local structure.
    param_combinations = [
        (n, omega, phi, d)
        for n in [1.0, 2.0, 3.0]
        for omega in [0.005, 0.01]
        for phi in [0.1, 0.3, 0.5, 0.7, 0.9]
        for d in [0.15, 0.85]
    ]

    print(f"Starting microrank hyperparameter search, total {len(param_combinations)} parameter combinations")

    # Check completed experiments to avoid duplicate runs
    pending = []
    for p in param_combinations:
        n, omega, phi, d = p
        param_folder = f"n{n:.1f}_omega{omega:.3f}_phi{phi:.1f}_d{d:.2f}"
        result_file = os.path.join(output_dir, param_folder, "evaluation_results.csv")
        if not os.path.exists(result_file):
            pending.append(p)

    print(f"Remaining {len(pending)} parameter combinations to run")

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
                    result = future.result()  # 10 minute timeout
                    if result:
                        all_results.append(result)
                        print(f"Progress [{completed}/{len(pending)}]: Completed {result}")
                    else:
                        print(f"Progress [{completed}/{len(pending)}]: Skipped")
                except Exception as e:
                    params = futures[future]
                    print(f"Progress [{completed}/{len(pending)}]: Failed - parameters{params}, error: {str(e)}")

        print(f"Hyperparameter tuning completed, successfully ran {len(all_results)} parameter combinations")

    end_time = time.time()  # End timing
    run_time = end_time - start_time
    print(f"\nTotal runtime: {run_time:.4f} seconds")
    print("=== Program execution completed ===")

    # Summarize HR@1 results from all parameter folders and sort output
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
