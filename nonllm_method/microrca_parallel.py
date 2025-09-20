import pandas as pd
import numpy as np
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
import networkx as nx
import os
import re
import json
import warnings
import concurrent.futures
from utils.CSVFormat import *
import time

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Global hyperparameter configuration - ported from improved algorithm
THRESHOLD = 0.045  # BIRCH clustering threshold
ALPHA = 0.55  # Anomaly edge weight parameter
C = 0.15  # PageRank restart probability
MAX_ITERATIONS = 100  # PageRank maximum iterations
TOLERANCE = 1e-6  # PageRank convergence tolerance
ROLLING_WINDOW = 6  # Data smoothing window size
BRANCHING_FACTOR = 50  # BIRCH clustering branching factor
N_CLUSTERS = None  # BIRCH clustering number
COMPUTE_LABELS = True  # Whether to compute clustering labels


def create_attributed_graph(dependency_graph, host_mapping=None):
    """Build attributed graph containing service nodes and host nodes - from improved algorithm"""
    G = nx.DiGraph()

    # Add directed edges between services, representing call dependencies
    for parent, children in dependency_graph.items():
        for child in children:
            G.add_edge(parent, child, edge_type="service")

    # Add service-host directed edges, representing deployment relationships
    if host_mapping:
        for service, host in host_mapping.items():
            G.add_edge(service, host, edge_type="host")

    return G


def birch_anomaly_detection_train(train_data, threshold=None):
    """Train anomaly detection using BIRCH - training function from improved algorithm"""
    if threshold is None:
        threshold = THRESHOLD

    if len(train_data) == 0:
        raise ValueError("Training data is empty")

    # Ensure input is pandas Series format
    if not isinstance(train_data, pd.Series):
        train_data = pd.Series(train_data)

    # Apply rolling average smoothing to response time to reduce noise
    smoothed_data = train_data.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    x = smoothed_data.fillna(0).values  # Fill missing values and convert to numpy array

    # Data standardization to eliminate dimensional effects
    scaler = StandardScaler()
    normalized_x = scaler.fit_transform(x.reshape(-1, 1))

    # BIRCH unsupervised online clustering to establish normal behavior patterns
    birch = Birch(
        branching_factor=BRANCHING_FACTOR,
        n_clusters=N_CLUSTERS,
        threshold=threshold,
        compute_labels=COMPUTE_LABELS,
    )
    birch.fit(normalized_x)

    return birch, scaler


def birch_anomaly_detection_predict(predict_data, birch_model, scaler):
    """Use trained BIRCH model for anomaly detection prediction - improved algorithm"""
    # Ensure input is pandas Series format
    if not isinstance(predict_data, pd.Series):
        predict_data = pd.Series(predict_data)

    # Data smoothing processing
    smoothed_data = predict_data.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    x = smoothed_data.fillna(0).values  # Convert to numpy array

    # Use standardization parameters from training for standardization
    normalized_x = scaler.transform(x.reshape(-1, 1))

    # Predict cluster labels, labels > 0 indicate anomalies
    labels = birch_model.predict(normalized_x)

    # Return anomaly identification array
    is_anomaly = (labels > 0).astype(int)

    return is_anomaly


def identify_anomalous_edges_and_nodes(service_data_dict):
    """Anomalous edge and node identification - from improved algorithm"""
    anomaly_edges = set()  # Anomalous edge set
    anomaly_services = set()  # Anomalous service set

    for service, data in service_data_dict.items():
        if "is_anomaly" in data.columns and data["is_anomaly"].any():
            anomaly_services.add(service)
            # Calculate average response time during anomalies
            anomaly_mask = data["is_anomaly"] == 1
            if anomaly_mask.any():
                anomaly_rt_mean = data.loc[anomaly_mask, "duration"].mean()
            else:
                anomaly_rt_mean = data["duration"].mean()

            # Add anomalous response time mean to service data
            service_data_dict[service] = data.assign(rt_anomaly_mean=anomaly_rt_mean)

    return anomaly_edges, anomaly_services


def extract_anomalous_subgraph(graph, anomaly_services):
    """Extract connected subgraph containing anomaly propagation paths - from improved algorithm"""
    subgraph_nodes = set()

    # For each anomalous service, collect its neighbor nodes (predecessors and successors)
    for service in anomaly_services:
        subgraph_nodes.add(service)
        subgraph_nodes.update(graph.neighbors(service))  # Successor nodes
        subgraph_nodes.update(graph.predecessors(service))  # Predecessor nodes

    # Build anomalous subgraph
    anomalous_subgraph = graph.subgraph(subgraph_nodes).copy()
    return anomalous_subgraph


def calculate_edge_weights(graph, service_data_dict, anomaly_services, alpha=None):
    """Anomalous subgraph weighting - edge weight calculation from improved algorithm"""
    if alpha is None:
        alpha = ALPHA

    for u, v, data in graph.edges(data=True):
        edge_type = data.get("edge_type", "service")

        # Anomalous edge weight priority judgment - if edge connects anomalous services, give high weight
        if (u in anomaly_services and v in service_data_dict) or (
            v in anomaly_services and u in service_data_dict
        ):
            is_anomaly_edge = False
            if u in service_data_dict and "is_anomaly" in service_data_dict[u].columns:
                if service_data_dict[u]["is_anomaly"].any():
                    is_anomaly_edge = True

            if is_anomaly_edge:
                graph[u][v]["weight"] = alpha
                continue

        # Normal service node weights - based on response time correlation calculation
        if edge_type == "service" and u in service_data_dict and v in service_data_dict:
            try:
                # Get response time data for both services, unified to minimum length
                u_data = service_data_dict[u]["duration"].values
                v_data = service_data_dict[v]["duration"].values
                min_len = min(len(u_data), len(v_data))

                if min_len <= 0:
                    graph[u][v]["weight"] = 0.1
                    continue

                # Truncate to same length for correlation calculation
                u_data = u_data[:min_len]
                v_data = v_data[:min_len]

                # Calculate different correlations based on anomaly status
                if u not in anomaly_services and v in anomaly_services:
                    rt_a_j = service_data_dict[v].get(
                        "rt_anomaly_mean", service_data_dict[v]["duration"].mean()
                    )
                    rt_a_j_array = np.full(min_len, rt_a_j)
                    corr = np.corrcoef(u_data, rt_a_j_array)[0, 1]
                    graph[u][v]["weight"] = max(0, corr) if not np.isnan(corr) else 0.1
                elif v not in anomaly_services and u in anomaly_services:
                    rt_a_i = service_data_dict[u].get(
                        "rt_anomaly_mean", service_data_dict[u]["duration"].mean()
                    )
                    rt_a_i_array = np.full(min_len, rt_a_i)
                    corr = np.corrcoef(v_data, rt_a_i_array)[0, 1]
                    graph[u][v]["weight"] = max(0, corr) if not np.isnan(corr) else 0.1
                else:
                    # Check if standard deviation is 0 to avoid correlation calculation errors
                    if np.std(u_data) == 0 or np.std(v_data) == 0:
                        graph[u][v]["weight"] = 0.1
                    else:
                        corr = np.corrcoef(u_data, v_data)[0, 1]
                        graph[u][v]["weight"] = max(0, corr) if not np.isnan(corr) else 0.1
            except Exception as e:
                print(f"Weight calculation failed {u}->{v}: {e}")
                graph[u][v]["weight"] = 0.1
        else:
            graph[u][v]["weight"] = 0.1


def compute_service_anomaly_scores(anomaly_services, service_data_dict):
    """Service anomaly score assignment - anomaly score calculation from improved algorithm"""
    anomaly_scores = {}

    for service in anomaly_services:
        if service not in service_data_dict:
            anomaly_scores[service] = 0
            continue

        data = service_data_dict[service]
        w_service = 1.0
        container_metrics = [col for col in data.columns if col.startswith("metric@")]
        rt_a_sj = data.get("rt_anomaly_mean", data["duration"].mean())
        max_corr = 0

        for metric in container_metrics:
            if metric in data.columns:
                try:
                    uk_array = data[metric].values
                    if len(uk_array) > 0:
                        rt_a_array = np.full(len(uk_array), rt_a_sj)
                        # Check standard deviation to avoid division by zero error
                        if np.std(uk_array) > 0:
                            corr = np.corrcoef(uk_array, rt_a_array)[0, 1]
                            if not np.isnan(corr):
                                max_corr = max(max_corr, abs(corr))
                except Exception:
                    continue

        anomaly_scores[service] = w_service * max_corr

    return anomaly_scores


def personalized_pagerank(graph, preference_vector, c=None, max_iterations=None, tolerance=None):
    """Personalized PageRank - PageRank implementation from improved algorithm"""
    if c is None:
        c = C
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS
    if tolerance is None:
        tolerance = TOLERANCE

    nodes = list(graph.nodes())
    n = len(nodes)

    if n == 0:
        return {}

    # Build transition probability matrix
    P = np.zeros((n, n))
    for i, node in enumerate(nodes):
        out_edges = list(graph.successors(node))
        if out_edges:
            total_weight = sum(graph[node][neighbor].get("weight", 1) for neighbor in out_edges)
            if total_weight > 0:
                for neighbor in out_edges:
                    j = nodes.index(neighbor)
                    P[j][i] = graph[node][neighbor].get("weight", 1) / total_weight

    # Initialize vectors
    v = np.ones(n) / n
    u = np.array([preference_vector.get(node, 0) for node in nodes])

    # Normalize preference vector
    if np.sum(u) > 0:
        u = u / np.sum(u)
    else:
        u = np.ones(n) / n

    # Iterative calculation
    for iteration in range(max_iterations):
        v_new = (1 - c) * np.dot(P, v) + c * u

        if np.linalg.norm(v_new - v) < tolerance:
            break
        v = v_new

    return dict(zip(nodes, v))


def detect_anomalies_and_build_service_data(
    nodes, train_data_dict, predict_data_dict, threshold=None
):
    """Detect anomalies and build service data dictionary - anomaly detection process from improved algorithm"""
    if threshold is None:
        threshold = THRESHOLD

    service_data_dict = {}

    for service in nodes:
        if service in predict_data_dict:
            data = predict_data_dict[service].copy()

            if service in train_data_dict and len(train_data_dict[service]) > 0:
                try:
                    if (
                        "duration" in train_data_dict[service].columns
                        and len(train_data_dict[service]["duration"]) > 0
                    ):
                        birch_model, scaler = birch_anomaly_detection_train(
                            train_data_dict[service]["duration"], threshold
                        )
                        is_anomaly = birch_anomaly_detection_predict(
                            data["duration"], birch_model, scaler
                        )

                        # Ensure length matches exactly
                        expected_len = len(data)
                        if len(is_anomaly) != expected_len:
                            print(
                                f"Length mismatch {service}: is_anomaly={len(is_anomaly)}, data={expected_len}"
                            )
                            # Adjust to correct length
                            if len(is_anomaly) > expected_len:
                                is_anomaly = is_anomaly[:expected_len]
                            else:
                                padding = np.zeros(expected_len - len(is_anomaly), dtype=int)
                                is_anomaly = np.concatenate([is_anomaly, padding])

                        data = data.copy()  # Ensure original data is not modified
                        data["is_anomaly"] = is_anomaly
                    else:
                        data["is_anomaly"] = np.zeros(len(data), dtype=int)
                except Exception as e:
                    print(f"Service {service} anomaly detection failed: {e}")
                    data["is_anomaly"] = np.zeros(len(data), dtype=int)
            else:
                data["is_anomaly"] = np.zeros(len(data), dtype=int)

            service_data_dict[service] = data

    return service_data_dict


def localize_faulty_services(dependency_graph, service_data_dict, host_mapping=None):
    """Faulty service localization - complete fault localization process using improved algorithm"""
    # Build attributed graph
    attributed_graph = create_attributed_graph(dependency_graph, host_mapping)

    # Anomalous subgraph extraction
    anomaly_edges, anomaly_services = identify_anomalous_edges_and_nodes(service_data_dict)

    # If no anomalous services, return default ordering based on communication dependencies
    if not anomaly_services:
        print("No anomalous services detected, using services from dependency graph as default ordering")
        # Extract all services from dependency graph as candidates
        all_services = set(dependency_graph.keys())
        for deps in dependency_graph.values():
            all_services.update(deps)

        # Filter services that exist in service_data_dict
        available_services = [s for s in all_services if s in service_data_dict]

        # If dependency graph is empty, return all available services
        if not available_services:
            available_services = list(service_data_dict.keys())

        return [(service, 1.0) for service in available_services]

    anomalous_subgraph = extract_anomalous_subgraph(attributed_graph, anomaly_services)

    # Faulty service localization
    calculate_edge_weights(anomalous_subgraph, service_data_dict, anomaly_services)
    anomaly_scores = compute_service_anomaly_scores(anomaly_services, service_data_dict)

    # Reverse edges between service nodes
    reversed_graph = anomalous_subgraph.reverse()

    # Apply personalized PageRank
    pagerank_scores = personalized_pagerank(reversed_graph, anomaly_scores)

    # Get service scores
    service_scores = {}
    all_services = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        all_services.update(deps)

    for node, score in pagerank_scores.items():
        if node in all_services:
            service_scores[node] = score

    sorted_services = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_services


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate HR@k - maintain original evaluation metrics"""
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # Check if predicted top k root causes contain actual root causes
        hit = any(root in predicted_roots[:k] for root in actual_roots)
        hr_k.append(hit)
    return np.mean(hr_k)


def ndcg_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate NDCG@k - maintain original evaluation metrics"""
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
    """Main processing function - integrate improved algorithm into original workflow"""
    print(f"\n=== Starting folder processing ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    # Load data - maintain original data loading logic
    print("Loading data...")
    train_df = pd.read_csv(train_path + "/data_final.csv")
    predict_df = pd.read_csv(predict_path + "/data_final.csv")
    if len(train_df) > 100000:
        print("Training set rows > 100k, switching to data_final_sampled.csv")
        train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
    if len(predict_df) > 100000:
        print("Prediction set rows > 100k, switching to data_final_sampled.csv")
        predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")
    print(f"Training data size: {train_df.shape}")
    print(f"Prediction data size: {predict_df.shape}")

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

    # Get all unique name values in data
    data_names = set(train_df["name"].unique()) | set(predict_df["name"].unique())
    # Take intersection of nodes and names in data
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    # Keep common columns - maintain original data preprocessing logic
    print("Processing common columns...")
    common_columns = train_df.columns.intersection(predict_df.columns)
    train_df = train_df[common_columns]
    predict_df = predict_df[common_columns]
    print(f"Number of common columns: {len(common_columns)}")

    # Use improved algorithm for anomaly detection and fault localization
    print("Using improved microrca algorithm for fault localization...")

    # Group data by service
    train_data_dict = {}
    predict_data_dict = {}
    for service in nodes:
        train_data = train_df[train_df["name"] == service]
        predict_data = predict_df[predict_df["name"] == service]

        if len(train_data) > 0:
            train_data_dict[service] = train_data
        if len(predict_data) > 0:
            predict_data_dict[service] = predict_data

    # Use improved anomaly detection algorithm
    service_data_dict = detect_anomalies_and_build_service_data(
        nodes, train_data_dict, predict_data_dict
    )

    # Use improved fault localization algorithm
    sorted_faulty_services = localize_faulty_services(dependency_graph, service_data_dict)
    service_list = [service[0] for service in sorted_faulty_services]
    print(f"Detected service root causes: {service_list}")

    # Get labels - maintain original label retrieval logic
    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]
    print(f"Actual service root causes: {service_label}")

    return service_list, service_label


def write_results_to_csv(
    service_labels, service_lists, ks=[1, 3, 5], filename="evaluation_results.csv"
):
    """Save evaluation results - maintain original result saving format"""
    results = []
    for k in ks:
        hr_k = hit_ratio_at_k(service_labels, service_lists, k)
        ndcg_k = ndcg_at_k(service_labels, service_lists, k)
        # Save results
        results.append([k, hr_k, ndcg_k])

    # Use pandas DataFrame to store results
    df = pd.DataFrame(results, columns=["k", "HR@k", "NDCG@k"])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

    # Add this line to print results
    print("\nEvaluation Results:")
    print(df)


def run_single_params(params):
    """Run single hyperparameter configuration"""
    threshold, alpha = params

    # Set global hyperparameters
    global THRESHOLD, ALPHA
    THRESHOLD, ALPHA = threshold, alpha

    print(f"Running parameters: threshold={threshold}, alpha={alpha}")

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
        param_folder = f"threshold{threshold:.3f}_alpha{alpha:.2f}"
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
                print(f"Error processing folder {folder_name}: {str(e)}")
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

            print(f"Completed parameter combination: threshold={threshold}, alpha={alpha}")
            return param_folder

        return None

    except Exception as e:
        print(f"Parameter combination failed: threshold={threshold}, alpha={alpha}, error: {e}")
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

    # Hyperparameter search configuration - threshold: BIRCH threshold, alpha: anomaly edge weight
    param_combinations = [(t, a) for t in [0.01, 0.1, 0.5] for a in [0.3, 0.5, 0.7]]

    print(f"Starting microrca hyperparameter search, total {len(param_combinations)} parameter sets")

    # Check completed experiments to avoid duplicate runs
    pending = []
    for p in param_combinations:
        threshold, alpha = p
        param_folder = f"threshold{threshold:.3f}_alpha{alpha:.2f}"
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

    else:
        print("All hyperparameter combinations completed, running default parameter experiment...")

        folder_names = [
            f
            for f in os.listdir(data_final_dir)
            if os.path.isdir(os.path.join(data_final_dir, f)) and f != "00"
        ]
        folder_names.sort()

        # Filter completed folders
        for folder_name in folder_names[:]:
            eval_path = os.path.join(output_dir, folder_name, "evaluation_results.csv")
            if os.path.exists(eval_path):
                folder_names.remove(folder_name)
        print(f"\nFound {len(folder_names)} data folders to process")

        train_path = normal_data_dir
        print(f"Using {train_path} as training data")

        service_lists = []
        service_labels = []

        def process_folder(args):
            i, folder_name = args
            predict_path = os.path.join(data_final_dir, folder_name)
            label_match = re.search(r"(\d+)_", predict_path)
            if not label_match:
                return None
            label_str = label_match.group(1)

            print(f"\n[{i}/{len(folder_names)}] Processing folder: {folder_name}")
            if int(label_str) in jump_folder:
                print(f"Skipping folder {folder_name} (in jump list)")
                return None

            try:
                service_list, service_label = sub_main(
                    train_path, predict_path, failure_label_path, nodes, dependency_graph
                )
                service_list = [item for item in service_list if item != "query"]
                return (service_list, service_label)
            except Exception as e:
                print(f"Error processing folder {folder_name}: {str(e)}")
                return None

        folder_args = [(i, folder_name) for i, folder_name in enumerate(folder_names, 1)]

        max_workers = max(int(os.cpu_count() / 10 - 1), 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_folder, arg): arg for arg in folder_args}
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except concurrent.futures.TimeoutError:
                    print("Task timeout")
                    future.cancel()
                except Exception as e:
                    print(f"Task execution failed: {str(e)}")
                    future.cancel()

            for future in futures:
                if not future.done():
                    future.cancel()

        for res in results:
            if res:
                service_lists.append(res[0])
                service_labels.append(res[1])

        print("\n=== Processing final results ===")
        print(f"Number of root cause lists: {len(service_lists)}")
        print(f"Root cause lists: {service_lists}")

        evaluation_results_path = os.path.join(output_dir, "evaluation_results.csv")
        write_results_to_csv(service_labels, service_lists, filename=evaluation_results_path)
        print(f"Results saved to {evaluation_results_path}")

    # Summarize results from all parameter folders, generate all_results.csv
    import glob

    result_rows = []
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
                    if "nonserviceHR@k" in r.index:
                        row[f"nonserviceHR@{k}"] = r["nonserviceHR@k"]
                    if "nonserviceNDCG@k" in r.index:
                        row[f"nonserviceNDCG@{k}"] = r["nonserviceNDCG@k"]
                result_rows.append(row)
            except Exception as e:
                print(f"Failed to read {eval_csv}: {e}")

    if result_rows:

        def colkey(x):
            if "@" in x:
                return (int(x.split("@")[1]), x.split("@")[0])
            return (-1, x)

        all_cols = set()
        for r in result_rows:
            all_cols.update(r.keys())
        all_cols.discard("param_folder")
        columns = ["param_folder"] + sorted(all_cols, key=colkey)
        df_all = pd.DataFrame(result_rows)[columns]
        if "HR@1" in df_all.columns:
            df_all = df_all.sort_values("HR@1", ascending=False)
        summary_path = os.path.join(output_dir, "all_results.csv")
        df_all.to_csv(summary_path, index=False)
        print(f"Output {len(df_all)} rows to all_results.csv")
    else:
        print("No results")

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.4f} seconds")
    print("=== Program execution completed ===")
