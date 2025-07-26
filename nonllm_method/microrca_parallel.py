import pandas as pd
import numpy as np
from sklearn.cluster import Birch
from sklearn.preprocessing import normalize
import networkx as nx
import os
import re
import json
import warnings
import concurrent.futures
from sklearn.preprocessing import StandardScaler
from nonllm_method.utils.CSVFormat import *

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def create_graph_from_dependency_graph(dependency_graph):
    G = nx.DiGraph()  # Create a directed graph
    for parent, children in dependency_graph.items():
        for child in children:
            G.add_edge(parent, child)  # Add edge to represent dependency
    return G


def birch_ad_with_smoothing_train(train_df, threshold=0.045):
    if len(train_df) == 0:
        raise ValueError("Training data is empty")
    """
    Use training data (train_df) to fit a BIRCH model, and return standardized data and cluster labels for prediction.
    """
    # Smooth 'duration' column
    latency = train_df["duration"].rolling(window=12, min_periods=1).mean()
    # Replace missing values with 0
    x = np.array(latency)
    x = np.where(np.isnan(x), 0, x)

    # Data standardization: calculate standardization parameters (mean and std) during training
    scaler = StandardScaler()
    normalized_x = scaler.fit_transform(x.reshape(-1, 1))  # Use fit_transform to get standardization params
    # Build and train BIRCH model
    X = normalized_x.reshape(-1, 1)
    brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
    brc.fit(X)
    # Get cluster labels
    labels = brc.labels_

    # Return BIRCH model, scaler, and normalized training data
    return brc, scaler, normalized_x, labels


def birch_ad_with_smoothing_predict(predict_df, model, scaler, threshold=0.045):
    """
    Use the trained BIRCH model to perform anomaly detection on prediction data (predict_df), and mark as new column 'is_anomaly'.
    """
    # Smooth prediction data
    latency = predict_df["duration"].rolling(window=12, min_periods=1).mean()
    # Replace missing values with 0
    x = np.array(latency)
    x = np.where(np.isnan(x), 0, x)

    # Standardize prediction data using training parameters
    normalized_x = scaler.transform(x.reshape(-1, 1))  # Use transform to ensure consistency with training data
    X = normalized_x.reshape(-1, 1)

    # Predict using trained model
    brc_labels = model.predict(X)

    # Mark each data point as anomaly if cluster label > 0
    predict_df["is_anomaly"] = (brc_labels > 0).astype(int)  # Or use `n_clusters` to judge
    return predict_df


def node_weight(name, predict_all_dfs, dependency_graph):
    # 1. Calculate weighted average of incoming edges for the node
    in_edges_weight_avg = 0.0
    num_edges = 0

    # Traverse dependencies to calculate weighted value of incoming edges
    for parent_name in dependency_graph:
        if name in dependency_graph[parent_name]:  # Check if parent node depends on current node
            # Directly calculate weighted value, assume all edge weights are 1, or adjust as needed
            num_edges += 1
            in_edges_weight_avg += 1  # Adjust weight as needed, e.g., use a value instead of 1

    if num_edges > 0:
        in_edges_weight_avg /= num_edges  # Calculate weighted average

    # 2. Calculate max correlation with resource columns (using prediction data)
    resource_columns = [col for col in predict_all_dfs[f"predict_{name}_df"].columns if col.startswith("metric@")]

    # Initialize max correlation and corresponding metric
    max_corr = -1
    metric = "node_cpu"  # Default metric

    # Extract 'duration' data for the service node from prediction data
    service_data = predict_all_dfs[f"predict_{name}_df"]["duration"]

    # Calculate correlation with each resource column
    for col in resource_columns:
        temp = abs(service_data.corr(predict_all_dfs[f"predict_{name}_df"][col]))  # Calculate correlation
        if temp > max_corr:
            max_corr = temp
            metric = col

    # Calculate final node weighted value
    data = in_edges_weight_avg * max_corr

    # 3. Adjust or mark anomaly node based on 'is_anomaly' column
    if predict_all_dfs[f"predict_{name}_df"]["is_anomaly"].any():  # If node has anomaly
        data *= 1.5  # For example, anomaly node's weight may be higher (adjust as needed)

    return data, metric


# 2. Calculate anomaly edge weight
def compute_anomaly_edge_weight(service_node, dependency_graph, predict_all_dfs):
    anomaly_weight = predict_all_dfs[f"predict_{service_node}_df"]["is_anomaly"].mean()
    return anomaly_weight


# 3. Calculate anomaly score for service node
def compute_service_anomaly_score(service_node, anomaly_weight, predict_all_dfs, resource_columns):
    service_data = predict_all_dfs[f"predict_{service_node}_df"]["duration"]
    max_corr = -1
    for col in resource_columns:
        corr_value = service_data.corr(predict_all_dfs[f"predict_{service_node}_df"][col])
        max_corr = max(max_corr, abs(corr_value))
    anomaly_score = anomaly_weight * max_corr
    return anomaly_score


# 4. Calculate Personalized PageRank
def personalized_pagerank(graph, anomaly_scores, c=0.15):
    nodes = list(graph.nodes)
    num_nodes = len(nodes)
    P = np.zeros((num_nodes, num_nodes))

    # Assign weight to each edge
    for i, node in enumerate(nodes):
        neighbors = list(graph.neighbors(node))
        weight_sum = sum([graph[node][neighbor].get("weight", 1) for neighbor in neighbors])
        for j, neighbor in enumerate(neighbors):
            P[i, nodes.index(neighbor)] = graph[node][neighbor].get("weight", 1) / weight_sum

    w = np.array([anomaly_scores.get(node, 0) for node in nodes])

    # Personalized PageRank iterative calculation
    u = np.ones(num_nodes) / num_nodes  # Initial probability distribution
    for _ in range(100):
        u = (1 - c) * np.dot(P, u) + c * w

    return u


# 5. Reverse edges in the graph (for PageRank)
def reverse_graph_edges(graph, nodes):
    reversed_graph = graph.copy()
    for node in nodes:
        for neighbor in list(graph.neighbors(node)):
            reversed_graph.remove_edge(node, neighbor)
            reversed_graph.add_edge(neighbor, node)
    return reversed_graph


def localize_faulty_services(dependency_graph, predict_all_dfs, nodes):
    anomaly_scores = {}
    # Use the first available node to get resource columns
    first_node = next(iter(predict_all_dfs.keys()))
    resource_columns = [col for col in predict_all_dfs[first_node].columns if col.startswith("metric@")]

    for node in nodes:
        df_key = f"predict_{node}_df"
        if df_key not in predict_all_dfs:
            continue
        if "is_anomaly" not in predict_all_dfs[df_key].columns:
            continue

        anomaly_weight = compute_anomaly_edge_weight(node, dependency_graph, predict_all_dfs)
        anomaly_score = compute_service_anomaly_score(node, anomaly_weight, predict_all_dfs, resource_columns)
        anomaly_scores[node] = anomaly_score

    # Create directed graph
    graph = create_graph_from_dependency_graph(dependency_graph)
    reversed_graph = reverse_graph_edges(graph, nodes)
    faulty_services = personalized_pagerank(reversed_graph, anomaly_scores)

    sorted_services = sorted(zip(nodes, faulty_services), key=lambda x: x[1], reverse=True)
    return sorted_services


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # Check if any actual root cause is in the top k predicted root causes
        hit = any(root in predicted_roots[:k] for root in actual_roots)
        hr_k.append(hit)
    return np.mean(hr_k)


def ndcg_at_k(actual_root_causes, predicted_top_k, k):
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
    print(f"\n=== Start processing folder ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    # Load data
    print("\n[1/4] Loading data...")
    train_df = pd.read_csv(train_path + "/data_final.csv")
    predict_df = pd.read_csv(predict_path + "/data_final.csv")
    if len(train_df) > 100000:
        print("Training set has more than 100k rows, switching to data_final_sampled.csv")
        train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
    if len(predict_df) > 100000:
        print("Prediction set has more than 100k rows, switching to data_final_sampled.csv")
        predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")
    print(f"Training data shape: {train_df.shape}")
    print(f"Prediction data shape: {predict_df.shape}")
    
    # Get all unique 'name' values in the data
    data_names = set(train_df['name'].unique()) | set(predict_df['name'].unique())
    # Take intersection of nodes and data_names
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    # Add data validation
    for name in nodes:
        train_data = train_df[train_df["name"] == name]
        if len(train_data) == 0:
            print(f"Warning: No data found for service {name} in training data")
            continue
        predict_data = predict_df[predict_df["name"] == name]
        if len(predict_data) == 0:
            print(f"Warning: No data found for service {name} in prediction data")
            continue

    # Keep only common columns
    print("\n[2/4] Processing common columns...")
    common_columns = train_df.columns.intersection(predict_df.columns)
    train_df = train_df[common_columns]
    predict_df = predict_df[common_columns]
    print(f"Number of common columns: {len(common_columns)}")

    # Remove redundant log columns
    print("\n[3/4] Removing redundant log columns...")
    drop_log = ["log@httpcore.connection", "log@urllib3.connectionpool", "log@httpx", "log@httpcore.http11"]
    columns_to_drop = [col for col in train_df.columns if any(col.startswith(prefix) for prefix in drop_log)]
    train_df = train_df.drop(columns=columns_to_drop)
    predict_df = predict_df.drop(columns=columns_to_drop)
    print(f"Number of log columns removed: {len(columns_to_drop)}")

    # Anomaly detection
    print("\n[4/4] Anomaly detection...")
    train_all_dfs = {}
    predict_all_dfs = {}
    threshold = 0.98

    for name in nodes:
        train_all_dfs[f"train_{name}_df"] = train_df[train_df["name"] == name]
        predict_all_dfs[f"predict_{name}_df"] = predict_df[predict_df["name"] == name]
        if len(train_all_dfs[f"train_{name}_df"]) > 0:
            try:
                brc, scaler, normalized_x, labels = birch_ad_with_smoothing_train(train_all_dfs[f"train_{name}_df"], threshold=threshold)
                predict_all_dfs[f"predict_{name}_df"] = birch_ad_with_smoothing_predict(predict_all_dfs[f"predict_{name}_df"], brc, scaler, threshold=threshold)
            except Exception as e:
                print(f"Error processing service {name}: {str(e)}")

    sorted_faulty_services = localize_faulty_services(dependency_graph, predict_all_dfs, nodes)
    service_list = [service[0] for service in sorted_faulty_services]
    print(f"Detected root cause services: {service_list}")

    # Get labels
    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]
    print(f"Actual root cause services: {service_label}")

    return service_list, service_label


def write_results_to_csv(service_labels, service_lists, ks=[1, 3, 5], filename="evaluation_results.csv"):
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

    # Print results
    print("\nEvaluation Results:")
    print(df)


if __name__ == "__main__":
    # Start timing
    import time

    start_time = time.time()

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    dependency_graph = config["dependency_graph"]
    jump_folder = config["jump_folder"]
    data_final_dir = config["data_final_dir"]
    normal_data_dir = config["normal_data_dir"]
    output_dir = config["output_dir"]
    failure_label_path = config["failure_label_path"]
    nodes = config["nodes"]

    folder_names = [f for f in os.listdir(data_final_dir) if os.path.isdir(os.path.join(data_final_dir, f)) and f != "00"]
    folder_names.sort()
    # folder_names = folder_names[:3]
    for folder_name in folder_names:
        if os.path.exists(os.path.join(output_dir, folder_name, "evaluation_results.csv")):
            folder_names.remove(folder_name)
    print(f"\nFound {len(folder_names)} data folders")

    # The first one is the training set
    train_path = normal_data_dir
    print(f"Using {train_path} as training data")

    evaluation_results_path = os.path.join(output_dir, "evaluation_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output path: {output_dir}")

    service_lists = []
    service_labels = []

    print("\nStart processing each folder...")

    # Define function to process a single folder
    def process_folder(args):
        i, folder_name = args
        predict_path = data_final_dir + "/" + folder_name
        label_match = re.search(r"(\d+)_", predict_path)
        label_str = label_match.group(1)

        print(f"\n[{i}/{len(folder_names)-1}] Processing folder: {folder_name}")
        if int(label_str) in jump_folder:
            print(f"Skipping folder {folder_name} (in jump list)")
            return None

        try:
            service_list, service_label = sub_main(train_path, predict_path, failure_label_path, nodes, dependency_graph)
            process_and_write_to_csv(service_list,[],label_str,f"{output_dir}/result.csv")
            print(f"Successfully processed folder {folder_name}")
            return (service_list, service_label)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing folder {folder_name}: {str(e)}")
            return None

    # Prepare parameters for parallel processing
    folder_args = [(i, folder_name) for i, folder_name in enumerate(folder_names, 1)]

    max_workers = max(int(os.cpu_count()/4 - 1), 1)
    # Use process pool for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use as_completed to get completed tasks
        futures = {executor.submit(process_folder, arg): arg for arg in folder_args}
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=30)  # Set overall timeout
                if result:
                    results.append(result)
            except concurrent.futures.TimeoutError:
                print(f"Task timed out")
                # Cancel timed out task
                future.cancel()
                continue
            except Exception as e:
                print(f"Task failed: {str(e)}")
                # Cancel failed task
                future.cancel()
                continue

        # Wait for all unfinished tasks
        for future in futures:
            if not future.done():
                future.cancel()

    # Process results
    for result in results:
        if result:
            service_list, service_label = result
            service_lists.append(service_list)
            service_labels.append(service_label)

    print("\n=== Processing final results ===")
    service_lists = [[item for item in sublist if item != "query"] for sublist in service_lists]
    print(f"Number of root cause lists: {len(service_lists)}")
    print(f"Root cause lists: {service_lists}")
    write_results_to_csv(service_labels, service_lists, filename=evaluation_results_path)
    print(f"Results saved to {evaluation_results_path}")

    end_time = time.time()  # End timing
    run_time = end_time - start_time
    print(f"\nTotal run time: {run_time:.4f} seconds")
    print("=== Program finished ===")
