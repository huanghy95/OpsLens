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
from nonllm_method.utils.CSVFormat import *

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def compute_similarity(df, nodes):

    df["start_time"] = df["start_time"].dt.tz_localize(None)
    similarity_matrix = np.zeros((len(nodes), len(nodes)))
    timestamps = pd.Series(df["start_time"].unique()).sort_values().values

    node_data = {}
    for node in nodes:
        node_df = df[df["name"] == node]
        if len(node_df) > 0:
            try:
                node_df = node_df.groupby("start_time")["duration"].mean().reset_index()
                node_data[node] = node_df.set_index("start_time")["duration"].reindex(timestamps)
            except Exception as e:
                node_df = node_df.groupby("start_time")["elapsedTime"].mean().reset_index()
                node_data[node] = node_df.set_index("start_time")["elapsedTime"].reindex(timestamps)               
               
                continue
        else:
            print(f"Warning: Node {node} has no data")
            continue

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i <= j:
                if node_i not in node_data or node_j not in node_data:
                    similarity_matrix[i, j] = 0
                    similarity_matrix[j, i] = 0
                    continue

                data_i = node_data[node_i].values
                data_j = node_data[node_j].values

                # Remove rows containing NaN
                mask = ~(np.isnan(data_i) | np.isnan(data_j))
                data_i = data_i[mask]
                data_j = data_j[mask]

                if len(data_i) > 1:
                    try:
                        sim = np.corrcoef(data_i, data_j)[0, 1]
                        if np.isnan(sim):  # Handle case of all zeros or identical values
                            sim = 0
                            print("Warning: Calculation result is NaN, setting to 0")
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                    except Exception as e:
                        similarity_matrix[i, j] = 0
                        similarity_matrix[j, i] = 0
                else:
                    similarity_matrix[i, j] = 0
                    similarity_matrix[j, i] = 0

    return similarity_matrix


def random_walk(G, nodes, similarity_matrix, alpha=0.15, iterations=100):
    """
    Execute random walk algorithm to calculate root cause scores for each node.
    """
    n = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    u = np.zeros(n)
    u = similarity_matrix[0] / np.sum(similarity_matrix[0]) if np.sum(similarity_matrix[0]) > 0 else np.ones(n) / n

    P = np.zeros((n, n))
    for i, node_from in enumerate(nodes):
        for j, node_to in enumerate(nodes):
            if G.has_edge(node_from, node_to):
                P[i, j] = similarity_matrix[i, j]

    row_sums = P.sum(axis=1)
    P = np.divide(P, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
    P = np.nan_to_num(P)

    pi = np.ones(n) / n

    for iter_num in range(iterations):
        pi_old = pi.copy()
        pi = alpha * np.dot(P, pi) + (1 - alpha) * u
        pi = pi / np.sum(pi)

    return pi


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # Check if predicted top k root causes contain actual root causes
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
    print(f"\n=== Starting to process folder ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    # Load data
    print("\n[1/4] Loading data...")
    load_start_time = time.time()
    train_df = pd.read_csv(train_path + "/data_final.csv")
    predict_df = pd.read_csv(predict_path + "/data_final.csv")
    if len(train_df) > 100000:
        print("Training set has more than 100k rows, switching to data_final_sampled.csv")
        train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
    if len(predict_df) > 100000:
        print("Prediction set has more than 100k rows, switching to data_final_sampled.csv")
        predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")
    load_end_time = time.time()
    print(f"Training data size: {train_df.shape}",flush=True)
    print(f"Prediction data size: {predict_df.shape}",flush=True)
    print(f"Data loading time: {load_end_time - load_start_time:.2f} seconds",flush=True)

    # Convert time column
    print("\n[2/4] Processing time data...")
    time_process_start = time.time()
    try:
        for df in [train_df, predict_df]:
            try:
                df["start_time"] = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S.%f%z")
            except:
                try:
                    df["start_time"] = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S%z")
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
    print(f"Time data processing time: {time_process_end - time_process_start:.2f} seconds",flush=True)

    # Get similarity matrix
    print("\n[3/4] Computing node similarity...")
    similarity_start_time = time.time()
    # Get all unique name values from data
    data_names = set(train_df['name'].unique()) | set(predict_df['name'].unique())
    # Get intersection of nodes and name values from data
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")
    similarity_matrix = compute_similarity(predict_df, nodes)
    similarity_end_time = time.time()
    print(f"Similarity calculation time: {similarity_end_time - similarity_start_time:.2f} seconds",flush=True)

    # Build dependency graph
    print("\n[4/4] Executing random walk...",flush=True)
    random_walk_start_time = time.time()
    G = nx.DiGraph()
    for node, dependencies in dependency_graph.items():
        for dep in dependencies:
            G.add_edge(node, dep)

    try:
        root_cause_scores = random_walk(G, nodes, similarity_matrix)
        sorted_indices = np.argsort(root_cause_scores)[::-1]
        sorted_nodes = [nodes[i] for i in sorted_indices]
        service_list = sorted_nodes

        print("Detected service root causes:")
        for i, idx in enumerate(sorted_indices):
            print(f"{nodes[idx]}: {root_cause_scores[idx]:.4f} -> Rank {i+1}",flush=True)

    except Exception as e:
        print(f"Warning: Random walk calculation error: {str(e)}",flush=True)
        # If random walk fails, use simple sorting method
        print("Using alternative sorting method...")
        service_list = nodes.copy()
        print("Using original node order as alternative sorting")
    random_walk_end_time = time.time()
    print(f"Random walk time: {random_walk_end_time - random_walk_start_time:.2f} seconds",flush=True)

    # Get labels
    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]
    print(f"Actual service root cause: {service_label}",flush=True)

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

    # Add this line to print results
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

    print("\nStarting to process each folder...")

    # Define function to process individual folders
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
            print(f"Error occurred while processing folder {folder_name}: {str(e)}")
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
                print(f"Task timeout")
                # Cancel timeout tasks
                future.cancel()
                continue
            except Exception as e:
                print(f"Task execution failed: {str(e)}")
                # Cancel failed tasks
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
    print(f"Results saved to {evaluation_results_path}",flush=True)

    end_time = time.time()  # End timing
    run_time = end_time - start_time
    print(f"\nTotal runtime: {run_time:.4f} seconds")
    print("=== Program execution completed ===")
