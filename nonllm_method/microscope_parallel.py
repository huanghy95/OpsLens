import pandas as pd
import os
import numpy as np
import re
import json
import time
import concurrent.futures
import signal
import sys
from sklearn.preprocessing import StandardScaler
import warnings
from nonllm_method.utils.CSVFormat import *
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Add signal handler function
def signal_handler(signum, frame):
    print(f"Received signal {signum}, cleaning up and exiting...")
    sys.exit(0)

# Set signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def apply_3sigma(series, mean=None, std_dev=None):
    if mean is None or std_dev is None:
        mean = series.mean()
        std_dev = series.std()
    if std_dev == 0:
        return series.apply(lambda x: 0), mean, std_dev
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    return series.apply(lambda x: 1 if x < lower_bound or x > upper_bound else 0), mean, std_dev


# Calculate the number of anomalies for a row's node and its neighbors
def calculate_anomalies_for_row(row, df, dependency_graph):
    anomaly_count = 0
    node = row["name"]
    # 1. Check if the node itself is anomalous
    if row["anomaly_detection"] == 1:
        anomaly_count += 1
    # 2. Count the number of anomalies in neighbor nodes
    if node in dependency_graph:
        for neighbor in dependency_graph[node]:
            # Count neighbor node anomalies
            anomaly_count += (df[df["name"] == neighbor]["anomaly_detection"] == 1).sum()
    return anomaly_count



def calculate_neighbour_anomalies(name, dependency_graph, predict_all_dfs, nodes):
    connected_neighbours = dependency_graph.get(name, [])
    reverse_neighbours = [key for key, values in dependency_graph.items() if name in values]
    neighbours = set(connected_neighbours + reverse_neighbours)

    neighbour_anomalies = 0
    for neighbour in neighbours:
        df_name = f"predict_{neighbour}_df"
        if df_name in predict_all_dfs and neighbour in predict_all_dfs[df_name]["name"].values:
            try:
                anomaly = predict_all_dfs[df_name][predict_all_dfs[df_name]["name"] == neighbour]["anomaly_detection"].values[0]
                if anomaly:
                    neighbour_anomalies += 1
            except Exception as e:
                print(f"Error occurred while calculating anomalies for node {name}'s neighbor {neighbour}: {str(e)}")
                continue
    return neighbour_anomalies


# Function for sorting
def sort_nodes_by_anomaly(predict_all_dfs, dependency_graph, nodes):
    # Initialize list to store information for all nodes
    service_lists_sub = []

    # Check data integrity
    for name in nodes:
        df_name = f"predict_{name}_df"
        if df_name not in predict_all_dfs:
            print(f"Error: Missing data for node {name}")
            continue

    df_len = predict_all_dfs[f"predict_{nodes[0]}_df"].shape[0]

    # Iterate through rows of each df
    for row_id in range(df_len):
        all_nodes_info = []
        for name in nodes:
            df_name = f"predict_{name}_df"
            if row_id >= len(predict_all_dfs[df_name]):
                continue

            try:
                anomaly = predict_all_dfs[df_name].iloc[row_id]["anomaly_detection"]
                # Calculate number of anomalies in neighbor nodes
                neighbour_anomalies = calculate_neighbour_anomalies(name, dependency_graph, predict_all_dfs, nodes)
                # Store node information (node name, anomaly status, number of neighbor anomalies)
                all_nodes_info.append((name, anomaly, neighbour_anomalies))
            except Exception as e:
                continue

        if all_nodes_info:
            all_nodes_info_sorted = sorted(all_nodes_info, key=lambda x: (x[1] == False, -x[2]))
            service_lists_sub.append([node_info[0] for node_info in all_nodes_info_sorted])

    return service_lists_sub


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # Check if the actual root causes are in the predicted top k
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

        # Calculate IDCG@k (DCG@k in ideal case)
        idcg = 0
        ideal_relevance = [1 if root in actual_roots else 0 for root in predicted_roots[:k]]
        ideal_relevance.sort(reverse=True)
        for i, rel in enumerate(ideal_relevance):
            idcg += (2**rel - 1) / np.log2(i + 2)  # i+2 for 1-based index

        # Calculate NDCG@k
        ndcg_k.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcg_k)


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


def sub_main(train_path, predict_path, failure_label_path, nodes, dependency_graph):
    print(f"\n=== Start processing folder ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    # Load data
    print("\n[1/4] Loading data...")
    
    try:
        # Load data
        print("\n[1/6] Loading data...",flush=True)
        # Prefer reading data_final.csv, if rows > 100k switch to data_final_sampled.csv
        train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
        predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")
        if len(train_df) > 100000:
            print("Training set has more than 100k rows, switching to data_final_sampled.csv")
            train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
            if len(train_df) > 30000:
                print("Training set has more than 30k rows, switching to data_final_sampled_005.csv")
                train_df = pd.read_csv(train_path + "/data_final_sampled_005.csv")
        if len(predict_df) > 100000:
            print("Prediction set has more than 100k rows, switching to data_final_sampled.csv")
            predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")
            if len(predict_df) > 30000:
                print("Prediction set has more than 30k rows, switching to data_final_sampled_005.csv")
                predict_df = pd.read_csv(predict_path + "/data_final_sampled_005.csv")

        print(f"Training data size: {train_df.shape}")
        print(f"Prediction data size: {predict_df.shape}")
    except:
        print("Failed to load data")
        return None, None, None, None

        
    print(f"Training data size: {train_df.shape}")
    print(f"Prediction data size: {predict_df.shape}")

    # Keep only common columns
    print("\n[2/4] Processing common columns...")
    common_columns = train_df.columns.intersection(predict_df.columns)
    train_df = train_df[common_columns]
    predict_df = predict_df[common_columns]
    print(f"Number of common columns: {len(common_columns)}")

    # Remove extra log columns
    print("\n[3/4] Removing extra log columns...")
    drop_log = ["log@httpcore.connection", "log@urllib3.connectionpool", "log@httpx", "log@httpcore.http11"]
    columns_to_drop = [col for col in train_df.columns if any(col.startswith(prefix) for prefix in drop_log)]
    train_df = train_df.drop(columns=columns_to_drop)
    predict_df = predict_df.drop(columns=columns_to_drop)
    print(f"Number of log columns removed: {len(columns_to_drop)}")
    
    
        # Get all unique name values in the data
    data_names = set(train_df['name'].unique()) | set(predict_df['name'].unique())
    # Take the intersection of nodes and data_names
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    print("\n[4/4] Calculating anomaly detection...")
    train_all_dfs = {}
    predict_all_dfs = {}
    for name in nodes:
        train_all_dfs[f"train_{name}_df"] = train_df[train_df["name"] == name]
        predict_all_dfs[f"predict_{name}_df"] = predict_df[predict_df["name"] == name]

        if len(train_all_dfs[f"train_{name}_df"]) == 0:
            continue
        if len(predict_all_dfs[f"predict_{name}_df"]) == 0:
            continue

        try:
            train_all_dfs[f"train_{name}_df"]["anomaly_detection"], mean_train, std_dev_train = apply_3sigma(train_all_dfs[f"train_{name}_df"]["duration"])
            predict_all_dfs[f"predict_{name}_df"]["anomaly_detection"], mean_train, std_dev_train = apply_3sigma(predict_all_dfs[f"predict_{name}_df"]["duration"], mean_train, std_dev_train)
        except Exception as e:
            train_all_dfs[f"train_{name}_df"]["anomaly_detection"], mean_train, std_dev_train = apply_3sigma(train_all_dfs[f"train_{name}_df"]["elapsedTime"])
            predict_all_dfs[f"predict_{name}_df"]["anomaly_detection"], mean_train, std_dev_train = apply_3sigma(predict_all_dfs[f"predict_{name}_df"]["elapsedTime"], mean_train, std_dev_train)            
            # print(f"Error occurred while processing anomaly detection for node {name}: {str(e)}")
            continue

    service_lists_sub = sort_nodes_by_anomaly(predict_all_dfs, dependency_graph, nodes)

    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]
    service_labels_sub = np.tile(service_label, (len(service_lists_sub), 1))
    
    
        # Only keep the top 3 services
    from collections import Counter
    merged_service = []
    for lst in service_lists_sub:
        merged_service.extend(lst)
    counter_service = Counter(merged_service)
    top3_service = [item for item, _ in counter_service.most_common(3)]
    service_lists_sub = top3_service
    service_labels_sub = service_labels_sub[0]  # Only keep one label    
    # print(f"Detected root cause services: {service_lists_sub}")
    # print(f"Actual root cause services: {service_label}")

    return service_lists_sub, service_labels_sub


if __name__ == "__main__":
    # Start timing
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

    def process_folder(args):
        try:
            i, folder_name = args
            predict_path = data_final_dir + "/" + folder_name
            label_match = re.search(r"(\d+)_", predict_path)
            label_str = label_match.group(1)

            print(f"\n[{i}/{len(folder_names)-1}] Processing folder: {folder_name}")
            if int(label_str) in jump_folder:
                print(f"Skipping folder {folder_name} (in jump list)")
                return None

            service_lists_sub, service_labels_sub = sub_main(train_path, predict_path, failure_label_path, nodes, dependency_graph)
            process_and_write_to_csv(service_lists_sub,[],label_str,f"{output_dir}/result.csv")
            print(f"Successfully processed folder {folder_name}")
            return (service_lists_sub, service_labels_sub)
        except Exception as e:
            print(f"Error occurred while processing folder {folder_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Ensure the process can exit properly
            sys.stdout.flush()
            sys.stderr.flush()

    folder_args = [(i, folder_name) for i, folder_name in enumerate(folder_names, 1)]
    max_workers = max(int(os.cpu_count()/2 - 1), 1)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_folder = {executor.submit(process_folder, arg): arg for arg in folder_args}
            timeout = 300  # 5 minutes timeout

            for future in concurrent.futures.as_completed(future_to_folder, timeout=timeout):
                folder_arg = future_to_folder[future]
                try:
                    result = future.result()
                    if result:
                        service_lists_sub, service_labels_sub = result
                        service_lists.extend(service_lists_sub)
                        service_labels.extend(service_labels_sub)
                except concurrent.futures.TimeoutError:
                    print(f"Timeout while processing folder {folder_arg[1]}")
                except Exception as e:
                    print(f"Error occurred while processing folder {folder_arg[1]}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                finally:
                    future.cancel()
                    # Ensure the process is terminated
                    if future.running():
                        future.cancel()
                        try:
                            future.result(timeout=1)
                        except:
                            pass

    except KeyboardInterrupt:
        print("\nReceived interrupt signal, cleaning up and exiting...")
        executor.shutdown(wait=False)
        sys.exit(0)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        executor.shutdown(wait=False)
        sys.exit(1)

    print("\n=== Processing final results ===")
    service_lists = [[item for item in sublist if item != "query"] for sublist in service_lists]
    print(f"Number of root cause lists: {len(service_lists)}")
    print(f"Root cause lists: {service_lists}")
    write_results_to_csv(service_labels, service_lists, filename=evaluation_results_path)
    print(f"Results saved to {evaluation_results_path}")

    end_time = time.time()
    run_time = end_time - start_time
    print(f"\nTotal run time: {run_time:.4f} seconds")
    print("=== Program completed ===")
