# from llmrca.utils import project_root
import os
import json
import pandas as pd
import numpy as np
import re
import warnings
import concurrent.futures
from nonllm_method.utils.CSVFormat import *
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def apply_3sigma(series, mean=None, std_dev=None):
    if mean is None or std_dev is None:
        mean = series.mean()
        std_dev = series.std()
    if std_dev == 0:
        return series.apply(lambda x: 0), mean, std_dev
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    return series.apply(lambda x: 1 if x < lower_bound or x > upper_bound else 0), mean, std_dev


def detect_log_change(series):
    first_value = series.iloc[0]
    return series.apply(lambda x: 1 if x != first_value else 0)


def process_data(df_data, column_stats, log_first_values=None):
    # print("df_data:",df_data)
    try:
        df_data["duration_processed"], column_stats["duration"]["mean"], column_stats["duration"]["std_dev"] = apply_3sigma(df_data["duration"], column_stats["duration"]["mean"], column_stats["duration"]["std_dev"])
    except:
        df_data["duration_processed"], column_stats["elapsedTime"]["mean"], column_stats["elapsedTime"]["std_dev"] = apply_3sigma(df_data["elapsedTime"], column_stats["elapsedTime"]["mean"], column_stats["elapsedTime"]["std_dev"])
        
    
    log_columns = [col for col in df_data.columns if col.startswith("log@")]
    if log_first_values is None:
        log_first_values = {}
    for col in log_columns:
        if col not in log_first_values:
            log_first_values[col] = df_data[col].iloc[0]
        df_data[f"{col}_processed"] = df_data[col].apply(lambda x: 1 if x != log_first_values[col] else 0)
    metric_columns = [col for col in df_data.columns if col.startswith("metric@")]
    for col in metric_columns:
        mean, std_dev = column_stats.get(col, {"mean": None, "std_dev": None}).values()
        df_data[f"{col}_processed"], column_stats[col]["mean"], column_stats[col]["std_dev"] = apply_3sigma(df_data[col], mean, std_dev)
    return df_data, column_stats, log_first_values


def calculate_processed_frequency(df):
    processed_columns = [col for col in df.columns if col.endswith("_processed")]
    frequency = {}
    for col in processed_columns:
        ones_count = df[col].sum()
        total_count = len(df[col])
        frequency[col] = ones_count / total_count
    return frequency


def expected_pattern_ranker(train_ones_frequency, predict_ones_frequency):
    scores_e = {}
    for col in train_ones_frequency:
        s_c = train_ones_frequency[col]
        s_p = predict_ones_frequency.get(col, 0)
        if s_c + s_p != 0:
            scores_e[col] = s_c / (s_c + s_p)
        else:
            scores_e[col] = 0
    return scores_e


def actual_pattern_ranker(train_ones_frequency, predict_ones_frequency):
    scores_a = {}
    for col in predict_ones_frequency:
        s_p = predict_ones_frequency[col]
        s_c = train_ones_frequency.get(col, 0)
        if s_c + s_p != 0:
            scores_a[col] = s_p / (s_p + s_c)
        else:
            scores_a[col] = 0
    return scores_a


def aggregate_redundant_patterns(expected_scores, actual_scores, threshold=0.7):
    aggregated_patterns = {}
    for col in expected_scores:
        score_e = expected_scores.get(col, 0)
        score_a = actual_scores.get(col, 0)
        if score_e > threshold or score_a > threshold:
            aggregated_patterns[col] = max(score_e, score_a)
    if "duration_processed" in aggregated_patterns:
        duration_score = aggregated_patterns.pop("duration_processed")
        sorted_patterns = sorted(aggregated_patterns.items(), key=lambda item: item[1], reverse=True)
        sorted_patterns.append(("duration_processed", duration_score))
    else:
        sorted_patterns = sorted(aggregated_patterns.items(), key=lambda item: item[1], reverse=True)
    return sorted_patterns


def calculate_node_fault_mode(train_df, predict_df, node_name, column_stats=None):
    # print("*****enter calculate_node_fault_mode*****")
    train_node_df = train_df[train_df["name"] == node_name]
    # print("train_node_df:",train_node_df)
    predict_node_df = predict_df[predict_df["name"] == node_name]
    if column_stats is None:
        try:
            column_stats = {"duration": {"mean": train_node_df["duration"].mean(), "std_dev": train_node_df["duration"].std()}}
        except:
            column_stats = {"elapsedTime": {"mean": train_node_df["elapsedTime"].mean(), "std_dev": train_node_df["elapsedTime"].std()}}
           
    metric_columns = [col for col in train_node_df.columns if col.startswith("metric@")]
    for col in metric_columns:
        column_stats[col] = {"mean": train_node_df[col].mean(), "std_dev": train_node_df[col].std()}
    processed_train_df, column_stats, log_first_values = process_data(train_node_df, column_stats)
    processed_predict_df, _, _ = process_data(predict_node_df, column_stats, log_first_values)
    train_ones_frequency = calculate_processed_frequency(processed_train_df)
    predict_ones_frequency = calculate_processed_frequency(processed_predict_df)
    expected_scores = expected_pattern_ranker(train_ones_frequency, predict_ones_frequency)
    actual_scores = actual_pattern_ranker(train_ones_frequency, predict_ones_frequency)
    return aggregate_redundant_patterns(expected_scores, actual_scores, threshold=0.7)


def calculate_all_node_fault_modes(train_df, predict_df, nodes, column_stats=None):
    all_sorted_patterns = []
    # print("*****enter calculate_all_node_fault_modes*****")

    for node in nodes:
        # Get the ranking pattern for each node
        sorted_patterns = calculate_node_fault_mode(train_df, predict_df, node, column_stats)

        # print("sorted_patterns:",sorted_patterns)

        # Extract metrics from sorted_patterns except duration_processed
        aggregated_patterns = {col: score for col, score in sorted_patterns if col != "duration_processed"}

        # Extract duration_processed score
        duration_score = next((score for col, score in sorted_patterns if col == "duration_processed"), 0)

        # Multiply other metric scores by duration_processed score
        aggregated_patterns = {f"{node}>{col}": score * duration_score for col, score in aggregated_patterns.items()}

        # Add the processed metrics to the total sorted_patterns list
        for col, score in aggregated_patterns.items():
            all_sorted_patterns.append((col, score))

    # Finally, sort all metrics by score in descending order
    all_sorted_patterns = sorted(all_sorted_patterns, key=lambda item: item[1], reverse=True)
    # print("**********all_sorted_patterns:",all_sorted_patterns)
    # Extract node names and metric names
    service_list = []
    metrics_list = []
    nonservice_list = []

    for col, score in all_sorted_patterns:
        col_new = re.sub(r"(_processed|_final)$", "", col)
        node_name, metric_name = col_new.split(">", 1)

        # Add node names and metric names to their respective lists, avoiding duplicates
        if node_name not in service_list:
            service_list.append(node_name)
        if metric_name not in metrics_list:
            metrics_list.append(metric_name)
        nonservice_list.append(col_new)
    return service_list, nonservice_list


def reduce_nonservice_list(list2reduce, reduce_json):
    # Create a new list to store deduplicated elements
    reduced_list = []

    # Record services that have already appeared to prevent duplicate additions
    seen_services = set()

    # Iterate through each item in nonservice_list
    for item in list2reduce:
        service, metric = item.split(">")

        # Check if the metric exists in reduce_json
        if metric in reduce_json:
            # Find the equivalent service group for this metric
            for group in reduce_json[metric]:
                # If the current service is in the group, and the first service in the group has not been added
                if service in group:
                    first_service = group[0]
                    # Generate the equivalent item and ensure only the first occurrence of the service is added
                    if f"{first_service}>{metric}" not in seen_services:
                        reduced_list.append(f"{first_service}>{metric}")
                        seen_services.add(f"{first_service}>{metric}")
                    break
        else:
            # If the metric of the service item is not in reduce_json, add it directly
            if item not in seen_services:
                reduced_list.append(item)
                seen_services.add(item)

    return reduced_list


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # Check if the predicted top k root causes contain the actual root cause
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


def sub_main(train_path, predict_path, failure_label_path, nodes):
    print(f"\n=== Start processing folder ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    # Load data
    print("\n[1/4] Loading data...")
    try:
        train_df = pd.read_csv(train_path + "/data_final.csv", on_bad_lines='skip')
        if len(train_df) > 100000:
            print("Training set has more than 100k rows, switching to data_final_sampled.csv")
            train_df = pd.read_csv(train_path + "/data_final_sampled.csv", on_bad_lines='skip')
    except Exception as e:
        print(f"Failed to read training data: {str(e)}")
        return None, None, None, None

    try:
        predict_df = pd.read_csv(predict_path + "/data_final.csv", on_bad_lines='skip')
        if len(predict_df) > 100000:
            print("Prediction set has more than 100k rows, switching to data_final_sampled.csv")
            try:
                predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv", on_bad_lines='skip')
            except FileNotFoundError:
                print(f"Sampled file not found, using original file")
                pass
    except Exception as e:
        print(f"Failed to read prediction data: {str(e)}")
        return None, None, None, None

    if train_df is None or predict_df is None or len(train_df) == 0 or len(predict_df) == 0:
        print("Data read failed or is empty")
        return None, None, None, None

    print(f"Training data shape: {train_df.shape}")
    print(f"Prediction data shape: {predict_df.shape}")

    # Keep common columns
    print("\n[2/4] Processing common columns...")
    common_columns = train_df.columns.intersection(predict_df.columns)
    if len(common_columns) == 0:
        print("No common columns")
        return None, None, None, None
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

    # Calculate fault modes for all nodes
    print("\n[4/4] Calculating fault modes...")
    data_names = set(train_df['name'].unique()) | set(predict_df['name'].unique())
    nodes = list(set(nodes) & data_names)
    if len(nodes) == 0:
        print("No matching nodes")
        return None, None, None, None
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")
    
    try:
        service_list, nonservice_list = calculate_all_node_fault_modes(train_df, predict_df, nodes)
        print(f"Detected service root causes: {service_list}")
        print(f"Detected non-service root causes: {nonservice_list}")
    except Exception as e:
        print(f"Failed to calculate fault modes: {str(e)}")
        return None, None, None, None

    # Get labels
    try:
        label_match = re.search(r"(\d+)_", predict_path)
        label_str = label_match.group(1)
        with open(failure_label_path, "r") as f:
            failure_label = json.load(f)
        service_label = failure_label[label_str]["service"]
        nonservice_label = failure_label[label_str]["nonservice"]
        print(f"Actual service root causes: {service_label}")
        print(f"Actual non-service root causes: {nonservice_label}")
    except Exception as e:
        print(f"Failed to read labels: {str(e)}")
        return None, None, None, None

    return service_list, nonservice_list, service_label, nonservice_label


def write_results_to_csv(service_labels, service_lists, nonservice_labels, nonservice_lists, ks=[1, 3, 5], filename="evaluation_results.csv"):
    results = []
    for k in ks:
        hr_k = hit_ratio_at_k(service_labels, service_lists, k)
        ndcg_k = ndcg_at_k(service_labels, service_lists, k)
        nonservicehr_k = hit_ratio_at_k(nonservice_labels, nonservice_lists, k)
        nonservicendcg_k = ndcg_at_k(nonservice_labels, nonservice_lists, k)
        # Save results
        results.append([k, hr_k, ndcg_k, nonservicehr_k, nonservicendcg_k])

    # Use pandas DataFrame to store results
    df = pd.DataFrame(results, columns=["k", "HR@k", "NDCG@k", "nonserviceHR@k", "nonserviceNDCG@k"])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

    # Add this line to print results
    print("\nEvaluation Results:")
    print(df)


if __name__ == "__main__":
    # Start timing
    import time

    start_time = time.time()

    # dependency_graph = {
    #     "apache01": ["IG01", "IG02"],
    #     "apache02": ["IG01", "IG02"],
    #     "IG01": ["Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04"],
    #     "IG02": ["Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04"],
    #     "Tomcat01": ["Redis01", "Mysql01", "MG01", "MG02"],
    #     "Tomcat02": ["Redis01", "Mysql01", "MG01", "MG02"],
    #     "Tomcat03": ["Redis01", "Mysql01", "MG01", "MG02"],
    #     "Tomcat04": ["Redis01", "Mysql01", "MG01", "MG02"],
    #     "MG01": ["Redis02", "Mysql02"],
    #     "MG02": ["Redis02", "Mysql02"],
    # }
    # jump_folder = [0, 1, 6, 13, 15, 23, 24, 25, 27, 28, 35, 37, 57, 60, 64, 68, 69, 70, 73, 80, 86, 89, 90, 93, 110, 119, 120, 126, 128, 129, 132, 134, 135]

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
    nonservice_lists = []
    service_labels = []
    nonservice_labels = []

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
            result = sub_main(train_path, predict_path, failure_label_path, nodes)
            if result is None:
                print(f"Failed to process folder {folder_name}")
                return None
            service_list, nonservice_list, service_label, nonservice_label = result
            # The original code had process_and_write_to_csv here, but it's not defined.
            # Assuming it's a placeholder for a function that writes results to CSV.
            # For now, we'll just print success and return the result.
            print(f"Successfully processed folder {folder_name}")
            return (service_list, nonservice_list, service_label, nonservice_label)
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
                print(f"Task timed out")
                continue
            except Exception as e:
                print(f"Task execution failed: {str(e)}")
                continue

    # Process results
    for result in results:
        if result:
            service_list, nonservice_list, service_label, nonservice_label = result
            service_lists.append(service_list)
            nonservice_lists.append(nonservice_list)
            service_labels.append(service_label)
            nonservice_labels.append(nonservice_label)

    print("\n=== Processing final results ===")
    service_lists = [[item for item in sublist if item != "query"] for sublist in service_lists]
    print(f"Number of root cause lists: {len(service_lists)}")
    print(f"Root cause lists: {service_lists}")
    write_results_to_csv(service_labels, service_lists, nonservice_labels, nonservice_lists, filename=evaluation_results_path)
    print(f"Results saved to {evaluation_results_path}")

    end_time = time.time()  # End timing
    run_time = end_time - start_time
    print(f"\nTotal run time: {run_time:.4f} seconds")
    print("=== Program completed ===")
