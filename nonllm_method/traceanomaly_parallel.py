import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
import json
import warnings
import concurrent.futures
from sklearn.neighbors import KernelDensity
from nonllm_method.utils.CSVFormat import *

pd.set_option("mode.chained_assignment", None)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # log variance
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), torch.log1p(torch.exp(self.fc22(h1)))  # SoftPlus activation

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)  # No output range restriction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction="sum")(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar)
    return recon_loss + kl_loss


# Training process
def train_vae(train_data, input_dim, hidden_dim=16, latent_dim=8, epochs=50, batch_size=128, lr=1e-3):
    vae = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    train_tensor = torch.tensor(train_data.values, dtype=torch.float32).view(-1, input_dim)
    for epoch in range(epochs):
        vae.train()
        permutation = torch.randperm(train_tensor.size(0))
        for i in range(0, train_tensor.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch = train_tensor[indices]
            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
    return vae


def calculate_log_px(vae, data, kde):
    data_tensor = torch.tensor(data, dtype=torch.float32).view(-1, 1)
    vae.eval()
    with torch.no_grad():
        recon_data, mu, logvar = vae(data_tensor)
        recon_data_np = recon_data.numpy()
        if np.isnan(recon_data_np).any():
            recon_data_np = np.nan_to_num(recon_data_np, nan=0)
        log_px = kde.score_samples(recon_data_np)  # KDE distribution evaluation
    return log_px


def fit_kde(data):
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=0)
    kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(data.reshape(-1, 1))
    return kde


def calculate_reconstruction_error(vae, predict_df, duration_col="duration"):
    durations = predict_df[duration_col].values
    predict_data_tensor = torch.tensor(durations, dtype=torch.float32).view(-1, 1)
    vae.eval()
    with torch.no_grad():
        reconstructions, _, _ = vae(predict_data_tensor)
    # Calculate reconstruction error (using squared error)
    reconstruction_errors = (reconstructions.view(-1).numpy() - durations) ** 2  # MSE
    # Add the calculated reconstruction error to the prediction dataframe
    predict_df["reconstruction_error"] = reconstruction_errors
    return predict_df


def anomaly_detection_pipeline(train_df, predict_df, duration_col="duration", latent_dim=8):
    print(f"\nProcessing service: {train_df['name'].iloc[0] if not train_df.empty else 'unknown'}")
    print(f"Training data size: {train_df.shape}")
    print(f"Prediction data size: {predict_df.shape}")
    
    train_data = train_df[duration_col].values
    predict_data = predict_df[duration_col].values
    
    # Check if data is empty
    if train_data.shape[0] == 0:
        print("Warning: Training data is empty")
        return predict_df
    if predict_data.shape[0] == 0:
        print("Warning: Prediction data is empty")
        return predict_df
        
    # Check if data contains invalid values
    if np.isnan(train_data).any() or np.isnan(predict_data).any():
        print("Warning: Data contains NaN values")
        train_data = np.nan_to_num(train_data, nan=0)
        predict_data = np.nan_to_num(predict_data, nan=0)
    
    # Normalize data
    train_min, train_max = train_data.min(), train_data.max()
    if train_max == train_min:
        print("Warning: All training data values are the same")
        return predict_df
        
    train_data_norm = (train_data - train_min) / (train_max - train_min)
    predict_data_norm = (predict_data - train_min) / (train_max - train_min)
    
    # Train VAE
    try:
        vae = train_vae(pd.DataFrame(train_data_norm), input_dim=1, latent_dim=latent_dim)
    except Exception as e:
        print(f"VAE training failed: {str(e)}")
        return predict_df
        
    # Learn KDE distribution
    try:
        predict_tensor = torch.tensor(predict_data_norm, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            recon_data, _, _ = vae(predict_tensor)
            
        # Check if reconstructed data is empty
        if recon_data.shape[0] == 0:
            print("Warning: Reconstructed data is empty")
            return predict_df
            
        kde = fit_kde(recon_data.numpy())
        # Calculate log p(x)
        predict_df["log_px"] = calculate_log_px(vae, predict_data_norm, kde)
    except Exception as e:
        print(f"KDE calculation failed: {str(e)}")
        predict_df["log_px"] = float('inf')  # Use infinity as default value
        
    return predict_df


def min_max_normalize(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    return (df[column] - min_val) / (max_val - min_val)


def sort_names_by_reconstruction_error(predict_all_dfs, nodes):
    # Create an empty list to store the sorted name order for each row
    sorted_names = []
    
    # Use the number of rows in the first DataFrame as a reference
    first_df = predict_all_dfs[f"predict_{nodes[0]}_df"]
    num_rows = len(first_df)
    print(f"Reference DataFrame row count: {num_rows}")
    
    # Operate on each row
    for i in range(num_rows):
        # Create a dictionary to store each name's reconstruction_error
        reconstruction_errors = {}
        # Extract reconstruction_error for each node and save
        for name in nodes:
            df = predict_all_dfs[f"predict_{name}_df"]
            if i < len(df):  # Check if index is in range
                if "log_px" in df.columns:  # Check if column exists
                    reconstruction_errors[name] = df["log_px"].iloc[i]
                else:
                    print(f"Warning: DataFrame {name} does not have log_px column")
                    reconstruction_errors[name] = float('inf')  # Use infinity as default value
            else:
                print(f"Warning: DataFrame {name} row count is less than {i}")
                reconstruction_errors[name] = float('inf')  # Use infinity as default value
        
        # Sort by reconstruction_error in ascending order
        sorted_names_for_row = sorted(reconstruction_errors, key=reconstruction_errors.get, reverse=False)
        sorted_names.append(sorted_names_for_row)
    
    return sorted_names


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


def sub_main(train_path, predict_path, failure_label_path, nodes):
    print(f"\n=== Start processing folder ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    try:
        # Load data
        print("\n[1/6] Loading data...",flush=True)        # Prefer reading data_final.csv, if rows > 100k then switch to data_final_sampled.csv
        train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
        predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")
        if len(train_df) > 100000:
            print("Training set row count > 100k, switching to data_final_sampled.csv")
            train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
        if len(predict_df) > 100000:
            print("Prediction set row count > 100k, switching to data_final_sampled.csv")
            predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")

        print(f"Training data size: {train_df.shape}")
        print(f"Prediction data size: {predict_df.shape}")
    except:
        print("Failed to load data")
        return None, None, None, None

    # Get all unique name values in the data
    data_names = set(train_df['name'].unique()) | set(predict_df['name'].unique())
    # Take the intersection of nodes and data_names
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    print(f"Training data size: {train_df.shape}")
    print(f"Prediction data size: {predict_df.shape}")

    # Keep common columns
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
    print(f"Number of removed log columns: {len(columns_to_drop)}")

    # Anomaly detection calculation
    print("\n[4/4] Anomaly detection calculation...")
    train_all_dfs = {}
    predict_all_dfs = {}
    
    for name in nodes:
        train_all_dfs[f"train_{name}_df"] = train_df[train_df["name"] == name]
        predict_all_dfs[f"predict_{name}_df"] = predict_df[predict_df["name"] == name]
        # Select the column name to use
        duration_col = "duration" if "duration" in train_all_dfs[f"train_{name}_df"].columns else "elapsedTime"
        
        # Normalize training data
        train_all_dfs[f"train_{name}_df"]["duration_normalized"] = min_max_normalize(train_all_dfs[f"train_{name}_df"], duration_col)
            
        # Normalize prediction data using training data parameters
        predict_all_dfs[f"predict_{name}_df"] = anomaly_detection_pipeline(
            train_all_dfs[f"train_{name}_df"], 
            predict_all_dfs[f"predict_{name}_df"], 
            duration_col=duration_col
        )
            
    service_list = sort_names_by_reconstruction_error(predict_all_dfs, nodes)

    # Get labels
    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]
    print(f"Actual service root cause: {service_label}")
    
    # Only keep the top 3 services
    from collections import Counter
    merged_service = []
    for lst in service_list:
        merged_service.extend(lst)
    counter_service = Counter(merged_service)
    top3_service = [item for item, _ in counter_service.most_common(3)]
    service_list = top3_service
    service_label = service_label[0]  # Only keep one label
    

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
            service_list, service_label = sub_main(train_path, predict_path, failure_label_path, nodes)
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

    max_workers = max(int(os.cpu_count()/2 - 1), 1)
    # Use process pool for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use map instead of directly calling executor.map
        future_to_folder = {executor.submit(process_folder, arg): arg for arg in folder_args}
        
        # Set timeout (seconds)
        timeout = 300  # 5 minutes timeout
        
        for future in concurrent.futures.as_completed(future_to_folder, timeout=timeout):
            folder_arg = future_to_folder[future]
            try:
                result = future.result()
                if result:
                    service_list, service_label = result
                    service_lists.extend(service_list)
                    service_labels.extend([service_label] * len(service_list))
            except concurrent.futures.TimeoutError:
                print(f"Timeout processing folder {folder_arg[1]}")
            except Exception as e:
                print(f"Error processing folder {folder_arg[1]}: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                # Ensure future is cancelled
                future.cancel()

    print("\n=== Processing final results ===")
    service_lists = [[item for item in sublist if item != "query"] for sublist in service_lists]
    print(f"Number of root cause lists: {len(service_lists)}")
    print(f"Root cause lists: {service_lists}")
    write_results_to_csv(service_labels, service_lists, filename=evaluation_results_path)
    print(f"Results saved to {evaluation_results_path}")

    end_time = time.time()  # End timing
    run_time = end_time - start_time
    print(f"\nTotal run time: {run_time:.4f} seconds")
    print("=== Program execution completed ===")
