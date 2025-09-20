# traceanomaly_parallel.py
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
from utils.CSVFormat import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import glob

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option("mode.chained_assignment", None)

# Check CUDA device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global hyperparameter variables - using improved algorithm parameters
HIDDEN_DIM = 16  # Hidden layer dimension
LATENT_DIM = 8  # Latent space dimension
EPOCHS = 1  # Training epochs
BATCH_SIZE = 128  # Batch size
LEARNING_RATE = 1e-3  # Learning rate
L_SAMPLES = 5  # Importance sampling times
KDE_BANDWIDTH = 0.5  # KDE bandwidth
SIGNIFICANCE_LEVEL = 0.001  # Significance level


class VAE(nn.Module):
    """Improved VAE model - using more stable architecture"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance layer

        # Decoder network
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc41 = nn.Linear(hidden_dim, input_dim)  # Reconstruction mean
        self.fc42 = nn.Linear(hidden_dim, input_dim)  # Reconstruction log variance

    def encode(self, x):
        """Encoder: input -> latent space parameters"""
        h1 = torch.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sampling from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decoder: latent space -> reconstruction distribution parameters"""
        h3 = torch.relu(self.fc3(z))
        mu_x = self.fc41(h3)
        logvar_x = self.fc42(h3)
        return mu_x, logvar_x

    def forward(self, x):
        """Forward propagation: complete VAE computation flow"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mu_x, logvar_x = self.decode(z)
        return mu_x, logvar_x, mu, logvar


def loss_function(mu_x, logvar_x, x, mu_z, logvar_z):
    """Improved VAE loss function: reconstruction loss + KL divergence"""
    # Numerical stability handling: limit log variance range
    logvar_x = torch.clamp(logvar_x, -10, 10)
    logvar_z = torch.clamp(logvar_z, -10, 10)

    # Reconstruction loss: negative log likelihood
    recon_loss = 0.5 * torch.sum((logvar_x + (x - mu_x) ** 2 / torch.exp(logvar_x)))

    # KL divergence loss: regularization term
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    # Total loss: reconstruction + regularization
    total_loss = recon_loss + kl_loss

    # Ensure loss value is finite
    if not torch.isfinite(total_loss):
        return torch.tensor(1000.0, requires_grad=True)

    return total_loss


def train_vae(
    train_data, input_dim, hidden_dim=None, latent_dim=None, epochs=None, batch_size=None, lr=None
):
    """Train VAE model - using improved training process"""
    # Use global parameters as defaults
    if hidden_dim is None:
        hidden_dim = HIDDEN_DIM
    if latent_dim is None:
        latent_dim = LATENT_DIM
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if lr is None:
        lr = LEARNING_RATE

    # Initialize model and optimizer
    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Data preprocessing: convert to tensor and create data loader
    train_tensor = torch.tensor(train_data.values, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        vae.train()
        for (batch_data,) in dataloader:
            optimizer.zero_grad()
            mu_x, logvar_x, mu_z, logvar_z = vae(batch_data)
            loss = loss_function(mu_x, logvar_x, batch_data, mu_z, logvar_z)
            loss.backward()
            optimizer.step()

    return vae


# Cache normalization parameters to improve efficiency
_normalization_cache = {}


def get_normalization_params(train_data, cache_key=None):
    """Get or cache data normalization parameters"""
    if cache_key and cache_key in _normalization_cache:
        return _normalization_cache[cache_key]

    train_min, train_max = train_data.min(), train_data.max()
    if train_max - train_min == 0:
        train_max = train_min + 1

    params = (train_min, train_max)
    if cache_key:
        _normalization_cache[cache_key] = params

    return params


def calculate_log_px_importance_sampling(vae, data, L_samples=None, max_batch_size=1000):
    """Calculate log p(x) using batch importance sampling - improved efficient implementation"""
    if L_samples is None:
        L_samples = L_SAMPLES

    # Process large datasets in batches to avoid memory overflow
    if len(data) > max_batch_size:
        results = []
        for i in range(0, len(data), max_batch_size):
            batch_data = data[i : i + max_batch_size]
            batch_result = calculate_log_px_importance_sampling(
                vae, batch_data, L_samples, max_batch_size
            )
            results.append(batch_result)
        return np.concatenate(results)

    vae.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    batch_size = len(data_tensor)

    with torch.no_grad():
        # Encode to get latent distribution parameters
        mu_z, logvar_z = vae.encode(data_tensor)
        logvar_z = torch.clamp(logvar_z, -10, 10)

        # Batch processing: expand dimensions to support L sampling
        mu_z_expanded = mu_z.unsqueeze(1).expand(-1, L_samples, -1)
        logvar_z_expanded = logvar_z.unsqueeze(1).expand(-1, L_samples, -1)
        data_expanded = data_tensor.unsqueeze(1).expand(-1, L_samples, -1)

        # Batch reparameterization sampling
        std_z = torch.exp(0.5 * logvar_z_expanded)
        eps = torch.randn_like(std_z)
        z_samples = mu_z_expanded + eps * std_z

        # Batch decoding
        z_flat = z_samples.view(-1, z_samples.size(-1))
        mu_x_flat, logvar_x_flat = vae.decode(z_flat)

        mu_x = mu_x_flat.view(batch_size, L_samples, -1)
        logvar_x = torch.clamp(logvar_x_flat.view(batch_size, L_samples, -1), -10, 10)

        # Batch calculation of various probabilities
        log_p_x_given_z = -0.5 * torch.sum(
            (data_expanded - mu_x) ** 2 / torch.exp(logvar_x) + logvar_x, dim=-1
        )
        log_p_z = -0.5 * torch.sum(z_samples**2, dim=-1)
        log_q_z_given_x = -0.5 * torch.sum(
            (z_samples - mu_z_expanded) ** 2 / torch.exp(logvar_z_expanded) + logvar_z_expanded,
            dim=-1,
        )

        # Calculate importance weights and use logsumexp to avoid numerical instability
        log_weights = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_weights = torch.clamp(log_weights, -50, 50)
        log_px_estimates = torch.logsumexp(log_weights, dim=1) - np.log(L_samples)

        return log_px_estimates.cpu().numpy()


def fit_kde_on_normal_traces(log_px_scores, bandwidth=None):
    """Fit KDE distribution on log p(x) scores of normal traces"""
    if bandwidth is None:
        bandwidth = KDE_BANDWIDTH

    log_px_scores = np.array(log_px_scores).reshape(-1, 1)

    # Filter invalid values
    valid_mask = np.isfinite(log_px_scores).all(axis=1)
    valid_scores = log_px_scores[valid_mask]

    if len(valid_scores) == 0:
        valid_scores = np.array([[-10.0]])

    # Numerical range limitation to avoid KDE fitting issues
    valid_scores = np.clip(valid_scores, -100, 100)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(valid_scores)
    return kde


def detect_anomaly_with_pvalue_vectorized(kde, log_px_scores, significance_level=None):
    """Vectorized version of anomaly detection - using KDE and p-values"""
    if significance_level is None:
        significance_level = SIGNIFICANCE_LEVEL

    log_px_array = np.array(log_px_scores).reshape(-1, 1)
    log_densities = kde.score_samples(log_px_array)
    p_values = np.exp(log_densities)
    return p_values < significance_level


def get_data_columns_by_type(df):
    """Automatically classify columns by name into metric, log, trace categories"""
    columns = df.columns.tolist()

    metric_cols = []
    log_cols = []
    trace_cols = []

    for col in columns:
        col_lower = col.lower()
        if any(
            keyword in col_lower
            for keyword in ["metric", "count", "rate", "latency", "cpu", "memory", "disk"]
        ):
            metric_cols.append(col)
        elif any(keyword in col_lower for keyword in ["log", "error", "warn", "info", "debug"]):
            log_cols.append(col)
        elif col in ["duration", "name", "trace_id"] or any(
            keyword in col_lower for keyword in ["trace", "span", "duration"]
        ):
            trace_cols.append(col)
        else:
            metric_cols.append(col)

    return {"metric": metric_cols, "log": log_cols, "trace": trace_cols}


def extract_duration_column(df, column_mapping):
    """Automatically extract duration-related columns"""
    trace_cols = column_mapping["trace"]
    duration_candidates = [col for col in trace_cols if "duration" in col.lower()]

    if duration_candidates:
        return duration_candidates[0]
    elif "duration" in df.columns:
        return "duration"
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]

    raise ValueError("Duration-related column not found")


def anomaly_detection_pipeline(train_df, predict_df, service_name, latent_dim=None):
    """Optimized anomaly detection pipeline - using improved detection algorithm"""
    if latent_dim is None:
        latent_dim = LATENT_DIM

    try:
        # Automatically identify data column types and extract duration column
        column_mapping = get_data_columns_by_type(train_df)
        duration_col = extract_duration_column(train_df, column_mapping)

        train_data = train_df[duration_col].values
        predict_data = predict_df[duration_col].values

        # Vectorized data cleaning: filter invalid values
        valid_train_mask = np.isfinite(train_data) & (train_data > 0)
        valid_predict_mask = np.isfinite(predict_data) & (predict_data > 0)

        train_data = train_data[valid_train_mask]
        predict_data = predict_data[valid_predict_mask]

        if len(train_data) == 0 or len(predict_data) == 0:
            # Return default results
            result_df = predict_df.copy()
            result_df["log_px"] = np.full(len(predict_df), -10.0)
            result_df["is_anomaly"] = np.full(len(predict_df), False)
            return result_df

        # Use cached normalization parameters to improve efficiency
        cache_key = f"{service_name}_{len(train_data)}_{train_data.min()}_{train_data.max()}"
        train_min, train_max = get_normalization_params(train_data, cache_key)

        # Vectorized normalization processing
        train_data_norm = np.clip((train_data - train_min) / (train_max - train_min), 0, 1)
        predict_data_norm = np.clip((predict_data - train_min) / (train_max - train_min), 0, 1)

        # Train VAE model
        train_df_norm = pd.DataFrame(train_data_norm.reshape(-1, 1), columns=["duration_norm"])
        vae = train_vae(train_df_norm, input_dim=1, latent_dim=latent_dim)

        # Calculate log p(x) scores for training and prediction sets
        train_log_px = calculate_log_px_importance_sampling(vae, train_data_norm.reshape(-1, 1))
        kde = fit_kde_on_normal_traces(train_log_px)
        predict_log_px = calculate_log_px_importance_sampling(vae, predict_data_norm.reshape(-1, 1))

        # Ensure result DataFrame has consistent length with input predict_df
        result_df = predict_df.copy()

        # Handle length inconsistency
        if len(predict_log_px) != len(predict_df):
            if len(predict_log_px) < len(predict_df):
                padded_log_px = np.full(len(predict_df), -10.0)
                padded_log_px[: len(predict_log_px)] = predict_log_px
                predict_log_px = padded_log_px
            else:
                predict_log_px = predict_log_px[: len(predict_df)]

        result_df["log_px"] = predict_log_px
        result_df["is_anomaly"] = detect_anomaly_with_pvalue_vectorized(kde, predict_log_px)

        return result_df

    except Exception as e:
        # Return default results in case of exception
        result_df = predict_df.copy()
        result_df["log_px"] = np.full(len(predict_df), -10.0)
        result_df["is_anomaly"] = np.full(len(predict_df), False)
        return result_df


def sub_main(train_path, predict_path, failure_label_path, nodes):
    """Main processing function - maintaining compatibility with old interface"""
    print(f"\n=== Starting folder processing ===")
    print(f"Training data path: {train_path}")
    print(f"Prediction data path: {predict_path}")

    try:
        # Data loading phase
        print("\nLoading data...", flush=True)
        train_df = pd.read_csv(train_path + "/data_final_sampled.csv")
        predict_df = pd.read_csv(predict_path + "/data_final_sampled.csv")

        # Data volume control
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
        # Modified sampling section
        train_df = sample_data(train_df, TRAIN_SIZE, select_outstanding=False)  # Select stable data
        predict_df = sample_data(predict_df, PREDICT_SIZE, select_outstanding=True)  # Select outstanding data



        print(f"Training data size: {train_df.shape}")
        print(f"Prediction data size: {predict_df.shape}")
    except:
        print("Failed to load data")
        return None, None

    # Get node intersection
    data_names = set(train_df["name"].unique()) | set(predict_df["name"].unique())
    nodes = list(set(nodes) & data_names)
    print(f"Number of services in data: {len(data_names)}")
    print(f"Number of services after intersection: {len(nodes)}")
    print(f"Service list after intersection: {nodes}")

    # Data preprocessing phase
    print("\nProcessing common columns...")
    common_columns = train_df.columns.intersection(predict_df.columns)
    train_df = train_df[common_columns]
    predict_df = predict_df[common_columns]
    print(f"Number of common columns: {len(common_columns)}")

    # Anomaly detection calculation phase
    print("\nCalculating anomaly detection...")
    predict_all_dfs = {}

    # Run improved anomaly detection pipeline for each service
    for name in nodes:
        train_service_df = train_df[train_df["name"] == name]
        predict_service_df = predict_df[predict_df["name"] == name]

        if len(train_service_df) == 0 or len(predict_service_df) == 0:
            continue

        predict_all_dfs[f"predict_{name}_df"] = anomaly_detection_pipeline(
            train_service_df, predict_service_df, name
        )

    # Calculate service ranking - based on average log_px scores
    valid_nodes = [name for name in nodes if f"predict_{name}_df" in predict_all_dfs]

    if not valid_nodes:
        service_scores = {}

    service_scores = {}
    for name in valid_nodes:
        df = predict_all_dfs[f"predict_{name}_df"]
        if "log_px" in df.columns and len(df) > 0:
            avg_score = df["log_px"].mean()
            service_scores[name] = avg_score

    if not service_scores:
        # Fallback 1: Sort by anomaly rate (higher is more anomalous) in descending order
        anomaly_scores = {}
        for name in nodes:
            df_tmp = predict_all_dfs.get(f"predict_{name}_df")
            if df_tmp is not None and "is_anomaly" in df_tmp.columns and len(df_tmp) > 0:
                anomaly_scores[name] = df_tmp["is_anomaly"].mean()
        if anomaly_scores:
            service_list = sorted(
                anomaly_scores.keys(), key=lambda x: anomaly_scores[x], reverse=True
            )
        else:
            # Fallback 2: Sort by failure phase sample size in descending order
            size_scores = {n: len(predict_df[predict_df["name"] == n]) for n in nodes}
            service_list = sorted(size_scores.keys(), key=lambda x: size_scores[x], reverse=True)
    else:
        # Normal path: smaller log_px is more anomalous, maintain anomaly degree from high to low
        service_list = sorted(service_scores.keys(), key=lambda x: service_scores[x])

    # Get true labels
    label_match = re.search(r"(\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(failure_label_path, "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]

    return service_list, service_label


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate Hit Ratio@k evaluation metric"""
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        hit = any(root in predicted_roots[:k] for root in actual_roots)
        hr_k.append(hit)
    return np.mean(hr_k)


def ndcg_at_k(actual_root_causes, predicted_top_k, k):
    """Calculate NDCG@k evaluation metric"""
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
    service_labels, service_lists, ks=[1, 3, 5], filename="evaluation_results.csv"
):
    """Save evaluation results to CSV file - maintaining compatibility with old interface"""
    results = []
    for k in ks:
        hr_k = hit_ratio_at_k(service_labels, service_lists, k)
        ndcg_k = ndcg_at_k(service_labels, service_lists, k)
        results.append([k, hr_k, ndcg_k])

    df = pd.DataFrame(results, columns=["k", "HR@k", "NDCG@k"])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

    print("\nEvaluation Results:")
    print(df)

    return df


def run_single_params(params):
    """Run single hyperparameter configuration"""
    hd, ld, ep, bs, lr = params

    # Set global hyperparameter variables
    global HIDDEN_DIM, LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE
    HIDDEN_DIM, LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE = hd, ld, ep, bs, lr

    print(f"Running parameters: hd={hd}, ld={ld}, ep={ep}, bs={bs}, lr={lr}")

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
        param_folder = f"hd{hd}_ld{ld}_ep{ep}_bs{bs}_lr{lr:.0e}"
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
                    train_path, predict_path, failure_label_path, nodes
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

            # Calculate and save evaluation metrics
            results = []
            for k in [1, 3, 5]:
                hr_k = hit_ratio_at_k(service_labels, service_lists, k)
                ndcg_k = ndcg_at_k(service_labels, service_lists, k)
                nonservicehr_k = hit_ratio_at_k(nonservice_labels, nonservice_lists, k)
                nonservicendcg_k = ndcg_at_k(nonservice_labels, nonservice_lists, k)
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

            print(f"Completed parameter combination: hd={hd}, ld={ld}, ep={ep}, bs={bs}, lr={lr}")
            return param_folder

        return None

    except Exception as e:
        print(f"Parameter combination failed: hd={hd}, ld={ld}, ep={ep}, bs={bs}, lr={lr}, error: {e}")
        return None


def write_hyperparameter_summary(all_results, output_dir):
    """Summarize all hyperparameter experiment results"""
    if not all_results:
        print("No hyperparameter experiment results to summarize")
        return

    # Summarize all result files
    files = [f for f in all_results if f is not None]
    if not files:
        print("No successfully completed experiments")
        return

    print(f"\n=== Hyperparameter Tuning Summary Results ===")
    print(f"Number of successfully completed experiments: {len(files)}")

    # Find best result files
    best_hr_files = []
    for filename in files:
        try:
            result_path = os.path.join(output_dir, filename)
            if os.path.exists(result_path):
                df = pd.read_csv(result_path)
                hr1_score = df[df["k"] == 1]["HR@k"].values[0] if len(df[df["k"] == 1]) > 0 else 0
                best_hr_files.append((filename, hr1_score))
        except:
            continue

    if best_hr_files:
        best_hr_files.sort(key=lambda x: x[1], reverse=True)
        best_file, best_score = best_hr_files[0]
        print(f"Best result file: {best_file}")
        print(f"Best HR@1: {best_score:.4f}")

    return len(files)


if __name__ == "__main__":
    """Main function - hyperparameter tuning and parallel processing"""
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

    # ep: training epochs
    # ld: latent space dimension, smaller means stronger compression, easier to detect anomalies
    # Define hyperparameter search space
    param_combinations = [
        (hd, ld, ep, bs, lr)
        for hd in [8]
        for ld in [1, 2, 3]
        for ep in [5, 20, 50]
        for bs in [128]
        for lr in [1e-3]
    ]

    print(f"Starting traceanomaly hyperparameter search, total {len(param_combinations)} parameter sets")

    # Check completed experiments to avoid duplicate runs
    pending = []
    for p in param_combinations:
        hd, ld, ep, bs, lr = p
        param_folder = f"hd{hd}_ld{ld}_ep{ep}_bs{bs}_lr{lr:.0e}"
        result_file = os.path.join(output_dir, param_folder, "evaluation_results.csv")
        if not os.path.exists(result_file):
            pending.append(p)

    print(f"Remaining {len(pending)} parameter sets to run")

    if len(pending) > 0:
        # Execute hyperparameter tuning in parallel
        max_workers = max(int(os.cpu_count() / 10 - 1), 1)
        all_results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_params, p): p for p in pending}
            completed = 0

            for future in as_completed(futures):
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
