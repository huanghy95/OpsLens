def get_generated_functions():
    """
    Get all dynamically generated functions in the current module.
    
    Returns:
        list: Returns a list of all dynamically generated function objects.
    """
    import sys
    import inspect

    # Get the current module object
    current_module = sys.modules[__name__]

    # Get all functions whose names start with 'function_'
    generated_funcs = []
    for name, obj in inspect.getmembers(current_module):
        if name.startswith('function_') and inspect.isfunction(obj):
            generated_funcs.append(obj)

    return generated_funcs



def function_1_20250725_234816():
    """
    Generated time: 2025-07-25 23:48:16
    Tool description: List all available KPIs and components in metric_container.csv and metric_app.csv for the time window 2021-03-04 18:00 to 18:30 (UTC+8) (timestamps: 1614852000 to 1614853800)
    """
    import pandas as pd
    import numpy as np
    import os
    import pytz
    from datetime import timedelta
    
    class MetricAnomalyDetector:
        """
        A class to detect metric anomalies based on a defined workflow.
        """
        def __init__(self, start_str, end_str, data_path_template):
            self.tz = pytz.timezone('Asia/Shanghai')
            self.diagnosis_start = self.tz.localize(pd.to_datetime(start_str))
            self.diagnosis_end = self.tz.localize(pd.to_datetime(end_str))
            
            # Extend window by 15 mins before and after for context
            self.extended_start = self.diagnosis_start - timedelta(minutes=15)
            self.extended_end = self.diagnosis_end + timedelta(minutes=15)
            
            self.diagnosis_start_ts = int(self.diagnosis_start.timestamp())
            self.diagnosis_end_ts = int(self.diagnosis_end.timestamp())
            self.extended_start_ts = int(self.extended_start.timestamp())
            self.extended_end_ts = int(self.extended_end.timestamp())
            
            self.data_path = data_path_template.format(date=self.diagnosis_start.strftime('%Y_%m_%d'))
            
            self.kpis_to_check = [
                'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
                'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
                'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
                'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
                'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
                'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
                'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
            ]
            
            # Detection parameters
            self.STABLE_PHASE_LEN = 5
            self.ANOMALY_PHASE_MAX_LEN = 8
            self.RECOVERY_PHASE_LEN = 5
            self.STABILITY_STD_THRESHOLD = 10.0
            self.SLOPE_THRESHOLD = 15.0
            self.NOISE_REDUCTION_W = 0.6
            self.CLUSTER_TIME_GAP_MINS = 3
            self.MAX_CLUSTERS = 3
            self.FINAL_FILTER_X = 0.2
    
        def load_data(self):
            """Loads and filters data for the extended time window."""
            if not os.path.exists(self.data_path):
                return None
            df = pd.read_csv(self.data_path)
            df_filtered = df[(df['timestamp'] >= self.extended_start_ts) & (df['timestamp'] <= self.extended_end_ts)].copy()
            df_filtered['datetime'] = pd.to_datetime(df_filtered['timestamp'], unit='s', utc=True).dt.tz_convert(self.tz)
            return df_filtered
    
        def normalize_metric(self, df_metric):
            """Normalizes metric values to a 0-100 scale."""
            min_val = df_metric['value'].min()
            max_val = df_metric['value'].max()
            if max_val == min_val:
                df_metric['norm_value'] = 0.0
            else:
                df_metric['norm_value'] = 100 * (df_metric['value'] - min_val) / (max_val - min_val)
            return df_metric
    
        def find_anomalies_in_series(self, series, is_spike):
            """
            Finds anomaly events (stable-anomaly-recovery pattern) in a single time series.
            """
            events = []
            search_series = series[(series.index >= self.diagnosis_start) & (series.index <= self.diagnosis_end)]
    
            for i in range(len(search_series)):
                for anomaly_len in range(1, self.ANOMALY_PHASE_MAX_LEN + 1):
                    anomaly_start_idx_abs = series.index.get_loc(search_series.index[i])
                    
                    if anomaly_start_idx_abs < self.STABLE_PHASE_LEN or \
                       anomaly_start_idx_abs + anomaly_len + self.RECOVERY_PHASE_LEN > len(series):
                        continue
                    
                    anomaly_end_time = series.index[anomaly_start_idx_abs + anomaly_len - 1]
                    if not (self.diagnosis_start <= anomaly_end_time <= self.diagnosis_end):
                        continue
    
                    stable_slice = series.iloc[anomaly_start_idx_abs - self.STABLE_PHASE_LEN : anomaly_start_idx_abs]
                    anomaly_slice = series.iloc[anomaly_start_idx_abs : anomaly_start_idx_abs + anomaly_len]
                    recovery_slice = series.iloc[anomaly_start_idx_abs + anomaly_len : anomaly_start_idx_abs + anomaly_len + self.RECOVERY_PHASE_LEN]
    
                    if len(stable_slice) < self.STABLE_PHASE_LEN or len(recovery_slice) < self.RECOVERY_PHASE_LEN:
                        continue
    
                    if stable_slice.std() < self.STABILITY_STD_THRESHOLD and recovery_slice.std() < self.STABILITY_STD_THRESHOLD:
                        stable_avg = stable_slice.mean()
                        
                        if is_spike:
                            if anomaly_slice.iloc[0] - stable_slice.iloc[-1] > self.SLOPE_THRESHOLD:
                                delta = anomaly_slice.max() - stable_avg
                                if delta > 0:
                                    events.append({'start_time': anomaly_slice.index[0], 'peak_time': anomaly_slice.idxmax(), 'delta': delta})
                        else: # Drop
                            if stable_slice.iloc[-1] - anomaly_slice.iloc[0] > self.SLOPE_THRESHOLD:
                                delta = stable_avg - anomaly_slice.min()
                                if delta > 0:
                                    events.append({'start_time': anomaly_slice.index[0], 'peak_time': anomaly_slice.idxmin(), 'delta': delta})
            
            if not events: return []
            return pd.DataFrame(events).sort_values('delta', ascending=False).drop_duplicates('start_time').to_dict('records')
    
        def find_events_for_metric(self, df_metric, kpi_name):
            """
            Orchestrates the anomaly detection process for a single metric.
            """
            df_metric = self.normalize_metric(df_metric.copy())
            pivot_df = df_metric.pivot_table(index='datetime', columns='cmdb_id', values='norm_value')
            diag_window_df = pivot_df.loc[self.diagnosis_start:self.diagnosis_end]
            if diag_window_df.empty: return []
                
            sorted_components = diag_window_df.max().sort_values(ascending=False).index
            
            metric_anomalies = []
            for cmdb_id in sorted_components:
                component_series = pivot_df[cmdb_id].dropna()
                if len(component_series) < (self.STABLE_PHASE_LEN + 1 + self.RECOVERY_PHASE_LEN): continue
                
                all_events = self.find_anomalies_in_series(component_series, is_spike=True) + \
                             self.find_anomalies_in_series(component_series, is_spike=False)
                
                for event in sorted(all_events, key=lambda x: x['delta'], reverse=True)[:2]:
                    metric_anomalies.append({'cmdb_id': cmdb_id, 'kpi_name': kpi_name, **event})
    
            if not metric_anomalies: return []
    
            max_delta = max(e['delta'] for e in metric_anomalies)
            denoised_anomalies = [e for e in metric_anomalies if e['delta'] >= self.NOISE_REDUCTION_W * max_delta]
            if not denoised_anomalies: return []
                
            denoised_anomalies.sort(key=lambda x: x['start_time'])
            
            clusters = []
            if denoised_anomalies:
                current_cluster = [denoised_anomalies[0]]
                clusters.append(current_cluster)
                for i in range(1, len(denoised_anomalies)):
                    time_diff = (denoised_anomalies[i]['start_time'] - current_cluster[-1]['start_time']).total_seconds()
                    if time_diff <= self.CLUSTER_TIME_GAP_MINS * 60:
                        current_cluster.append(denoised_anomalies[i])
                    else:
                        current_cluster = [denoised_anomalies[i]]
                        clusters.append(current_cluster)
            
            if len(clusters) > self.MAX_CLUSTERS: return []
    
            denoised_anomalies.sort(key=lambda x: x['delta'], reverse=True)
            top_events = denoised_anomalies[:2]
            
            def get_event_id(event):
                return (event['cmdb_id'], event['kpi_name'], event['start_time'])
    
            top_event_ids = {get_event_id(e) for e in top_events}
            
            final_anomalies = []
            added_event_ids = set()
    
            for cluster in clusters:
                is_root_cause_cluster = False
                for event_in_cluster in cluster:
                    if get_event_id(event_in_cluster) in top_event_ids:
                        is_root_cause_cluster = True
                        break
                
                if is_root_cause_cluster:
                    for event_in_cluster in cluster:
                        event_id = get_event_id(event_in_cluster)
                        if event_id not in added_event_ids:
                            final_anomalies.append(event_in_cluster)
                            added_event_ids.add(event_id)
            
            return final_anomalies
    
        def run(self):
            """
            Main execution function.
            """
            full_df = self.load_data()
            if full_df is None or full_df.empty:
                print("anomaly_events = []")
                pd.DataFrame(columns=["data_source", "timestamp", "cmdb_id", "description"]).to_csv("anomaly_events.csv", index=False)
                return
    
            all_anomalies = []
            for kpi in self.kpis_to_check:
                kpi_df = full_df[full_df['kpi_name'] == kpi]
                if not kpi_df.empty:
                    all_anomalies.extend(self.find_events_for_metric(kpi_df, kpi))
    
            if not all_anomalies:
                print("anomaly_events = []")
                pd.DataFrame(columns=["data_source", "timestamp", "cmdb_id", "description"]).to_csv("anomaly_events.csv", index=False)
                return
    
            overall_max_delta = max(e['delta'] for e in all_anomalies)
            final_anomalies = [e for e in all_anomalies if e['delta'] >= self.FINAL_FILTER_X * overall_max_delta]
            
            output_events = [{
                "data_source": "Metric",
                "timestamp": event['peak_time'].strftime('%Y-%m-%d %H:%M:%S'),
                "cmdb_id": event['cmdb_id'],
                "description": f"Metric '{event['kpi_name']}' is abnormally high/low, delta={event['delta']:.2f}"
            } for event in final_anomalies]
                
            output_events.sort(key=lambda x: float(x['description'].split('=')[-1]), reverse=True)
    
            print("anomaly_events = [")
            for i, event in enumerate(output_events):
                print(f"    {event}" + ("," if i < len(output_events) - 1 else ""))
            print("]")
            
            if output_events:
                pd.DataFrame(output_events).to_csv("anomaly_events.csv", index=False)
            else:
                pd.DataFrame(columns=["data_source", "timestamp", "cmdb_id", "description"]).to_csv("anomaly_events.csv", index=False)
    
    
    def main():
        start_time_str = "2021-03-04 18:00:00"
        end_time_str = "2021-03-04 18:30:00"
        data_path_template = "./dataset/Bank/telemetry/{date}/metric/metric_container.csv"
        
        detector = MetricAnomalyDetector(start_time_str, end_time_str, data_path_template)
        detector.run()
    
    if __name__ == "__main__":
        main()
    


def function_1_20250725_235114():
    """
    Generated time: 2025-07-25 23:51:14
    Tool description: Analyze all relevant KPIs (CPU, memory, disk, network) for each component using metric_container.csv and metric_app.csv within the time window 2021-03-04 18:00:00 to 18:30:00 (UTC+8) to detect any abnormal resource usage or performance degradation.
    """
    import pandas as pd
    import numpy as np
    import pytz
    from datetime import datetime, timedelta
    import os
    
    def find_anomalies():
        """
        Main function to orchestrate the anomaly detection process.
        """
        # 1. Configuration
        TARGET_KPIS = [
            'OSLinux-CPU_CPU_CPUCpuUtil', 'OSLinux-CPU_CPU_CPUUserTime',
            'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc', 'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
            'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT', 'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
            'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
            'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity', 'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
            'JVM-Operating System_7779_JVM_JVM_CPULoad', 'JVM-Operating System_7778_JVM_JVM_CPULoad',
            'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed', 'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
        ]
        DROP_KPIS = ['OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem']
        
        # Time window configuration
        tz = pytz.timezone('Asia/Shanghai')
        start_time_str = "2021-03-04 18:00:00"
        end_time_str = "2021-03-04 18:30:00"
        
        target_start_dt = tz.localize(datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S"))
        target_end_dt = tz.localize(datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S"))
        
        extended_start_dt = target_start_dt - timedelta(minutes=15)
        extended_end_dt = target_end_dt + timedelta(minutes=15)
    
        # 2. Data Loading and Preparation
        try:
            df = load_and_prepare_data(extended_start_dt, extended_end_dt, TARGET_KPIS)
            if df.empty:
                print("anomaly_events = []")
                return
        except FileNotFoundError:
            print(f"Error: Data file not found for date {extended_start_dt.strftime('%Y_%m_%d')}.")
            print("anomaly_events = []")
            return
    
        # 3. Normalization
        df_normalized = normalize_metrics(df)
    
        # 4. Anomaly Detection Loop
        all_kpi_events = []
        for kpi_name, group in df_normalized.groupby('kpi_name'):
            pivoted_df = group.pivot_table(index='datetime', columns='cmdb_id', values='normalized_value')
            pivoted_df = pivoted_df.resample('1min').mean().ffill().bfill()
    
            if pivoted_df.empty:
                continue
    
            target_window_df = pivoted_df.loc[target_start_dt:target_end_dt]
            if target_window_df.empty:
                continue
                
            peak_values = target_window_df.max().sort_values(ascending=False)
            sorted_components = peak_values.index
    
            kpi_events = []
            is_drop = kpi_name in DROP_KPIS
            
            for cmdb_id in sorted_components:
                series = pivoted_df[cmdb_id].dropna()
                if series.empty:
                    continue
                
                events = find_events_in_series(series, target_start_dt, target_end_dt, is_drop)
                for event in events:
                    event['kpi_name'] = kpi_name
                kpi_events.extend(events)
            
            # 5. Per-KPI Filtering
            denoised_events = denoise_kpi_events(kpi_events, w=0.6)
            clustered_events = cluster_and_filter_events(denoised_events, k_minutes=3, max_clusters=3)
            all_kpi_events.extend(clustered_events)
    
        # 6. Global Filtering and Final Output
        final_events = final_denoise(all_kpi_events, x=0.2)
        output_results(final_events)
    
    def load_and_prepare_data(start_dt, end_dt, kpis):
        """Loads and filters metric data."""
        date_str = start_dt.strftime('%Y_%m_%d')
        file_path = f'./dataset/Bank/telemetry/{date_str}/metric/metric_container.csv'
        
        df = pd.read_csv(file_path)
        df = df[df['kpi_name'].isin(kpis)]
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
        df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
        
        return df
    
    def normalize_metrics(df):
        """Normalizes metric values to a 0-100 scale."""
        df_out = df.copy()
        df_out['normalized_value'] = 0.0
        for kpi_name, group in df.groupby('kpi_name'):
            min_val, max_val = group['value'].min(), group['value'].max()
            if max_val == min_val:
                df_out.loc[group.index, 'normalized_value'] = 0.0
            else:
                df_out.loc[group.index, 'normalized_value'] = 100 * (group['value'] - min_val) / (max_val - min_val)
        return df_out
    
    def find_events_in_series(series, target_start, target_end, is_drop, STABLE_WINDOW=5, ANOMALY_WINDOW_MAX=8, STABLE_RANGE_THRESH=15, SPIKE_DELTA_THRESH=25):
        """Detects anomaly events in a single time series."""
        events = []
        series_len = len(series)
        
        try:
            target_indices = series.loc[target_start:target_end].index
            if len(target_indices) == 0: return []
            start_idx_loc = series.index.get_loc(target_indices[0])
            end_idx_loc = series.index.get_loc(target_indices[-1])
        except KeyError:
            return []
    
        covered_indices = set()
        for i in range(start_idx_loc, end_idx_loc + 1):
            if i in covered_indices: continue
    
            stable_start_loc = i - STABLE_WINDOW
            if stable_start_loc < 0: continue
            
            stable_series = series.iloc[stable_start_loc:i]
            if stable_series.isnull().any() or len(stable_series) < STABLE_WINDOW: continue
    
            if (stable_series.max() - stable_series.min()) >= STABLE_RANGE_THRESH: continue
            
            avg_stable = stable_series.mean()
            is_significant_change = (series.iloc[i] > avg_stable + SPIKE_DELTA_THRESH) if not is_drop else (series.iloc[i] < avg_stable - SPIKE_DELTA_THRESH)
            
            if not is_significant_change: continue
    
            anomaly_end_loc = i
            for j in range(i + 1, min(i + ANOMALY_WINDOW_MAX, series_len)):
                if series.index[j] > target_end: break
                val_j = series.iloc[j]
                is_still_anomalous = (val_j > avg_stable + SPIKE_DELTA_THRESH) if not is_drop else (val_j < avg_stable - SPIKE_DELTA_THRESH)
                if is_still_anomalous:
                    anomaly_end_loc = j
                else:
                    break
            
            recovery_start_loc = anomaly_end_loc + 1
            recovery_end_loc = recovery_start_loc + STABLE_WINDOW
            if recovery_end_loc > series_len: continue
                
            recovery_series = series.iloc[recovery_start_loc:recovery_end_loc]
            if recovery_series.isnull().any() or len(recovery_series) < STABLE_WINDOW: continue
            if (recovery_series.max() - recovery_series.min()) >= STABLE_RANGE_THRESH: continue
    
            anomaly_series = series.iloc[i : anomaly_end_loc + 1]
            peak_or_valley = anomaly_series.max() if not is_drop else anomaly_series.min()
            delta = abs(peak_or_valley - avg_stable)
            
            events.append({'timestamp': series.index[i], 'cmdb_id': series.name, 'delta': delta})
            for k in range(i, anomaly_end_loc + 1): covered_indices.add(k)
    
        return sorted(events, key=lambda x: x['delta'], reverse=True)[:2]
    
    def denoise_kpi_events(events, w):
        """Filters events for a single KPI based on delta."""
        if not events: return []
        max_delta = max(event['delta'] for event in events)
        return [event for event in events if event['delta'] >= w * max_delta]
    
    def cluster_and_filter_events(events, k_minutes, max_clusters):
        """Clusters events by time and filters if too many clusters."""
        if not events: return []
        
        events.sort(key=lambda x: x['timestamp'])
        clusters = []
        if events:
            current_cluster = [events[0]]
            for i in range(1, len(events)):
                time_diff = (events[i]['timestamp'] - current_cluster[-1]['timestamp']).total_seconds() / 60
                if time_diff <= k_minutes:
                    current_cluster.append(events[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [events[i]]
            clusters.append(current_cluster)
    
        if len(clusters) > max_clusters:
            return []
    
        # Retain events from the top two clusters by max delta
        if not clusters: return []
        clusters.sort(key=lambda c: max(e['delta'] for e in c), reverse=True)
        
        final_events = []
        for cluster in clusters[:2]:
            final_events.extend(cluster)
            
        return final_events
    
    def final_denoise(events, x):
        """Filters all events globally based on delta."""
        if not events: return []
        global_max_delta = max(event['delta'] for event in events)
        return [event for event in events if event['delta'] >= x * global_max_delta]
    
    def output_results(events):
        """Formats and prints the final anomaly events."""
        if not events:
            print("anomaly_events = []")
            # Create empty csv
            pd.DataFrame(columns=['data_source', 'timestamp', 'cmdb_id', 'description']).to_csv('anomaly_events.csv', index=False)
            return
    
        output_list = []
        for event in events:
            description = f"Metric '{event['kpi_name']}' on component '{event['cmdb_id']}' is abnormal, delta={event['delta']:.2f}"
            output_list.append({
                "data_source": "Metric",
                "timestamp": event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "cmdb_id": event['cmdb_id'],
                "description": description
            })
        
        # Sort by timestamp for final presentation
        output_list.sort(key=lambda x: x['timestamp'])
        
        print("anomaly_events = [")
        for i, item in enumerate(output_list):
            print(f"    {item}" + ("," if i < len(output_list) - 1 else ""))
        print("]")
    
        # Save to CSV
        df_out = pd.DataFrame(output_list)
        df_out.to_csv('anomaly_events.csv', index=False)
    
    if __name__ == '__main__':
        find_anomalies()
    


def function_1_20250725_235341():
    """
    Generated time: 2025-07-25 23:53:41
    Tool description: Extract and analyze all trace spans within 2021-03-04 18:00:00 to 18:30:00 (UTC+8), focusing on failed or slow transactions, especially those involving Redis02, Tomcat03, and MG01, to correlate trace anomalies with the detected metric anomalies.
    """
    import pandas as pd
    import pytz
    from datetime import datetime, timedelta
    import os
    import csv
    
    def detect_trace_anomalies():
        """
        Detects trace-based anomalies by identifying significant drops in span counts.
        """
        # --- Configuration ---
        DATA_DIR = './dataset/Bank/telemetry/'
        DATE = '2021_03_04'
        START_TIME_STR = f'{DATE.replace("_", "-")} 18:00:00'
        END_TIME_STR = f'{DATE.replace("_", "-")} 18:30:00'
        TIMEZONE = pytz.timezone('Asia/Shanghai')
    
        # Anomaly detection parameters - INCREASED THRESHOLDS to reduce noise
        EXTEND_MINUTES = 5
        COUNT_DROP_AGG_FREQ_INITIAL = "1min"
        COUNT_DROP_WINDOW_MINUTES = 3
        # Increased from 0.50 to 0.75 to focus on severe drops
        COUNT_DROP_THRESHOLD_RATIO = 0.75
        # Increased from 1 to 2 to require multiple components to fail simultaneously
        COUNT_DROP_MIN_COMPONENTS = 2
        TARGET_COMPONENTS = ['Tomcat01', 'Tomcat02', 'Tomcat03', 'Tomcat04', 'IG01', 'IG02', 'MG01', 'MG02']
    
        # --- Main Logic ---
    
        # 1. Time setup
        try:
            start_time = TIMEZONE.localize(datetime.strptime(START_TIME_STR, '%Y-%m-%d %H:%M:%S'))
            end_time = TIMEZONE.localize(datetime.strptime(END_TIME_STR, '%Y-%m-%d %H:%M:%S'))
            extended_start_time = start_time - timedelta(minutes=EXTEND_MINUTES)
        except Exception as e:
            print(f"Error parsing time: {e}")
            return
    
        # 2. Load and preprocess data
        trace_file_path = os.path.join(DATA_DIR, DATE, 'trace', 'trace_span.csv')
        if not os.path.exists(trace_file_path):
            print(f"Trace file not found: {trace_file_path}")
            print("anomaly_events = []")
            return
    
        df_trace = pd.read_csv(trace_file_path)
    
        # Filter for target components
        df_trace = df_trace[df_trace['cmdb_id'].isin(TARGET_COMPONENTS)]
    
        # Convert timestamp and filter by extended time window
        df_trace['datetime'] = pd.to_datetime(df_trace['timestamp'], unit='ms', utc=True).dt.tz_convert(TIMEZONE)
        df_trace = df_trace[(df_trace['datetime'] >= extended_start_time) & (df_trace['datetime'] < end_time)]
    
        if df_trace.empty:
            print("anomaly_events = []")
            with open("anomaly_events.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['data_source', 'timestamp', 'cmdb_id', 'description'])
            return
    
        # 3. Aggregate span counts
        df_trace.set_index('datetime', inplace=True)
        span_counts_1min = df_trace.groupby('cmdb_id').resample(COUNT_DROP_AGG_FREQ_INITIAL).size().unstack(level=0).fillna(0)
    
        # 4. Calculate rolling sums and detect drops
        span_counts_3min = span_counts_1min.rolling(window=COUNT_DROP_WINDOW_MINUTES, min_periods=1).sum()
        previous_span_counts_3min = span_counts_3min.shift(COUNT_DROP_WINDOW_MINUTES)
    
        # Condition: current < previous * (1 - ratio) AND previous > 0
        is_drop = (span_counts_3min < previous_span_counts_3min * (1 - COUNT_DROP_THRESHOLD_RATIO)) & (previous_span_counts_3min > 0)
    
        # 5. Identify anomaly events
        simultaneous_drops = is_drop.sum(axis=1)
        anomaly_timestamps = simultaneous_drops[simultaneous_drops >= COUNT_DROP_MIN_COMPONENTS].index
    
        # 6. Format and report results
        anomaly_events = []
        final_anomaly_timestamps = anomaly_timestamps[(anomaly_timestamps >= start_time) & (anomaly_timestamps < end_time)]
    
        for ts in final_anomaly_timestamps:
            dropping_components = is_drop.loc[ts][is_drop.loc[ts]].index
            
            for comp in dropping_components:
                current_count = span_counts_3min.loc[ts, comp]
                previous_count = previous_span_counts_3min.loc[ts, comp]
                
                if pd.notna(current_count) and pd.notna(previous_count):
                    description = (f"Span count dropped by more than {COUNT_DROP_THRESHOLD_RATIO*100}%. "
                                   f"Current 3-min count: {int(current_count)}, "
                                   f"Previous 3-min count: {int(previous_count)}.")
                    
                    anomaly_event = {
                        "data_source": "Trace",
                        "timestamp": ts.strftime('%Y-%m-%d %H:%M:%S'),
                        "cmdb_id": comp,
                        "description": description
                    }
                    anomaly_events.append(anomaly_event)
    
        # 7. Final Output
        print(f"anomaly_events = {str(anomaly_events)}")
    
        output_df = pd.DataFrame(anomaly_events)
        if output_df.empty:
            output_df = pd.DataFrame(columns=['data_source', 'timestamp', 'cmdb_id', 'description'])
        
        output_df.to_csv("anomaly_events.csv", index=False)
    
    if __name__ == "__main__":
        detect_trace_anomalies()
    


def function_1_20250725_235629():
    """
    Generated time: 2025-07-25 23:56:29
    Tool description: Extract and analyze all log entries for Redis02, Tomcat03, and MG01 within 2021-03-04 18:00:00 to 18:30:00 (UTC+8), focusing on error, warning, or critical events, especially those related to memory or network issues.
    """
    import pandas as pd
    import pytz
    import datetime
    import os
    import numpy as np
    import csv
    import math
    
    def is_stable(counts, threshold):
        """
        Checks if a list of peer counts is stable based on the coefficient of variation.
        """
        counts = np.array(counts, dtype=float)
        if len(counts) < 1:
            return True
        
        mean = np.mean(counts)
        std_dev = np.std(counts)
        
        if mean == 0:
            return True
            
        cv = std_dev / mean
        return cv <= threshold
    
    def find_log_anomalies_refined():
        """
        Detects network anomalies based on significant log count drops, following a structured workflow.
        """
        # --- Core Parameters (stricter thresholds to reduce noise) ---
        WINDOW_MINUTES = 5
        THRESHOLD_RATIO = 0.6
        STABILITY_THRESHOLD = 0.2
        MIN_AVG_COUNT = 500
        HISTORICAL_DECREASE_RATIO = 0.6
    
        # --- Time and Path Definitions ---
        target_date = '2021_03_04'
        log_file_path = f'./dataset/Bank/telemetry/{target_date}/log/log_service.csv'
        start_time_str = '2021-03-04 17:55:00' 
        end_time_str = '2021-03-04 18:30:00'
        
        anomaly_events = []
        tz = pytz.timezone('Asia/Shanghai')
    
        # --- 1. Data Preparation ---
        if not os.path.exists(log_file_path):
            return []
    
        df = pd.read_csv(log_file_path)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(tz)
    
        start_time = tz.localize(datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
        end_time = tz.localize(datetime.datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'))
        
        df = df[(df['datetime'] >= start_time) & (df['datetime'] < end_time)]
    
        df['component_type'] = df['cmdb_id'].str.extract(r'([a-zA-Z]+)')[0]
        target_types = ['Tomcat', 'apache']
        df = df[df['component_type'].isin(target_types)]
    
        if df.empty:
            return []
    
        # --- 2. Create Time-Windowed Count Table ---
        time_bins = pd.date_range(start=start_time, end=end_time, freq=f'{WINDOW_MINUTES}min', tz=tz)
        df['time_window'] = pd.cut(df['datetime'], bins=time_bins, right=False, labels=time_bins[:-1])
        df.dropna(subset=['time_window'], inplace=True)
    
        # Create a DataFrame with time_window as index and cmdb_id as columns
        counts_df = df.groupby(['time_window', 'cmdb_id'], observed=True).size().unstack(fill_value=0)
        
        # --- 3. Iterate Through Windows and Apply Rules ---
        analysis_start_time = tz.localize(datetime.datetime.strptime('2021-03-04 18:00:00', '%Y-%m-%d %H:%M:%S'))
    
        for window_start in counts_df.index:
            if window_start < analysis_start_time:
                continue
    
            previous_window_start = window_start - pd.Timedelta(minutes=WINDOW_MINUTES)
            
            for comp_type in target_types:
                all_components_of_type = [c for c in counts_df.columns if c.startswith(comp_type)]
                
                if len(all_components_of_type) < 2:
                    continue
    
                current_counts_list = [counts_df.loc[window_start, comp] for comp in all_components_of_type if comp in counts_df.columns]
                if not current_counts_list:
                    continue
                
                avg_count = np.mean(current_counts_list)
    
                if avg_count < MIN_AVG_COUNT:
                    continue
    
                for cmdb_id in all_components_of_type:
                    if cmdb_id not in counts_df.columns:
                        continue
                    
                    count = counts_df.loc[window_start, cmdb_id]
    
                    if count > 0 and count < (avg_count * THRESHOLD_RATIO):
                        peer_counts = [counts_df.loc[window_start, p] for p in all_components_of_type if p != cmdb_id and p in counts_df.columns]
                        
                        if is_stable(peer_counts, STABILITY_THRESHOLD):
                            if previous_window_start in counts_df.index:
                                historical_count = counts_df.loc[previous_window_start].get(cmdb_id, 0)
                                
                                if count < (historical_count * HISTORICAL_DECREASE_RATIO):
                                    peer_avg = np.mean(peer_counts) if peer_counts else 0
                                    if count > 1 and peer_avg > 1:
                                        delta = math.log(peer_avg) / math.log(count)
                                    else:
                                        delta = float('inf')
    
                                    description = (f"Log count for '{cmdb_id}' dropped to {count}, "
                                                   f"which is significantly below the peer average of {peer_avg:.0f} "
                                                   f"and its own previous count of {historical_count}. Delta: {delta:.2f}")
    
                                    anomaly_events.append({
                                        'data_source': 'Log',
                                        'timestamp': window_start.strftime('%Y-%m-%d %H:%M:%S'),
                                        'cmdb_id': cmdb_id,
                                        'description': description
                                    })
        return anomaly_events
    
    if __name__ == '__main__':
        anomaly_events = find_log_anomalies_refined()
    
        print("anomaly_events = [")
        if anomaly_events:
            for i, event in enumerate(anomaly_events):
                print(f"    {event}" + ("," if i < len(anomaly_events) - 1 else ""))
        print("]")
    
        if anomaly_events:
            output_file_path = 'anomaly_events.csv'
            try:
                with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['data_source', 'timestamp', 'cmdb_id', 'description']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(anomaly_events)
            except IOError as e:
                pass
    

