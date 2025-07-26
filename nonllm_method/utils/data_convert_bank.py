import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import json
import pandas as pd
from functools import lru_cache
import warnings
from tqdm import tqdm
import os

class DataConverter:
    def __init__(self, base_path, record_path):
        self.base_path = base_path
        self.record_path = record_path
        self.tz = pytz.timezone('Asia/Shanghai')
        
        self.trace_path = None
        self.metric_container_path = None
        self.metric_app_path = None
        self.log_path = None
        
        self.trace_df = None
        self.metric_container_df = None
        self.metric_app_df = None
        self.log_df = None
        self.log_templates = None
        
        self.time_windows = []
        
        self._metric_container_groups = None
        self._log_groups = None
        
        self._time_window_cache = {}
        
        self.key_kpis = [
            # CPU
            'OSLinux-CPU_CPU_CPUCpuUtil',
            'OSLinux-CPU_CPU_CPUUserTime',
            # Memory
            'OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc',
            'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc',
            'OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem',
            # Network
            'OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT',
            'OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum',
            # Disk
            'OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite',
            'OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity',
            'OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity',
            # JVM
            'JVM-Operating System_7779_JVM_JVM_CPULoad',
            'JVM-Operating System_7778_JVM_JVM_CPULoad',
            'JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed',
            'JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed'
        ]
        
    def load_fault_records(self):
        """Load fault records and generate time windows"""
        print("\n=== Loading fault records ===")
        records_df = pd.read_csv(self.record_path)
        records_df['datetime'] = pd.to_datetime(records_df['datetime'])
        
        # Generate time windows (each fault point's half-hour interval)
        self.time_windows = []  # Reset time window list
        for _, record in records_df.iterrows():
            fault_time = record['datetime']
            # Ensure fault_time has timezone info
            if fault_time.tz is None:
                fault_time = fault_time.tz_localize('Asia/Shanghai')

            # Align time to the start of the current half-hour
            window_start = fault_time - timedelta(
                minutes=fault_time.minute % 30,
                seconds=fault_time.second,
                microseconds=fault_time.microsecond
            )
            window_end = window_start + timedelta(minutes=30)

            # Ensure window_start and window_end have timezone info
            if window_start.tz is None:
                window_start = window_start.tz_localize('Asia/Shanghai')
            if window_end.tz is None:
                window_end = window_end.tz_localize('Asia/Shanghai')

            self.time_windows.append((window_start, window_end))
        print(f"Found {len(self.time_windows)} fault time windows")
        
        # Merge adjacent time windows
        # Use timestamp values for comparison to avoid timezone issues
        self.time_windows.sort(key=lambda x: x[0].timestamp())
        merged_windows = []
        for start, end in self.time_windows:
            if not merged_windows or merged_windows[-1][1].timestamp() < start.timestamp():
                merged_windows.append([start, end])
            else:
                merged_windows[-1][1] = max(merged_windows[-1][1], end, key=lambda x: x.timestamp())
                
        self.time_windows = merged_windows
        
        print(f"After merging, {len(self.time_windows)} fault time windows remain")
        for i, (start, end) in enumerate(self.time_windows):
            print(f"Window {i+1}: {start.strftime('%Y-%m-%d %H:%M:%S')} - {end.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def filter_by_time_windows(self, df, time_col='timestamp'):
        """Efficiently filter data within time windows"""
        if len(self.time_windows) == 0:
            return df
            
        # Create an empty boolean mask
        mask = pd.Series(False, index=df.index)
        
        # Ensure the time column has timezone info
        if df[time_col].dt.tz is None:
            df[time_col] = df[time_col].dt.tz_localize('Asia/Shanghai')
        
        # Update mask for each time window
        for start, end in self.time_windows:
            # Ensure start and end have timezone info
            if start.tz is None:
                start = start.tz_localize('Asia/Shanghai')
            if end.tz is None:
                end = end.tz_localize('Asia/Shanghai')
                
            window_mask = (df[time_col] >= start) & (df[time_col] <= end)
            mask = mask | window_mask
            
        return df[mask]
        
    def load_data(self):
        """Load all data files, only load data within time windows"""
        print("\n=== Start loading data ===")
        
        # First load fault records
        # self.load_fault_records()
        
        print(f"Time window: {self.time_windows[0][0]} - {self.time_windows[0][1]}")
        
        # Load trace data
        print("Loading trace data...")
        self.trace_df = pd.read_csv(self.trace_path)
        self.trace_df['start_time'] = pd.to_datetime(self.trace_df['timestamp'], unit='ms', utc=True)
        self.trace_df['start_time'] = self.trace_df['start_time'].dt.tz_convert('Asia/Shanghai')
        self.trace_df['end_time'] = self.trace_df['start_time'] + pd.to_timedelta(self.trace_df['duration'], unit='ms')
        
        # Filter data within time windows - special handling for trace as it has start and end time
        mask = pd.Series(False, index=self.trace_df.index)
        for start, end in self.time_windows:
            # Ensure start and end have timezone info
            if start.tz is None:
                start = start.tz_localize('Asia/Shanghai')
            if end.tz is None:
                end = end.tz_localize('Asia/Shanghai')
                
            window_mask = ((self.trace_df['start_time'] >= start) & (self.trace_df['start_time'] <= end)) | \
                          ((self.trace_df['end_time'] >= start) & (self.trace_df['end_time'] <= end))
            mask = mask | window_mask
        self.trace_df = self.trace_df[mask]
        print(f"Trace data loaded, {len(self.trace_df)} records")
        
        # Load metric_container data - filter key KPIs first
        print("\nLoading metric_container data...")
        self.metric_container_df = pd.read_csv(self.metric_container_path)
        print(f"Original metric_container data: {len(self.metric_container_df)} records")
        
        # Filter key KPIs
        self.metric_container_df = self.metric_container_df[
            self.metric_container_df['kpi_name'].isin(self.key_kpis)
        ]
        print(f"After filtering key KPIs: {len(self.metric_container_df)} records")
        
        # Convert timestamp and filter by time window
        self.metric_container_df['timestamp'] = pd.to_datetime(self.metric_container_df['timestamp'], unit='s', utc=True)
        self.metric_container_df['timestamp'] = self.metric_container_df['timestamp'].dt.tz_convert('Asia/Shanghai')
        self.metric_container_df = self.filter_by_time_windows(self.metric_container_df)
        print(f"Metric_container data loaded, {len(self.metric_container_df)} records")
        
        # Load metric_app data
        print("\nLoading metric_app data...")
        self.metric_app_df = pd.read_csv(self.metric_app_path)
        self.metric_app_df['timestamp'] = pd.to_datetime(self.metric_app_df['timestamp'], unit='s', utc=True)
        self.metric_app_df['timestamp'] = self.metric_app_df['timestamp'].dt.tz_convert('Asia/Shanghai')
        
        # Filter data within time windows
        self.metric_app_df = self.filter_by_time_windows(self.metric_app_df)
        print(f"Metric_app data loaded, {len(self.metric_app_df)} records")
        
        # Load log data
        print("\nLoading log data...")
        self.log_df = pd.read_csv(self.log_path)
        self.log_df['timestamp'] = pd.to_datetime(self.log_df['timestamp'], unit='s', utc=True)
        self.log_df['timestamp'] = self.log_df['timestamp'].dt.tz_convert('Asia/Shanghai')
        
        # Filter data within time windows
        self.log_df = self.filter_by_time_windows(self.log_df)
        print(f"Log data loaded, {len(self.log_df)} records")
        
        # Preprocessing: group by cmdb_id to speed up subsequent operations
        print("\nPreprocessing data...")
        self._metric_container_groups = dict(list(self.metric_container_df.groupby('cmdb_id')))
        print("Data preprocessing complete")
        
    @lru_cache(maxsize=1000)
    def extract_template(self, log_content):
        """Extract log template, use stricter rules to limit template count, use LRU cache for speedup"""
        # Handle non-string data
        if not isinstance(log_content, str):
            log_content = str(log_content)
            
        # Convert to lowercase
        template = log_content.lower()
        
        # Remove timestamps
        template = re.sub(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{1,2}:\d{1,2}(\.\d+)?', '<TIME>', template)
        
        # Remove all numbers
        template = re.sub(r'\d+(\.\d+)?', '<NUM>', template)
        
        # Remove all hex
        template = re.sub(r'[0-9a-f]{4,}', '<HASH>', template)
        
        # Remove all file paths
        template = re.sub(r'[/\\][\w\-\.]+(?:[/\\][\w\-\.]+)*', '<PATH>', template)
        
        # Remove all IP addresses
        template = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', template)
        
        # Remove all URLs
        template = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', '<URL>', template)
        
        # Remove all emails
        template = re.sub(r'[\w\.-]+@[\w\.-]+', '<EMAIL>', template)
        
        # Remove all content in brackets (keep brackets)
        template = re.sub(r'\([^)]*\)', '()', template)
        template = re.sub(r'\[[^\]]*\]', '[]', template)
        template = re.sub(r'\{[^}]*\}', '{}', template)
        
        # Remove extra spaces and punctuation
        template = re.sub(r'\s+', ' ', template)
        template = re.sub(r'[^\w\s<>]', ' ', template)
        template = re.sub(r'\s+', ' ', template).strip()
        
        return template
        
    def extract_log_templates(self):
        """Extract log templates, select top 15 templates without clustering"""
        print("\n=== Start extracting log templates ===")
        
        print("Extracting log templates...")
        # Extract templates for all logs
        self.log_df['template'] = self.log_df['value'].apply(self.extract_template)
        
        # Count template frequency
        template_counts = self.log_df['template'].value_counts()
        print(f"Initial template extraction complete, {len(template_counts)} unique templates")
        
        # Select top 15 most frequent templates
        top_templates = template_counts.head(15)
        self.log_templates = {i: template for i, template in enumerate(top_templates.index)}
        
        # Assign template ID to each log - use mapping instead of loop
        template_to_id = {template: tid for tid, template in self.log_templates.items()}
        self.log_df['template_id'] = self.log_df['template'].map(template_to_id).fillna(-1).astype(int)
        
        # Recreate log groups, ensure template_id column exists
        self._log_groups = dict(list(self.log_df.groupby('cmdb_id')))
        
        print(f"Selection complete, {len(self.log_templates)} templates")
        print("\nTemplate examples:")
        for template_id, template in self.log_templates.items():
            print(f"Template {template_id}: {template}")
            # Show example logs for this template
            example_logs = self.log_df[self.log_df['template_id'] == template_id]['value'].head(2)
            for j, log in enumerate(example_logs):
                print(f"  Example {j+1}: {log}")
            print()
        
        # Count unmatched logs
        unmatched = (self.log_df['template_id'] == -1).sum()
        print(f"\nNumber of logs not matched to any template: {unmatched} ({unmatched/len(self.log_df)*100:.2f}%)")
        
        # Ensure all required columns exist
        required_columns = ['template', 'template_id']
        missing_columns = [col for col in required_columns if col not in self.log_df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in log data: {missing_columns}")
            return False
            
        return True
        
    def get_metrics_for_time_window(self, start_time, end_time, cmdb_id):
        """Get metric data for a given time window, use cache for optimization"""
        # Generate cache key
        cache_key = (
            start_time.timestamp(),
            end_time.timestamp(),
            str(cmdb_id)
        )
        if cache_key in self._time_window_cache:
            return self._time_window_cache[cache_key]
        
        result = {}
        
        # Handle container metrics
        if cmdb_id in self._metric_container_groups:
            container_metrics = self._metric_container_groups[cmdb_id]
            span_metrics = container_metrics[
                (container_metrics['timestamp'] >= start_time) &
                (container_metrics['timestamp'] <= end_time)
            ]
            
            if not span_metrics.empty:
                for kpi_name, group in span_metrics.groupby('kpi_name'):
                    result[f'metric@{kpi_name}'] = group['value'].mean()
        
        # Handle app metrics
        span_app_metrics = self.metric_app_df[
            (self.metric_app_df['timestamp'] >= start_time) &
            (self.metric_app_df['timestamp'] <= end_time)
        ]
        
        if not span_app_metrics.empty:
            for metric in ['rr', 'sr', 'cnt', 'mrt']:
                if metric in span_app_metrics.columns:
                    result[f'metric@{metric}'] = span_app_metrics[metric].mean()
        
        # Cache result
        self._time_window_cache[cache_key] = result
        return result

    def batch_merge_metrics(self):
        """Batch process metric data, use cache for optimization"""
        print("\nBatch merging metric data...")
        
        # Create result DataFrame
        result_df = pd.DataFrame(index=self.trace_df.index)
        
        # Process each span
        for idx, span in tqdm(self.trace_df.iterrows(), total=len(self.trace_df), desc="Merging metrics"):
            start_time = span['start_time']
            end_time = span['end_time']
            cmdb_id = span['cmdb_id']
            
            # Get metrics for this time window (use cache)
            metrics = self.get_metrics_for_time_window(start_time, end_time, cmdb_id)
            
            # Add metrics to result DataFrame
            for metric_name, value in metrics.items():
                result_df.at[idx, metric_name] = value
        
        # Fill missing values with 0
        result_df = result_df.fillna(0)
        return result_df
    
    def batch_merge_logs(self):
        """Batch process log data, use grouping for optimization"""
        print("\nBatch merging log data...")
        
        # Create result DataFrame
        result_df = pd.DataFrame(index=self.trace_df.index)
        result_df['log_entries'] = None
        
        # Check if log data exists
        if self.log_df is None or len(self.log_df) == 0:
            print("Warning: No log data found, skipping log merge")
            return result_df
            
        # Check if log templates exist
        if self.log_templates is None or len(self.log_templates) == 0:
            print("Warning: No log templates found, skipping log merge")
            return result_df
            
        # Check if template_id column exists, if not, create it
        if 'template_id' not in self.log_df.columns:
            print("Warning: template_id column missing in log data, creating...")
            # Extract templates for all logs
            self.log_df['template'] = self.log_df['value'].apply(self.extract_template)
            
            # Assign template ID to each log
            template_to_id = {template: tid for tid, template in self.log_templates.items()}
            self.log_df['template_id'] = self.log_df['template'].map(template_to_id).fillna(-1).astype(int)
        
        # Create count column for each template
        for template_id, template in self.log_templates.items():
            template_name = f"log@{template}"
            result_df[template_name] = 0
        
        # Iterate over each span
        for idx, span in tqdm(self.trace_df.iterrows(), total=len(self.trace_df), desc="Merging logs"):
            cmdb_id = span['cmdb_id']
            start_time = span['start_time']
            end_time = span['end_time']
            
            # Get logs for this cmdb_id
            if cmdb_id in self._log_groups:
                cmdb_logs = self._log_groups[cmdb_id]
                
                # Get logs within time window
                span_logs = cmdb_logs[(cmdb_logs['timestamp'] >= start_time) & 
                                    (cmdb_logs['timestamp'] <= end_time)]
                
                if not span_logs.empty:
                    # Collect log entries
                    result_df.at[idx, 'log_entries'] = json.dumps(span_logs['value'].tolist())
                    
                    # Count each template
                    template_counts = span_logs['template_id'].value_counts()
                    for template_id, count in template_counts.items():
                        if template_id != -1:  # Skip noise
                            template_name = f"log@{self.log_templates[template_id]}"
                            result_df.at[idx, template_name] = count
        
        # Fill empty log_entries with empty list
        result_df['log_entries'] = result_df['log_entries'].fillna('[]')
        
        return result_df
        
    def process_single_record(self, record, record_idx):
        """Process a single fault record"""
        print(f"\n{'='*50}")
        print(f"Processing fault record {record_idx + 1}")
        print(f"{'='*50}")
        print(f"Fault time: {record['datetime']}")
        
        # Set time window
        fault_time = pd.to_datetime(record['datetime'])
        # Align time to the start of the current half-hour
        window_start = fault_time - timedelta(
            minutes=fault_time.minute % 30,
            seconds=fault_time.second,
            microseconds=fault_time.microsecond
        )
        window_end = window_start + timedelta(minutes=30)
        
        self.time_windows = [(window_start, window_end)]
        print(f"\nProcessing time window: {window_start} to {window_end}")
        
        # Set file paths
        date_str = fault_time.strftime('%Y_%m_%d')
        current_base_path = os.path.join(os.path.dirname(self.base_path), date_str)
        
        self.trace_path = f"{current_base_path}/trace/trace_span.csv"
        self.metric_container_path = f"{current_base_path}/metric/metric_container.csv"
        self.metric_app_path = f"{current_base_path}/metric/metric_app.csv"
        self.log_path = f"{current_base_path}/log/log_service.csv"
        
        # Load data
        print("\n[1/4] Start loading data...")
        self.load_data()
        
        # Extract log templates
        print("\n[2/4] Start extracting log templates...")
        self.extract_log_templates()
        
        print("\n[3/4] Start merging data...")
        # Create result DataFrame
        result_df = self.trace_df.copy()
        total_spans = len(result_df)
        
        if total_spans == 0:
            print(f"Warning: No relevant data found for fault record {record_idx + 1} in time window")
            return None
        
        print(f"Found {total_spans} related span data")
        
        # Batch merge metric data
        print("\n[3/4] Start merging metric data...")
        metric_data = self.batch_merge_metrics()
        for col in metric_data.columns:
            result_df[col] = metric_data[col]
        print(f"Metric data merged, {len(metric_data.columns)} metric columns added")
        
        # Batch merge log data
        print("\n[3/4] Start merging log data...")
        log_data = self.batch_merge_logs()
        for col in log_data.columns:
            result_df[col] = log_data[col]
        print(f"Log data merged, {len(log_data.columns)} log-related columns added")
        
        print("\n[4/4] Processing final data...")
        # Add other necessary columns
        result_df['id'] = range(1, len(result_df) + 1)
        result_df['trace_rowid'] = result_df['id']
        result_df['start_time_unix'] = result_df['start_time'].dt.tz_localize(None).astype(np.int64) // 10**9
        result_df['end_time_unix'] = result_df['end_time'].dt.tz_localize(None).astype(np.int64) // 10**9
        
        # Map cmdb_id to name
        result_df['name'] = result_df['cmdb_id']
        
        # Add events, status_code, and status_message columns
        result_df['events'] = '[]'
        result_df['status_code'] = 0
        result_df['status_message'] = ''
        
        # Select and reorder columns
        columns = ['id', 'trace_rowid', 'span_id', 'parent_id', 'name', 'start_time', 'end_time',
                  'events', 'status_code', 'status_message', 'start_time_unix', 'end_time_unix']
        
        # Add all metric columns
        metric_columns = [col for col in result_df.columns if col.startswith('metric@')]
        columns.extend(metric_columns)
        
        # Add all log template columns
        log_columns = [col for col in result_df.columns if col.startswith('log@')]
        columns.extend(log_columns)
        
        # Add log_entries and duration columns
        columns.extend(['log_entries', 'duration'])
        
        # Reorder
        result_df = result_df[columns]
        
        # Create output directory
        output_dir = "YOURPATH/Bank/datafinal"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate file name: use fault record index and time as file name
        file_name = f"{record_idx}_{fault_time.strftime('%Y%m%d_%H%M%S')}_datafinal.csv"
        output_filename = os.path.join(output_dir, file_name)
        print(f"\nSaving result to {output_filename}...")
        result_df.to_csv(output_filename, index=False)
        print(f"Saved {len(result_df)} records to {output_filename}")
        
        print(f"\n{'='*50}")
        print(f"Fault record {record_idx + 1} processed")
        print(f"{'='*50}")
        print(f"Total spans processed: {total_spans}")
        print(f"Final data columns: {len(columns)}")
        print(f"Metric columns: {len(metric_columns)}")
        print(f"Log template columns: {len(log_columns)}")
        
        return result_df

    def convert(self):
        """Run data conversion"""
        print("\n" + "="*50)
        print("Start data conversion")
        print("="*50)
        
        # Read all fault records
        records_df = pd.read_csv(self.record_path)
        records_df['datetime'] = pd.to_datetime(records_df['datetime'])
        total_records = len(records_df)
        print(f"Read {total_records} fault records")
        
        # Process each fault record
        processed_records = 0
        successful_records = 0
        failed_records = 0
        
        print("\n" + "="*50)
        print("Start processing fault records")
        print("="*50)
        
        for idx, record in records_df.iterrows():
            processed_records += 1
            print(f"\n{'='*50}")
            print(f"Progress: {processed_records}/{total_records} ({processed_records/total_records*100:.1f}%)")
            print(f"{'='*50}")
            
            try:
                result = self.process_single_record(record, idx)
                if result is not None:
                    successful_records += 1
                else:
                    failed_records += 1
            except Exception as e:
                import traceback
                print(f"Error occurred while processing fault record {idx + 1}: {str(e)}")
                print("Stack trace:")
                traceback.print_exc()
                failed_records += 1
            
            # Print current stats
            print(f"\nCurrent stats:")
            print(f"Total records: {total_records}")
            print(f"Processed: {processed_records}")
            print(f"Successfully processed: {successful_records}")
            print(f"Failed: {failed_records}")
            print(f"Remaining: {total_records - processed_records}")
            print(f"Current success rate: {successful_records/processed_records*100:.1f}%")
            
        print("\n" + "="*50)
        print("All fault records processed")
        print("="*50)
        print(f"Total records: {total_records}")
        print(f"Successfully processed: {successful_records}")
        print(f"Failed: {failed_records}")
        print(f"Final success rate: {successful_records/total_records*100:.1f}%")

if __name__ == "__main__":
    # Example usage
    import os

    # Set paths
    record_path = "YOURPATH/Bank/record.csv"
    base_path_root = "YOURPATH/Bank/telemetry/"
    
    # Create converter instance and run conversion
    converter = DataConverter(
        base_path=base_path_root,
        record_path=record_path
    )
    converter.convert()


