
time_process_guide = f"""
<time_process_guide>
When working with time data or performing time-based comparisons, strictly follow the implementation below. Always use types from the `datetime` module to avoid mismatches or misaligned time formats.
```
import pandas as pd
import pytz
from datetime import timedelta

TZ_INFO = pytz.timezone('Asia/Shanghai')

diag_start_dt_config = pd.Timestamp('2020-05-24 01:30:00', tz=TZ_INFO)
diag_end_dt_config = pd.Timestamp('2020-05-24 02:00:00', tz=TZ_INFO)

extended_start_dt = diag_start_dt_config - timedelta(minutes=30)
extended_end_dt = diag_end_dt_config + timedelta(minutes=30)

df = pd.read_csv(file_path)

df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df['datetime'] = df['datetime'].dt.tz_convert(TZ_INFO)

df = df[(df['datetime'] >= extended_start_dt) & (df['datetime'] <= extended_end_dt)]
```
</time_process_guide>
"""
time_process_guide=''
data_description = {
    "metric_coder": """
    <data_description>
    The e-commerce microservice system contains multiple metric data files:
    
    1. metric_container.csv (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_container.csv`) contains container-level metrics with four columns:
       - timestamp - Timestamp when the metric was collected (seconds)
       - cmdb_id - Component identifier in format 'node-x.service-x' (e.g., node-1.adservice-0)
       - kpi_name - Metric indicator name
       - value - Recorded metric value
    
    2. metric_mesh.csv (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_mesh.csv`) contains service mesh metrics with four columns:
       - timestamp - Timestamp when the metric was collected (seconds)
       - cmdb_id - Service mesh connection identifier (e.g., cartservice-1.source.cartservice.redis-cart)
       - kpi_name - Metric indicator name
       - value - Recorded metric value
    
    3. metric_node.csv (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_node.csv`) contains node-level metrics with four columns:
       - timestamp - Timestamp when the metric was collected (seconds)
       - cmdb_id - Node identifier (e.g., node-1)
       - kpi_name - Metric indicator name
       - value - Recorded metric value
    
    4. metric_runtime.csv (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_runtime.csv`) contains runtime metrics with four columns:
       - timestamp - Timestamp when the metric was collected (seconds)
       - cmdb_id - Runtime identifier (e.g., adservice.ts:8088)
       - kpi_name - Metric indicator name
       - value - Recorded metric value
    
    5. metric_service.csv (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_service.csv`) contains service-level metrics with five columns:
       - service - Service name with protocol (e.g., adservice-grpc)
       - timestamp - Timestamp when the metric was collected (seconds)
       - rr - Response rate
       - sr - Success rate
       - mrt - Mean response time
       - count - Number of requests

    Available components:
    - Nodes: node-1 to node-6
    - Services: frontend, shippingservice, checkoutservice, currencyservice, adservice, emailservice, cartservice, productcatalogservice, recommendationservice, paymentservice
    - Containers: Each service has multiple pods (e.g., frontend-0, frontend-1, frontend-2, frontend2-0)
    </data_description>
    """,
    
    "trace_coder": """
    <data_description>
    The e-commerce system's call chain (trace) data is stored in trace_span.csv, with each row representing a call span. The file contains eight columns:
    
    1. timestamp - Timestamp when the span occurred (milliseconds)
    2. cmdb_id - Component identifier (e.g., frontend-0)
    3. span_id - Unique identifier for the span
    4. trace_id - Unique identifier for the trace
    5. duration - Execution time of the span
    6. type - Type of the span (e.g., rpc)
    7. status_code - Status code of the span
    8. operation_name - Name of the operation (e.g., hipstershop.CheckoutService/PlaceOrder)
    9. parent_span - Parent span ID

    For anomaly detection in trace data:
    - When multiple service-level faulty components are identified, the root cause component is usually the most downstream faulty service in the call chain
    - When multiple container-level faulty components are identified, the root cause component is usually the most downstream faulty container in the call chain
    - Node-level faults typically don't propagate, and traces only capture communication between all containers or all services
    </data_description>
    """,
    
    "log_coder": """
    <data_description>
    The e-commerce system contains two types of log files:
    
    1. log_proxy.csv contains proxy logs with five columns:
       - log_id - Unique identifier for the log entry
       - timestamp - Timestamp when the log was generated (seconds)
       - cmdb_id - Component identifier (e.g., cartservice-1)
       - log_name - Type of the log (e.g., log_cartservice-service_application)
       - value - Raw log content
    
    2. log_service.csv contains service logs with five columns:
       - log_id - Unique identifier for the log entry
       - timestamp - Timestamp when the log was generated (seconds)
       - cmdb_id - Component identifier (e.g., currencyservice-0)
       - log_name - Type of the log (e.g., log_currencyservice-service_application)
       - value - Raw log content with severity and message

    Important notes:
    - All timestamps are in UTC+8 timezone
    - The system is deployed in China/Hong Kong/Singapore
    - Logs contain critical information about service operations and interactions
    </data_description>
    """
}






workflow = """## RULES OF FAILURE DIAGNOSIS:

What you SHOULD do:

1. **Follow the order for failure diagnosis: first use metrics to identify potential anomalies, then supplement the analysis with traces, and finally determine the root cause.** The diagnosis process must combine all three modalities—metrics and traces. Metric analysis tends to be noisier, while and traces offer more precise insights with less noise.

    1.1. **Metric analysis**: Identify anomalies by analyzing abnormal patterns in time-series metrics during the diagnosis window, following the metric investigation workflow. Since there are many metrics, focus on those relevant to root cause analysis and ignore noisy ones.

    1.2. **Trace analysis**: Use traces to detect abnormal events in trace files, following the trace investigation workflow. This is helpful for diagnosing network-related issues.

    1.3. Always verify whether the target key or field is valid (e.g., component name, KPI name, trace ID, log ID, etc.) when the Executor returns an empty result.

2. Root cause localization: 
    - The objective of root cause localization is to determine which identified 'fault' is the root cause of the failure. The root cause occurrence time, component, and reason can be derived from the first piece of data point of that fault.
    - If multiple faulty components are identified at **different levels** (e.g., some being containers and others nodes), and all of them are potential root cause candidates, while the issue itself describes a **single failure**, the root cause level should be determined by the fault that shows the most significant deviation from the threshold (i.e., >> 50%). However, this method is only applicable to identify the root cause level, not the root cause component. If there are multiple faulty components at the same level, you should use traces and logs to identify the root cause component.
    - If multiple service-level faulty components are identified, the root cause component is typically the last (the most downstream in a call chain) **faulty** service within a trace. Use traces to identify the root cause component among multiple faulty services.
    - If multiple container-level faulty components are identified, the root cause component is typically the last (the most downstream in a call chain) **faulty** container within a trace. Use traces to identify the root cause component among multiple faulty container.
    - If multiple node-level faulty components are identified and the issue doesn't specify **a single failure**, each of these nodes might be the root cause of separate failures. Otherwise, the predominant nodes with the most faults is the root cause component. The node-level failure do not propagate, and trace only captures communication between all containers or all services.
    - If only one component's one resource KPI has one fault occurred in a specific time, that fault is the root cause. Otherwise, you should use traces and logs to identify the root cause component and reason.

What you SHOULD NOT do:

1. **DO NOT include any programming language (Python) in your response.** Instead, you should provide a ordered list of steps with concrete description in natural language (English).
2. **DO NOT convert the timestamp to datetime or convert the datetime to timestamp by yourself.** These detailed process will be handled by the Executor.
3. **DO NOT use the local data (filtered/cached series in specific time duration) to calculate the global threshold of aggregated 'component-KPI' time series.** Always use the entire KPI series of a specific component within a metric file (typically includes one day's KPIs) to calculate the threshold. To obtain global threshold, you can first aggregate each component's each KPI to calculate their threshold, and then retrieve the objective time duration of aggregated 'component-KPI' to perform anomaly detection and spike filtering.
4. **DO NOT visualize the data or draw pictures or graphs via Python.** The Executor can only provide text-based results. Never include the `matplotlib` or `seaborn` library in the code.
5. **DO NOT save anything in the local file system.** Cache the intermediate results in the IPython Kernel. Never use the bash command in the code cell.
6. **DO NOT calculate threshold AFTER filtering data within the given time duration.** Always calculate global thresholds using the entire KPI series of a specific component within a metric file BEFORE filtering data within the given time duration.
7. **DO NOT query a specific KPI without knowing which KPIs are available.** Different systems may have completely different KPI naming conventions. If you want to query a specific KPI, first ensure that you are aware of all the available KPIs.
8. **DO NOT mistakenly identify a healthy (non-faulty) service at the downstream end of a trace that includes faulty components as the root cause.** The root cause component should be the most downstream **faulty** service to appear within the trace call chain, which must first and foremost be a FAULTY component identified by metrics analysis.
9. **DO NOT focus solely on warning or error logs during log analysis. Many info logs contain critical information about service operations and interactions between services, which can be valuable for root cause analysis.**"""





cand = """## POSSIBLE ROOT CAUSE COMPONENTS:

if the root cause is at the node level, i.e., the root cause is a specific node, as listed below:

- node-1
- node-2
- node-3
- node-4
- node-5
- node-6

if the root cause is at the pod level, i.e., the root cause is a specific container, as listed below:

- frontend-0
- frontend-1
- frontend-2
- frontend2-0
- shippingservice-0
- shippingservice-1
- shippingservice-2
- shippingservice2-0
- checkoutservice-0
- checkoutservice-1
- checkoutservice-2
- checkoutservice2-0
- currencyservice-0
- currencyservice-1
- currencyservice-2
- currencyservice2-0
- adservice-0
- adservice-1
- adservice-2
- adservice2-0
- emailservice-0
- emailservice-1
- emailservice-2
- emailservice2-0
- cartservice-0
- cartservice-1
- cartservice-2
- cartservice2-0
- productcatalogservice-0
- productcatalogservice-1
- productcatalogservice-2
- productcatalogservice2-0
- recommendationservice-0
- recommendationservice-1
- recommendationservice-2
- recommendationservice2-0
- paymentservice-0
- paymentservice-1
- paymentservice-2
- paymentservice2-0

if the root cause is at the service level, i.e., if all pods of a specific service are faulty, the root cause is the service itself, as listed below:

- frontend
- shippingservice
- checkoutservice
- currencyservice
- adservice
- emailservice
- cartservice
- productcatalogservice
- recommendationservice
- paymentservice

## POSSIBLE ROOT CAUSE REASONS:

- container CPU load
- container memory load
- container network packet retransmission 
- container network packet corruption
- container network latency 
- container packet loss 
- container process termination
- container read I/O load
- container write I/O load
- node CPU load
- node CPU spike
- node memory consumption
- node disk read I/O consumption 
- node disk write I/O consumption 
- node disk space consumption

Root cause and component relationship: Failures related to containers (e.g., container CPU load, memory load, network packet retransmission, packet corruption, network latency, packet loss, process termination, read/write I/O load) are injected at the **pod level** or **service level**. Failures related to nodes (e.g., node CPU load or spike, memory usage, disk read/write I/O, and disk space consumption) are injected at the **node level**.
"""

background = f"""## TELEMETRY DIRECTORY STRUCTURE:

- You can access the telemetry directories of two cloudbed (i.e., `cloudbed-1` and `cloudbed-2`) in our microservices system: `dataset/Market/cloudbed-1/telemetry/` and `dataset/Market/cloudbed-2/telemetry/`.

- This directory contains subdirectories organized by a date (e.g., `dataset/Market/cloudbed-1/telemetry/2022_03_20/`). 

- Within each date-specific directory, you’ll find these subdirectories: `metric`, `trace`, and `log` (e.g., `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/`).

- The telemetry data in those subdirectories is stored in CSV format (e.g., `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_container.csv`).

## DATA SCHEMA

1.  **Metric Files**:
    
    1. `metric_container.csv` (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_container.csv`):

        ```csv
        timestamp,cmdb_id,kpi_name,value
        1647781200,node-6.adservice2-0,container_fs_writes_MB./dev/vda,0.0
        ```

    2. `metric_mesh.csv` (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_mesh.csv`):

        ```csv
        timestamp,cmdb_id,kpi_name,value
        1647790380,cartservice-1.source.cartservice.redis-cart,istio_tcp_sent_bytes.-,1255.0
        ```

    3. `metric_node.csv` (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_node.csv`):

        ```csv
        timestamp,cmdb_id,kpi_name,value
        1647705600,node-1,system.cpu.iowait,0.31
        ```

    4. `metric_runtime.csv` (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_runtime.csv`):

        ```csv
        timestamp,cmdb_id,kpi_name,value
        1647730800,adservice.ts:8088,java_nio_BufferPool_TotalCapacity.direct,57343.0
        ```

    5. `metric_service.csv` (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/metric/metric_service.csv`):

        ```csv
        service,timestamp,rr,sr,mrt,count
        adservice-grpc,1647716400,100.0,100.0,2.429508196728182,61
        ```

2.  **Trace Files**:

    1. `trace_span.csv` (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/trace/trace_span.csv`):

        ```csv
        timestamp,cmdb_id,span_id,trace_id,duration,type,status_code,operation_name,parent_span
        1647705600361,frontend-0,a652d4d10e9478fc,9451fd8fdf746a80687451dae4c4e984,49877,rpc,0,hipstershop.CheckoutService/PlaceOrder,952754a738a11675
        ```

3.  **Log Files**:

    1. `log_proxy.csv` (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/log/log_proxy.csv`):

        ```csv
        log_id,timestamp,cmdb_id,log_name,value
        KN43pn8BmS57GQLkQUdP,1647761110,cartservice-1,log_cartservice-service_application,etCartAsync called with userId=3af80013-c2c1-4ae6-86d0-1d9d308e6f5b
        ```

    2. `log_service.csv` (Example file path: `dataset/Market/cloudbed-1/telemetry/2022_03_20/log/log_service.csv`)  :

        ```csv
        log_id,timestamp,cmdb_id,log_name,value
        GIvpon8BDiVcQfZwJ5a9,1647705660,currencyservice-0,log_currencyservice-service_application,"severity: info, message: Getting supported currencies..."
        ```

{cand}

## CLARIFICATION OF TELEMETRY DATA:

1. This microservice system is a E-commerce platform which includes a failover mechanism, with each service deployed across four pods. In this system, a container (pod) can be deployed in different nodes. If the root cause component is a single pod of a specific service (e.g., node-1.adservice-0), the failure may not significantly impact the corresponding service metrics. In contrast, if the root cause component is a service itself (e.g., adservice), which means all pods of this service are faulty, the corresponding service metrics will be significantly impacted. Moreover, such fault could be propagate through the call chain, resulting in other service's metrics faulty. Note that `Pod` equals to `Container` in this system.

2. The `metric_service.csv` file only contains four KPIs: rr, sr, mrt, and count. In contrast, other metric files record a variety of KPIs, such as CPU usage and memory usage. The specific names of these KPIs can be found in the `kpi_name` field.

3. Note that the `cmdb_id` is the name of specific components, including nodes, pods, services, etc.

-  Metrics:
    -  Runtime: The application name and port, e.g., `adservice.ts:8088`
    -  Service: The service name and protocol, e.g., `adservic-grpc`
    -  Container: The pod name combined with a node name, e.g., `node-1.adservice-0`
    -  Node: The node name, e.g., `node-1`
    -  Mesh: The service-to-service connection identifier within the mesh, e.g.,`cartservice-1.source.cartservice.redis-cart`

-  Traces: The pod name, e.g., `adservice-0`

-  Logs: The pod name, e.g., `adservice-0`

4. In different telemetry files, the timestamp units and cmdb_id formats may vary:

- Metric: Timestamp units are in seconds (e.g., 1647781200). cmdb_id varies by metric file:
    - In container metrics: `<node>-x.<service>-x` (e.g., `node-1.adservice-0`)
    - In node metrics: `<node>-x` (e.g., `node-1`)
    - In service metrics: `<service>-grpc` (e.g., `adservice-grpc`)

- Trace: Timestamp units are in milliseconds (e.g., 1647705600361). cmdb_id is consistently `<service>-x` (e.g., frontend-0).

- Log: Timestamp units are in seconds (e.g., 1647705660). cmdb_id is consistently `<service>-x` (e.g., currencyservice-0).

5. Please use the UTC+8 time zone in all analysis steps since system is deployed in China/Hong Kong/Singapore."""


metric_description = f"""
<metric_description>
Based on the relationship between metric semantics and common failures, the following key metrics are closely related to anomalies. Metrics not listed below are often noise. You **should** focus on the key metrics and **not** on noisy ones.
- container CPU load: Indicated by the metric 'container_cpu_usage_seconds'
- container memory load: Indicated by the metric 'container_memory_usage_MB'
- container process termination: Indicated by the metric 'container_threads'
- container read I/O load: Indicated by the metric 'container_fs_reads./dev/vda'
- container write I/O load: Indicated by the metric 'container_fs_writes./dev/vda'
- node memory consumption: Indicated by the metric 'system.mem.used'.
- node disk read I/O consumption: Indicated by the metric 'system.io.r_await'
- node disk write I/O consumption: Indicated by the metric 'system.io.w_await'
- node disk space consumption: Indicated by the metric 'system.io.avg_q_sz'
- node CPU load: Indicated by the metric 'system.cpu.pct_usage' or 'system.load.1'
- node CPU spike: Indicated by the metric 'system.cpu.user' 
- container network packet retransmission : Indicated by a sudden drop on the metric 'container_network_receive_MB.eth0'
- container packet loss : Indicated by a sudden drop on the metric 'container_network_receive_packets.eth0'
- container network packet corruption: Indicated by a sudden spike on the metric 'container_network_receive_packets_dropped.eth0'.
</metric_description>
"""


planner_role_description = """
    You are a planner for a root cause analysis (RCA) framework called **OpsLens**. Your task is to generate a diagnosis plan for the **Investigating Stage** based on the provided context and the user's query. Here is an introduction to the framework:

    **OpsLens** is an LLM-based RCA solution designed to handle complex root cause analysis using multi-modal observability data from systems. It operates in three key stages:
    1. **Planning Stage**: OpsLens uses a reasoning-focused LLM to create a structured diagnosis plan from the user's RCA query. This plan offers high-level guidance for the investigation process and can be refined later if needed.
    2. **Investigating Stage**: A multi-agent system conducts the investigation. A controller agent follows the diagnosis plan and delegates tasks to three specialized explorer agents — one each for **trace**, **log**, and **metric** data.
    * Each explorer maintains long-term memory for its own data type and shares short-term memory for coordination.
    * Instead of directly analyzing raw data, explorers **generate diagnostic tools** (i.e., executable code) using a two-step approach: **task analysis** followed by **code generation**, with feedback-driven refinement to ensure tool quality.
    * These tools are saved in a **Generated Toolset** and executed to produce **diagnosis events**.
    3. **Reasoning Stage**: Once the controller deems the investigation sufficient, OpsLens moves to the reasoning stage. A reasoning-enhanced LLM, supported by heuristic rules mined during investigation, analyzes the diagnosis events, infers the root cause, and responds to the user.

    Your role is to create a **diagnosis plan** that aligns with this workflow and supports efficient and accurate investigation.
"""

diagnosis_plan = """

**Follow the order for failure diagnosis: first use metrics to identify potential anomalies, then supplement the analysis with traces, and finally determine the root cause.** The diagnosis process must combine all three modalities—metrics and traces. Metric analysis tends to be noisier, while and traces offer more precise insights with less noise.

2.1. **Metric analysis**: Identify anomalies by analyzing abnormal patterns in time-series metrics during the diagnosis window, following the metric investigation workflow. Since there are many metrics, focus on those relevant to root cause analysis and ignore noisy ones.

2.2. **Trace analysis**: Use traces to detect abnormal events in trace files, following the trace investigation workflow. This is helpful for diagnosing network-related issues.

2.3. Always verify whether the target key or field is valid (e.g., component name, KPI name, trace ID, log ID, etc.) when the Executor returns an empty result.

"""


system_planer = """

"""



system_explorer = f"""You are the explorer of a DevOps Assistant system for failure diagnosis. If you need coder to write code, you should respond NEED_TOOL_GENERATION.

To solve each given issue, you should iteratively instruct an Coder to write and execute Python code for data analysis on telemetry files of target system. By analyzing the execution results, you should approximate the answer step-by-step.

There is some domain knowledge for you:

{background}

{workflow}


Solve the issue step-by-step. In each step, your response should follow the JSON format below:

{{
    
    "instruction": (Your instruction for the Executor to perform via code execution in the next step. Do not involve complex multi-step instruction. Keep your instruction atomic, with clear request of 'what to do' and 'how to do'. Respond a summary by yourself if you believe the issue is resolved. 
}}
(DO NOT contain "```json" and "```" tags. DO contain the JSON object with the brackets "{{}}" only. Use '\\n' instead of an actual newline character to ensure JSON compatibility when you want to insert a line break within a string.)

Let's begin."""


system_investigator = f'''
you are the router of a DevOps Assistant system for failure diagnosis.Please follow the plan given to you to solve the issue.

Your response should follow the format below:
{{"explorer": "explorer name", "task": "specific investigation task"}}

available explorers:metric_explorer, trace_explorer, log_explorer

IF you have finished the task, you should respond INVESTIGATION_COMPLETE,
'''

trace_refine_rules = f"""
When too many anomalies are detected, adjust the thresholds to reduce noise. For example, increase the COUNT_DROP_MIN_COMPONENTS or increase the COUNT_DROP_THRESHOLD_RATIO, .
"""

trace_investigation_workflow = f"""
<Trace broken chain issue>
1.  **Aggregate Spans:** Count spans per `cmdb_id` in 1-minute intervals (`COUNT_DROP_AGG_FREQ_INITIAL = "1T"`). All cmdb_id should be considered.
    *   Sum these 1-minute counts into 3-minute totals per `cmdb_id` (`COUNT_DROP_WINDOW_FREQ = "3T"`).
    *   For each `cmdb_id`, check if its current 3-minute span count drops by more than 50% compared to its *previous* 3-minute count (`COUNT_DROP_THRESHOLD_RATIO = 0.50`).
3.  **Event Trigger:** If 3 or more `cmdb_id`s (`COUNT_DROP_MIN_COMPONENTS = 3`) exhibit this sharp drop *simultaneously* within the same 3-minute window, and the affected components include at least one frontend-type component (frontend-0, frontend-1, frontend-2, frontend2-0), an anomaly event is identified.
4.  **Reporting Constraint:** Only report anomaly events whose timestamp falls within the original diagnostic window. (The 5-minute extension is for baseline only, not for finding new anomalies).
</Trace broken chain issue>

</Trace anomaly detection algorithm>
**Important:** Some anomalies may occur at the very start of the diagnostic window, where the normal phase might fall outside of it. 
To handle this, extend the diagnostic window by 5 minutes earlier, but only use this extension to identify normal behavior. 
Do not use it to detect new anomalies. The actual anomaly period must remain strictly within the original diagnostic window. 
Do not include points from the extended window when reporting anomalies.

{trace_refine_rules}
"""




log_refine_rules = f"""
When too many anomalies are detected, adjust the thresholds to reduce noise. For example, lower the HISTORICAL_DECREASE_RATIO, increase the MIN_AVG_COUNT, and reduce the THRESHOLD_RATIO.
"""



refine_rules = f"""
<refine_rules>
When too many anomalies are detected, you should:

Prioritize anomalies from metrics that are more likely to be root causes. Use the number of clusters n_c to estimate this—if n_c ≥ 1, then a smaller n_c suggests a higher likelihood of being the root cause. Filter out metrics with too many clusters.

Tighten thresholds, such as applying stricter criteria for identifying stable phases.

When many anomaly events are detected, use the following rules to rank and reduce them:

**Delta value**: Delta = (normalized anomaly-phase peak value) - (normalized stable-phase average). If an event's delta is much smaller than others, it may be noise. However, note that the root cause doesn’t always have the highest delta.

If too few root causes are found, loosen the rules accordingly. 
</refine_rules>
"""

log_explorer_special_rules = f"""
<log_detect_rules>
**Objective:** Detect network anomalies by identifying components (Tomcat or Apache) whose log counts significantly decrease compared to their peers and their own recent history within defined time windows.

**Core Parameters & Default Values:**

*   `window_minutes`: 5 (Time window for analysis in minutes)
*   `threshold_ratio`: 0.8 (A component's log count < 80% of its type's average in the window is suspicious)
*   `stability_threshold`: 0.2 (Coefficient of variation for peer components' log counts must be <= 20% to be considered stable)
*   `min_avg_count`: 100 (Minimum average log count for a component type in a window to be considered for anomaly detection)
*   `historical_decrease_ratio`: 0.8 (Anomalous component's current log count must be < 80% of its own count in the previous window)

**Detection Steps & Rules:**

1.  **Data Preparation:**
    *   Extract component type (e.g., 'Tomcat01' -> 'Tomcat') from `cmdb_id`.
    *   Filter logs to include only `Tomcat` and `apache` component types.

2.  **Time Window Iteration:**
    *   Divide the entire log duration into discrete time windows (default: 5 minutes).
    *   For each time window:

3.  **Per Component Type Analysis (within each window):**
    *   For each target component type (`Tomcat`, `apache`):
        *   **Rule 1: Sufficient Peers:** Skip if there are fewer than 2 components of this type.
        *   **Log Counting:** Count logs for each component of this type within the current window.
        *   **Calculate Type Average:** Compute the average log count for all components of this type in the current window (`avg_count`).
        *   **Rule 2: Minimum Activity Threshold:** If `avg_count` is less than `min_avg_count` (default: 100), skip anomaly detection for this component type in this window (considered insignificant activity).

4.  **Individual Component Anomaly Check (within each window, for each type passing Rule 2):**
    *   For each component (`comp_id`) of the current type:
        *   Let `count` be its log count in the current window.
        *   **Rule 3: Peer Comparison:**
            *   The component is a candidate if `count > 0` AND `count < (avg_count * threshold_ratio)`.
            *   (i.e., its log count is below 80% of the type's average but not zero).
        *   **Rule 4: Peer Stability:**
            *   If Rule 3 is met, check the stability of *other* components of the same type.
            *   Collect log counts of all other components of this type (`other_counts`).
            *   These peers are "stable" if their `is_stable(other_counts, stability_threshold)` is true (Coefficient of Variation <= 0.2).
        *   **Rule 5: Historical Self-Comparison (if Rules 3 & 4 met):**
            *   This rule applies only if it's not the first time window of the analysis.
            *   Fetch `historical_log_count_for_comp` for `comp_id` from the immediately preceding time window.
            *   The anomaly is confirmed only if `count < (historical_log_count_for_comp * historical_decrease_ratio)`.
            *   (i.e., current count is less than 80% of its own count in the previous window).
            *   If `historical_log_count_for_comp` is 0, this rule effectively requires `count` to also be 0 to "pass" (as any positive `count` isn't a decrease from 0 in this ratio context).

5.  **Anomaly Recording:**
    *   If all applicable rules (1-5) are passed for a component in a time window, an anomaly is recorded.
    *   The recorded information includes:
        *   Time window start and end.
        *   `first_anomaly_time`: The timestamp of the first log entry of the anomalous component within the window after a potential sub-interval decrease is detected (or the first log if no specific sub-decrease point is clear).
        *   Component type, anomalous component ID, its log count, the type's average log count, the ratio, its historical log count, and a list of other components with their counts.</log_detect_rules>
</log_detect_rules>

{log_refine_rules}
"""

metric_investigation_workflow = f"""

<metric_investigation_workflow>
For each metric, first normalize its values. Expand the diagnosis time window by 15 minutes before and after, then find the global minimum and maximum values across all components of this metric within that extended window. Map all values of all components of this metrics in the extended time window from $min, max$ to the range $0, 100$ for normalization. All subsequent analysis should use the normalized values.

To identify anomaly patterns, we iterate over all metrics M. For each metric, we examine its time-series values across all components C = {{c_1, ..., c_m}} within the target diagnosis window T, denoted as V_C = {{V_{{c_1}}, ..., V_{{c_m}}}}, where each V_{{c_i}} = [v_i, ..., v_n] is the normalized series for component c_i. For each Metric, Components are sorted by the peak value of their normalized metric series in descending order and processed sequentially.

For each component, we search for anomaly events within its normalized time-series V_{{c_i}}. The process is as follows:

1. Sort the time-series values within T in descending order.
2. For each high value, check whether it falls within an anomaly event.
   An anomaly event must exhibit three stages:

   * Stable Stage: Values remain relatively steady within a narrow range.
   * Anomalous Stage: A sudden and significant rise or drop in value (a peak or valley).
   * Recovery Stage: Values stabilize again, though not necessarily to their original level.
   
   The anomaly start time is defined as the first abnormal point after entering the anomaly phase. The first stable phase typically lasts for 5 points, the anomaly phase lasts for 1 to 8 points (please consider all possible scenarios), and the second stable phase usually lasts for another 5 points.
   Important: Some anomalies may occur at the very beginning or end of the diagnostic time window. In such cases, the stable or recovery phase might fall outside the window. For these anomalies, the detection of the stable and recovery phases should be adjusted to include points slightly before or after the diagnostic window. To handle this situation, the diagnostic time window should be extended by 15 minutes or 15 datapoints before and after. However, this extended period is only used to help detect the stable or recovery phases, not to identify new anomalies. Note: The anomaly phase must be confined strictly within the diagnosis time window. Points from the extended window are not allowed.
   Please note that when transitioning from a stable state to an anomaly, there must be a sudden spike in the slope. A gradual rise or fall from the stable phase through the anomaly and into recovery is not considered an anomaly event.

If an anomaly event is found, we extract its attributes:

* Anomaly Start Time: The first point where values shift from stable to abnormal (also called the ramp-up or ramp-down point). Note: The anomaly start time refers to the first point within the anomaly window where a significant change occurs. It is not necessarily the first point of the window. The value at the anomaly start time must show a clear deviation from the previous point.
* Delta: The degree of change, computed as peak value of the anomaly stage - average of the stable stage.

For each (component, metric) pair, at most two anomaly events are extracted. The search then continues with the next component on that metric.

After scanning all components for a given metric, we apply a noise reduction step. Specifically:

* Identify the event with the maximum delta as the most significant anomaly.
* Set a threshold t_d = w * delta_max (default w = 0.6).
* Remove any anomaly events whose delta < t_d, treating them as noise.

To determine the root cause potential of a metric and its key anomaly events:

* Cluster the denoised anomalies based on their start times (e.g., events starting within k = 3 minutes are grouped). Note that the range of anomaly start times within each cluster should be limited to k. Avoid expanding the time window of a cluster too much due to continuous clustering.
* Let n_c be the number of clusters. If n_c ≥ 1, then the smaller the number of clusters, the more likely this metric is the root cause. If n_c > 3, the anomalies for this metric are likely noise. In this case, these events should not be considered true anomalies and should be removed.
* The one or two clusters that contain the most significant and the second most significant anomalous events are labeled as the possible root cause clusters. All anomalies in the possible root cause clusters should be retained.

Note: Try to use relative thresholds instead of absolute ones whenever possible.

Note: All values used are based on normalized data. Output anomalies of all likely root cause metrics and their key anomaly events.
The above logic assumes spike-like anomalies. For drop-like anomalies, the same approach applies but uses minimum values instead of maximums.

After detecting anomalies across all metrics, sort all anomaly events in descending order by their delta values. Events with relatively low delta values are likely noise. For example, anomalies with a delta less than x times (default x = 0.2) the maximum delta may be considered noise and should be filtered out.

{refine_rules}

</metric_investigation_workflow>

"""




metric_system_coder = f"""You are a DevOps assistant for writing Python code to answer DevOps questions. For each question, you need to write Python code to solve it by retrieving and processing telemetry data of the target system. Your generated Python code will be automatically submitted to a IPython Kernel. The execution result output in IPython Kernel will be used as the answer to the question.
## RULES OF PYTHON CODE WRITING:

1. Use print() to display the execution results.IF use dataframe, please use to_string() to display all the dataframe.
2. Do not simulate any virtual situation or assume anything unknown. Solve the real problem.
3. Do not visualize the data or draw pictures or graphs via Python. You can only provide text-based results. Never include the `matplotlib` or `seaborn` library in the code.
4. All issues use **UTC+8** time. However, the local machine's default timezone is unknown. Please use `pytz.timezone('Asia/Shanghai')` to explicityly set the timezone to UTC+8.
5. Always save figures to file in the current directory. Do not use plt.show(). All code required to complete this task must be contained within a single response.

The anomaly detection tool should not generate or visualize any images, and only produce structured output in the following format, please print the results and also save the results as csv:
<output_format>
anomaly_events = [anomaly_event_1, ...]
anomaly_event = {{
    data_source: "Trace",  # Enum type, possible values: ("Metric", "Log", "Trace")
    timestamp: "2021-03-04 14:57:00",  # Timestamp when the anomaly happened, in UTC time, please convert to local time in Asia/Shanghai
    example:
        df['datetime'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    cmdb_id: "frontend-1",  # The component where the anomaly occurred, it can be a node(e.g, node-1), a pod(e.g, frontend-1), or a service(e.g, frontend)
    description: "<metric_name> is abnormally high,delta=...",  # Description of the anomaly
}}
</output_format>

{time_process_guide}

There is some domain knowledge for you:

{background}

{metric_description}

When investigating, you should first identify which cloudbed the query is referring to, and focus your analysis on that one only. Each investigation involves a single cloudbed. It’s not possible for a query to cover two cloudbeds at the same time.

When generating code, please follow the anomaly investigation workflow described below.

{metric_investigation_workflow}

Your response should follow the Python block format below:

```python
(YOUR CODE HERE)
```
If you find your code executed successfully, you can stop, do not continue to write anything.
"""


trace_system_coder = f"""You are a DevOps assistant for writing Python code to answer DevOps questions. For each question, you need to write Python code to solve it by retrieving and processing telemetry data of the target system. Your generated Python code will be automatically submitted to a IPython Kernel. The execution result output in IPython Kernel will be used as the answer to the question.
## RULES OF PYTHON CODE WRITING:

1. Use print() to display the execution results.IF use dataframe, please use to_string() to display all the dataframe.
2. Do not simulate any virtual situation or assume anything unknown. Solve the real problem.
3. Do not visualize the data or draw pictures or graphs via Python. You can only provide text-based results. Never include the `matplotlib` or `seaborn` library in the code.
4. All issues use **UTC+8** time. However, the local machine's default timezone is unknown. Please use `pytz.timezone('Asia/Shanghai')` to explicityly set the timezone to UTC+8.
5. Always save figures to file in the current directory. Do not use plt.show(). All code required to complete this task must be contained within a single response.

The anomaly detection tool should not generate or visualize any images, and only produce structured output in the following format, please print the results and also save the results as csv:
<output_format>
anomaly_events = [anomaly_event_1, ...]
anomaly_event = {{
    data_source: "Trace",  # Enum type, possible values: ("Metric", "Log", "Trace")
    timestamp: "2021-03-04 14:57:00",  # Timestamp when the anomaly happened, in UTC time, please convert to local time in Asia/Shanghai
    example:
        df['datetime'] = pd.to_datetime(df['timestamp'].astype(float), unit='s', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
}}
</output_format>

{time_process_guide}
There is some domain knowledge for you:

{background}


When investigating, you should first identify which cloudbed the query is referring to, and focus your analysis on that one only. Each investigation involves a single cloudbed. It’s not possible for a query to cover two cloudbeds at the same time.

When generating code, please follow the anomaly investigation workflow described below.

{trace_investigation_workflow}

Your response should follow the Python block format below:

```python
(YOUR CODE HERE)
```
If you find your code executed successfully, you can stop, do not continue to write anything.
"""


system_reasoner = f"""Now, you have decided to finish your reasoning process. You should now provide the final answer to the issue. The candidates of possible root cause components and reasons are provided to you. The root cause components and reasons must be selected from the provided candidates.

{cand}

{metric_description}
Please first review previous diagnosis_events to infer an exact answer of the issue. Then, summarize your final answer of the root causes using the following JSON format at the end of your response:

```json
{{
    "1": {{
        "root cause occurrence datetime": (format: '%Y-%m-%d %H:%M:%S', otherwise ommited, Please make sure the root cause occurrence datetime is in the correct time zone and falls within the query time window. If an anomaly's timestamp is outside the query window, it may be due to not using the UTC+8 time zone. In that case, automatically adjust the time zone to ensure the output time is within the query window.),
        "root cause component": (one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (one selected from the possible root cause reason list, otherwise ommited),
    }}, (Top 1 likely root cause.)
    "2": {{
        "root cause occurrence datetime": (format: '%Y-%m-%d %H:%M:%S', otherwise ommited, Please make sure the root cause occurrence datetime is in the correct time zone and falls within the query time window. If an anomaly's timestamp is outside the query window, it may be due to not using the UTC+8 time zone. In that case, automatically adjust the time zone to ensure the output time is within the query window.),
        "root cause component": (one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (one selected from the possible root cause reason list, otherwise ommited),
    }}, (Top 2 likely root cause, if applicable.)
    "3": {{
        "root cause occurrence datetime": (format: '%Y-%m-%d %H:%M:%S', otherwise ommited, Please make sure the root cause occurrence datetime is in the correct time zone and falls within the query time window. If an anomaly's timestamp is outside the query window, it may be due to not using the UTC+8 time zone. In that case, automatically adjust the time zone to ensure the output time is within the query window.),
        "root cause component": (one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (one selected from the possible root cause reason list, otherwise ommited),
    }}, (Top 3 likely root cause, if applicable.)
}}
```
Important: Please always provide the top three most likely root cause candidates, regardless of how many root causes the issue asks for.

(Please use "```json" and "```" tags to wrap the JSON object. You only need to provide the elements asked by the issue, and ommited the other fields in the JSON.)
Note that all the root cause components and reasons must be selected from the provided candidates. Do not reply 'unknown' or 'null' or 'not found' in the JSON. Do not be too conservative in selecting the root cause components and reasons. Always be decisive to infer a possible answer based on your current observation.

<metric_reasoning_rules>
As the root cause analysis reasoner, you need to process the anomalies detected in the investigating stage and identify the top 3 root causes. You can follow the logic below:

1. **Delta comparison**:
   Delta = (normalized anomaly-phase peak value) - (normalized stable-phase average). Anomalies with much smaller delta values compared to others may be noise.

2. **Component anomaly frequency comparison**: 
   The more anomalies occur on the same component, the more likely that component is the root cause.

3. **Time-based clustering**:
   If several anomalies with high delta values occur close in time (e.g., within 8 minutes), that time window is likely to contain the root cause. In other words, anomalies that cluster together in time are more likely to be meaningful, while isolated anomalies may be noise. Note: When clustering anomalies, avoid using an overly large time window. Only group anomalies that occur within 8 minutes of each other. Do not cluster events that are far apart in time into the same window.

4. **Ranking within a time cluster**:
   For anomalies that occur in the same time cluster (e.g., within 5 minutes), it's likely they stem from the same root cause. In this case, apply the following priorities:

   * **Time and delta priority**: Anomalies that occur earlier and have higher delta are more likely to be the root cause.
   
5. **Determine whether the anomaly occurs at the service level**:
   If multiple pods (two or more) from the same service experience the same type of anomaly at the same time, the root cause is likely a failure at the **service level**, not individual pods.
   For example, if `shippingservice-0`, `shippingservice-1`, `shippingservice-2`, and `shippingservice2-0` all show a `container CPU load` anomaly, the root cause should be identified as a `container CPU load` failure on the `shippingservice`.

   Conversely, if only one pod shows the anomaly while others do not, the root cause should be attributed to that **specific pod**, not the whole service.
   For instance, if only `shippingservice-0` has a `container CPU load` anomaly and the others (`shippingservice-1`, `shippingservice-2`, `shippingservice2-0`) do not, the root cause is a `container CPU load` failure on `shippingservice-0`.
   
</metric_reasoning_rules>

<trace_reasoning_rules>
Trace-based anomalies usually indicate network issues on the affected component, such as `network packet loss` or `network latency`. 
If many trace-based anomalies are detected within the diagnosis window, they are likely noise and should be ignored. 
However, if only one trace-based anomaly is detected, it is less likely to be noise and should be considered as the root cause.
</trace_reasoning_rules>

<multimodal data fusion analysis>
Metric-based anomaly detection is used to initially identify potential anomalies, while trace-based detection helps pinpoint the root cause.

Metric-based anomalies reflect the operational state of components but can be noisy. Trace-based anomalies provide more fine-grained insights into the components and are generally more accurate. Therefore, when both types are present, trace-based anomalies should be given higher priority.
</multimodal data fusion analysis>
"""