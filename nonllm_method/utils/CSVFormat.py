import json
import csv
import threading
from typing import List, Dict, Optional

write_lock = threading.Lock()

kpi_mapping = {
    "high CPU usage": [
        "OSLinux-CPU_CPU_CPUCpuUtil",
        "OSLinux-CPU_CPU_CPUUserTime"
    ],
    "high memory usage": [
        "OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc",
        "OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc",
        "OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem"
    ],
    "network latency": [
        "OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT"
    ],
    "network packet loss": [
        "OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT"
    ],
    "high disk I/O read usage": [
        "OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite"
    ],
    "high disk space usage": [
        "OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity",
        "OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity"
    ],
    "high JVM CPU load": [
        "JVM-Operating System_7779_JVM_JVM_CPULoad",
        "JVM-Operating System_7778_JVM_JVM_CPULoad"
    ],
    "JVM Out of Memory (OOM) Heap": [
        "JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed",
        "JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed"
    ],
    
    "CPU fault": ["container_container_cpu_used", "container_container_mem_used",'container_cpu_used','container_mem_used'],
    "db connection limit": ["Proc_User_Used_Pct", "Login_Per_Sec"],
    "db close": ["On_Off_State"],
    "network delay": ["Sent_queue", "Received_queue"],
    "network loss": ["Sent_queue", "Received_queue"],
    
    "container CPU load": ["container_container_cpu_usage_seconds", "container_cpu_usage_seconds"],
    "container memory load": ["container_container_memory_usage_MB", "container_memory_usage_MB"],
    "container process termination": ["container_container_threads", "container_threads"],
    "container read I/O load": ["container_container_fs_reads_MB./dev/vda", "container_fs_reads_MB./dev/vda"],
    "container write I/O load": ["container_container_fs_writes_MB./dev/vda", "container_fs_writes_MB./dev/vda"],
    "node memory consumption": ["system.mem.used", "system.mem.free"],
    "node disk read I/O consumption": ["system.io.r_await"],
    "node disk write I/O consumption": ["system.io.w_await"],
    "node disk space consumption": ["system.io.avg_q_sz"],
    "node CPU load": ["system.cpu.pct_usage", "system.load.1"],
    "node CPU spike": ["system.cpu.user"],
    "container network packet retransmission": ["container_container_network_receive_MB.eth0", "container_network_receive_MB.eth0"],
    "container network latency": ["container_container_sockets", "container_sockets"],
    "container packet loss": ["container_container_network_receive_MB.eth0", "container_container_network_receive_packets.eth0", "container_network_receive_MB.eth0", "container_network_receive_packets.eth0"]
}

_reason_map = {}
for _reason, _kpis in kpi_mapping.items():
    for _kpi in _kpis:
        _reason_map[_kpi] = _reason


def process_and_write_to_csv(
    service_list: List[str],
    nonservice_list: List[str],
    label_str: str,
    csv_file_path: str = "output.csv",
    fieldnames: Optional[List[str]] = None
):
    if fieldnames is None:
        fieldnames = ['record_id', 'prediction']

    prediction = {}
    for i in range(3):
        component = service_list[i] if i < len(service_list) else ""
        kpi = nonservice_list[i] if i < len(nonservice_list) else None

        if isinstance(kpi, list) and len(kpi) > 0:
            kpi = kpi[0]

        if isinstance(kpi, str) and '@' in kpi:
            kpi_for_reason = kpi.split('@', 1)[1]
        else:
            kpi_for_reason = kpi

        reason = _reason_map.get(kpi_for_reason, "unknown") if kpi_for_reason else "unknown"

        prediction[str(i + 1)] = {
            "root cause component": component,
            "root cause reason": reason
        }


    output_row = {
        "record_id": label_str,
        "prediction": json.dumps(prediction, ensure_ascii=False)
    }

    with write_lock:
        file_exists = True
        try:
            with open(csv_file_path, mode='r', newline='', encoding='utf-8'):
                pass
        except FileNotFoundError:
            file_exists = False

        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(output_row)

    print(f"✅ record_id: {label_str}")
    
    
if __name__ == "__main__":
    process_and_write_to_csv(
    service_list=["ServiceA", "ServiceB", "ServiceC"],
    nonservice_list=[
        "OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc",
        "container_cpu_used",
        "system.cpu.pct_usage"
    ],
    label_str="0",
    csv_file_path="output.csv"
)