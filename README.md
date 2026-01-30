# OpsLens 

![Python Version](https://img.shields.io/badge/Python-3776AB?&logo=python&logoColor=white-blue&label=3.12)&ensp;

We propose OpsLens, a generalizable RCA solution that performs collaborative analysis on multi-modal observability data. OpsLens eliminates the need for prebuilt tools or historical incident reports. It uses an LLM-based multi-agent system (MAS) and the two-stage tool generation with quality-focused to better interact with observability data and perform collaborative analysis. 

Experiments on public datasets  from three real-world systems  containing 335  incidents show that OpsLens achieves state-of-the-art RCA performance, with a score rate of 0.80.

## ✨ Features
- LLM-based multi-agent system (MAS)
- Two-stage tool generation with quality-focused tools management
- Better interact with observability data and perform collaborative analysis. 

## 💡Prerequisites
OpsLens requires **Python >= 3.12**. 
```
# create conda env
conda create -n opslens python==3.12
conda activate opslens
pip install -r requirements.txt
```
### docker
To ensure environmental safety, we run the generated code in Docker. Therefore, please make sure that Docker is installed on your machine before running the code.
Run the following command to check if the Docker daemon is running:
```
docker version
```
If the terminal outputs `Cannot connect to the Docker daemon at xxx Is the docker daemon running?`, please check the running status of Docker again.

For Linux machines, please refer to: [Linux | Docker Docs](https://docs.docker.com/desktop/setup/install/linux/)
For Mac machines, in addition to Docker Desktop, you can also use Orbstack as an alternative:
```
brew install orbstack
```
### dataset
In addition to the environment, we use OpenRCA as the dataset. You can download the data from [Google Drive](https://drive.google.com/drive/folders/1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM) and then place it in the `coding/dataset` directory under the coding file set.

Note: Please maintain the following file structure format.
```
├── coding
│   └── dataset
│       └── Bank
│           └── telemetry
│               ├── 2021_03_04
│               │   ├── log
│               │   │   └── log_service.csv
│               │   ├── metric
│               │   │   ├── metric_app.csv
│               │   │   └── metric_container.csv
│               │   └── trace
│               │       └── trace_span.csv
│               └── 2021_03_05
│                   ├── log
│                   │   └── log_service.csv
│                   ├── metric
│                   │   ├── metric_app.csv
│                   │   └── metric_container.csv
│                   └── trace
│                       └── trace_span.csv
...
...

```
## 🛠️ How to Run
First, you need to add your api_key in `dataset_config.yaml`. 
```yaml
# global config
global_config:
  # OpenAI API config
  openai:
    api_key: "sk-1234"
```


Then,  you can run OpsLens to perform RCA on a given query by running:
```bash
python opslens.py --query "On March 4, 2021, between 18:00 and 18:30, there was a single failure observed in the system. The exact component that caused this failure is unknown, and the reason behind the failure is also undetermined. Your task is to identify the root cause component and the root cause reason for this failure."
```

## 📊 How to Evaluate
We evaluate OpsLens on three real-world systems: Bank, Market, and Telecom. 
You can reproduce the evaluation results by running:
```bash
python -m eval.evaluate \
    -p \
      ./archive/opslens-eval-bank.csv \
      ./archive/opslens-eval-telecom.csv \
      ./archive/opslens-eval-market1.csv \
      ./archive/opslens-eval-market2.csv \
    -q \
      ./query/Bank/query.csv \
      ./query/Telecom/query.csv \
      ./query/Market/cloudbed-1/query.csv \
      ./query/Market/cloudbed-2/query.csv \
    -r \
      ./report.csv
```

Alternatively, you can conduct an evaluation on just one of the datasets individually.
```bash
python -m eval.evaluate \
    -p \
      ./archive/opslens-eval-bank.csv \
    -q \
      ./query/Bank/query.csv \
    -r \
      ./report_bank.csv
```


## 📒Notes

If you wish to experiment with other datasets(Telecom, Market), please first switch the `current_dataset` in dataset_config.yaml and organize the new dataset in accordance with the specified format.
