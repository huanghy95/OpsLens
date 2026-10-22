# OpsLens: Runbook-Free Root Cause Analysis via Multi-agent Collaboration and Tool Management

[![ASE 2026](https://img.shields.io/badge/ASE_2026-Accepted-2ea44f)](https://conf.researchr.org/track/ase-2026/ase-2026-industry-showcase)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3832783.3834501-blue)](https://doi.org/10.1145/3832783.3834501)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)

Official code repository for the **ASE 2026** paper **OpsLens**. OpsLens is an LLM-based, runbook-free **root cause analysis (RCA)** framework for **AIOps**. It uses multi-agent collaboration to analyze multi-modal observability data—**logs, metrics, and traces**—without requiring prebuilt diagnostic tools or historical incident reports. Two-stage tool generation and quality-focused tool management support collaborative fault diagnosis.

Experiments on the OpenRCA benchmark, covering **335 incidents** from three real-world systems (Bank, Market, and Telecom), achieve a **score rate of 0.80**.

**[Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3832783.3834501)** · **[Citation](#citation)**


## ✨ Features

- LLM-based multi-agent collaboration for runbook-free root cause analysis.
- Multi-modal analysis of logs, metrics, and distributed traces.
- Two-stage tool generation with quality-focused tool management.
- Evaluation on the Bank, Market, and Telecom systems in OpenRCA.

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

## Citation

If you use OpsLens in your research, please cite our ASE 2026 paper:

```bibtex
@inproceedings{huang2026opslens,
  author    = {Huang, Haiyu and Lyu, Jiewei and Tan, Gou and Chen, Pengfei and Jiang, Zhihan and Yu, Guangba and Mi, Penghui and Wang, Fuqiang and Han, Jian and Tan, Chongkang and Lyu, Michael R.},
  title     = {{OpsLens}: Runbook-Free Root Cause Analysis via Multi-agent Collaboration and Tool Management},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering},
  series    = {ASE '26},
  year      = {2026},
  month     = oct,
  pages     = {4116--4127},
  publisher = {Association for Computing Machinery},
  location  = {Munich, Germany},
  doi       = {10.1145/3832783.3834501},
  url       = {https://doi.org/10.1145/3832783.3834501}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff) for GitHub's **Cite this repository** feature.
