# OpsLens

![Python Version](https://img.shields.io/badge/Python-3776AB?&logo=python&logoColor=white-blue&label=3.12)&ensp;

We propose OpsLens, a generalizable RCA solution that performs collaborative analysis on multi-modal observability data. OpsLens eliminates the need for prebuilt tools or historical incident reports. It uses an LLM-based multi-agent system (MAS) and the two-stage tool generation with feedback-based refinement to better interact with observability data and perform collaborative analysis. 

Experiments on public datasets  from three real-world systems  containing 335  incidents show that OpsLens achieves state-of-the-art RCA performance, with a score rate of 0.80.

## ✨ Features

- LLM-based multi-agent system (MAS)
- Two-stage tool generation with feedback-based refinement
- Better interact with observability data and perform collaborative analysis. 

## 🛠️ How to Run

OpsLens requires **Python >= 3.12**. 

You can run OpsLens to perform RCA on a given query by running:

```bash
python opslens.py --query "On March 10, 2021, between 15:00 and 15:30, two system failures were encountered. The components responsible for these failures and the reasons behind them are not yet known. Please identify the root cause components and the root cause reasons."
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
      ./query/bank_query.csv \
      ./query/telecom_query.csv \
      ./query/market1_query.csv \
      ./query/market2_query.csv \
    -r \
      ./report.csv
```
