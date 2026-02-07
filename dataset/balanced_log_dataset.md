# Balanced HDFS Event Traces Dataset

## Overview
This file, **`balanced_log_dataset.csv`**, is a balanced version of the **Event Traces dataset** derived from the **HDFS (Hadoop Distributed File System) log dataset**.  
It is prepared specifically for **log anomaly detection using event sequence modeling**.

Each data sample represents a sequence of log events generated during a system execution, labeled as either normal or anomalous.

---

## Dataset Source
- Original Dataset Source: [HDFS Event Traces](https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1)
- Logs originate from a real Hadoop Distributed File System (HDFS) environment.

---

## Dataset Description
- Each record is an **event sequence** (e.g., `E1 → E5 → E12 → E3`)
- Sequences correspond to individual system execution sessions
- Labels:
  - `0` → Normal
  - `1` → Anomaly

---

## Class Distribution (Balanced)

| Class | Label | Samples |
|------|------|---------|
| Normal | 0 | 16,838 |
| Anomaly | 1 | 16,838 |

**Total samples:** 33,676

---

## Purpose
This dataset is used to:
- Train **Anomaly Detection models**
- Learn normal HDFS execution patterns
- Detect anomalous event sequences
- Support **RAG + LLM‑based root cause analysis**

---
