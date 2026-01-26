# LLM-Powered Log Anomaly Detection & Root Cause Analysis

## Overview
This project implements an **intelligent log analysis system** that detects anomalies in system logs and provides **explainable root cause analysis** using **Machine Learning, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs)**.

The system is designed as a **cloud-ready, modular application** suitable for academic and industry-oriented use cases.

---

## Problem
- Large-scale systems generate massive volumes of logs  
- Manual monitoring is inefficient and error-prone  
- Traditional anomaly detection lacks explanation and context  

---

## Solution
- Detect anomalous log patterns using ML  
- Retrieve similar historical incidents using RAG  
- Generate human-readable explanations and remediation suggestions using LLMs  

---

## Dataset
- **HDFS Structured Log Dataset (LogPai / LogHub)**
- File: `HDFS_2k.log_structured.csv`
- Contains parsed log events with anomaly labels

---

## Architecture (High Level)
Logs → Anomaly Detection → RAG Retrieval → LLM Explanation

---

## Key Features
- ML-based log anomaly detection  
- Vector similarity search for incident retrieval  
- LLM-powered root cause explanation  
- Modular and cloud-deployable design  

