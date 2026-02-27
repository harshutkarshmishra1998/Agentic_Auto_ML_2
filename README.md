# Agentic_Auto_ML_2

Agentic_Auto_ML_2 is a deterministic-first, modular AutoML pipeline designed for structured datasets.  
It automates the full machine learning lifecycle while preserving auditability, reproducibility, and strict stage control.

The system follows a staged architecture that converts raw tabular data into trained models through semantic understanding, intelligent preprocessing, model selection, training, evaluation, and closed-loop retraining.



## Pipeline Overview

1. **Schema Inference** — detects semantic column roles (numeric, categorical, datetime, target) using deterministic rules with LLM fallback for ambiguity.  
2. **Data Understanding** — performs deep diagnostics including correlations, redundancy detection, dependency graphs, and transformation hints.  
3. **Preprocessing (Two-Stage)**  
   - Model-independent transformations (safe cleaning, imputation, structural fixes)  
   - Model-dependent alignment (encoding, scaling, feature engineering)  
4. **Model Selection** — hybrid rule-based and LLM-assisted ranking of candidate model families.  
5. **Model Initialization** — dataset-aware hyperparameter generation.  
6. **Training Engine** — executes validation strategies and logs experiments.  
7. **Evaluation** — derives metrics and determines retraining need.  
8. **Retraining Loop** — adaptive closed-loop optimization until convergence.



## Design Principles

- Deterministic before LLM reasoning  
- Append-only lineage logging (JSONL)  
- Strict module isolation  
- Versioned datasets and artifacts  
- Reproducible experiments  
- Controlled retraining decisions  



## Purpose

This system is built for controlled AutoML, experiment management, and production-grade model lifecycle orchestration where transparency and repeatability are required.