# Sentiment Analysis on Imbalanced Persian User Reviews with LLMs

This repository contains a full sentiment analysis pipeline on **Persian user reviews**, focusing on:

- Handling a **highly imbalanced dataset** (many more positive than negative samples).
- Comparing **classical ML (TF–IDF)** vs **modern sentence embeddings** (BGE-M3, ParsBERT).
- Using a **Large Language Model (LLM)** to:
  - Generate **synthetic negative samples** to balance the data.
  - Act as a **judge** (“LLM-as-a-Judge”) to automatically label sentiment and compare with the original labels.

The core logic and experiments are implemented in a single notebook:

> `notebooks/persian_sentiment_llm_pipeline.ipynb`

---

## 1. Project Overview

Many real-world Persian review datasets are **imbalanced**: users tend to write more positive reviews than negative ones.  
This makes it harder for a classifier to learn the negative class properly.

In this project, we:

1. Load and clean a dataset of Persian user reviews.
2. Convert a numeric score into a **sentiment label** (`Positive` / `Negative`).
3. Normalize and clean the text (URLs, emojis, extra characters, etc.).
4. Train **classical ML models** on top of **TF–IDF** features.
5. Train models using **sentence embeddings**:
   - `BAAI/bge-m3`
   - ParsBERT-based embeddings
6. Handle data imbalance by generating **synthetic negative comments** using an LLM (via AvalAI API).
7. Use **LLM-as-a-Judge** to label each review as `Positive` or `Negative` and compare those labels with the original dataset labels.

