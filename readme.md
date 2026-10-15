# SynthShield

### Privacy-Preserving Synthetic Data Generation

SynthShield is a Python and Streamlit-based application that generates synthetic tabular data while incorporating privacy-aware processing and statistical evaluation.

## Features
- UCI Adult Income dataset loading and preprocessing
- Quasi-identifier-based privacy-risk analysis
- Five-year age generalization
- Group filtering using a minimum threshold of k = 5
- Baseline and Smoothed Distribution generation
- Exact-match checking
- Statistical comparison of original and synthetic datasets
- CSV download through a Streamlit interface

## Technologies
Python, Pandas, NumPy, Streamlit

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
The project uses the UCI Adult Income dataset.

## Limitations
The prototype does not implement differential privacy and does not guarantee protection against every privacy attack.
