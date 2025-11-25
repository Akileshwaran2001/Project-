# N-BEATS Time Series Forecasting — Full Expanded Project

This project provides a production-ready starting point for implementing and evaluating the N-BEATS model for multi-step time series forecasting.

Contents:
- `scripts/` — data generation, training, evaluation scripts
- `models/` — model implementation (N-BEATS)
- `data/` — example synthetic dataset (CSV)
- `report/` — project report and summary

Quick start:
1. Create a Python environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Generate example data:
   ```
   python scripts/generate_synthetic.py
   ```
3. Train the model:
   ```
   python scripts/train.py --epochs 10
   ```
4. Evaluate:
   ```
   python scripts/evaluate.py
   ```

Files are intentionally small; modify hyperparameters and dataset as needed.
