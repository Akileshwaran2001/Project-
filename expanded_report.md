# Expanded Project Report — N-BEATS Forecasting

## Project summary
This expanded project includes data generation, a modular N-BEATS model, training and evaluation scripts, and metrics.

## Architecture decisions
- Generic fully-connected N-BEATS blocks
- Stack of blocks with residual/backcast subtraction
- Mean Squared Error training; evaluate with MSE and MASE

## How to reproduce
1. Install requirements
2. Generate dataset: `python scripts/generate_synthetic.py`
3. Train: `python scripts/train.py --epochs 20`
4. Evaluate: `python scripts/evaluate.py`

## Next steps
- Replace synthetic data with ETT or M4 subsets
- Add seasonal/trend interpretable blocks
- Implement learning rate schedules, early stopping, and logging
