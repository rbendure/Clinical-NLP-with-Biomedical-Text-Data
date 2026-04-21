@echo off
REM Example usage:
REM run_experiment.bat
REM run_experiment.bat --compare_models --epochs 2 --train_size 1000 --val_size 200

python -m src.main %*
