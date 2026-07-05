#!/usr/bin/env bash
# ==============================================================
# Reproduction script: GTPA + Dynamic Fallback
# RoboCup 2026 Soccer Trajectory Prediction Challenge
# ==============================================================
# Usage: bash reproduce.sh
# Requirements: Python 3.8+, PyTorch, WSL/Linux (GPU optional)
# ==============================================================

set -euo pipefail

echo "=== Step 1: Install dependencies ==="
pip install torch numpy pandas matplotlib scikit-learn

echo "=== Step 2: Download weights ==="
# Weights are included at: STP-challenge-2025-main/weights/gtpa_robocup2D/16_20/

echo "=== Step 3: Generate challenge predictions ==="
cd STP-challenge-2025-main

python main.py --model gtpa \
  --data robocup2D \
  --data_dir robocup2d_data \
  --batchsize 16 \
  --totalTimeSteps 20 \
  --challenge_data challenge_input \
  --cont \
  --use_perturbation \
  --pert_noise_scale 0.2 \
  --pert_p_event 1.0 \
  --pf_alpha 0.5 \
  --pf_beta 0.5 \
  --pf_gamma 1.0 \
  --pf_num_particles 32 \
  --use_recursive_memory \
  --recursive_alpha 0.3 \
  --use_intercept \
  --intercept_beta 0.5 \
  --intercept_horizon 5 \
  --intercept_weight 0.5 \
  --use_fluid_ball \
  --fluid_ball_gamma 0.6 \
  --fluid_ball_sigma 0.02 \
  --use_dynamic_fallback \
  --fallback_w_dist 0.6 \
  --fallback_w_speed 0.2 \
  --fallback_w_horizon 0.2 \
  --fallback_w_accel 0.0

echo "=== Step 4: Evaluate ==="
python eval_dynamic.py

echo "=== Done ==="
echo "Predictions saved to results/test/submission/"
echo "Expected overall error: 14.94 m"
