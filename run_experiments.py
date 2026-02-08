"""
Run the Phase 4 Sensitivity Experiment Suite.

Executes a curated set of experiments varying architecture, loss
function, feature subsets, and regularisation strength.  Results are
saved to ``results/experiments/``.

Experiment Sets:
  1. Architecture Comparison (6 configs, 30 features)
  2. Feature Ablation (5 configs, MLP [128, 64])
  3. Regularisation (4 configs, varying dropout)

Usage:
    python run_experiments.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import config
from src.train import load_processed_data
from src.experiments import (
    ExperimentConfig,
    run_experiment_suite,
    generate_experiment_report,
)

# Phase 3 reference results
NN_V1_REFERENCE_MAE = 4.29


def define_experiments() -> list[ExperimentConfig]:
    """Return the full list of experiment configurations."""

    configs: list[ExperimentConfig] = []

    # ==================================================================
    # Set 1: Architecture Comparison (using current 30 features)
    # ==================================================================

    # 1. MLP [64, 32] (Phase 3 reproduction) -- MSE loss
    configs.append(ExperimentConfig(
        name="MLP [64,32] MSE",
        model_class="enhanced_mlp",
        hidden_sizes=[64, 32],
        dropout=0.0,
        loss_type="mse",
        max_epochs=50,
        patience=15,
    ))

    # 2. MLP [64, 32] -- Huber loss
    configs.append(ExperimentConfig(
        name="MLP [64,32] Huber",
        model_class="enhanced_mlp",
        hidden_sizes=[64, 32],
        dropout=0.0,
        loss_type="huber",
        max_epochs=50,
        patience=15,
    ))

    # 3. MLP [128, 64, 32] -- Huber loss
    configs.append(ExperimentConfig(
        name="MLP [128,64,32] Huber",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64, 32],
        dropout=0.1,
        loss_type="huber",
        max_epochs=50,
        patience=15,
    ))

    # 4. MLP [256, 128, 64] -- Huber loss
    configs.append(ExperimentConfig(
        name="MLP [256,128,64] Huber",
        model_class="enhanced_mlp",
        hidden_sizes=[256, 128, 64],
        dropout=0.1,
        loss_type="huber",
        max_epochs=50,
        patience=15,
    ))

    # 5. Enhanced MLP [128, 64] with batch norm -- Huber loss
    configs.append(ExperimentConfig(
        name="MLP [128,64] BN Huber",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.1,
        loss_type="huber",
        use_batch_norm=True,
        max_epochs=50,
        patience=15,
    ))

    # 6. LSTM hidden=64, 1 layer -- Huber loss
    configs.append(ExperimentConfig(
        name="LSTM h=64 Huber",
        model_class="lstm",
        hidden_sizes=[64],
        dropout=0.0,
        loss_type="huber",
        max_epochs=50,
        patience=15,
    ))

    # ==================================================================
    # Set 2: Feature Ablation (using MLP [128, 64], Huber loss)
    # ==================================================================

    # 7. TMAX features only
    configs.append(ExperimentConfig(
        name="TMAX only",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.1,
        loss_type="huber",
        features="tmax_only",
        max_epochs=50,
        patience=15,
    ))

    # 8. TMIN features only
    configs.append(ExperimentConfig(
        name="TMIN only",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.1,
        loss_type="huber",
        features="tmin_only",
        max_epochs=50,
        patience=15,
    ))

    # 9. TMAX+TMIN (current features, all) -- same arch for comparison
    configs.append(ExperimentConfig(
        name="TMAX+TMIN (all)",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.1,
        loss_type="huber",
        features="all",
        max_epochs=50,
        patience=15,
    ))

    # 10. Without date features
    configs.append(ExperimentConfig(
        name="No date features",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.1,
        loss_type="huber",
        features="no_date",
        max_epochs=50,
        patience=15,
    ))

    # 11. With date features (same as #9, for explicit comparison)
    configs.append(ExperimentConfig(
        name="With date features",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.1,
        loss_type="huber",
        features="all",
        max_epochs=50,
        patience=15,
    ))

    # ==================================================================
    # Set 3: Regularisation (MLP [128, 64], Huber loss)
    # ==================================================================

    # 12. dropout=0.0
    configs.append(ExperimentConfig(
        name="Dropout 0.0",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.0,
        loss_type="huber",
        max_epochs=50,
        patience=15,
    ))

    # 13. dropout=0.1
    configs.append(ExperimentConfig(
        name="Dropout 0.1",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.1,
        loss_type="huber",
        max_epochs=50,
        patience=15,
    ))

    # 14. dropout=0.2
    configs.append(ExperimentConfig(
        name="Dropout 0.2",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.2,
        loss_type="huber",
        max_epochs=50,
        patience=15,
    ))

    # 15. dropout=0.3
    configs.append(ExperimentConfig(
        name="Dropout 0.3",
        model_class="enhanced_mlp",
        hidden_sizes=[128, 64],
        dropout=0.3,
        loss_type="huber",
        max_epochs=50,
        patience=15,
    ))

    return configs


def main():
    """Run the full experiment suite."""
    print("=" * 70)
    print("NYC Temperature Prediction -- Phase 4 Sensitivity Experiments")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Load processed data
    # ------------------------------------------------------------------
    print("Step 1: Loading processed data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run the data pipeline first.")
        sys.exit(1)

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | "
          f"Test: {X_test.shape}")
    print(f"  Features: {list(X_train.columns)}")
    print()

    # ------------------------------------------------------------------
    # 2. Define experiments
    # ------------------------------------------------------------------
    configs = define_experiments()
    print(f"Step 2: {len(configs)} experiments defined:")
    for i, c in enumerate(configs, 1):
        print(f"  {i:2d}. {c.name} "
              f"(model={c.model_class}, loss={c.loss_type}, "
              f"feat={c.features}, drop={c.dropout})")
    print()

    # ------------------------------------------------------------------
    # 3. Run experiments
    # ------------------------------------------------------------------
    print("Step 3: Running experiments...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print()

    output_dir = os.path.join(config.RESULTS_DIR, "experiments")
    models_dir = os.path.join(config.MODELS_DIR, "experiments")

    t0 = time.time()
    results_df = run_experiment_suite(
        configs,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        device=device,
        models_dir=models_dir,
    )
    total_time = time.time() - t0

    print()
    print(f"  All experiments completed in {total_time:.1f} seconds")
    print()

    # ------------------------------------------------------------------
    # 4. Generate report
    # ------------------------------------------------------------------
    print("Step 4: Generating report...")
    report = generate_experiment_report(
        results_df,
        output_dir=output_dir,
        reference_mae=NN_V1_REFERENCE_MAE,
    )
    print()
    print(report)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print()
    ok = results_df[results_df["status"] == "success"]
    if len(ok) > 0:
        best = ok.sort_values("mae").iloc[0]
        print(f"Best experiment: {best['name']}")
        print(f"  MAE:  {best['mae']:.3f} F")
        print(f"  RMSE: {best['rmse']:.3f} F")
        print(f"  R2:   {best['r2']:.4f}")
        print()

        beats_ref = ok[ok["mae"] < NN_V1_REFERENCE_MAE]
        if len(beats_ref) > 0:
            print(f"{len(beats_ref)} experiment(s) beat the NN V1 "
                  f"reference (MAE < {NN_V1_REFERENCE_MAE}):")
            for _, row in beats_ref.sort_values("mae").iterrows():
                print(f"  - {row['name']}: MAE={row['mae']:.3f}")
        else:
            print("No experiment beat the NN V1 reference yet.")

    print()
    print(f"Results saved to: {output_dir}")
    print()
    saved = sorted(os.listdir(output_dir))
    for fname in saved:
        fpath = os.path.join(output_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname} ({size_kb:.1f} KB)")
    print()
    print("Phase 4 experiments complete.")


if __name__ == "__main__":
    main()
