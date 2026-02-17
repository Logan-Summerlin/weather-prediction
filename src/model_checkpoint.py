"""
Model Checkpoint Persistence for Multi-City Temperature Prediction.

Provides utilities to save and load trained models, scalers, calibrators,
and associated metadata for CHI/PHL/NYC pipelines. Supports:

  - PyTorch model state dicts (.pt files)
  - Scikit-learn models and scalers (.pkl files)
  - Column metadata and training configuration (.json files)
  - Model loading for inference without retraining

Directory structure:
    models/{city}/
        {model_name}.pt          -- PyTorch state dict
        {model_name}_meta.json   -- Architecture config + training info
        scaler.pkl               -- StandardScaler fit on training data
        col_metadata.json        -- Column names and order
        calibrators/
            isotonic.pkl         -- IsotonicRegression calibrator
            platt.pkl            -- LogisticRegression (Platt) calibrator
            regime.pkl           -- Dict of regime-conditional calibrators
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_pytorch_model(
    model: nn.Module,
    save_dir: str,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a PyTorch model state dict and optional metadata.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    save_dir : str
        Directory to save into (e.g., models/chicago/).
    model_name : str
        Base name for the saved files (e.g., "e3_nn_128_64").
    metadata : dict, optional
        Additional metadata to persist (architecture config,
        training metrics, etc.).

    Returns
    -------
    str
        Path to the saved .pt file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save state dict
    pt_path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), pt_path)
    logger.info("Saved PyTorch model to %s", pt_path)

    # Save metadata
    meta = metadata or {}
    meta["saved_at"] = datetime.utcnow().isoformat()
    meta["model_class"] = model.__class__.__name__
    meta["n_parameters"] = sum(p.numel() for p in model.parameters())

    # Capture architecture details if available
    if hasattr(model, "n_features"):
        meta["n_features"] = model.n_features
    if hasattr(model, "hidden_sizes"):
        meta["hidden_sizes"] = model.hidden_sizes
    if hasattr(model, "dropout_rate"):
        meta["dropout_rate"] = model.dropout_rate
    if hasattr(model, "n_stations"):
        meta["n_stations"] = model.n_stations
    if hasattr(model, "output_mode"):
        meta["output_mode"] = model.output_mode

    meta_path = os.path.join(save_dir, f"{model_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Saved model metadata to %s", meta_path)

    return pt_path


def save_sklearn_model(
    model: Any,
    save_dir: str,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a scikit-learn model (Ridge, LogisticRegression, etc.) as pickle.

    Parameters
    ----------
    model : sklearn estimator
        Trained scikit-learn model.
    save_dir : str
        Directory to save into.
    model_name : str
        Base name for the saved file (e.g., "e2_ridge").

    Returns
    -------
    str
        Path to the saved .pkl file.
    """
    os.makedirs(save_dir, exist_ok=True)

    pkl_path = os.path.join(save_dir, f"{model_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved sklearn model to %s", pkl_path)

    if metadata:
        meta_path = os.path.join(save_dir, f"{model_name}_meta.json")
        metadata["saved_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return pkl_path


def save_scaler(
    scaler: Any,
    save_dir: str,
    filename: str = "scaler.pkl",
) -> str:
    """Save a fitted StandardScaler or similar transformer.

    Parameters
    ----------
    scaler : sklearn transformer
        Fitted scaler (e.g., StandardScaler).
    save_dir : str
        Directory to save into.
    filename : str
        Filename for the scaler pickle.

    Returns
    -------
    str
        Path to the saved pickle file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to %s", path)
    return path


def save_column_metadata(
    columns: List[str],
    save_dir: str,
    filename: str = "col_metadata.json",
) -> str:
    """Save column names and ordering for feature matrix reconstruction.

    Parameters
    ----------
    columns : list of str
        Column names in order.
    save_dir : str
        Directory to save into.
    filename : str
        Filename for the metadata JSON.

    Returns
    -------
    str
        Path to the saved JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump({"columns": columns, "n_features": len(columns)}, f, indent=2)
    logger.info("Saved column metadata (%d features) to %s", len(columns), path)
    return path


def save_calibrator(
    calibrator: Any,
    save_dir: str,
    calibrator_name: str,
) -> str:
    """Save a calibration model (isotonic, Platt, regime-conditional).

    Parameters
    ----------
    calibrator : sklearn estimator or dict
        The calibration model or dict of calibrators.
    save_dir : str
        Directory to save into.
    calibrator_name : str
        Name for the calibrator (e.g., "isotonic", "platt", "regime").

    Returns
    -------
    str
        Path to the saved pickle file.
    """
    cal_dir = os.path.join(save_dir, "calibrators")
    os.makedirs(cal_dir, exist_ok=True)
    path = os.path.join(cal_dir, f"{calibrator_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info("Saved calibrator '%s' to %s", calibrator_name, path)
    return path


# ---------------------------------------------------------------------------
# Load utilities
# ---------------------------------------------------------------------------

def load_pytorch_model(
    model_class: type,
    save_dir: str,
    model_name: str,
    device: str = "cpu",
    **model_kwargs,
) -> nn.Module:
    """Load a PyTorch model from a saved state dict.

    Parameters
    ----------
    model_class : type
        The nn.Module subclass to instantiate (e.g., HeteroscedasticNet).
    save_dir : str
        Directory containing the saved model.
    model_name : str
        Base name of the saved model.
    device : str
        Device to load the model onto ("cpu" or "cuda").
    **model_kwargs
        Arguments passed to model_class constructor. If not provided,
        attempts to load from metadata JSON.

    Returns
    -------
    nn.Module
        The loaded model in eval mode.
    """
    pt_path = os.path.join(save_dir, f"{model_name}.pt")
    meta_path = os.path.join(save_dir, f"{model_name}_meta.json")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {pt_path}")

    # Load metadata for model construction if kwargs not provided
    if not model_kwargs and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "n_features" in meta:
            model_kwargs["n_features"] = meta["n_features"]
        if "hidden_sizes" in meta:
            model_kwargs["hidden_sizes"] = meta["hidden_sizes"]
        if "dropout_rate" in meta:
            model_kwargs["dropout"] = meta["dropout_rate"]

    model = model_class(**model_kwargs)
    state_dict = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Loaded PyTorch model from %s (%d params)",
                pt_path, sum(p.numel() for p in model.parameters()))
    return model


def load_sklearn_model(save_dir: str, model_name: str) -> Any:
    """Load a scikit-learn model from pickle.

    Parameters
    ----------
    save_dir : str
        Directory containing the saved model.
    model_name : str
        Base name of the saved model.

    Returns
    -------
    Any
        The loaded sklearn estimator.
    """
    pkl_path = os.path.join(save_dir, f"{model_name}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Model not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded sklearn model from %s", pkl_path)
    return model


def load_scaler(save_dir: str, filename: str = "scaler.pkl") -> Any:
    """Load a fitted scaler from pickle.

    Parameters
    ----------
    save_dir : str
        Directory containing the scaler.
    filename : str
        Filename of the scaler pickle.

    Returns
    -------
    Any
        The loaded scaler.
    """
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found: {path}")

    with open(path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Loaded scaler from %s", path)
    return scaler


def load_column_metadata(
    save_dir: str,
    filename: str = "col_metadata.json",
) -> Dict[str, Any]:
    """Load column metadata from JSON.

    Parameters
    ----------
    save_dir : str
        Directory containing the metadata.
    filename : str
        Filename of the metadata JSON.

    Returns
    -------
    dict
        Dictionary with "columns" and "n_features" keys.
    """
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Column metadata not found: {path}")

    with open(path) as f:
        meta = json.load(f)
    logger.info("Loaded column metadata (%d features) from %s",
                meta.get("n_features", -1), path)
    return meta


def load_calibrator(save_dir: str, calibrator_name: str) -> Any:
    """Load a calibration model from pickle.

    Parameters
    ----------
    save_dir : str
        Directory containing the calibrators/ subdirectory.
    calibrator_name : str
        Name of the calibrator (e.g., "isotonic", "platt", "regime").

    Returns
    -------
    Any
        The loaded calibrator.
    """
    path = os.path.join(save_dir, "calibrators", f"{calibrator_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibrator not found: {path}")

    with open(path, "rb") as f:
        calibrator = pickle.load(f)
    logger.info("Loaded calibrator '%s' from %s", calibrator_name, path)
    return calibrator


# ---------------------------------------------------------------------------
# Convenience: save full model suite
# ---------------------------------------------------------------------------

def save_city_model_suite(
    city_code: str,
    models_dir: str,
    nn_models: Optional[Dict[str, nn.Module]] = None,
    sklearn_models: Optional[Dict[str, Any]] = None,
    scaler: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    calibrators: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Save a complete model suite for a city.

    Parameters
    ----------
    city_code : str
        City identifier (e.g., "chi", "phl").
    models_dir : str
        Base models directory for the city.
    nn_models : dict, optional
        Dict mapping model names to PyTorch models.
    sklearn_models : dict, optional
        Dict mapping model names to sklearn models.
    scaler : sklearn transformer, optional
        Fitted feature scaler.
    columns : list of str, optional
        Feature column names.
    calibrators : dict, optional
        Dict mapping calibrator names to calibrator objects.
    metadata : dict, optional
        Additional metadata to include.

    Returns
    -------
    dict
        Mapping of artifact names to saved file paths.
    """
    os.makedirs(models_dir, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    meta = metadata or {}
    meta["city_code"] = city_code

    if nn_models:
        for name, model in nn_models.items():
            path = save_pytorch_model(model, models_dir, name, meta.copy())
            saved_paths[f"nn_{name}"] = path

    if sklearn_models:
        for name, model in sklearn_models.items():
            path = save_sklearn_model(model, models_dir, name, meta.copy())
            saved_paths[f"sklearn_{name}"] = path

    if scaler is not None:
        path = save_scaler(scaler, models_dir)
        saved_paths["scaler"] = path

    if columns is not None:
        path = save_column_metadata(columns, models_dir)
        saved_paths["col_metadata"] = path

    if calibrators:
        for name, cal in calibrators.items():
            path = save_calibrator(cal, models_dir, name)
            saved_paths[f"calibrator_{name}"] = path

    logger.info("Saved %d artifacts for %s to %s",
                len(saved_paths), city_code, models_dir)
    return saved_paths


def list_saved_models(models_dir: str) -> Dict[str, List[str]]:
    """List all saved model artifacts in a directory.

    Parameters
    ----------
    models_dir : str
        Directory to scan.

    Returns
    -------
    dict
        Dictionary with keys "pytorch", "sklearn", "calibrators", "other"
        mapping to lists of filenames.
    """
    result: Dict[str, List[str]] = {
        "pytorch": [],
        "sklearn": [],
        "calibrators": [],
        "other": [],
    }

    if not os.path.exists(models_dir):
        return result

    for f in sorted(os.listdir(models_dir)):
        if f.endswith(".pt"):
            result["pytorch"].append(f)
        elif f.endswith(".pkl"):
            result["sklearn"].append(f)
        elif f.endswith(".json"):
            result["other"].append(f)

    cal_dir = os.path.join(models_dir, "calibrators")
    if os.path.exists(cal_dir):
        for f in sorted(os.listdir(cal_dir)):
            if f.endswith(".pkl"):
                result["calibrators"].append(f)

    return result
