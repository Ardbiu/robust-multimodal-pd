import logging
import numpy as np
from pathlib import Path
from pd_fusion.data.schema import MODALITIES, TARGET_COL
from pd_fusion.data.preprocess import preprocess_features
from pd_fusion.data.feature_utils import get_modality_feature_cols, get_all_feature_cols
from pd_fusion.data.missingness import get_modality_mask_matrix
from pd_fusion.utils.io import load_yaml
from pd_fusion.paths import ROOT_DIR
import torch

def train_pipeline(config, df_train, df_val, mask_train, mask_val):
    logger = logging.getLogger("pd_fusion")
    model_type = config["model_type"]
    if "params" not in config or not isinstance(config.get("params"), dict):
        config["params"] = {}

    # Fallback to default params if missing
    def _load_default(path_str: str):
        p = Path(path_str)
        if not p.exists():
            p = ROOT_DIR / p
        try:
            return load_yaml(p).get("params", {})
        except Exception:
            return {}

    if model_type in ["fusion_late", "fusion_masked", "fusion_moddrop"]:
        defaults = _load_default("configs/model_fusion.yaml")
        if "hidden_dims" not in config["params"]:
            config["params"] = {**defaults, **config["params"]}
    elif model_type == "moe":
        defaults = _load_default("configs/model_moe.yaml")
        if "expert_hidden_dims" not in config["params"]:
            config["params"] = {**defaults, **config["params"]}
    elif model_type == "unimodal_gbdt":
        defaults = _load_default("configs/model_unimodal.yaml")
        if not config["params"]:
            config["params"] = {**defaults, **config["params"]}
    
    # Simple feature concatenation for non-MoE models
    all_features = get_all_feature_cols(df_train)
    
    if not all_features:
        raise ValueError("No feature columns found for any modality. Check dataset loader and schema.")

    X_train, imputer, scaler = preprocess_features(df_train, all_features)
    X_val, _, _ = preprocess_features(df_val, all_features, imputer, scaler)
    
    y_train = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values

    # Determine modality dims for advanced models
    mod_dims = {}
    for mod in MODALITIES:
        current_mod_feats = get_modality_feature_cols(df_train, mod)
        mod_dims[mod] = len(current_mod_feats)

    model = None
    prep_info = (imputer, scaler, all_features)
    calibrate_X_val = X_val
    calibrate_masks = None
    
    if model_type == "unimodal_gbdt":
        from pd_fusion.models.unimodal_gbdt import UnimodalGBDT
        modality = config.get("modality", "clinical")
        # Extract only this modality's features
        mod_features = get_modality_feature_cols(df_train, modality)
        if not mod_features:
            logger.warning(f"Unimodal '{modality}' has no features in dataset; using constant baseline.")
            from pd_fusion.models.dummy import ConstantProbabilityModel
            model = ConstantProbabilityModel()
            model.train(np.zeros((len(y_train), 1)), y_train, None)
            prep_info = (None, None, mod_features)
            calibrate_X_val = np.zeros((len(y_val), 1))
        else:
            X_train_mod, imp, scl = preprocess_features(df_train, mod_features)
            X_val_mod, _, _ = preprocess_features(df_val, mod_features, imp, scl)
            model = UnimodalGBDT(modality, config["params"])
            model.train(X_train_mod, y_train, (X_val_mod, y_val))
            prep_info = (imp, scl, mod_features)
            calibrate_X_val = X_val_mod
        
    elif model_type == "fusion_late":
        from pd_fusion.models.fusion_late import LateFusionModel
        model = LateFusionModel(len(all_features), config["params"])
        model.train(X_train, y_train, (X_val, y_val))

    elif model_type == "fusion_masked":
        from pd_fusion.models.fusion_masked import MaskedFusionModel
        # Append masks to inputs
        train_mask_mat = get_modality_mask_matrix(mask_train)
        val_mask_mat = get_modality_mask_matrix(mask_val)
        X_train_masked = np.concatenate([X_train, train_mask_mat], axis=1)
        X_val_masked = np.concatenate([X_val, val_mask_mat], axis=1)
        model = MaskedFusionModel(len(all_features), train_mask_mat.shape[1], config["params"])
        model.train(X_train_masked, y_train, (X_val_masked, y_val))
        calibrate_X_val = X_val_masked

    elif model_type == "fusion_moddrop":
        from pd_fusion.models.fusion_moddrop import ModalityDropoutModel
        # X is concatenated, but model needs dims
        model = ModalityDropoutModel(mod_dims, config["params"])
        model.train(X_train, y_train, (X_val, y_val))
        calibrate_masks = mask_val

    elif model_type == "moe":
        from pd_fusion.models.moe import MoEModel
        mod_dims = {}
        X_train_dict = {}
        X_val_dict = {}
        moe_preprocessors = {}
        mods_used = []
        
        for mod in MODALITIES:
            feats = get_modality_feature_cols(df_train, mod)
            if not feats:
                continue
            X_mod, imp_m, scl_m = preprocess_features(df_train, feats)
            X_mod_val, _, _ = preprocess_features(df_val, feats, imp_m, scl_m)
            
            X_train_dict[mod] = torch.FloatTensor(X_mod)
            X_val_dict[mod] = torch.FloatTensor(X_mod_val)
            mod_dims[mod] = len(feats)
            moe_preprocessors[mod] = (imp_m, scl_m, feats)
            mods_used.append(mod)
            
        mask_train_tensor = torch.FloatTensor(np.stack([mask_train[m] for m in mods_used], axis=1))
        mask_val_tensor = torch.FloatTensor(np.stack([mask_val[m] for m in mods_used], axis=1))
        
        model = MoEModel(mod_dims, config["params"])
        model.train(X_train_dict, y_train, mask_train_tensor, (X_val_dict, y_val, mask_val_tensor))
        prep_info = moe_preprocessors
        calibrate_X_val = X_val_dict
        calibrate_masks = mask_val_tensor

    # Calibration
    if config.get("calibrate", False):
        from pd_fusion.models.calibrate import CalibratedModel
        cal_model = CalibratedModel(model, method="isotonic")
        # Calibrate using validation data in the correct format for each model
        if model_type == "moe":
            if calibrate_X_val is not None and calibrate_masks is not None:
                cal_model.fit(calibrate_X_val, y_val, calibrate_masks)
                model = cal_model
            else:
                logger.warning("Skipping calibration for MoE; missing validation inputs or masks.")
        else:
            cal_model.fit(calibrate_X_val, y_val, calibrate_masks)
            model = cal_model

    return model, prep_info

    # TODO: Add other models
    raise ValueError(f"Unknown model type: {model_type}")
