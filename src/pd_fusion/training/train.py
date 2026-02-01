import logging
import numpy as np
from pd_fusion.data.schema import MODALITIES, TARGET_COL
from pd_fusion.data.preprocess import preprocess_features
import torch

def train_pipeline(config, df_train, df_val, mask_train, mask_val):
    logger = logging.getLogger("pd_fusion")
    model_type = config["model_type"]
    
    # Simple feature concatenation for non-MoE models
    all_features = []
    for mod in MODALITIES:
        all_features.extend([col for col in df_train.columns if col.startswith(f"{mod}_")])
    
    X_train, imputer, scaler = preprocess_features(df_train, all_features)
    X_val, _, _ = preprocess_features(df_val, all_features, imputer, scaler)
    
    y_train = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values
    
    if model_type == "unimodal_gbdt":
        from pd_fusion.models.unimodal_gbdt import UnimodalGBDT
        modality = config.get("modality", "clinical")
        mod_features = [col for col in df_train.columns if col.startswith(f"{modality}_")]
        X_train_mod, imp, scl = preprocess_features(df_train, mod_features)
        X_val_mod, _, _ = preprocess_features(df_val, mod_features, imp, scl)
        model = UnimodalGBDT(modality, config["params"])
        model.train(X_train_mod, y_train, (X_val_mod, y_val))
        return model, (imp, scl, mod_features)
        
    elif model_type == "fusion_late":
        from pd_fusion.models.fusion_late import LateFusionModel
        model = LateFusionModel(len(all_features), config["params"])
        model.train(X_train, y_train, (X_val, y_val))
        return model, (imputer, scaler, all_features)

    elif model_type == "moe":
        from pd_fusion.models.moe import MoEModel
        mod_dims = {}
        X_train_dict = {}
        for mod in MODALITIES:
            feats = [col for col in df_train.columns if col.startswith(f"{mod}_")]
            X_mod, _, _ = preprocess_features(df_train, feats)
            X_train_dict[mod] = torch.FloatTensor(X_mod)
            mod_dims[mod] = len(feats)
            
        mask_train_tensor = torch.FloatTensor(np.stack([mask_train[m] for m in MODALITIES], axis=1))
        model = MoEModel(mod_dims, config["params"])
        model.train(X_train_dict, y_train, mask_train_tensor)
        return model, (None, None, MODALITIES)

    # TODO: Add other models
    raise ValueError(f"Unknown model type: {model_type}")
活跃的
