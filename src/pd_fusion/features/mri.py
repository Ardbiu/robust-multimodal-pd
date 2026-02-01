def get_mri_features(df):
    # TODO: Map PPMI MRI columns to research features
    return [col for col in df.columns if col.startswith("mri_")]
