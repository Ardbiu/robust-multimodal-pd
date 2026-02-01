def get_clinical_features(df):
    # TODO: Map PPMI clinical columns to research features
    return [col for col in df.columns if col.startswith("clinical_")]
