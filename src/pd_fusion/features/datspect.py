def get_datspect_features(df):
    # TODO: Map PPMI DAT-SPECT columns to research features
    return [col for col in df.columns if col.startswith("datspect_")]
