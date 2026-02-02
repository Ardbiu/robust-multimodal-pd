import logging
from pd_fusion.data.dev_datasets.uci_parkinsons import load_uci_parkinsons
from pd_fusion.data.dev_datasets.uci_telemonitoring import load_uci_telemonitoring
from pd_fusion.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("pd_fusion.verify")

def verify_loaders():
    print("-" * 50)
    print("Verifying UCI Parkinsons...")
    try:
        df, masks = load_uci_parkinsons()
        print(f"SUCCESS. Shape: {df.shape}")
        print(f"Masks keys: {list(masks.keys())}")
        print(f"Clinical Present: {masks['clinical'].sum()}/{len(df)}")
    except Exception as e:
        print(f"FAILED: {e}")

    print("-" * 50)
    print("Verifying UCI Telemonitoring...")
    try:
        df, masks = load_uci_telemonitoring()
        print(f"SUCCESS. Shape: {df.shape}")
        print(f"Masks keys: {list(masks.keys())}")
        print(f"Clinical Present: {masks['clinical'].sum()}/{len(df)}")
    except Exception as e:
        print(f"FAILED: {e}")
    print("-" * 50)

if __name__ == "__main__":
    verify_loaders()
