import json
from pathlib import Path

import pandas as pd

from pd_fusion.data.ppmi_studydata import build_ppmi_datasets


def _write_csv(path: Path, rows: list):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_ppmi_merge_baseline_and_splits(tmp_path: Path):
    raw_dir = tmp_path / "raw_ppmi" / "study_data"
    processed_dir = tmp_path / "processed_ppmi"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Label table
    _write_csv(
        raw_dir / "Participant_Status.csv",
        [
            {"PATNO": 1, "COHORT": "PD"},
            {"PATNO": 2, "COHORT": "HC"},
            {"PATNO": 3, "COHORT": "PD"},
        ],
    )

    # Clinical table
    _write_csv(
        raw_dir / "MDS_UPDRS.csv",
        [
            {"PATNO": 1, "EVENT_ID": "BL", "UPDRSIII": 20},
            {"PATNO": 2, "EVENT_ID": "BL", "UPDRSIII": 5},
            {"PATNO": 3, "EVENT_ID": "BL", "UPDRSIII": 15},
        ],
    )

    config = {
        "study_data_dir": str(raw_dir),
        "processed_ppmi_dir": str(processed_dir),
        "extract_zips": False,
        "tables": {
            "participant_status": {
                "patterns": ["*Participant_Status*.csv"],
                "group": "labels",
            },
            "mds_updrs": {
                "patterns": ["*MDS_UPDRS*.csv"],
                "group": "clinical",
            },
        },
        "splits": {
            "seeds": [42],
            "train_size": 0.67,
            "val_size": 0.16,
            "test_size": 0.17,
        },
    }

    logger = __import__("logging").getLogger("test")
    outputs = build_ppmi_datasets(config, logger)

    baseline = pd.read_csv(outputs["baseline"])
    assert "label" in baseline.columns
    assert baseline["label"].nunique() == 2
    assert baseline["subject_id"].nunique() == baseline.shape[0]

    split_path = processed_dir / "ppmi_splits_seed42.json"
    splits = json.loads(split_path.read_text())
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
