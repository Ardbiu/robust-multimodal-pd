import logging
import datetime
from pathlib import Path
from pd_fusion.utils.io import load_yaml, save_yaml, save_pickle
from pd_fusion.utils.seed import set_seed
from pd_fusion.paths import RUNS_DIR, get_run_dir
from pd_fusion.data.ppmi_loader import load_ppmi_data
from pd_fusion.data.splits import stratified_split, get_subset_masks
from pd_fusion.training.train import train_pipeline
from pd_fusion.evaluation.evaluate import evaluate_model
from pd_fusion.evaluation.plots import plot_degradation_curve

def run_full_pipeline(config_path, synthetic=False):
    logger = logging.getLogger("pd_fusion")
    config = load_yaml(Path(config_path))
    data_config = load_yaml(Path("configs/data_ppmi.yaml"))
    
    set_seed(config.get("seed", 42))
    
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = get_run_dir(run_id)
    logger.info(f"Starting experiment: {run_id}")
    
    # 1. Load Data
    df, masks = load_ppmi_data(data_config, synthetic=synthetic)
    
    # 2. Split
    train_df, val_df, test_df = stratified_split(df)
    train_masks = get_subset_masks(masks, train_df.index)
    val_masks = get_subset_masks(masks, val_df.index)
    test_masks = get_subset_masks(masks, test_df.index)
    
    # 3. Train
    model, prep_info = train_pipeline(config, train_df, val_df, train_masks, val_masks)
    model.save(run_dir / "model.pt")
    save_pickle(prep_info, run_dir / "preprocess.pkl")
    
    # 4. Evaluate
    eval_config = load_yaml(Path("configs/eval_missingness.yaml"))
    results = evaluate_model(model, test_df, test_masks, prep_info, eval_config)
    save_yaml(results, run_dir / "results.yaml")
    
    # 5. Plot
    plot_degradation_curve(results, run_dir / "degradation.png")
    
    logger.info(f"Experiment finished. Results saved in {run_dir}")
活跃的
