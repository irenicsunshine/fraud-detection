# =============================================================================
# 🎓 STEP 2.3: MAIN PIPELINE — The Orchestrator
# =============================================================================
#
# WHAT IS A PIPELINE ORCHESTRATOR?
# ----------------------------------
# Individual modules handle specific tasks:
#   - data_preprocessing → cleans data
#   - feature_engineering → creates features
#   - model_training → trains models
#   - model_evaluation → evaluates models
#   - business_insights → generates insights
#
# The orchestrator (this file) connects them in the right order,
# like a conductor leading an orchestra.
#
# 🎓 WHAT IS ARGPARSE?
# ----------------------
# argparse lets you control your script from the command line:
#
#   python main.py --mode train                    # Full training
#   python main.py --mode train --sample-frac 0.1  # Quick test with 10%
#   python main.py --mode evaluate                 # Evaluate existing models
#   python main.py --mode predict --input data.csv # Batch prediction
#
# This is much better than editing code to change behavior!
# =============================================================================

import argparse
import time
import sys
import logging
from pathlib import Path

from src.config import Config, setup_logging
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.business_insights import BusinessAnalyzer

logger = logging.getLogger("fraud_detection.main")


def train(config: Config) -> None:
    """
    Run the full training pipeline.
    
    🎓 THE TRAINING PIPELINE (in order):
    
    1. LOAD & CLEAN → raw CSV becomes a clean DataFrame
    2. ENGINEER FEATURES → 11 raw columns become 34+ engineered features
    3. ENCODE CATEGORICALS → "TRANSFER" becomes 4 (integer)
    4. TRAIN MODELS → 3 models trained with cross-validation
    5. EVALUATE → honest metrics on held-out test set
    6. GENERATE INSIGHTS → business-ready analysis
    
    Each step feeds into the next. If any step fails, we stop early
    with a clear error message (instead of crashing mysteriously).
    """
    pipeline_start = time.time()
    
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # --- Step 1: Load and clean data ---
    logger.info("\n📊 STEP 1/6: Loading and cleaning data...")
    step_start = time.time()
    
    preprocessor = DataPreprocessor(config)
    df = preprocessor.load_and_clean()
    
    logger.info(f"   ✓ Complete in {time.time() - step_start:.1f}s")
    
    # --- Step 2: Feature engineering ---
    logger.info("\n🔧 STEP 2/6: Engineering features...")
    step_start = time.time()
    
    engineer = FeatureEngineer(config)
    df_features = engineer.create_features(df)
    
    logger.info(f"   ✓ Complete in {time.time() - step_start:.1f}s")
    
    # --- Step 3: Encode categoricals ---
    logger.info("\n🔤 STEP 3/6: Encoding categorical features...")
    step_start = time.time()
    
    engineer.fit_encoder(df_features)
    df_encoded = engineer.encode_categoricals(df_features)

    # Save encoder so inference can use it without retraining
    import joblib
    encoder_path = config.paths.models_dir / "encoder.pkl"
    joblib.dump(engineer.encoder, encoder_path)
    logger.info(f"   Encoder saved to: {encoder_path}")

    logger.info(f"   ✓ Complete in {time.time() - step_start:.1f}s")
    
    # --- Step 4: Train models ---
    logger.info("\n🏋️ STEP 4/6: Training models...")
    step_start = time.time()
    
    trainer = ModelTrainer(config)
    models, X_test, y_test = trainer.train_pipeline(df_encoded)
    
    logger.info(f"   ✓ Complete in {time.time() - step_start:.1f}s")
    
    # --- Step 5: Evaluate models ---
    logger.info("\n📈 STEP 5/6: Evaluating models...")
    step_start = time.time()
    
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_models(models, X_test, y_test)
    
    logger.info(f"   ✓ Complete in {time.time() - step_start:.1f}s")
    
    # --- Step 6: Business insights ---
    logger.info("\n💼 STEP 6/6: Generating business insights...")
    step_start = time.time()
    
    analyzer = BusinessAnalyzer(config)
    best_model = evaluator.get_best_model_name()
    insights = analyzer.generate_insights(
        results, best_model, preprocessor.get_stats()
    )
    
    logger.info(f"   ✓ Complete in {time.time() - step_start:.1f}s")
    
    # --- Summary ---
    total_time = time.time() - pipeline_start
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"{'='*60}")
    logger.info(f"Champion model: {best_model}")
    logger.info(f"Models saved to: {config.paths.models_dir}")
    logger.info(f"Charts saved to: {config.paths.results_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Start the API:  python api.py")
    logger.info(f"  2. View docs:      http://localhost:8000/docs")
    logger.info(f"  3. Run tests:      pytest tests/ -v")


def evaluate(config: Config) -> None:
    """
    Evaluate existing trained models (without retraining).
    
    Useful when you want to re-evaluate with different thresholds
    or cost assumptions without waiting for training.
    """
    import joblib
    
    logger.info("Loading pre-trained models for evaluation...")
    
    models_path = config.paths.models_dir / "trained_models.pkl"
    if not models_path.exists():
        logger.error(f"No trained models found at {models_path}")
        logger.info("Run 'python main.py --mode train' first")
        return
    
    # Note: This requires the test set to be available
    # In a full implementation, you'd save X_test/y_test alongside the models
    logger.info("Evaluation of existing models requires the dataset to be loaded.")
    logger.info("Running full pipeline with evaluation focus...")
    train(config)


def predict(config: Config, input_file: str) -> None:
    """
    Run batch predictions on a CSV file.
    """
    import pandas as pd
    from src.inference import FraudPredictor
    
    logger.info(f"Loading transactions from: {input_file}")
    
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} transactions")
    
    predictor = FraudPredictor(model_name="xgboost", config=config)
    results = predictor.predict_batch(df)
    
    output_path = input_path.parent / f"{input_path.stem}_predictions.csv"
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")
    
    # Summary
    n_fraud = (results["prediction"] == "fraud").sum()
    logger.info(f"\nSummary: {n_fraud:,} fraud detected in {len(results):,} transactions")


def serve(config: Config, port: int = 8000) -> None:
    """
    Start the API server.
    """
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn fastapi")
        return
    
    logger.info(f"Starting API server on port {port}...")
    logger.info(f"API docs: http://localhost:{port}/docs")
    
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)


def main():
    """
    Main entry point with argument parsing.
    
    🎓 ARGUMENT PARSING
    ---------------------
    argparse creates a command-line interface for your script:
    
    Usage:
      python main.py --mode train                    # Full training pipeline
      python main.py --mode train --sample-frac 0.1  # Quick 10% test
      python main.py --mode predict --input data.csv # Batch prediction
      python main.py --mode serve --port 8000        # Start API
      python main.py --help                          # Show all options
    """
    parser = argparse.ArgumentParser(
        description="Fraud Detection ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train                     Full training pipeline
  python main.py --mode train --sample-frac 0.1   Quick test with 10% data
  python main.py --mode predict --input data.csv  Batch prediction
  python main.py --mode serve --port 8000         Start API server
        """,
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict", "serve"],
        default="train",
        help="Pipeline mode (default: train)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to use (0.1 = 10%%, for quick testing)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file for prediction mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup
    config = Config()
    config.training.sample_fraction = args.sample_frac
    config.log_level = args.log_level
    setup_logging(args.log_level)
    
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Sample fraction: {args.sample_frac}")
    
    # Dispatch to the appropriate function
    if args.mode == "train":
        train(config)
    elif args.mode == "evaluate":
        evaluate(config)
    elif args.mode == "predict":
        if not args.input:
            logger.error("--input is required for predict mode")
            parser.print_help()
            sys.exit(1)
        predict(config, args.input)
    elif args.mode == "serve":
        serve(config, args.port)


if __name__ == "__main__":
    main()
