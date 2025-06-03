import argparse
from pathlib import Path

from src.eda import run_eda
from src.logistic import run_logistic
from src.roc_curve import run_roc
from src.decision_tree import run_decision_tree_regressor
from src.decision_tree_gini import run_decision_tree_gini
from src.decision_tree_entropy import run_decision_tree_entropy
from src.regression import run_regression
from src.overfitting_analysis import run_overfitting_analysis
from src.feature_importance import run_feature_importance


def parse_args():
    parser = argparse.ArgumentParser(description="Run selected stages of the ML pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "eda", "logistic", "roc", "tree", "tree_gini", "tree_entropy",
            "regression", "overfit", "importance", "term", "all"
        ],
        help="Select specific stage to run or 'all'"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Path("outputs").mkdir(exist_ok=True)

    if args.stage in ("eda", "all"):
        run_eda()

    if args.stage in ("logistic", "all"):
        run_logistic()

    if args.stage in ("roc", "all"):
        run_roc()

    if args.stage in ("tree", "all"):
        run_decision_tree_regressor()

    if args.stage in ("tree_gini", "all"):
        run_decision_tree_gini()

    if args.stage in ("tree_entropy", "all"):
        run_decision_tree_entropy()

    if args.stage in ("regression", "all"):
        run_regression()

    if args.stage in ("overfit", "all"):
        run_overfitting_analysis()

    if args.stage in ("importance", "all"):
        run_feature_importance()

    print("\n✅ Pipeline execution completed.")


if __name__ == "__main__":
    main()
