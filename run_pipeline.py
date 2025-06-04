import argparse
from pathlib import Path

from preprocessing.preprocessing import run_preprocessing
from eda.eda import run_eda
from regression.decision_tree import run_decision_tree_regression
from regression.feature_importance import run_feature_importance
from regression.overfitting_analysis import run_overfitting_analysis
from regression.regression import run_regression
from classification.decision_tree_gini import run_decision_tree_gini
from classification.decision_tree_entropy import run_decision_tree_entropy
from classification.logistic import run_logistic
from classification.roc_curve import run_roc

def parse_args():
    parser = argparse.ArgumentParser(description="Run selected stages of the ML pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "preprocessing", "eda", "logistic", "roc", "tree", "tree_gini", "tree_entropy",
            "regression", "overfit", "importance", "term", "all"
        ],
        help="Select specific stage to run or 'all'"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Path("outputs").mkdir(exist_ok=True)

    if args.stage in ("preprocessing", "all"):
        run_preprocessing()
        print("Step 1: Data preprocessing completed")

    if args.stage in ("eda", "all"):
        run_eda()
        print("Step 2: EDA completed")


    if args.stage in ("tree", "all"):
        run_decision_tree_regression()
        print("Step 3: Regression Decision Tree completed")

    if args.stage in ("importance", "all"):
        run_feature_importance()
        print("Step 4: Regression Feature Importance completed")

    if args.stage in ("overfit", "all"):
        run_overfitting_analysis()
        print("Step 5: Regression Overfitting Analysis completed")

    if args.stage in ("regression", "all"):
        run_regression()
        print("Step 6: Regression completed")

    if args.stage in ("tree_gini", "all"):
        run_decision_tree_gini()
        print("Step 7: Classification Decision Tree Gini completed")

    if args.stage in ("tree_entropy", "all"):
        run_decision_tree_entropy()
        print("Step 8: Classification Decision Tree Entropy completed")

    if args.stage in ("logistic", "all"):
        run_logistic()
        print("Step 9: Classification Logistic completed")

    if args.stage in ("roc", "all"):
        run_roc()
        print("Step 10: Classification Roc Curve completed")

    print("\nPipeline execution completed.")


if __name__ == "__main__":
    main()
