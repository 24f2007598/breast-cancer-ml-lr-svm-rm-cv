from src.data_loader import load_data
from src.preprocessing import split_and_scale
from src.models import get_models
from src.evaluate import evaluate_model

import pandas as pd

def main():
    # Load data
    X, y = load_data()

    # Split and scale
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(X, y)

    models = get_models()

    results = []

    for name, model in models.items():
        if name == "Random Forest":
            acc, cm, cv = evaluate_model(
                model, X_train, y_train, X_test, y_test
            )
        else:
            acc, cm, cv = evaluate_model(
                model, X_train_scaled, y_train, X_test_scaled, y_test
            )

        results.append({
            "Model": name,
            "Accuracy": acc,
            "CV Accuracy": cv
        })

        print(f"\n{name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"CV Accuracy: {cv:.4f}")
        print("Confusion Matrix:")
        print(cm)

    results_df = pd.DataFrame(results)
    print("\nFinal Model Comparison:")
    print(results_df)

    # Save results
    results_df.to_csv("results/metrics.txt", index=False)

if __name__ == "__main__":
    main()
