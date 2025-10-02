from src.data_loader import load_data
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.predictor import load_model, predict_model

def main():
    print("=== Digit Classifier CLI ===")
    digits, df = load_data()

    print("Training Model...")
    model, X_test, y_test = train_model(df)

    print("Evaluating model...")
    acc, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.2f}")
    print(report)

    print("Testing single prediction...")
    sample = X_test.iloc[0]
    pred = predict_single(model, sample)
    print(f"Predicted: {pred}, Actual: {y_test.iloc[0]}")

if __name__ == "__main__":
    main()