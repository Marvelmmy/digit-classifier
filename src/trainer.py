from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

def train_model(df, model_path="models/my_model.pkl"):
    X = df.drop(df[['target', 'target desc']], axis='columns')
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 random_state=42)
    model = SVC()
    model.fit(X_train, y_train)

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test
