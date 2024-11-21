# IU CSCI-B 455, Fall 2024
# PRJ2: Spam detection
# File written by Gabe Shores and Daniel Yang
# Tests the ensembles with KFold Cross Validation, printing the results
#
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


"""
Constructs  rather simple neural network with Layers input_size -> 128 -> 64 -> 1
"""
class SpamDetectionNN(nn.Module):
    def __init__(self, input_size):
        super(SpamDetectionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


"""
Vectorize email into numeric representations using CountVectorizer.
Args:
    emails: Emails.
    vectorizer: Existing CountVectorizer for transformation.
Returns:
    Transformed text data.
    Fitted or existing CountVectorizer instance.
"""
def preprocess_data(emails, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()
        transformed_emails = vectorizer.fit_transform(emails)
    else:
        transformed_emails = vectorizer.transform(emails)
    return transformed_emails, vectorizer

"""
Train a classifier, svm, rf.
Args:
    model_type: Type of classifier ('svm' or 'rf').
    X_train: Training feature matrix.
    y_train: Training labels.
Returns:
    Trained classifier.
"""
def train_classifier(model_type):
    if model_type == 'svm':
        return SVC(kernel='linear', C=0.1, random_state=20)
    elif model_type == 'rf':
        return RandomForestClassifier(n_estimators=100, max_depth=100, random_state=20)
    else:
        raise ValueError("Invalid classifier type. Use 'svm' or 'rf'.")

"""
Evaluate the model for given inputs, returning the predictions
"""
def train_nn_model(X_train, y_train, input_size, epochs=12):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SpamDetectionNN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model

"""
Evaluate the model for given inputs, returning the predictions
"""
def evaluate_nn_model(model, X_val_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor)
        predictions = (outputs > 0.5).float().squeeze()
    return predictions.numpy()

"""
Given the predictions determine the majority vote for each, and return it
"""
def majority_vote(predictions):
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)


def main():
    if len(sys.argv) != 2:
        print("Usage: python ensemble_classify.py data_filename")
        sys.exit(1)

    data_filename = sys.argv[1]
    data = pd.read_csv(data_filename)

    # Convert 'Category' to binary labels: 1 for 'spam', 0 for 'ham'
    data['spam'] = np.where(data['Category'] == 'spam', 1, 0)

    # Preprocess data
    X, vectorizer = preprocess_data(data['Message'])
    y = data['spam'].values

    # Prepare k-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
    ensemble_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold+1}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Convert data for PyTorch
        X_train_tensor = torch.FloatTensor(X_train.toarray())
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test.toarray())

        # Train models
        classifiers = {
            "SVM": train_classifier('svm'),
            "Random Forest": train_classifier('rf'),
        }

        predictions = []
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            predictions.append(preds)
            acc = accuracy_score(y_test, preds)
            print(f"  {name} Accuracy: {acc:.4f}")

        # Train Neural Network
        input_size = X_train_tensor.shape[1]
        nn_model = train_nn_model(X_train_tensor, y_train_tensor, input_size, epochs=12)
        nn_predictions = evaluate_nn_model(nn_model, X_test_tensor)
        nn_predictions = (nn_predictions > 0.5).astype(int)  # Convert probabilities to binary
        predictions.append(nn_predictions)

        nn_accuracy = accuracy_score(y_test, nn_predictions)
        print(f"  Neural Network Accuracy: {nn_accuracy:.4f}")

        # Ensemble majority voting
        predictions = np.array(predictions).T
        ensemble_predictions = majority_vote(predictions)

        # Evaluate ensemble model
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        ensemble_accuracies.append(ensemble_accuracy)
        print(f"  Ensemble Accuracy: {ensemble_accuracy:.4f}")

    # Final k-fold ensemble accuracy
    print("\nFinal Ensemble Accuracy (Mean ± Std):")
    print(f"{np.mean(ensemble_accuracies):.4f} ± {np.std(ensemble_accuracies):.4f}")


if __name__ == "__main__":
    main()
