# IU CSCI-B 455, Fall 2024
# PRJ2: Spam detection - Ensemble
# File written by Gabe Shores and Daniel Yang

import sys
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
def train_classifier(model_type, X_train, y_train):
    if model_type == 'svm':
        classifier =  SVC(kernel='linear', C=0.1, random_state=20)
    elif model_type == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=20)
    else:
        raise ValueError("Invalid classifier type. Use 'svm' or 'rf'.")
    classifier.fit(X_train, y_train)  # Fit the classifier to the training data
    return classifier


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
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return model


"""
Evaluate the model for given inputs, returning the predictions
"""
def evaluate_nn_model(model, X_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs > 0.5).float().squeeze()
    return predictions.numpy()


"""
Given the predictions determine the majority vote for each, and return it
"""
def majority_vote(predictions):
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)


"""
Save the predictions to an output file, appending a 'Predicted' column.
"""
def save_predictions(input_file, output_file, predictions):
    data = pd.read_csv(input_file)
    if len(data) != len(predictions):
        raise ValueError(f"Length of predictions ({len(predictions)}) does not match data ({len(data)}).")
    category_mapping = {1: 'spam', 0: 'ham'}
    data['Predicted'] = [category_mapping[pred] for pred in predictions]
    data.to_csv(output_file, index=False)


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 ensemble_classify.py train_filename test_filename train_output_filename test_output_filename")
        sys.exit(1)
    # File input/output paths as specified in file_io_standard.py
    INPUT_DIR = Path('./data').resolve()
    OUTPUT_DIR = Path('./').resolve()
    
    # Command-line arguments
    train_filename = INPUT_DIR / sys.argv[1]
    test_filename = INPUT_DIR / sys.argv[2]
    train_output_filename = OUTPUT_DIR / sys.argv[3]
    test_output_filename = OUTPUT_DIR / sys.argv[4]

    if not train_filename.exists() or not test_filename.exists():
        print("Error: One or more input files not found.")
        sys.exit(1)

    # Load and preprocess training data
    train_data = pd.read_csv(train_filename)
    train_data['Category'] = train_data['Category'].map({'spam': 1, 'ham': 0})
    X_train, vectorizer = preprocess_data(train_data['Message'])
    y_train = train_data['Category'].values

    # Load and preprocess test data using the same vectorizer
    test_data = pd.read_csv(test_filename)
    X_test = vectorizer.transform(test_data['Message'])

    # Train classifiers
    classifiers = {
        "SVM": train_classifier('svm', X_train, y_train),
        "Random Forest": train_classifier('rf', X_train, y_train),
    }

    # Train Neural Network
    X_train_tensor = torch.FloatTensor(X_train.toarray())
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    input_size = X_train_tensor.shape[1]
    nn_model = train_nn_model(X_train_tensor, y_train_tensor, input_size)

    # Generate predictions
    train_predictions = []
    test_predictions = []
    for clf_name, clf in classifiers.items():
        train_predictions.append(clf.predict(X_train))
        test_predictions.append(clf.predict(X_test))

    train_nn_predictions = evaluate_nn_model(nn_model, X_train_tensor)
    test_nn_predictions = evaluate_nn_model(nn_model, torch.FloatTensor(X_test.toarray()))
    train_predictions.append((train_nn_predictions > 0.5).astype(int))
    test_predictions.append((test_nn_predictions > 0.5).astype(int))

    # Ensemble predictions
    train_ensemble_predictions = majority_vote(np.array(train_predictions).T)
    test_ensemble_predictions = majority_vote(np.array(test_predictions).T)

    # Save predictions
    save_predictions(train_filename, train_output_filename, train_ensemble_predictions)
    save_predictions(test_filename, test_output_filename, test_ensemble_predictions)


if __name__ == "__main__":
    main()
