import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

"""
Vectorize email into numeric representations using CountVectorizer.
Args:
    messages: Emails.
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
Train a classifier, svm, dt or rf.
Args:
    model_type: Type of classifier ('svm' or 'dt' or rf).
    X_train: Training feature matrix.
    y_train: Training labels.
Returns:
    Trained classifier.
"""
def train_classifier(model_type):
    if model_type == 'svm':
        classifier = SVC(
            kernel='linear',
            C=.1,
            random_state=20
        )
    elif model_type == 'dt':
        classifier = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=20,
            min_samples_split=2,
            random_state=20
        )
    elif model_type == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=100,
            random_state=20
        )
    else:
        raise ValueError("Invalid classifier type. Use 'svm', 'dt', or 'rf'.")
    return classifier


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_svm_dt.py model_type data_filename")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    if model_type not in ['svm', 'dt', 'rf']:
        print("Invalid model type. Use 'svm', 'dt', or 'rf'.")
        sys.exit(1)
    data_filename = sys.argv[2]

    data = pd.read_csv(data_filename)

    # Convert 'Category' to binary labels: 1 for 'spam', 0 for 'ham'
    data['spam'] = np.where(data['Category'] == 'spam', 1, 0)

    # Preprocess data
    X, vectorizer = preprocess_data(data['Message'])
    y = data['spam']

    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
    classifier = train_classifier(model_type)

    train_accuracies = []
    test_accuracies = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i + 1}:")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the classifier on the current fold
        classifier.fit(X_train, y_train)

        # Evaluate on train and test data
        train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
        test_accuracy = accuracy_score(y_test, classifier.predict(X_test))

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy:  {test_accuracy:.4f}\n")

    # Summary statistics
    mean_train_accuracy = np.mean(train_accuracies)
    mean_test_accuracy = np.mean(test_accuracies)
    std_test_accuracy = np.std(test_accuracies)

    print(f"\nFinal 5-Fold Cross-Validation Results:")
    print(f"Mean Train Accuracy: {mean_train_accuracy:.4f}")
    print(f"Mean Test Accuracy:  {mean_test_accuracy:.4f}")
    print(f"Test Accuracy Std Dev: {std_test_accuracy:.4f}")


if __name__ == "__main__":
    main()
