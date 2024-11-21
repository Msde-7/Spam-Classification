# IU CSCI-B 455, Fall 2024
# PRJ2: Spam detection
# File written by Gabe Shores and Daniel Yang
#

import sys
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


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
Train a classifier, svm or dt.
Args:
    model_type: Type of classifier ('svm' or 'dt').
    X_train: Training feature matrix.
    y_train: Training labels.
Returns:
    Trained classifier.
"""
def train_classifier(model_type, X_train, y_train):
    # Added a random state to make it easier to reproduce results
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
    else:
        raise ValueError("Invalid classifier type. Use 'svm' or 'dt'!!!")
    classifier.fit(X_train, y_train)  # Fit the classifier to the training data
    return classifier

"""
Save the predictions to an output file, appending a 'Predicted' column.
"""
def save_predictions(input_file, output_file, predictions):
    data = pd.read_csv(input_file)
    if len(data) != len(predictions):
        raise ValueError(f"Length of predictions ({len(predictions)}) does not match data ({len(data)}).")
    data['Predicted'] = predictions
    data.to_csv(output_file, index=False)

"""
Main!
"""
def main():  
    if len(sys.argv) != 6:
        print("Usage: python classify.py svm_or_dt train_filename test_filename train_output_filename test_output_filename")
        sys.exit(1)

    # File input/output paths as specified in file_io_standard.py
    INPUT_DIR = Path('./data').resolve()
    OUTPUT_DIR = Path('./').resolve()

    # Command-line arguments
    model_type = sys.argv[1]  # 'svm' or 'dt'
    train_filename = INPUT_DIR / sys.argv[2]
    test_filename = INPUT_DIR / sys.argv[3]
    train_output_filename = OUTPUT_DIR / sys.argv[4]
    test_output_filename = OUTPUT_DIR / sys.argv[5]

    # Load training and test data
    train_data = pd.read_csv(train_filename)
    test_data = pd.read_csv(test_filename)

    # Preprocess the data
    X_train, vectorizer = preprocess_data(train_data['Message'])
    y_train = train_data['Category']
    X_test = preprocess_data(test_data['Message'], vectorizer=vectorizer)[0]

    # Train the specified classifier
    classifier = train_classifier(model_type, X_train, y_train)

    # Make predictions
    train_predictions = classifier.predict(X_train)
    test_predictions = classifier.predict(X_test)

    # Save predictions to output files
    save_predictions(train_filename, train_output_filename, train_predictions)
    save_predictions(test_filename, test_output_filename, test_predictions)

if __name__ == "__main__":
    main()
