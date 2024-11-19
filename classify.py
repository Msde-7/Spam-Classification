# IU CSCI-B 455, Fall 2024
# PRJ2: Spam detection
# File written by Gabe Shores 
#

import sys
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


"""
Vectorize email/text messages into numeric representations using CountVectorizer.
Args:
    messages (list): List of email/text messages.
    vectorizer (CountVectorizer, optional): Existing CountVectorizer for transformation.
Returns:
    sparse_matrix: Transformed text data.
    CountVectorizer: Fitted or existing CountVectorizer instance.
"""
def preprocess_data(messages, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()
        transformed_messages = vectorizer.fit_transform(messages)  # Fits and transforms the input text
    else:
        transformed_messages = vectorizer.transform(messages)  # Only transform using existing vectorizer

    return transformed_messages, vectorizer

"""
Train a classifier based on the specified type (SVM or Decision Tree).
Args:
    model_type (str): Type of classifier ('svm' or 'dt').
    X_train (sparse_matrix): Training feature matrix.
    y_train (array): Training labels.
Returns:
    classifier: Trained classifier.
"""
def train_classifier(model_type, X_train, y_train):
    # Added a random state to make it easier to reproduce results
    if model_type == 'svm':
        classifier = SVC(kernel='linear', random_state=42)
    elif model_type == 'dt':
        classifier = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Invalid classifier type. Use 'svm' or 'dt'.")
    classifier.fit(X_train, y_train)
    return classifier

"""
Save the predictions to an output file, appending a 'Predicted' column.
Args:
    input_file (str): Path to the input file.
    output_file (str): Path to the output file.
    predictions (list): Predicted labels.
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

    # File output/input standard
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

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"Training Accuracy: {train_accuracy:.2f}")

    if 'Category' in test_data.columns:
        y_test = test_data['Category']
        test_accuracy = accuracy_score(y_test, test_predictions)
        print(f"Testing Accuracy: {test_accuracy:.2f}")
        print("\nClassification Report on Test Data:")
        print(classification_report(y_test, test_predictions))


if __name__ == "__main__":
    main()
