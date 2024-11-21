# IU CSCI-B 455, Fall 2024
# PRJ2: Spam detection
# File written by Gabe Shores and Daniel Yang
# Tests the nn with KFold Cross Validation, printing the results
#

import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
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

def train_model(model, train_loader, criterion, optimizer, epochs=12):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

def compute_pos_weight(y):
    # Compute the ratio of negative to positive samples
    pos_count = sum(y)
    neg_count = len(y) - pos_count
    pos_weight = neg_count / pos_count
    return torch.tensor(pos_weight, dtype=torch.float)

def main():
    if len(sys.argv) != 2:
        print("Usage: python spam_nn.py <data_file>")
        sys.exit(1)

    # Load data
    data_file = sys.argv[1]
    data = pd.read_csv(data_file)

    # Split data into training and validation sets
    X = data['Message']
    y = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    X_train_transformed, vectorizer = preprocess_data(X_train)
    X_val_transformed = preprocess_data(X_val, vectorizer=vectorizer)[0]
    
    pos_weight = compute_pos_weight(y_train)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_transformed.toarray())
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val_transformed.toarray())
    y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model, loss function, and optimizer
    input_size = X_train_transformed.shape[1]
    model = SpamDetectionNN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=30)

    # Evaluate the model
    train_accuracy = evaluate_model(model, train_loader)
    val_accuracy = evaluate_model(model, val_loader)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()