
import csv
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Test size for splitting the dataset
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data and split into training and testing sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, random_state=42
    )

    # Train the model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate the model
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate (Sensitivity): {100 * sensitivity:.2f}%")
    print(f"True Negative Rate (Specificity): {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    month_mapping = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    evidence = []
    labels = []
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        evidence.append([
            row["Administrative"], row["Administrative_Duration"],
            row["Informational"], row["Informational_Duration"],
            row["ProductRelated"], row["ProductRelated_Duration"],
            row["BounceRates"], row["ExitRates"], row["PageValues"],
            row["SpecialDay"], month_mapping[row["Month"]],
            row["OperatingSystems"], row["Browser"], row["Region"],
            row["TrafficType"], 1 if row["VisitorType"] == "Returning_Visitor" else 0,
            1 if row["Weekend"] else 0
        ])
        labels.append(1 if row["Revenue"] else 0)
    return evidence, labels

def train_model(X_train, y_train):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positive = sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 1)
    true_negative = sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 0)
    total_positive = sum(1 for actual in labels if actual == 1)
    total_negative = sum(1 for actual in labels if actual == 0)
    sensitivity = true_positive / total_positive if total_positive > 0 else 0
    specificity = true_negative / total_negative if total_negative > 0 else 0
    return sensitivity, specificity

if __name__ == "__main__":
    main()
