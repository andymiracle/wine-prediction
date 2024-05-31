import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import seaborn as sb
import argparse

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['logistic', 'softmax', 'random_forest', 'xgboost'])
    parser.add_argument('--version', '-v', choices=['binary', 'multi'])
    return parser.parse_args()

def count_quality(wine_data):
    quality_counts = wine_data['quality'].value_counts().sort_index()

    plt.bar(quality_counts.index, quality_counts.values)

    for i, count in enumerate(quality_counts.values):
        plt.text(quality_counts.index[i], count, str(count), ha='center', va='bottom')

    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.title('Distribution of Quality')
    plt.show()

def count_label(y):
    quality_counts = y.value_counts().sort_index()

    plt.bar(quality_counts.index, quality_counts.values)

    for i, count in enumerate(quality_counts.values):
        plt.text(quality_counts.index[i], count, str(count), ha='center', va='bottom')

    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.title('Distribution of Quality')
    plt.show()

def get_quality_counts(X, y, set_name):
    quality_counts = pd.Series(y).value_counts().sort_index()
    return pd.DataFrame({set_name: quality_counts})

def show_distribution_labels(X_train, X_dev, X_test, y_train, y_dev, y_test):
    train_counts = get_quality_counts(X_train, y_train, 'Training Set')
    dev_counts = get_quality_counts(X_dev, y_dev, 'Development Set')
    test_counts = get_quality_counts(X_test, y_test, 'Testing Set')
    combined_counts = pd.concat([train_counts, dev_counts, test_counts], axis=1)

    plot = combined_counts.plot(kind='bar')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.title('Quality Counts for Different Sets')
    plt.legend(title='Set')

    for i in plot.patches:
        plot.text(i.get_x() + i.get_width() / 2, i.get_height(), str(int(i.get_height())), ha='center', va='bottom')

    plt.show()

def analyze_pattern(dev_predictions, y_dev, X_dev):
    correct_indices = [i for i in range(len(y_dev)) if dev_predictions[i] == y_dev.iloc[i]]
    incorrect_indices = [i for i in range(len(y_dev)) if dev_predictions[i] != y_dev.iloc[i]]

    # Randomly select a subset of incorrect predictions for manual inspection
    random.seed(42)  # Set random seed for reproducibility
    num_samples = 10  # Number of examples to inspect
    incorrect_sample_indices = random.sample(incorrect_indices, min(num_samples, len(incorrect_indices)))

    random.seed(42)  # Set random seed for reproducibility
    num_samples = 10  # Number of examples to inspect
    correct_sample_indices = random.sample(correct_indices, min(num_samples, len(correct_indices)))

    
    for idx in correct_sample_indices:
        print("Example Index:", idx)
        print("Predicted Label:", dev_predictions[idx])
        print("True Label:", y_dev.iloc[idx])
        formatted_features = ['{:.8f}'.format(x) for x in X_dev[idx]]
        print("Features:", formatted_features)
        print()
    
    
    for idx in incorrect_sample_indices:
        print("Example Index:", idx)
        print("Predicted Label:", dev_predictions[idx])
        print("True Label:", y_dev.iloc[idx])
        formatted_features = ['{:.8f}'.format(x) for x in X_dev[idx]]
        print("Features:", formatted_features)
        print()
 

def f1_score_wrapper(*args, **kwargs):
    return f1_score(*args, **kwargs, average="weighted")

def logistic_regression(wine_data):
    #1. Convert the label of wine from 0 ~ 10, to either 0 or 1 (for the sake of binary classification)
    wine_data['best quality'] = [1 if x > 5 else 0 for x in wine_data.quality]

    #2. Extract X (feature vector) and y (best quality (0 or 1))
    X = wine_data.drop(columns=['quality', 'best quality'], axis=1)
    y = wine_data['best quality']

    #3. Split data into training, development, and testing set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

    #4. Show the distribution of labels (quality) to respectively training, development, testing set
    #show_distribution_labels(X_train, X_dev, X_test, y_train, y_dev, y_test)

    #5. Normalize the data before training to achieve stable and fast training of the model
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    X_dev = norm.transform(X_dev)
    X_test = norm.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #6. List the hyperparameter to tune (C = inverse of regularization strength)
    hyperparameters = [
        {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}, 
        {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}, 
        {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}, 
        {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs'}, 
        {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}, 
        {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'},

        {'C': 0.001, 'penalty': 'l1', 'solver': 'saga'}, 
        {'C': 0.01, 'penalty': 'l1', 'solver': 'saga'}, 
        {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'}, 
        {'C': 1, 'penalty': 'l1', 'solver': 'saga'}, 
        {'C': 10, 'penalty': 'l1', 'solver': 'saga'}, 
        {'C': 100, 'penalty': 'l1', 'solver': 'saga'},
    ]

    #7. Training (with Logistic Regression)
    best_accuracy = 0
    best_hyperparameters = None 

    for params in hyperparameters:
        # (1) Create a logistic regression model with the current hyperparameter
        logistic_regression_model = LogisticRegression(**params)
        
        # (2) Train the model on the training set
        logistic_regression_model.fit(X_train, y_train)
        
        # (3) Make predictions on the development set
        dev_predictions = logistic_regression_model.predict(X_dev)

        # (4) Analyze the result
        #analyze_pattern(dev_predictions, y_dev, X_dev)

        # (5) Calculate accuracy on the development set
        dev_accuracy = f1_score(y_dev, dev_predictions, average="weighted")

        print("Hyperparameter is ", params)
        print("Development set accuracy: ", dev_accuracy)
        
        # (6) Check if the current hyperparameters lead to a better accuracy
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_hyperparameters = params

    #8. Train the best model on the combined training and development sets
    best_logistic_regression_model = LogisticRegression(**best_hyperparameters)
    best_logistic_regression_model.fit(np.concatenate([X_train, X_dev]), np.concatenate([y_train, y_dev]))

    print()
    print("Best hyperparameter is ", best_hyperparameters)

    #9. Make predictions on the testing set using the best model
    test_predictions = best_logistic_regression_model.predict(X_test)

    #analyze_pattern(test_predictions, y_test, X_test)

    #10. Calculate accuracy on the testing set
    test_accuracy = f1_score(y_test, test_predictions, average="weighted")
    print("Testing set accuracy with best model:", test_accuracy)

def softmax_regression(wine_data):
    #1. Extract X (feature vector) and y (quality of wine that ranges from 0 to 10))
    X = wine_data.drop(columns=['quality'], axis=1)
    y = wine_data['quality']

    #2. Split data into training, development, and testing set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

    #3. Normalize the data before training to achieve stable and fast training of the model
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    X_dev = norm.transform(X_dev)
    X_test = norm.transform(X_test)

    #4. Use SMOTE (oversampling technique) to resample the data as the original data is very unbalanced
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #6. List the hyperparameter to tune (C = inverse of regularization strength, max_iter = maximum number of iterations taken for the solvers to converge)
    hyperparameters = [
        {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}, 
        {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}, 
        {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}, 
        {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}, 
        {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}, 
        {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000},

        {'C': 0.001, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000}, 
        {'C': 0.01, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000}, 
        {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000}, 
        {'C': 1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000}, 
        {'C': 10, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000}, 
        {'C': 100, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000},
    ]

    #7. Training (with Softmax Regression)
    best_accuracy = 0
    best_hyperparameters = None 

    for params in hyperparameters: 
        # (1) Create a softmax regression model with the current hyperparameter
        softmax_regression_model = LogisticRegression(**params, multi_class='multinomial')
        
        # (2) Train the model on the training set
        softmax_regression_model.fit(X_train, y_train)

        # (3) Make predictions on the development set
        dev_predictions = softmax_regression_model.predict(X_dev)

        # (4) Analyze the result
        #analyze_pattern(dev_predictions, y_dev, X_dev)
        
        # (5) Calculate accuracy on the development set
        dev_accuracy = f1_score(y_dev, dev_predictions, average="weighted")

        print("Hyperparameter is ", params)
        print("Development set accuracy: ", dev_accuracy)
        
        # (6) Check if the current hyperparameters lead to a better accuracy
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_hyperparameters = params

    best_softmax_regression_model = LogisticRegression(**best_hyperparameters, multi_class='multinomial')
    best_softmax_regression_model.fit(np.concatenate([X_train, X_dev]), np.concatenate([y_train, y_dev]))

    print()
    print("Best hyperparameter is ", best_hyperparameters)

    # Make predictions on the testing set using the best model
    test_predictions = best_softmax_regression_model.predict(X_test)

    #analyze_pattern(test_predictions, y_test, X_test)

    # Calculate accuracy on the testing set
    test_accuracy = f1_score(y_test, test_predictions, average="weighted")
    print("Testing set accuracy with best model:", test_accuracy)

def random_forest_binary(wine_data):
    #1. Extract X (feature vector) and y (quality of wine that ranges from 0 to 10))
    wine_data['best quality'] = [1 if x > 5 else 0 for x in wine_data.quality]

     #2. Extract X (feature vector) and y (best quality (0 or 1))
    X = wine_data.drop(columns=['quality', 'best quality'], axis=1)
    y = wine_data['best quality']

    #2. Split data into training, development, and testing set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

    #3. Normalize the data before training to achieve stable and fast training of the model
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    X_dev = norm.transform(X_dev)
    X_test = norm.transform(X_test)

    #4. Use SMOTE (oversampling technique) to resample the data as the original data is very unbalanced
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #5. List the hyperparameter to tune
    hyperparameters = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 150, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 5},
        {'n_estimators': 250, 'max_depth': 5},
        {'n_estimators': 300, 'max_depth': 5},

        {'n_estimators': 50, 'max_depth': 6},
        {'n_estimators': 100, 'max_depth': 6},
        {'n_estimators': 150, 'max_depth': 6},
        {'n_estimators': 200, 'max_depth': 6},
        {'n_estimators': 250, 'max_depth': 6},
        {'n_estimators': 300, 'max_depth': 6},

        {'n_estimators': 50, 'max_depth': 7},
        {'n_estimators': 100, 'max_depth': 7},
        {'n_estimators': 150, 'max_depth': 7},
        {'n_estimators': 200, 'max_depth': 7},
        {'n_estimators': 250, 'max_depth': 7},
        {'n_estimators': 300, 'max_depth': 7},

        {'n_estimators': 50, 'max_depth': 8},
        {'n_estimators': 100, 'max_depth': 8},
        {'n_estimators': 150, 'max_depth': 8},
        {'n_estimators': 200, 'max_depth': 8},
        {'n_estimators': 250, 'max_depth': 8},
        {'n_estimators': 300, 'max_depth': 8},

        {'n_estimators': 50, 'max_depth': 9},
        {'n_estimators': 100, 'max_depth': 9},
        {'n_estimators': 150, 'max_depth': 9},
        {'n_estimators': 200, 'max_depth': 9},
        {'n_estimators': 250, 'max_depth': 9},
        {'n_estimators': 300, 'max_depth': 9},
        
        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 150, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 10},
        {'n_estimators': 250, 'max_depth': 10},
        {'n_estimators': 300, 'max_depth': 10},
    ]

    #6. Training (with Random Forest)
    best_accuracy = 0
    best_hyperparameters = None 

    for params in hyperparameters:
        # (1) Create a random forest model with the current hyperparameter
        random_forest_model = RandomForestClassifier(**params, random_state = 42)
        
        # (2) Train the model on the training set
        random_forest_model.fit(X_train, y_train)
        
        # (3) Make predictions on the development set
        dev_predictions = random_forest_model.predict(X_dev)

        # (4) Analyze the result
        #analyze_pattern(dev_predictions, y_dev, X_dev)
        
        # (5) Calculate accuracy on the development set
        dev_accuracy = f1_score(y_dev, dev_predictions, average="weighted")

        print("Hyperparameter is ", params)
        print("Development set accuracy: ", dev_accuracy)
        
        # (6) Check if the current hyperparameters lead to a better accuracy
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_hyperparameters = params

    # Train the best model on the combined training and development sets
    best_random_forest_model = RandomForestClassifier(**best_hyperparameters, random_state=42)
    best_random_forest_model.fit(np.concatenate([X_train, X_dev]), np.concatenate([y_train, y_dev]))

    # Make predictions on the testing set using the best model
    test_predictions = best_random_forest_model.predict(X_test)

    print()
    print("Best hyperparameter is ", best_hyperparameters)

    #analyze_pattern(test_predictions, y_test, X_test)

    # Calculate accuracy on the testing set
    test_accuracy = f1_score(y_test, test_predictions, average="weighted")
    print("Testing set accuracy with best model:", test_accuracy)

def random_forest_multi(wine_data):
    #1. Extract X (feature vector) and y (quality of wine that ranges from 0 to 10))
    X = wine_data.drop(columns=['quality'], axis=1)
    y = wine_data['quality']

    #2. Split data into training, development, and testing set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

    #3. Normalize the data before training to achieve stable and fast training of the model
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    X_dev = norm.transform(X_dev)
    X_test = norm.transform(X_test)

    #4. Use SMOTE (oversampling technique) to resample the data as the original data is very unbalanced
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #5. List the hyperparameter to tune
    hyperparameters = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 150, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 5},
        {'n_estimators': 250, 'max_depth': 5},
        {'n_estimators': 300, 'max_depth': 5},
       
        {'n_estimators': 50, 'max_depth': 6},
        {'n_estimators': 100, 'max_depth': 6},
        {'n_estimators': 150, 'max_depth': 6},
        {'n_estimators': 200, 'max_depth': 6},
        {'n_estimators': 250, 'max_depth': 6},
        {'n_estimators': 300, 'max_depth': 6},

        {'n_estimators': 50, 'max_depth': 7},
        {'n_estimators': 100, 'max_depth': 7},
        {'n_estimators': 150, 'max_depth': 7},
        {'n_estimators': 200, 'max_depth': 7},
        {'n_estimators': 250, 'max_depth': 7},
        {'n_estimators': 300, 'max_depth': 7},

        {'n_estimators': 50, 'max_depth': 8},
        {'n_estimators': 100, 'max_depth': 8},
        {'n_estimators': 150, 'max_depth': 8},
        {'n_estimators': 200, 'max_depth': 8},
        {'n_estimators': 250, 'max_depth': 8},
        {'n_estimators': 300, 'max_depth': 8},

        {'n_estimators': 50, 'max_depth': 9},
        {'n_estimators': 100, 'max_depth': 9},
        {'n_estimators': 150, 'max_depth': 9},
        {'n_estimators': 200, 'max_depth': 9},
        {'n_estimators': 250, 'max_depth': 9},
        {'n_estimators': 300, 'max_depth': 9},

        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 150, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 10},
        {'n_estimators': 250, 'max_depth': 10},
        {'n_estimators': 300, 'max_depth': 10},
    ]

    #6. Training (with Random Forest)
    best_accuracy = 0
    best_hyperparameters = None 

    for params in hyperparameters:
        # (1) Create a random forest model with the current hyperparameter
        random_forest_model = RandomForestClassifier(**params, random_state = 42)
        
        # (2) Train the model on the training set
        random_forest_model.fit(X_train, y_train)
        
        # (3) Make predictions on the development set
        dev_predictions = random_forest_model.predict(X_dev)

        # (4) Analyze the result
        #analyze_pattern(dev_predictions, y_dev, X_dev)
        
        # (5) Calculate accuracy on the development set
        dev_accuracy = f1_score(y_dev, dev_predictions, average="weighted")

        print("Hyperparameter is ", params)
        print("Development set accuracy: ", dev_accuracy)
        
        # (6) Check if the current hyperparameters lead to a better accuracy
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_hyperparameters = params

    # Train the best model on the combined training and development sets
    best_random_forest_model = RandomForestClassifier(**best_hyperparameters, random_state=42)
    best_random_forest_model.fit(np.concatenate([X_train, X_dev]), np.concatenate([y_train, y_dev]))

    # Make predictions on the testing set using the best model
    test_predictions = best_random_forest_model.predict(X_test)

    #analyze_pattern(test_predictions, y_test, X_test)

    print()
    print("Best hyperparameter is ", best_hyperparameters)

    # Calculate accuracy on the testing set
    test_accuracy = f1_score(y_test, test_predictions, average="weighted")
    print("Testing set accuracy with best model:", test_accuracy)

def xgboost_learning_binary(wine_data):
    #1. Extract X (feature vector) and y (quality of wine that ranges from 0 to 10))
    wine_data['best quality'] = [1 if x > 5 else 0 for x in wine_data.quality]
    X = wine_data.drop(columns=['quality', 'best quality'], axis=1)
    y = wine_data['best quality']

    #2. Split data into training, development, and testing set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

    #3. Normalize the data before training to achieve stable and fast training of the model
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    X_dev = norm.transform(X_dev)
    X_test = norm.transform(X_test)

    #4. Use SMOTE (oversampling technique) to resample the data as the original data is very unbalanced
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #6. List the hyperparameter to tune
    hyperparameters = [
        {'learning_rate': 0.001, 'max_depth': 5}, 
        {'learning_rate': 0.01, 'max_depth': 5}, 
        {'learning_rate': 0.1, 'max_depth': 5}, 
        {'learning_rate': 1, 'max_depth': 5}, 
        {'learning_rate': 10, 'max_depth': 5}, 
        {'learning_rate': 100, 'max_depth': 5},

        {'learning_rate': 0.001, 'max_depth': 6}, 
        {'learning_rate': 0.01, 'max_depth': 6}, 
        {'learning_rate': 0.1, 'max_depth': 6}, 
        {'learning_rate': 1, 'max_depth': 6}, 
        {'learning_rate': 10, 'max_depth': 6}, 
        {'learning_rate': 100, 'max_depth': 6},

        {'learning_rate': 0.001, 'max_depth': 7}, 
        {'learning_rate': 0.01, 'max_depth': 7}, 
        {'learning_rate': 0.1, 'max_depth': 7}, 
        {'learning_rate': 1, 'max_depth': 7}, 
        {'learning_rate': 10, 'max_depth': 7}, 
        {'learning_rate': 100, 'max_depth': 7},
        
        {'learning_rate': 0.001, 'max_depth': 8}, 
        {'learning_rate': 0.01, 'max_depth': 8}, 
        {'learning_rate': 0.1, 'max_depth': 8}, 
        {'learning_rate': 1, 'max_depth': 8}, 
        {'learning_rate': 10, 'max_depth': 8}, 
        {'learning_rate': 100, 'max_depth': 8}, 

        {'learning_rate': 0.001, 'max_depth': 9}, 
        {'learning_rate': 0.01, 'max_depth': 9}, 
        {'learning_rate': 0.1, 'max_depth': 9}, 
        {'learning_rate': 1, 'max_depth': 9}, 
        {'learning_rate': 10, 'max_depth': 9}, 
        {'learning_rate': 100, 'max_depth': 9},
        
        {'learning_rate': 0.001, 'max_depth': 10}, 
        {'learning_rate': 0.01, 'max_depth': 10}, 
        {'learning_rate': 0.1, 'max_depth': 10}, 
        {'learning_rate': 1, 'max_depth': 10}, 
        {'learning_rate': 10, 'max_depth': 10}, 
        {'learning_rate': 100, 'max_depth': 10}, 
    ]

    #7. Training (with XGBoost)
    best_accuracy = 0
    best_hyperparameters = None 
        
    for params in hyperparameters: 
        # (1) Create a xgb_classifier model with the current hyperparameter
        xgb_classifier = XGBClassifier(**params, eval_metric=f1_score_wrapper)
    
        # (2) Train the model on the training set
        xgb_classifier.fit(X_train, y_train)

        # (3) Make predictions on the development set
        dev_predictions = xgb_classifier.predict(X_dev)

        # (4) Analyze the result
        #analyze_pattern(dev_predictions, y_dev, X_dev)

        # (5) Calculate accuracy on the development set
        dev_accuracy = f1_score(y_dev, dev_predictions, average="weighted")

        print("Hyperparameter is ", params)
        print("Development set accuracy: ", dev_accuracy)
        
        # (6) Check if the current hyperparameters lead to a better accuracy
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_hyperparameters = params

    best_xgbclassifier_model = XGBClassifier(**best_hyperparameters, eval_metric=f1_score_wrapper)
    best_xgbclassifier_model.fit(np.concatenate([X_train, X_dev]), np.concatenate([y_train, y_dev]))

    print()
    print("Best hyperparameter is ", best_hyperparameters)

    # Make predictions on the testing set using the best model
    test_predictions = best_xgbclassifier_model.predict(X_test)

    #analyze_pattern(test_predictions, y_test, X_test)

    # Calculate accuracy on the testing set
    test_accuracy = f1_score(y_test, test_predictions, average="weighted")
    print("Testing set accuracy with best model:", test_accuracy)

def xgboost_learning_multi(wine_data):
    #1. Extract X (feature vector) and y (quality of wine that ranges from 0 to 10))
    X = wine_data.drop(columns=['quality'], axis=1)
    y = wine_data['quality']

    #2. Split data into training, development, and testing set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

    #3. Normalize the data before training to achieve stable and fast training of the model
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    X_dev = norm.transform(X_dev)
    X_test = norm.transform(X_test)

    #4. Use SMOTE (oversampling technique) to resample the data as the original data is very unbalanced
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    #6. List the hyperparameter to tune
    hyperparameters = [
        {'learning_rate': 0.001, 'max_depth': 5}, 
        {'learning_rate': 0.01, 'max_depth': 5}, 
        {'learning_rate': 0.1, 'max_depth': 5}, 
        {'learning_rate': 1, 'max_depth': 5}, 
        {'learning_rate': 10, 'max_depth': 5}, 
        {'learning_rate': 100, 'max_depth': 5},

        {'learning_rate': 0.001, 'max_depth': 6}, 
        {'learning_rate': 0.01, 'max_depth': 6}, 
        {'learning_rate': 0.1, 'max_depth': 6}, 
        {'learning_rate': 1, 'max_depth': 6}, 
        {'learning_rate': 10, 'max_depth': 6}, 
        {'learning_rate': 100, 'max_depth': 6},

        {'learning_rate': 0.001, 'max_depth': 7}, 
        {'learning_rate': 0.01, 'max_depth': 7}, 
        {'learning_rate': 0.1, 'max_depth': 7}, 
        {'learning_rate': 1, 'max_depth': 7}, 
        {'learning_rate': 10, 'max_depth': 7}, 
        {'learning_rate': 100, 'max_depth': 7},
        
        {'learning_rate': 0.001, 'max_depth': 8}, 
        {'learning_rate': 0.01, 'max_depth': 8}, 
        {'learning_rate': 0.1, 'max_depth': 8}, 
        {'learning_rate': 1, 'max_depth': 8}, 
        {'learning_rate': 10, 'max_depth': 8}, 
        {'learning_rate': 100, 'max_depth': 8}, 

        {'learning_rate': 0.001, 'max_depth': 9}, 
        {'learning_rate': 0.01, 'max_depth': 9}, 
        {'learning_rate': 0.1, 'max_depth': 9}, 
        {'learning_rate': 1, 'max_depth': 9}, 
        {'learning_rate': 10, 'max_depth': 9}, 
        {'learning_rate': 100, 'max_depth': 9},
        
        {'learning_rate': 0.001, 'max_depth': 10}, 
        {'learning_rate': 0.01, 'max_depth': 10}, 
        {'learning_rate': 0.1, 'max_depth': 10}, 
        {'learning_rate': 1, 'max_depth': 10}, 
        {'learning_rate': 10, 'max_depth': 10}, 
        {'learning_rate': 100, 'max_depth': 10}, 
    ]

    #7. Training (with XGBoost)
    best_accuracy = 0
    best_hyperparameters = None 

    y_train = y_train.apply(lambda x:x-3)
    y_dev = y_dev.apply(lambda x:x-3)
    y_test = y_test.apply(lambda x:x-3)
        
    for params in hyperparameters: 
        # (1) Create a xgb_classifier model with the current hyperparameter
        xgb_classifier = XGBClassifier(**params, eval_metric=f1_score_wrapper, objective="multi:softmax")
    
        # (2) Train the model on the training set
        xgb_classifier.fit(X_train, y_train)

        # (3) Make predictions on the development set
        dev_predictions = xgb_classifier.predict(X_dev)

        # (4) Analyze the result
        #analyze_pattern(dev_predictions, y_dev, X_dev)

        # (5) Calculate accuracy on the development set
        dev_accuracy = f1_score(y_dev, dev_predictions, average="weighted")
        #dev_accuracy = accuracy_score(y_dev, dev_predictions)

        print("Hyperparameter is ", params)
        print("Development set accuracy: ", dev_accuracy)
        
        # (6) Check if the current hyperparameters lead to a better accuracy
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_hyperparameters = params

    best_xgbclassifier_model = XGBClassifier(**best_hyperparameters, eval_metric=f1_score_wrapper, objective="multi:softmax")
    best_xgbclassifier_model.fit(np.concatenate([X_train, X_dev]), np.concatenate([y_train, y_dev]))

    print()
    print("Best hyperparameter is ", best_hyperparameters)

    # Make predictions on the testing set using the best model
    test_predictions = best_xgbclassifier_model.predict(X_test)

    #analyze_pattern(test_predictions, y_test, X_test)

    # Calculate accuracy on the testing set
    test_accuracy = f1_score(y_test, test_predictions, average="weighted")
    print("Testing set accuracy with best model:", test_accuracy)

def main():
    #1. read the csv file "winequality-red.csv"
    wine_data = pd.read_csv('winequality-red.csv')

    #2. show the distribution of labels (qualities of wine) in wine_data
    #count_quality(wine_data)

    #3. 
    if OPTS.mode == 'logistic':
        logistic_regression(wine_data)
        print("----------------------------------------")
    elif OPTS.mode == 'softmax':
        softmax_regression(wine_data)
        print("----------------------------------------")
    elif OPTS.mode == 'random_forest':
        if OPTS.version == 'binary':
            random_forest_binary(wine_data)
            print("----------------------------------------")
        elif OPTS.version == 'multi':
            random_forest_multi(wine_data)
            print("----------------------------------------")
    elif OPTS.mode == 'xgboost':
        if OPTS.version == 'binary':
            xgboost_learning_binary(wine_data)
            print("----------------------------------------")
        elif OPTS.version == 'multi':
            xgboost_learning_multi(wine_data)
            print("----------------------------------------")

OPTS = parse_args()
main()
    
