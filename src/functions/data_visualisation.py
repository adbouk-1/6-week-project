from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
import seaborn as sns
import numpy as np
import graphviz

def plotCorrelationMatrix(df):
    """
    Plots a correlation matrix heatmap for the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame for which the correlation matrix heatmap needs to be plotted.

    Returns:
    None
    """

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, cmap='coolwarm', mask=mask, annot=True, fmt='.1g', xticklabels=corr.columns, yticklabels=corr.columns, cbar=False)
    plt.xticks(fontsize=8)  # Rotate labels by 45 degrees
    plt.show()

def plotConfusionMatrix(model, X_test, y_test, title):
    """
    Plots the confusion matrix for a given model's predictions on the test data.

    Parameters:
    - model: The trained model object.
    - X_test: The test data features.
    - y_test: The true labels for the test data.
    - title: The title of the confusion matrix plot.

    Returns:
    - cm: The confusion matrix as a NumPy array.
    """

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.title('Confusion Matrix for ' + title.strip('_unprocessed.csv') + ' dataset trained on the ' + title.strip('_unprocessed.csv') + ' dataset')
    plt.show()

    # Plot key metrics from confusion matrix
    print(classification_report(y_test, model.predict(X_test)))

    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print('Classification error : {0:0.4f}'.format(classification_error))

    precision = TP / float(TP + FP)
    print('Precision : {0:0.4f}'.format(precision))

    recall = TP / float(TP + FN)
    print('Recall or Sensitivity : {0:0.4f}'.format(recall))

    false_positive_rate = FP / float(FP + TN)
    print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

    specificity = TN / (TN + FP)
    print('Specificity : {0:0.4f}'.format(specificity))

    return cm

def calculateClassProbabilities(model, X_test):
    """
    Calculate the predicted probabilities of a binary classification model and plot a histogram of the predicted probabilities.

    Parameters:
    model (object): The trained binary classification model.
    X_test (array-like): The input features for which to calculate the predicted probabilities.

    Returns:
    None
    """

    # Predict the probability of each class
    y_pred1 = model.predict_proba(X_test)[:, 1]

    # plot histogram of predicted probabilities
    # adjust the font size 
    plt.rcParams['font.size'] = 12

    # plot histogram with 10 bins
    plt.hist(y_pred1, bins=10)

    # set the title of predicted probabilities
    plt.title('Histogram of predicted probabilities of deaths within 4 months')

    # set the x-axis limit
    plt.xlim(0, 1)

    # set the title
    plt.xlabel('Predicted probabilities of deaths within 4 months')
    plt.ylabel('Frequency')
    plt.show()

def plotROCCurve(model, X_test, y_test):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for a binary classifier.

    Parameters:
    model (object): The trained binary classifier model.
    X_test (array-like): The feature matrix of the test data.
    y_test (array-like): The true labels of the test data.

    Returns:
    float: The ROC AUC score for the classifier.

    """

    # Predict the probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.figure(figsize=(6,4))

    # Plot the ROC curve
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )

    # Set the limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.rcParams['font.size'] = 12
    # Set the labels and title
    plt.title('ROC curve for deaths within 4 months classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')

    # Add a grid
    plt.grid(True)

    # Add a diagonal line
    plt.plot([0, 1], [0, 1], 'k--')

    # Show the plot
    plt.show()

    if PRINT_INFO:
        # Calculate the ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        print('ROC AUC : {:.4f}'.format(roc_auc))

        Cross_validated_ROC_AUC = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc').mean()
        print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

    return roc_auc_score(y_test, y_pred_prob)

def kfoldCrossValidateModel(model, X_train, y_train):
    """
    Perform k-fold cross-validation on a given model using the provided training data.

    Parameters:
    model (object): The machine learning model to be evaluated.
    X_train (array-like): The input features for training the model.
    y_train (array-like): The target variable for training the model.

    Returns:
    cv_scores (array-like): The cross-validation scores for each fold.
    """

    # Setup the cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

    # Print the cross-validation scores
    print('Cross-validation scores:{}'.format(cv_scores))
    # Print the average cross-validation score
    print('Average cross-validation score: {:.4f}'.format(cv_scores.mean()))

    return cv_scores

def averagePrecisionScore(model, X_test, y_test):
    """
    Calculate the average precision score and plot the precision-recall curve.

    Parameters:
    - model: The trained model used for prediction.
    - X_test: The input features for testing.
    - y_test: The true labels for testing.

    Returns:
    None
    """

    # Calculate the predicted probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate the average precision score
    average_precision = average_precision_score(y_test, y_pred_prob)

    # Calculate the precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

    # Plot the precision-recall curve
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(f'Precision Recall Curve. AUPRC: {average_precision}')
    plt.show()


def randomForestGraphs(model, X_train):
    """
    Generate and save visualizations of individual decision trees in a random forest model.

    Parameters:
    - model: The trained random forest model.

    Returns:
    None
    """

    for i in range(len(model.estimators_)):
        tree = model.estimators_[i]
        dot_data = export_graphviz(tree,
                                feature_names=X_train.columns,  
                                filled=True,  
                                max_depth=2, 
                                impurity=False, 
                                proportion=True)
        graph = graphviz.Source(dot_data)
        graph.render('./figures/Source_' + str(i) + '.gv', format='jpg',view=True)

def create_ttest(df, y):
    """
    Perform t-test on the given DataFrame columns based on the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        y (pd.Series): The target variable.

    Returns:
        pd.DataFrame: A DataFrame containing the p-values and other statistics for each column.
    """

    categorical = [col for col in df.columns if df[col].dtypes == 'O']

    # Create a list of p-values
    p_values = []

    # Loop through the columns
    for col in df.columns:
        # Perform t-test
        x = df[col]

        print(col)

        if col in categorical:
            encoder = ce.OneHotEncoder(cols=col, use_cat_names=True)
            x = encoder.fit_transform(x)

        df_0 = x[y == 0]
        df_1 = x[y == 1]
        
        p_values = []
        counts = []
        original_values = []
        deaths = []

        if col in categorical:
            for cols in x.columns:

                #For each unique value record the p_value, the value count, the unique value and the number of deaths
                ttest = ttest_ind(df_0[cols], df_1[cols])
                count = x[cols].sum()
                original_value = cols.split('_')[1]

                p_values.append(ttest.pvalue)
                counts.append(count)
                original_values.append(original_value)
                deaths.append(df_1[cols].sum())

            p_values = pd.DataFrame(p_values, index=x.columns, columns=['p_value'])
            p_values['count'] = counts
            p_values['original_value'] = original_values
            p_values['deaths'] = deaths
        else:
            ttest = ttest_ind(df_0, df_1)
            p_values = pd.DataFrame([ttest.pvalue], columns=['p_value'])

        # Sort the p-values
        p_values = p_values.sort_values(by='p_value')
        p_values.to_csv('src/data/p_values/' + col + '_ttest_results.csv')

    return p_values

