from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn import svm # type: ignore
import community.community_louvain as community_louvain # type: ignore
import matplotlib.pyplot as plt # type: ignore
import category_encoders as ce # type: ignore
import networkx as nx # type: ignore
import pandas as pd
import numpy as np

PRINT_INFO = False

def splitTrainingSet(x, y, test_size = 0.3, random_state = 0):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    x (pandas.DataFrame): The input features.
    y (pandas.Series): The target variable.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.3.
    random_state (int, optional): The seed used by the random number generator. Default is 0.
    
    Returns:
    X_train (pandas.DataFrame): The training set features.
    X_test (pandas.DataFrame): The testing set features.
    y_train (pandas.Series): The training set target variable.
    y_test (pandas.Series): The testing set target variable.
    """
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)

    #One hot encode the categorical variables
    categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
    encoder = ce.OneHotEncoder(cols=categorical)

    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test, y_train, y_test

def trainModel(X_train, X_test, y_train, y_test, classifier="LogisticRegression"):
    """
    Trains a machine learning model using the specified classifier.

    Parameters:
    - X_train (array-like): The training data features.
    - X_test (array-like): The testing data features.
    - y_train (array-like): The training data labels.
    - y_test (array-like): The testing data labels.
    - classifier (str, optional): The classifier to use. Defaults to "LogisticRegression".

    Returns:
    - model: The trained machine learning model.

    """

    if classifier == "LogisticRegression":
        #Hyperparameters to use for each target variable
        #Readmission rate: 4.281332398719396
        #Dead within 4 months: 545.5594781168514    
        model = LogisticRegression(solver='liblinear', C=545.5594781168514, random_state=0, penalty='l1')

        model.fit(X_train, y_train)

        if PRINT_INFO:
            print(model.classes_)
            print(model.intercept_)
            print(model.coef_)

            model.predict_proba(X_test)

            model.predict(X_test)

            print(model.score(X_test, y_test))

            y_pred = model.predict(X_test)
            print(y_pred)
            print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

            y_pred_train = model.predict(X_train)
            print(y_pred_train)
            print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


            #Check null acuracy
            print(y_test.value_counts())
            print(np.unique(y_pred))
            null_accuracy = (y_test.value_counts().max()/(y_test.value_counts().max()+y_test.value_counts().min()))
            print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

    elif classifier == "NaiveBayes":

        model = GaussianNB(var_smoothing=8.111308307896872e-05)
        model.fit(X_train, y_train)

        if PRINT_INFO:
            #Predict the results
            y_pred = model.predict(X_test)
            print(y_pred)
            print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

            y_pred_train = model.predict(X_train)
            print(y_pred_train)
            print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


            #Check null acuracy
            print(y_test.value_counts())
            print(np.unique(y_pred))
            null_accuracy = (y_test.value_counts().max()/(y_test.value_counts().max()+y_test.value_counts().min()))
            print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

    elif classifier=="SVM":
        #NOTE: This doesnt work and was not attempted to be fixed
        model = svm.SVC(kernel='linear', C=1, random_state=0, verbose=1)
        model.fit(X_train, y_train)

        if PRINT_INFO:
            print(model.classes_)
            print(model.intercept_)
            print(model.coef_)

            model.predict_proba(X_test)

            model.predict(X_test)

            print(model.score(X_test, y_test))

    elif classifier=="RandomForest":
        #Testing with limiting the max depth
        # model = RandomForestClassifier(max_depth=14, n_estimators=190)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        if PRINT_INFO:

            print(model.score(X_test, y_test))

            y_pred = model.predict(X_test)
            print(y_pred)
            print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

            y_pred_train = model.predict(X_train)
            print(y_pred_train)
            print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


            #Check null acuracy
            print(y_test.value_counts())
            print(np.unique(y_pred))
            null_accuracy = (y_test.value_counts().max()/(y_test.value_counts().max()+y_test.value_counts().min()))
            print('Null accuracy score: {0:0.4f}'. format(null_accuracy))



    return model

# Setup the logistic regression model
def optimiseHyperParameters(X_train, X_test, y_train, y_test, classifier="LogisticRegression"):
    """
    Optimize hyperparameters for a given classifier using grid search or randomized search.

    Parameters:
    - x: The input features.
    - y: The target variable.
    - classifier: The classifier to use for hyperparameter optimization. Default is "LogisticRegression".

    Returns:
    None
    """

    if classifier == "LogisticRegression":

        model = LogisticRegression()

        # Define the grid of hyperparameters to search
        param_grid = {
            'C': np.logspace(-100, 500, 50),  # Regularization strength
            'penalty': ['l1', 'l2'],      # Types of regularization
            'solver': ['liblinear']       # Solver capable of handling l1 penalty
        }

        # Setup the grid search with cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    elif classifier == "NaiveBayes":

        model = GaussianNB()

        # Define the grid of hyperparameters to search
        param_grid = {
            'var_smoothing': np.logspace(0,-9, num=100) # Variance smoothing parameter
        }

        # Setup the grid search with cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    elif classifier == "RandomForest":

        model = RandomForestClassifier()

        param_grid = {
            # 'n_estimators': randint(50,500), # Number of trees in the forest
            'max_depth': np.arange(65, 80, 1).tolist()
        }

        grid_search = RandomizedSearchCV(model, param_distributions = param_grid, n_iter=5, cv=5)

    # Fit the random search object to the data
    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy:", test_accuracy)


def testRFDepth(X_train, X_test, y_train, y_test):
    """
    This function tests the Random Forest classifier with different depths.

    Parameters:
    X_train (array-like): The training input samples.
    X_test (array-like): The test input samples.
    y_train (array-like): The target values for the training set.
    y_test (array-like): The target values for the test set.

    Returns:
    None
    """
        
    random_state = 123
    output_rf = []

    # build initial forest
    initial_forest = RandomForestClassifier(
        max_features="sqrt", random_state=random_state
    )
    initial_forest_fit = initial_forest.fit(X_train, y_train)

    # get max depth
    max_depth_rf = max(
        [estimator.get_depth() for estimator in initial_forest_fit.estimators_]
    )

    #construct depth grid
    list_depths_rf = np.arange(1, max_depth_rf, 1).tolist()
    depth_pct_rf = [100] + [(n / max_depth_rf) * 100 for n in list_depths_rf]

    # fit rf with smaller depth
    list_forests = [initial_forest_fit]
    print(max_depth_rf)
    for depth in list_depths_rf:
        tmp_forest = RandomForestClassifier(
            max_features="sqrt", max_depth=depth, random_state=random_state
        )
        list_forests.append(tmp_forest.fit(X_train, y_train))

    train_accuracy_rf = []
    test_accuracy_rf = []

    for forest in list_forests:
        # get accuracy for the forest
        yhat_train = forest.predict(X_train)
        yhat_test = forest.predict(X_test)
        train_accuracy_rf.append(accuracy_score(y_train, yhat_train))
        test_accuracy_rf.append(accuracy_score(y_test, yhat_test))

    df_rf = pd.DataFrame(
        list(
            zip(
                depth_pct_rf,
                train_accuracy_rf,
                test_accuracy_rf,
            )
        ),
        columns=["depth", "train_accuracy", "test_accuracy"],
    )
    output_rf.append(df_rf)

    df_rf.to_csv(path_or_buf="sim_shift_results_rf_check.csv", index=False)

def graph_clustering(df):
    """
    Perform graph clustering on a DataFrame. Note that this function does not work, it still needs work.
    It is a starting point for implementing graph clustering on the NHS data, but with the time constraints it was not possible to complete it.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    None
    """

    # Create a graph
    G = nx.Graph()

    categorical = [col for col in df.columns if df[col].dtypes == 'O']
    encoder = ce.OneHotEncoder(cols=categorical)

    df = encoder.fit_transform(df)
    
    # Add nodes
    for col in df.columns:
        G.add_node(col)

    # Add edges based on correlation threshold
    threshold = 0.5  # Set a threshold for strong correlation
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            correlation = df[df.columns[i]].corr(df[df.columns[j]])
            if abs(correlation) > threshold:  # only consider strong correlations
                G.add_edge(df.columns[i], df.columns[j], weight=correlation)

    # Detect communities
    partition = community_louvain.best_partition(G)

    # Add community info to the graph
    for node, comm_id in partition.items():
        G.nodes[node]['community'] = comm_id

    # Print the community of each node
    print("Node communities:")
    for node in G.nodes(data=True):
        print(node)

    color_map = [G.nodes[node]['community'] for node in G]
    pos = nx.spring_layout(G)  # positions for all nodes

    nx.draw(G, pos, node_color=color_map, with_labels=True, cmap=plt.get_cmap('viridis'), node_size=500, font_size=12)
    plt.show()

