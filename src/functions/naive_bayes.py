# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split, GridSearchCV
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('src/data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.

# data = os.path.join(dirname, filenames[0])

df = pd.read_csv('src/data/unprocessed_csvs/M47_unprocessed.csv')

print(df.shape)
print(df.info())

# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

for var in categorical: 
    
    # print(df[var].value_counts())
    print(var, ' contains ', len(df[var].unique()), ' labels')


print(df['IsDeadWithin4Months'].value_counts())
df['IsDeadWithin4Months'] = df['IsDeadWithin4Months'].replace('#NUM!', "NO")

print(df['IsDeadWithin4Months'].value_counts())

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

X = df.drop(['IsDeceased', 'EmergencyReadmissionDateTime', 'DateOfDeath', 'UniqueEpisodeID', 
             'EpisodeStartDateTime', 'EpisodeEndDateTime', 'Ward', 'EthnicOrigin', 'SourceSystem', 'TreatmentFunction.1', 
             'AdmissionDate', 'EpisodeDiagnosisCodeList', 'WardStartDateTime', 'WardEndDateTime', 'EpisodeProcedureCodeList', 'PatientPseudoNo', 
             'TreatmentFunctionCode.1', 'MonthsAwayFromDeath', 'IsDeadWithin4Months'], axis=1)

y = df['IsDeadWithin4Months']

print(X.shape)
print(y.shape)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print(X_train.shape) 
print(X_test.shape)

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

print(X_train[categorical].isnull().mean())

# for df2 in [X_train, X_test]:
#     df2['DateOfDeath'].fillna(X_train['DateOfDeath'].mode()[0], inplace=True)
#     df2['EmergencyReadmissionDateTime'].fillna(X_train['EmergencyReadmissionDateTime'].mode()[0], inplace=True)


# encode remaining variables with one-hot encoding

#Cols=['UniqueEpisodeID', 'EpisodeStartDateTime', 'EpisodeEndDateTime', 'EpisodeProcedureCodeList', 'Ward', 
#  'WardCode', 'WardStartDateTime', 'WardEndDateTime', 'TreatmentFunction', 'EpisodeDiagnosisCodeList', 'PrimaryDiagnosis',
#  'AdmissionDate', 'Gender', 'EthnicOriginCode', 'EthnicOrigin', 'EthnicOriginGroup', 'DischargeMethod', 'DischargeDestination', 'AdmissionSource',
#  'AdmissionMethod', 'SourceSystem', 'LocalSpecialtyCode', 'LocalSpecialty',
#  'TreatmentFunction.1', 'TreatmentFunctionCombined', 'IPRDepartment', 'IsConsultantLedService', 'IsASISpecialty', 'SpecialtyOwner', 'TrustDivision']
encoder = ce.OneHotEncoder(cols=categorical)

print(X_train.shape)
print(X_test.shape)

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)



#feature scaling
cols = X_train.columns

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])

X_train.head()

#Model training
# model = GaussianNB()

# #Define the grid of hyperparameters to search
# params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}


# # Setup the grid search with cross-validation
# grid_search = GridSearchCV(estimator=model, param_grid=params_NB, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)


# grid_search.fit(X_train, y_train)

# # Best parameters and best score
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation accuracy:", grid_search.best_score_)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# # Calculate accuracy
# test_accuracy = accuracy_score(y_test, y_pred)
# print("Test accuracy:", test_accuracy)

# instantiate the model
gnb = GaussianNB(var_smoothing=8.111308307896872e-05)


# fit the model
gnb.fit(X_train, y_train)

#Predict the results
y_pred = gnb.predict(X_test)

print(y_pred)

#Check the model accuracy
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = gnb.predict(X_train)

print(y_pred_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


#Check null acuracy
print(y_test.value_counts())
print(y_test.value_counts().max())

null_accuracy = (16651/(16651+5275))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


#Classification metrices
print(classification_report(y_test, y_pred))


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

y_pred_prob = gnb.predict_proba(X_test)[0:10]
print(y_pred_prob)