# %%
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("Machine_Downtime.csv")


# %%
df = df.drop(['Date', 'Machine_ID', 'Assembly_Line_No'],axis=1)
df.head()

# %%
df['Downtime'].value_counts()

# %%
df['Downtime'] = df['Downtime'].map({
    'Machine_Failure': 0,
    'No_Machine_Failure': 1}).fillna(0).astype(int)


# %%
df = df.dropna()

# %%
parameters = []
for columns in df.columns:
    if df[columns].nunique()>7:
        parameters.append(columns)


# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# %%
df[parameters] = sc.fit_transform(pd.DataFrame(df[parameters]))

# %%


# %% [markdown]
# # Feature Engineering

# %%
from sklearn.model_selection import train_test_split

# %%
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)

# %%
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)

# %%
X = df[parameters]
y = df['Downtime']


# %%
X.skew()


# %% [markdown]
# # One Hot Encoding

# %%
from sklearn.feature_extraction import DictVectorizer

# %%
dv = DictVectorizer(sparse = False)

# %%
y_train = df_train['Downtime'].values.ravel()
y_val = df_val['Downtime'].values.ravel()
y_test = df_test['Downtime'].values.ravel()

# %%
train_dicts = df_train[parameters].to_dict(orient='records')
val_dicts = df_val[parameters].to_dict(orient='records')
test_dicts = df_test[parameters].to_dict(orient='records')

# %%
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)


# %% [markdown]
# # Modelling

# %%
import xgboost as xgb
from lightgbm import LGBMClassifier # type: ignore
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
# Set random seed for reproducibility
RANDOM_STATE = 42

# %%
classifier = {
    "KNN_Model": KNeighborsClassifier(n_neighbors=5),
    "SVM_Model": SVC(kernel='rbf', C=1.0, gamma='auto', probability=True, random_state=42),
    "ada_model": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
    "GB_model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=RANDOM_STATE),
    "LGBMClassifier": LGBMClassifier(n_etsimators=100, learning_rate=0.1, max_depth=5, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1),
    "logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=20, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1),
}

# %%
for name, clf in classifier.items():
    print(f'\n============={name}=============')
    clf.fit(X_train, y_train)

    # Make predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred =  clf.predict(X_val)
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    print(f'\n Accuracy: {accuracy_score(y_val, y_val_pred)}')
    print(f'\n classification_report: {classification_report(y_val, y_val_pred)}')
    print(f'\n confusion_matrix: {confusion_matrix(y_val, y_val_pred)}')
    print(f'\n roc_auc_score: {roc_auc_score(y_val, y_val_pred)}')

# %%


# %% [markdown]
# # XGBOOST

# %%
features = dv.get_feature_names_out().tolist()

# Create Dmatrices
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

# %%
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 42,
    'verbosity': 1,

}
model = xgb.train(xgb_params, dtrain, num_boost_round=200)

# %%
model.predict(dtrain)

# %%
y_pred_proba = model.predict(dval)

# 2. Create Hard Class Predictions (y_pred)
# Convert probabilities to 0 or 1 based on a threshold (0.5 is standard)
y_pred = (y_pred_proba >= 0.5).astype(int)

# %%
print (f"Prediction probability array shape: {y_pred_proba.shape}")
print (f"Sample predictions (probabilities): {y_pred_proba[:5]}")

# 3. Calculate the ROC AUC Score
# roc_auc_score requires the true labels (y_val) and the prediction probabilities (y_pred_proba).
roc_auc_score(y_val, y_pred_proba)

print(f'\n Accuracy: {accuracy_score(y_val, y_pred)}')
print(f'\n classification_report: {classification_report(y_val, y_pred)}')
print(f'\n confusion_matrix: {confusion_matrix(y_val, y_pred)}')
print(f'\n roc_auc_score: {roc_auc_score(y_val, y_pred)}')

# %%


# %% [markdown]
# # Updated XGBoost

# %%
watchlist = [(dtrain,"train"), (dval,"val")]

# %%
xgb_params = {
    'eta': 0.9,
    'max_depth': 10,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 2,

    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
}

model = xgb.train(xgb_params,
                  dtrain,
                  early_stopping_rounds=10,
                  verbose_eval = 2,
                  evals = watchlist,
                  num_boost_round = 200)
    

# %%
model.predict(dtrain)

# %%
y_pred_proba = model.predict(dval)

# 2. Create Hard Class Predictions (y_pred)
# Convert probabilities to 0 or 1 based on a threshold (0.5 is standard)
y_pred = (y_pred_proba >= 0.5).astype(int)

# %%
print (f"Prediction probability array shape: {y_pred_proba.shape}")
print (f"Sample predictions (probabilities): {y_pred_proba[:5]}")

# 3. Calculate the ROC AUC Score
# roc_auc_score requires the true labels (y_val) and the prediction probabilities (y_pred_proba).
roc_auc_score(y_val, y_pred_proba)

print(f'\n Accuracy: {accuracy_score(y_val, y_pred)}')
print(f'\n classification_report: {classification_report(y_val, y_pred)}')
print(f'\n confusion_matrix: {confusion_matrix(y_val, y_pred)}')
print(f'\n roc_auc_score: {roc_auc_score(y_val, y_pred)}')

# %%


# %% [markdown]
# # Best Model - XGBoost

# %%
### Create a Pickle file using serialization 
import pickle

with open ('machine_failure_prediction.pkl', 'wb') as f:
    pickle.dump(dv, f)
    pickle.dump(model, f)
    pickle.dump(sc, f)

# %%


# %% [markdown]
# # Validating Pickle File

# %%
import pickle

with open('machine_failure_prediction.pkl', 'rb') as f:
    dv = pickle.load(f)
    model = pickle.load(f)
    sc = pickle.load(f)

    print(f"Loaded dv:{type(dv)}")
    print(f"Loaded model:{type(model)}")
    print(f"Loaded sc:{type(sc)}")

# %%



