import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from scipy.stats import boxcox
import pickle

# Load dataset
df = pd.read_csv('"C:\Users\Naufal\Downloads\Copper_Set.xlsx - Result 1.csv"')

# Handle missing values and outliers
df.replace('00000', np.nan, inplace=True)
df['material_ref'] = df['material_ref'].fillna('Unknown')

# Handle outliers
iso_forest = IsolationForest(contamination=0.01)
df['outlier'] = iso_forest.fit_predict(df[['selling_price']])
df = df[df['outlier'] == 1].drop(columns=['outlier'])

# Treat Skewness
for column in df.select_dtypes(include=[np.number]).columns:
    skewness = df[column].skew()
    if abs(skewness) > 0.5:
        df[column], _ = boxcox(df[column] + 1)  # Box-Cox transformation

# Encode categorical variables
label_encoder = LabelEncoder()
df['customer'] = label_encoder.fit_transform(df['customer'])
df['country'] = label_encoder.fit_transform(df['country'])
df['status'] = df['status'].map({'WON': 1, 'LOST': 0})

# Drop highly correlated columns
correlation_matrix = df.corr()
drop_columns = [column for column in correlation_matrix.columns if any(correlation_matrix[column] > 0.9)]
df.drop(columns=drop_columns, inplace=True)

# Split data for regression
X_reg = df.drop(columns=['selling_price'])
y_reg = df['selling_price']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Train Regression Model
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg_scaled, y_train_reg)

# Evaluate Regression Model
y_pred_reg = reg_model.predict(X_test_reg_scaled)
print("Regression MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("Regression R2 Score:", r2_score(y_test_reg, y_pred_reg))

# Save Regression Model
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(reg_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Split data for classification
X_clf = df.drop(columns=['status'])
y_clf = df['status']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Scale features
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# Train Classification Model
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf_scaled, y_train_clf)

# Evaluate Classification Model
y_pred_clf = clf_model.predict(X_test_clf_scaled)
print("Classification Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_clf))

# Save Classification Model
with open('classification_model.pkl', 'wb') as f:
    pickle.dump(clf_model, f)
