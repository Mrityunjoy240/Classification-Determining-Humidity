# Step 1 & 2: Import libraries and load dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the dataset
df = pd.read_csv('daily_weather.csv')  # Make sure the file is in the same directory

# Step 3: Exploreing the data
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())



# Checking for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values (we can use other strategies like mean, median, etc.)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove outliers (For cleaner results)
# Example: removing data where humidity_3pm is greater than 100% or less than 0%
df = df[(df['relative_humidity_3pm'] >= 0) & (df['relative_humidity_3pm'] <= 100)]

# Step 5: Createing target column
# 1 if humidity_3pm >= 25, else 0
df['humidity_level'] = np.where(df['relative_humidity_3pm'] >= 25, 1, 0)

# Step 6: Feature selection
# We'll use relative_humidity_9am as feature
X = df[['relative_humidity_9am']]
y = df['humidity_level']

# Step 7: Train/Test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
