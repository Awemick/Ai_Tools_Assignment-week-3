# task1_iris_sklearn.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and explore data from CSV
import pandas as pd
from sklearn.preprocessing import LabelEncoder

iris_df = pd.read_csv('datasets/Iris.csv')
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y_text = iris_df['Species'].values

# Encode species names to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target_names = label_encoder.classes_

print("Dataset loaded from CSV:")
print("Dataset Shape:", X.shape)
print("Feature Names:", feature_names)
print("Target Names:", target_names)
print("Class Distribution:", np.bincount(y))

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = [target_names[i] for i in y]

# Data visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='species_name')
plt.title('Sepal Length vs Sepal Width')

plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='species_name')
plt.title('Petal Length vs Petal Width')

plt.subplot(2, 2, 3)
# Create separate boxplots for each species
for i, species in enumerate(target_names):
    species_data = df[df['species_name'] == species]
    plt.subplot(2, 2, 3)
    species_data[feature_names].boxplot()
    plt.title(f'Feature Distribution - {species}')
    plt.xticks(rotation=45)
    break  # Just show one for now to avoid subplot issues

plt.tight_layout()
plt.show()

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
dt_classifier = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

dt_classifier.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = dt_classifier.predict(X_test_scaled)
y_pred_proba = dt_classifier.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Training Accuracy: {dt_classifier.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {dt_classifier.score(X_test_scaled, y_test):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=12)
plt.title('Decision Tree Visualization')
plt.show()

# Save model
import joblib
joblib.dump(dt_classifier, 'iris_decision_tree_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')
print("\nModel and scaler saved successfully!")