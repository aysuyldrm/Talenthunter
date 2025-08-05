# Business Problem:
# Predict whether a football player is 'average' or 'highlighted' based on feature scores given by scouts.

'''
The dataset consists of evaluations made by scouts based on the in-game performance of football players.
It contains the scores given to various attributes of the players by scouts during matches.
There are 8 variables and 10,730 observations.
- task_response_id: A group of evaluations by a scout for all players of a team in a specific match
- match_id: ID of the match
- evaluator_id: ID of the scout
- player_id: ID of the player
- position_id: Position ID of the player in that match
    1: Goalkeeper
    2: Center-back
    3: Right-back
    4: Left-back
    5: Defensive Midfielder
    6: Central Midfielder
    7: Right Winger
    8: Left Winger
    9: Attacking Midfielder
    10: Forward
- analysis_id: A group of evaluations by a scout for a player in a specific match
- attribute_id: The ID of the feature being evaluated (e.g., passing, shooting)
- attribute_value: The score given by the scout for that attribute
'''

# Step 1: Read the scoutium_attributes.csv and scoutium_potential_labels.csv files

# Convert the two CSV files into Pandas DataFrames.
# attributes_df: Contains in-game observed attribute scores
# labels_df: Contains each player's potential label (average, highlighted, below_average)

import pandas as pd

attributes_df = pd.read_csv(r'C:\Users\NEWUSER\Desktop\MIUUL\Machine_Learning\Scotium\scoutium_attributes.csv', sep=';')
labels_df = pd.read_csv(r'C:\Users\NEWUSER\Desktop\MIUUL\Machine_Learning\Scotium\scoutium_potential_labels.csv', sep=';')

print("Attributes:")
print(attributes_df.head())
print(attributes_df['position_id'].unique())

# Explanation:
# Each player will have multiple attribute_id rows (e.g., 4322, 4323...)
# attribute_id = Which feature is evaluated
# attribute_value = Score given for that feature
# Example: Player 1361061 has 5 features scored as 56, 56, 67, 56, 45

print("\nLabels:")
print(labels_df.head())
# Explanation:
# This is the target variable for our model (label).
# Each player has one row indicating their final evaluation: average or highlighted.

# Step 2: Merge the two DataFrames using 'task_response_id', 'match_id', 'evaluator_id', and 'player_id'
merged_df = pd.merge(attributes_df, labels_df,
                     on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'],
                     how='inner')

# Check merged data
print("Merged dataset:")
print(merged_df.head())

# Check shape
print(f"\nMerged data shape: {merged_df.shape}")

# This creates a unified dataset combining player attribute scores and their potential label.
# Now ready for supervised learning:
# Input (X): attribute scores, Output (y): potential label

# Step 3: Remove goalkeepers (position_id = 1) from the dataset
# Goalkeepers have different evaluation criteria and including them can hurt model performance.
merged_df = merged_df[merged_df['position_id'] != 1]

# Check remaining positions
print("Remaining positions:", merged_df['position_id'].unique())

# Check shape
print("New data shape:", merged_df.shape)

# Step 4: Remove rows with 'below_average' label (as they make up ~1% of dataset)
merged_df = merged_df[merged_df['potential_label'] != 'below_average']

# Now each row contains an attribute score and the general potential label (average / highlighted)

# Check remaining labels
print("Remaining labels:", merged_df['potential_label'].unique())
print("New data shape:", merged_df.shape)

# Step 5: Create a pivot table where each row corresponds to one player

# Step 5-1: Create pivot with player_id, position_id, and potential_label as index;
# attribute_id as columns and attribute_value as values (aggregated by mean)
pivot_df = merged_df.pivot_table(
    index=['player_id', 'position_id', 'potential_label'],
    columns='attribute_id',
    values='attribute_value',
    aggfunc='mean'  # Average if multiple scores exist for the same player-feature
)

print(pivot_df.head())

# Step 5-2: Reset index and convert attribute_id column names to string
pivot_df.reset_index(inplace=True)

# Rename attribute_id columns to string format
for col in pivot_df.columns:
    if col not in ['player_id', 'position_id', 'potential_label']:
        pivot_df.rename(columns={col: str(col)}, inplace=True)

print(pivot_df.head())

# Step 6: Use LabelEncoder to convert potential_label (average, highlighted) into numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
pivot_df['potential_label_encoded'] = le.fit_transform(pivot_df['potential_label'])

# Check encoded labels
print(pivot_df[['potential_label', 'potential_label_encoded']].drop_duplicates())
# Shows the mapping between string and numeric labels

# Step 7: Create list of numerical feature columns as num_cols
num_cols = pivot_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols = [col for col in num_cols if col not in ['player_id', 'position_id', 'index', 'potential_label', 'potential_label_encoded']]
print("Numerical columns (num_cols):")
print(num_cols)

# Step 8: Apply StandardScaler to scale numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pivot_df[num_cols] = scaler.fit_transform(pivot_df[num_cols])

# Check scaled values
print(pivot_df[num_cols].head())

# Step 9: Build a machine learning model to predict player potential labels with minimum error
# Evaluate using metrics: ROC AUC, F1, Precision, Recall, Accuracy

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Define features and target
X = pivot_df[num_cols]
y = pivot_df['potential_label_encoded']

# Split into training and test sets (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use 80% training, 20% testing
# stratify=y: Keeps class distribution the same in both sets
# random_state=42: For reproducibility

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# Store evaluation results
results = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability for positive class (highlighted)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save results
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

# Display results
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:\n")
print(results_df)

# If recall is a priority (e.g., identifying all highlighted players), XGBoost may be preferred.
# For overall balance and performance, Random Forest appears to be a strong choice.

# Step 10: Use feature_importance function to visualize feature importances
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot top-N feature importances
def feature_importance_plot(model, model_name, feature_names, top_n=10):
    """
    Plots the most important features for a trained model using a bar chart.
   
    :param model: Trained model (RandomForest, XGBoost, LightGBM)
    :param model_name: Name of the model (string)
    :param feature_names: List of feature names
    :param top_n: Number of top features to plot
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'{model_name} - Top {top_n} Important Features')
    plt.xlabel('Importance Level')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

# Plot feature importances for all models
for name, model in models.items():
    feature_importance_plot(model, name, num_cols, top_n=15)
