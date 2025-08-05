# 🎯 Football Player Potential Prediction

This project aims to predict whether a football player has **average** or **highlighted** potential based on attribute scores given by scouts during professional matches. The dataset is sourced from **Scoutium**, a platform for professional scouting and talent evaluation.

## 🧠 Problem Definition

The goal is to build a supervised machine learning model that can classify players into one of two categories:
- **average**
- **highlighted**

By using the attribute scores given by scouts, the model attempts to learn patterns that differentiate high-potential players from others.


## 📁 Dataset Description

The dataset is divided into two CSV files:

### 1. `scoutium_attributes.csv`
Contains attribute evaluations by scouts:
- `task_response_id`: Unique evaluation set per scout per match
- `match_id`: Match ID
- `evaluator_id`: Scout ID
- `player_id`: Player ID
- `position_id`: Player position (1–10)
- `analysis_id`: Attribute set per player per match
- `attribute_id`: Feature ID (e.g., pass, shot, speed)
- `attribute_value`: Score for that attribute

### 2. `scoutium_potential_labels.csv`
Contains final player evaluations:
- `task_response_id`, `match_id`, `evaluator_id`, `player_id`: Foreign keys
- `potential_label`: One of (`below_average`, `average`, `highlighted`)

---

## ⚙️ Data Preprocessing Steps

1. **Read both CSV files** using pandas.
2. **Merge datasets** using `task_response_id`, `match_id`, `evaluator_id`, and `player_id`.
3. **Remove goalkeepers** (`position_id == 1`) due to their unique attributes.
4. **Remove below_average players**, as they make up only ~1% of the dataset.
5. **Pivot the data** so that each row represents a player and columns represent average scores for each attribute.
6. **Encode categorical labels** (`average`, `highlighted`) using `LabelEncoder`.
7. **Scale numeric features** with `StandardScaler`.

---

## 🧪 Modeling

Three machine learning models were used:
- ✅ Random Forest
- ✅ XGBoost
- ✅ LightGBM

### Evaluation Metrics
Each model is evaluated based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC**

| Model          | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|----------------|----------|-----------|--------|----------|---------|
| Random Forest  | ...      | ...       | ...    | ...      | ...     |
| XGBoost        | ...      | ...       | ...    | ...      | ...     |
| LightGBM       | ...      | ...       | ...    | ...      | ...     |

> Note: Replace `...` with actual results after model evaluation.

---

## 📊 Feature Importance

Feature importance plots are generated for each model to highlight which player attributes are most useful in classification.

Top attributes include (example):
- Speed
- Vision
- Passing Accuracy
- Game Awareness
- Ball Control

---

## 🧰 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- Matplotlib, Seaborn

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/scoutium-player-potential.git
   cd scoutium-player-potential
