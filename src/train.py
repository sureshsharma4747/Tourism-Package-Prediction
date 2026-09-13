import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA_PATH = "data/tourism.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "tourism_package_model.pkl")

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# Remove unwanted CSV index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("Shape after removing Unnamed: 0:", df.shape)

# --------------------------------------------------
# 3. Target
# --------------------------------------------------

TARGET = "ProdTaken"

X = df.drop(
    columns=[
        TARGET,
        "CustomerID",
        "Unnamed: 0"
    ],
    errors="ignore"
)
y = df[TARGET]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

# --------------------------------------------------
# 4. Identify columns
# --------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# --------------------------------------------------
# 5. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)

# --------------------------------------------------
# 7. Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

# --------------------------------------------------
# 8. Complete pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# --------------------------------------------------
# 9. Train
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 10. Evaluate
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 11. Save model
# --------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)

print("\nTraining completed!")