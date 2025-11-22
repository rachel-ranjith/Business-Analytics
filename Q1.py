import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report

# ---------------------------------------
# 1. Load Excel File
# ---------------------------------------

df = pd.read_excel("Bank Marketing Data.xlsx")
df = df.dropna(how="all")

# ---------------------------------------
# 2. Clean / Prepare Data
# ---------------------------------------

binary_cols = ["housing", "loan", "y"]
for col in binary_cols:
    df[col] = df[col].astype(str).str.lower().map({"yes": 1, "no": 0})

df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
df = df.dropna(subset=["housing", "loan", "balance", "y"])

# ---------------------------------------
# 3. Feature Engineering: Interactions
# ---------------------------------------

df["both_loans"] = df["housing"] * df["loan"]
df["housing_balance"] = df["housing"] * df["balance"]
df["loan_balance"] = df["loan"] * df["balance"]
df["balance_bin"] = pd.qcut(df["balance"], q=4, labels=False)

# ---------------------------------------
# 4. Show Subscription Rates by Combo Groups
# ---------------------------------------

print("\n=== Subscription Rates by Housing vs Loan ===")
print(df.pivot_table(values="y", index="housing", columns="loan", aggfunc="mean"))

print("\n=== Subscription Rates by Balance Quartile ===")
print(df.pivot_table(values="y", index="balance_bin", aggfunc="mean"))

print("\n=== Subscription Rates by Both Loans ===")
print(df.groupby("both_loans")["y"].mean())

# ---------------------------------------
# 5. Prepare Model Data
# ---------------------------------------

X = df[
    [
        "housing",
        "loan",
        "balance",
        "both_loans",
        "housing_balance",
        "loan_balance",
        "balance_bin",
    ]
]

# balance_bin is categorical, encode it
X["balance_bin"] = X["balance_bin"].astype(int)

y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------
# 6. Balanced Logistic Regression
# ---------------------------------------

log_reg = LogisticRegression(class_weight="balanced", max_iter=300)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("\n=== Balanced Logistic Regression Report ===")
print(classification_report(y_test, y_pred_lr, zero_division=0))

print("\nLogistic Regression Coefficients:")
for feature, coef in zip(X.columns, log_reg.coef_[0]):
    print(f"{feature}: {coef}")

# ---------------------------------------
# 7. Balanced Decision Tree
# ---------------------------------------

tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced")
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("\n=== Balanced Decision Tree Report ===")
print(classification_report(y_test, y_pred_tree, zero_division=0))

# ---------------------------------------
# 8. Visualisations
# ---------------------------------------

# Class distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="y")
plt.title("Class Distribution (0 = No, 1 = Yes)")
plt.tight_layout()
plt.show()

# Balance distribution
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="balance", kde=True)
plt.title("Balance Distribution")
plt.tight_layout()
plt.show()

# Subscription rate by housing & loan
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="housing", y="y", hue="loan")
plt.title("Subscription Rate by Housing & Loan")
plt.tight_layout()
plt.show()

# Subscription rate by balance bin
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="balance_bin", y="y")
plt.title("Subscription Rate by Balance Quartile")
plt.tight_layout()
plt.show()

# Decision tree plot
plt.figure(figsize=(14, 8))
plot_tree(tree, feature_names=X.columns, class_names=["no", "yes"], filled=True)
plt.title("Balanced Decision Tree for Subscription Prediction")
plt.show()
