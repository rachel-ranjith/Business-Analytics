import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Load the data
df = pd.read_excel('Bank Marketing Data.xlsx')

print("="*80)
print("BANK MARKETING CAMPAIGN ANALYSIS")
print("Focus: Impact of Loans and Balance on Term Deposit Subscription")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"Target Variable Distribution:")
print(df['y'].value_counts())
print(f"\nSubscription Rate: {(df['y']=='yes').sum()/len(df)*100:.2f}%")

# Create loan categories
df['loan_category'] = 'None'
df.loc[(df['housing']=='yes') & (df['loan']=='no'), 'loan_category'] = 'Housing Only'
df.loc[(df['housing']=='no') & (df['loan']=='yes'), 'loan_category'] = 'Personal Only'
df.loc[(df['housing']=='yes') & (df['loan']=='yes'), 'loan_category'] = 'Both Loans'

# Balance categories for clearer analysis
df['balance_category'] = pd.cut(df['balance'], 
                                 bins=[-np.inf, 0, 500, 2000, np.inf],
                                 labels=['Negative/Zero', 'Low (1-500)', 'Medium (501-2000)', 'High (>2000)'])

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Subscription rates by loan category
print("\n1. SUBSCRIPTION RATES BY LOAN CATEGORY:")
print("-" * 50)
loan_sub = df.groupby('loan_category')['y'].apply(lambda x: (x=='yes').sum()/len(x)*100)
for cat, rate in loan_sub.items():
    count = len(df[df['loan_category']==cat])
    print(f"  {cat:20s}: {rate:6.2f}% (n={count})")

# Subscription rates by balance category
print("\n2. SUBSCRIPTION RATES BY BALANCE CATEGORY:")
print("-" * 50)
balance_sub = df.groupby('balance_category')['y'].apply(lambda x: (x=='yes').sum()/len(x)*100)
for cat, rate in balance_sub.items():
    count = len(df[df['balance_category']==cat])
    print(f"  {cat:20s}: {rate:6.2f}% (n={count})")

# Average balance by subscription
print("\n3. AVERAGE BALANCE BY SUBSCRIPTION STATUS:")
print("-" * 50)
print(f"  Subscribed (yes):    €{df[df['y']=='yes']['balance'].mean():,.2f}")
print(f"  Not subscribed (no): €{df[df['y']=='no']['balance'].mean():,.2f}")
print(f"  Difference:          €{df[df['y']=='yes']['balance'].mean() - df[df['y']=='no']['balance'].mean():,.2f}")

# Prepare data for modeling
df_model = df.copy()
df_model['y_binary'] = (df_model['y'] == 'yes').astype(int)
df_model['housing_binary'] = (df_model['housing'] == 'yes').astype(int)
df_model['loan_binary'] = (df_model['loan'] == 'yes').astype(int)

# Features for modeling
X = df_model[['housing_binary', 'loan_binary', 'balance']]
y = df_model['y_binary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)

# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

print("\nCOEFFICIENTS (Standardized):")
print("-" * 50)
for feature, coef in zip(X.columns, log_reg.coef_[0]):
    odds_ratio = np.exp(coef)
    print(f"  {feature:20s}: {coef:8.4f} (Odds Ratio: {odds_ratio:.4f})")
print(f"  {'Intercept':20s}: {log_reg.intercept_[0]:8.4f}")

print("\nINTERPRETATION:")
print("-" * 50)
print("  • Housing Loan:", "DECREASES" if log_reg.coef_[0][0] < 0 else "INCREASES", 
      f"odds by {abs((np.exp(log_reg.coef_[0][0])-1)*100):.1f}%")
print("  • Personal Loan:", "DECREASES" if log_reg.coef_[0][1] < 0 else "INCREASES", 
      f"odds by {abs((np.exp(log_reg.coef_[0][1])-1)*100):.1f}%")
print("  • Balance: For each €1000 increase,", "DECREASES" if log_reg.coef_[0][2] < 0 else "INCREASES", 
      f"odds by {abs((np.exp(log_reg.coef_[0][2]*1000)-1)*100):.1f}%")

y_pred_log = log_reg.predict(X_test_scaled)
y_pred_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\nPERFORMANCE METRICS:")
print("-" * 50)
print(f"  Training Accuracy: {log_reg.score(X_train_scaled, y_train)*100:.2f}%")
print(f"  Test Accuracy:     {log_reg.score(X_test_scaled, y_test)*100:.2f}%")
print(f"  ROC-AUC Score:     {roc_auc_score(y_test, y_pred_proba_log):.4f}")

cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"  Cross-Val AUC:     {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("\n" + "="*80)
print("MODEL 2: DECISION TREE")
print("="*80)

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=100, random_state=42)
dt.fit(X_train, y_train)

print("\nFEATURE IMPORTANCES:")
print("-" * 50)
for feature, importance in zip(X.columns, dt.feature_importances_):
    print(f"  {feature:20s}: {importance:.4f}")

y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]

print("\nPERFORMANCE METRICS:")
print("-" * 50)
print(f"  Training Accuracy: {dt.score(X_train, y_train)*100:.2f}%")
print(f"  Test Accuracy:     {dt.score(X_test, y_test)*100:.2f}%")
print(f"  ROC-AUC Score:     {roc_auc_score(y_test, y_pred_proba_dt):.4f}")

print("\n" + "="*80)
print("MODEL 3: RANDOM FOREST")
print("="*80)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50, random_state=42)
rf.fit(X_train, y_train)

print("\nFEATURE IMPORTANCES:")
print("-" * 50)
for feature, importance in zip(X.columns, rf.feature_importances_):
    print(f"  {feature:20s}: {importance:.4f}")

y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\nPERFORMANCE METRICS:")
print("-" * 50)
print(f"  Training Accuracy: {rf.score(X_train, y_train)*100:.2f}%")
print(f"  Test Accuracy:     {rf.score(X_test, y_test)*100:.2f}%")
print(f"  ROC-AUC Score:     {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"  Cross-Val AUC:     {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std()*2:.4f})")

# VISUALIZATIONS
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)

# Figure 1: Subscription rates by loan category
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Subscription rate by loan category
loan_sub_df = df.groupby('loan_category')['y'].apply(lambda x: (x=='yes').sum()/len(x)*100).sort_values(ascending=False)
colors = sns.color_palette("RdYlGn", len(loan_sub_df))
axes[0, 0].barh(loan_sub_df.index, loan_sub_df.values, color=colors)
axes[0, 0].set_xlabel('Subscription Rate (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Term Deposit Subscription Rate by Loan Category', fontsize=14, fontweight='bold', pad=20)
axes[0, 0].set_xlim(0, max(loan_sub_df.values) * 1.1)
for i, v in enumerate(loan_sub_df.values):
    axes[0, 0].text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')

# Plot 2: Subscription rate by balance category
balance_sub_df = df.groupby('balance_category')['y'].apply(lambda x: (x=='yes').sum()/len(x)*100)
axes[0, 1].bar(range(len(balance_sub_df)), balance_sub_df.values, color=sns.color_palette("viridis", len(balance_sub_df)))
axes[0, 1].set_xticks(range(len(balance_sub_df)))
axes[0, 1].set_xticklabels(balance_sub_df.index, rotation=45, ha='right')
axes[0, 1].set_ylabel('Subscription Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Term Deposit Subscription Rate by Balance Category', fontsize=14, fontweight='bold', pad=20)
for i, v in enumerate(balance_sub_df.values):
    axes[0, 1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')

# Plot 3: Balance distribution by subscription
axes[1, 0].violinplot([df[df['y']=='no']['balance'], df[df['y']=='yes']['balance']], 
                       positions=[0, 1], showmeans=True, showmedians=True)
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(['Not Subscribed', 'Subscribed'])
axes[1, 0].set_ylabel('Balance (€)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Balance Distribution by Subscription Status', fontsize=14, fontweight='bold', pad=20)
axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Balance')
axes[1, 0].legend()

# Plot 4: Feature importance comparison
feature_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Logistic Regression': np.abs(log_reg.coef_[0]),
    'Decision Tree': dt.feature_importances_,
    'Random Forest': rf.feature_importances_
})
feature_imp_df_melted = feature_imp_df.melt(id_vars='Feature', var_name='Model', value_name='Importance')
feature_labels = {'housing_binary': 'Housing Loan', 'loan_binary': 'Personal Loan', 'balance': 'Balance'}
feature_imp_df_melted['Feature'] = feature_imp_df_melted['Feature'].map(feature_labels)

x_pos = np.arange(len(feature_labels))
width = 0.25
models = feature_imp_df_melted['Model'].unique()
for i, model in enumerate(models):
    model_data = feature_imp_df_melted[feature_imp_df_melted['Model'] == model]
    axes[1, 1].bar(x_pos + i*width, model_data['Importance'], width, label=model, alpha=0.8)

axes[1, 1].set_xlabel('Features', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Importance (Normalized)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold', pad=20)
axes[1, 1].set_xticks(x_pos + width)
axes[1, 1].set_xticklabels(feature_labels.values())
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('loan_analysis_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: loan_analysis_overview.png")
plt.show()

# Figure 2: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

ax.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_pred_proba_log):.3f})', linewidth=2)
ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_score(y_test, y_pred_proba_dt):.3f})', linewidth=2)
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_pred_proba_rf):.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves.png")
plt.show()

# Figure 3: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_cm = [
    ('Logistic Regression', y_pred_log),
    ('Decision Tree', y_pred_dt),
    ('Random Forest', y_pred_rf)
]

for idx, (name, y_pred) in enumerate(models_cm):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")
plt.show()

# Figure 4: Interaction effects
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Housing loan effect by balance quartile
df['balance_quartile'] = pd.qcut(df['balance'], 4, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])
housing_balance = df.groupby(['balance_quartile', 'housing'])['y'].apply(lambda x: (x=='yes').sum()/len(x)*100).unstack()
housing_balance.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Housing Loan Effect Across Balance Quartiles', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Balance Quartile', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Subscription Rate (%)', fontsize=11, fontweight='bold')
axes[0, 0].legend(title='Housing Loan', labels=['No', 'Yes'])
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

# Personal loan effect by balance quartile
personal_balance = df.groupby(['balance_quartile', 'loan'])['y'].apply(lambda x: (x=='yes').sum()/len(x)*100).unstack()
personal_balance.plot(kind='bar', ax=axes[0, 1], color=['#3498db', '#f39c12'])
axes[0, 1].set_title('Personal Loan Effect Across Balance Quartiles', fontsize=13, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Balance Quartile', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Subscription Rate (%)', fontsize=11, fontweight='bold')
axes[0, 1].legend(title='Personal Loan', labels=['No', 'Yes'])
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# Combined loan effect
loan_combo = df.groupby(['housing', 'loan'])['y'].apply(lambda x: (x=='yes').sum()/len(x)*100)
loan_combo_df = loan_combo.reset_index()
loan_combo_df['combo'] = loan_combo_df['housing'] + ' / ' + loan_combo_df['loan']
loan_combo_df.columns = ['housing', 'loan', 'rate', 'combo']
loan_combo_df = loan_combo_df.sort_values('rate', ascending=False)

axes[1, 0].barh(loan_combo_df['combo'], loan_combo_df['rate'], 
                color=sns.color_palette("coolwarm", len(loan_combo_df)))
axes[1, 0].set_xlabel('Subscription Rate (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Subscription Rate by Loan Combination\n(Housing / Personal)', 
                     fontsize=13, fontweight='bold', pad=15)
for i, v in enumerate(loan_combo_df['rate'].values):
    axes[1, 0].text(v + 0.3, i, f'{v:.2f}%', va='center', fontweight='bold')

# Balance vs Subscription scatter
sample_df = df.sample(min(1000, len(df)), random_state=42)
colors_scatter = ['#e74c3c' if y == 'yes' else '#3498db' for y in sample_df['y']]
axes[1, 1].scatter(sample_df['balance'], sample_df.index, c=colors_scatter, alpha=0.5, s=20)
axes[1, 1].axvline(x=df[df['y']=='yes']['balance'].mean(), color='#e74c3c', 
                   linestyle='--', linewidth=2, label=f'Avg (Subscribed): €{df[df["y"]=="yes"]["balance"].mean():.0f}')
axes[1, 1].axvline(x=df[df['y']=='no']['balance'].mean(), color='#3498db', 
                   linestyle='--', linewidth=2, label=f'Avg (Not Subscribed): €{df[df["y"]=="no"]["balance"].mean():.0f}')
axes[1, 1].set_xlabel('Balance (€)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Balance Distribution by Subscription Status\n(Sample of 1000)', 
                     fontsize=13, fontweight='bold', pad=15)
axes[1, 1].legend()
axes[1, 1].set_ylabel('Sample Index', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('interaction_effects.png', dpi=300, bbox_inches='tight')
print("✓ Saved: interaction_effects.png")
plt.show()