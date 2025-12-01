"""
Bank Marketing Analytics: Term Deposit Subscription Prediction
Focused analysis of housing loans, personal loans, and balance impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score,
                             accuracy_score, f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Color palette - dusky blues and greens
COLORS = {
    'dark_blue': '#2C5F7C',
    'medium_blue': '#4A8DAB', 
    'light_blue': '#6BAED6',
    'dark_green': '#2D5A4A',
    'medium_green': '#4A8B7C',
    'light_green': '#7BC9B5',
    'slate': '#5C7080',
    'muted_teal': '#3D7A7A'
}

# Load and prepare data
df = pd.read_excel('Bank Marketing Data.xlsx')
df['y_encoded'] = (df['y'] == 'yes').astype(int)
df['housing_encoded'] = (df['housing'] == 'yes').astype(int)
df['loan_encoded'] = (df['loan'] == 'yes').astype(int)

def categorize_loans(row):
    if row['housing'] == 'yes' and row['loan'] == 'yes':
        return 'Both Loans'
    elif row['housing'] == 'yes':
        return 'Housing Only'
    elif row['loan'] == 'yes':
        return 'Personal Only'
    else:
        return 'No Loans'

df['loan_status'] = df.apply(categorize_loans, axis=1)

df['balance_category'] = pd.cut(df['balance'], 
                                 bins=[-float('inf'), 0, 500, 1500, 5000, float('inf')],
                                 labels=['Negative', 'Low\n(€0-500)', 'Medium\n(€500-1.5K)', 
                                        'High\n(€1.5K-5K)', 'Very High\n(€5K+)'])

# Encode features
le_dict = {}
for col in ['job', 'marital', 'education', 'contact', 'month', 'poutcome']:
    le_dict[col] = LabelEncoder()
    df[f'{col}_encoded'] = le_dict[col].fit_transform(df[col])

feature_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous',
                'housing_encoded', 'loan_encoded', 'job_encoded', 'marital_encoded', 
                'education_encoded', 'contact_encoded', 'month_encoded', 'poutcome_encoded']

X = df[feature_cols]
y = df['y_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42)
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train_smote)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

# -----------------------------------------------------------------------------
# FIGURE 1: Subscription Rate by Loan Status
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

loan_sub = df.groupby('loan_status')['y_encoded'].agg(['sum', 'count'])
loan_sub['rate'] = loan_sub['sum'] / loan_sub['count'] * 100
loan_order = ['No Loans', 'Housing Only', 'Personal Only', 'Both Loans']
loan_sub = loan_sub.reindex(loan_order)

bar_colors = [COLORS['dark_green'], COLORS['medium_green'], COLORS['medium_blue'], COLORS['dark_blue']]
bars = ax.bar(loan_sub.index, loan_sub['rate'], color=bar_colors, edgecolor='white', linewidth=1.5)

for bar, rate, count in zip(bars, loan_sub['rate'], loan_sub['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4, 
            f'{rate:.1f}%', ha='center', fontsize=14, fontweight='bold', color=COLORS['slate'])
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
            f'n={count}', ha='center', fontsize=11, color='white', fontweight='bold')

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_xlabel('')
ax.set_title('Term Deposit Subscription Rate by Existing Loan Status', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_ylim(0, max(loan_sub['rate']) * 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('01_subscription_by_loan_status.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 2: Balance Comparison (Boxplot only - clean)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

df_plot = df[df['balance'].between(-1000, 8000)].copy()
df_plot['Subscribed'] = df_plot['y'].map({'no': 'No', 'yes': 'Yes'})

bp = ax.boxplot([df_plot[df_plot['Subscribed'] == 'No']['balance'],
                  df_plot[df_plot['Subscribed'] == 'Yes']['balance']],
                 labels=['Did Not Subscribe', 'Subscribed'],
                 patch_artist=True, widths=0.5)

bp['boxes'][0].set_facecolor(COLORS['medium_blue'])
bp['boxes'][1].set_facecolor(COLORS['dark_green'])
for box in bp['boxes']:
    box.set_alpha(0.8)
for median in bp['medians']:
    median.set_color('white')
    median.set_linewidth(2)

# Add mean markers
means = [df_plot[df_plot['Subscribed'] == 'No']['balance'].mean(),
         df_plot[df_plot['Subscribed'] == 'Yes']['balance'].mean()]
ax.scatter([1, 2], means, color=COLORS['light_green'], s=100, zorder=5, marker='D', edgecolor='white', linewidth=1.5)

ax.set_ylabel('Account Balance (€)', fontsize=12, color=COLORS['slate'])
ax.set_title('Account Balance: Subscribers vs Non-Subscribers', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

# Legend for mean
ax.plot([], [], 'D', color=COLORS['light_green'], markersize=8, label='Mean')
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('02_balance_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 3: Subscription Rate by Balance Category
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

bal_sub = df.groupby('balance_category', observed=True)['y_encoded'].agg(['sum', 'count'])
bal_sub['rate'] = bal_sub['sum'] / bal_sub['count'] * 100

gradient_colors = [COLORS['dark_blue'], COLORS['medium_blue'], COLORS['muted_teal'], 
                   COLORS['medium_green'], COLORS['dark_green']]

bars = ax.bar(range(len(bal_sub)), bal_sub['rate'], color=gradient_colors, 
              edgecolor='white', linewidth=1.5)
ax.set_xticks(range(len(bal_sub)))
ax.set_xticklabels(bal_sub.index, fontsize=10)

for bar, rate, count in zip(bars, bal_sub['rate'], bal_sub['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold', color=COLORS['slate'])

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_xlabel('Account Balance Category', fontsize=12, color=COLORS['slate'])
ax.set_title('Higher Balance = Higher Subscription Likelihood', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('03_subscription_by_balance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 4: ROC Curves
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

model_colors = [COLORS['dark_blue'], COLORS['dark_green'], COLORS['muted_teal']]
for (name, res), color in zip(results.items(), model_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], color=COLORS['slate'], linestyle='--', lw=1.5, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, color=COLORS['slate'])
ax.set_ylabel('True Positive Rate', fontsize=12, color=COLORS['slate'])
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])
ax.set_facecolor('#f8fafa')

plt.tight_layout()
plt.savefig('04_roc_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 5: Precision-Recall Curves
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

for (name, res), color in zip(results.items(), model_colors):
    precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
    ap = average_precision_score(y_test, res['y_prob'])
    ax.plot(recall, precision, color=color, lw=2.5, label=f'{name} (AP = {ap:.3f})')

no_skill = len(y_test[y_test==1]) / len(y_test)
ax.axhline(y=no_skill, color=COLORS['slate'], linestyle='--', lw=1.5, alpha=0.5, label=f'Baseline ({no_skill:.2f})')

ax.set_xlabel('Recall', fontsize=12, color=COLORS['slate'])
ax.set_ylabel('Precision', fontsize=12, color=COLORS['slate'])
ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.legend(loc='upper right', fontsize=11, frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])
ax.set_facecolor('#f8fafa')

plt.tight_layout()
plt.savefig('05_precision_recall_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 6: Confusion Matrices
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (name, res), color in zip(axes, results.items(), model_colors):
    cm = confusion_matrix(y_test, res['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                annot_kws={'size': 14, 'fontweight': 'bold'},
                cbar=False, linewidths=2, linecolor='white')
    
    ax.set_xlabel('Predicted', fontsize=11, color=COLORS['slate'])
    ax.set_ylabel('Actual', fontsize=11, color=COLORS['slate'])
    ax.set_title(f'{name}\nAccuracy: {res["accuracy"]:.1%}  |  F1: {res["f1"]:.3f}', 
                 fontsize=11, fontweight='bold', color=COLORS['dark_blue'])
    ax.tick_params(colors=COLORS['slate'])

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], y=1.02)
plt.tight_layout()
plt.savefig('06_confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 7: Key Feature Impact (Loan & Balance Only)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

# Get Random Forest importance for key features
rf_model = results['Random Forest']['model']
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
})

key_features = importance_df[importance_df['feature'].isin(['balance', 'housing_encoded', 'loan_encoded'])]
key_features = key_features.copy()
key_features['label'] = key_features['feature'].map({
    'balance': 'Account Balance',
    'housing_encoded': 'Housing Loan',
    'loan_encoded': 'Personal Loan'
})
key_features = key_features.sort_values('importance', ascending=True)

bars = ax.barh(key_features['label'], key_features['importance'], 
               color=[COLORS['dark_green'], COLORS['medium_blue'], COLORS['dark_blue']],
               edgecolor='white', linewidth=1.5, height=0.6)

for bar, imp in zip(bars, key_features['importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
            f'{imp:.3f}', va='center', fontsize=12, fontweight='bold', color=COLORS['slate'])

ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12, color=COLORS['slate'])
ax.set_title('Impact of Loans & Balance on Subscription Prediction', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_xlim(0, max(key_features['importance']) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('07_key_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 8: Decision Tree (Simplified)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

dt_model = results['Decision Tree']['model']
feature_labels = ['Age', 'Balance', 'Duration', 'Campaign', 'Pdays', 'Previous',
                  'Housing Loan', 'Personal Loan', 'Job', 'Marital', 
                  'Education', 'Contact', 'Month', 'Prev Outcome']

plot_tree(dt_model, feature_names=feature_labels, class_names=['No', 'Yes'],
          filled=True, rounded=True, ax=ax, fontsize=9, max_depth=3,
          proportion=True, impurity=False)
ax.set_title('Decision Tree: How the Model Makes Predictions (Top 3 Levels)', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)

plt.tight_layout()
plt.savefig('08_decision_tree.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 9: Model Performance Summary
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(metrics))
width = 0.25

for i, ((name, res), color) in enumerate(zip(results.items(), model_colors)):
    values = [res['accuracy'], res['precision'], res['recall'], res['f1']]
    bars = ax.bar(x + i*width, values, width, label=name, color=color, 
                  edgecolor='white', linewidth=1)

ax.set_ylabel('Score', fontsize=12, color=COLORS['slate'])
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.set_ylim(0, 1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

# Add value labels
for i, (name, res) in enumerate(results.items()):
    values = [res['accuracy'], res['precision'], res['recall'], res['f1']]
    for j, val in enumerate(values):
        ax.text(x[j] + i*width, val + 0.02, f'{val:.2f}', ha='center', fontsize=8, color=COLORS['slate'])

plt.tight_layout()
plt.savefig('09_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 10: Cross-Validation Stability
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

cv_data = []
cv_labels = []
for name, model in models.items():
    if name == 'Logistic Regression':
        scores = cross_val_score(model, X_train_scaled, y_train_smote, cv=5, scoring='f1')
    else:
        scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='f1')
    cv_data.append(scores)
    cv_labels.append(name)

bp = ax.boxplot(cv_data, labels=cv_labels, patch_artist=True, widths=0.5)

for patch, color in zip(bp['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for median in bp['medians']:
    median.set_color('white')
    median.set_linewidth(2)

# Add individual points
for i, scores in enumerate(cv_data, 1):
    ax.scatter([i]*len(scores), scores, color='white', s=40, zorder=5, edgecolor=COLORS['slate'], linewidth=1)

ax.set_ylabel('F1 Score', fontsize=12, color=COLORS['slate'])
ax.set_title('5-Fold Cross-Validation: Model Stability', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('10_cross_validation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()