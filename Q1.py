"""
Bank Marketing Analytics: Question 1 - Financial Profile Impact
Does having existing loans (housing/personal) and balance affect term deposit subscription?
8 Data Visualization + 4 Model Comparison Charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
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

# Focus on financial profile features only
feature_cols = ['balance', 'housing_encoded', 'loan_encoded']
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
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
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

model_colors = [COLORS['dark_blue'], COLORS['dark_green'], COLORS['muted_teal']]

# =============================================================================
# DATA VISUALIZATION CHARTS (8 total) - All focused on financial profile
# =============================================================================

# -----------------------------------------------------------------------------
# FIGURE 1: Subscription Rate by Loan Status (Combined View)
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
# FIGURE 2: Housing Loan Impact
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

housing_sub = df.groupby('housing')['y_encoded'].agg(['sum', 'count'])
housing_sub['rate'] = housing_sub['sum'] / housing_sub['count'] * 100

colors = [COLORS['dark_green'], COLORS['dark_blue']]
bars = ax.bar(['No Housing Loan', 'Has Housing Loan'], 
              [housing_sub.loc['no', 'rate'], housing_sub.loc['yes', 'rate']], 
              color=colors, edgecolor='white', linewidth=1.5)

for bar, rate, idx in zip(bars, [housing_sub.loc['no', 'rate'], housing_sub.loc['yes', 'rate']], ['no', 'yes']):
    count = housing_sub.loc[idx, 'count']
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{rate:.1f}%', ha='center', fontsize=14, fontweight='bold', color=COLORS['slate'])
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
            f'n={count}', ha='center', fontsize=11, color='white', fontweight='bold')

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_title('Housing Loan Impact on Term Deposit Subscription', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('02_housing_loan_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 3: Personal Loan Impact
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

loan_sub_personal = df.groupby('loan')['y_encoded'].agg(['sum', 'count'])
loan_sub_personal['rate'] = loan_sub_personal['sum'] / loan_sub_personal['count'] * 100

colors = [COLORS['medium_green'], COLORS['medium_blue']]
bars = ax.bar(['No Personal Loan', 'Has Personal Loan'], 
              [loan_sub_personal.loc['no', 'rate'], loan_sub_personal.loc['yes', 'rate']], 
              color=colors, edgecolor='white', linewidth=1.5)

for bar, rate, idx in zip(bars, [loan_sub_personal.loc['no', 'rate'], loan_sub_personal.loc['yes', 'rate']], ['no', 'yes']):
    count = loan_sub_personal.loc[idx, 'count']
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{rate:.1f}%', ha='center', fontsize=14, fontweight='bold', color=COLORS['slate'])
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
            f'n={count}', ha='center', fontsize=11, color='white', fontweight='bold')

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_title('Personal Loan Impact on Term Deposit Subscription', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('03_personal_loan_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 4: Subscription Rate by Balance Category
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
plt.savefig('04_subscription_by_balance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 5: Balance Distribution by Subscription Status
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

df_plot = df[df['balance'].between(-1000, 8000)].copy()

bp = ax.boxplot([df_plot[df_plot['y'] == 'no']['balance'],
                  df_plot[df_plot['y'] == 'yes']['balance']],
                 labels=['Did Not Subscribe', 'Subscribed'],
                 patch_artist=True, widths=0.5)

bp['boxes'][0].set_facecolor(COLORS['medium_blue'])
bp['boxes'][1].set_facecolor(COLORS['dark_green'])
for box in bp['boxes']:
    box.set_alpha(0.8)
for median in bp['medians']:
    median.set_color('white')
    median.set_linewidth(2)

means = [df_plot[df_plot['y'] == 'no']['balance'].mean(),
         df_plot[df_plot['y'] == 'yes']['balance'].mean()]
ax.scatter([1, 2], means, color=COLORS['light_green'], s=100, zorder=5, marker='D', edgecolor='white', linewidth=1.5)

ax.set_ylabel('Account Balance (€)', fontsize=12, color=COLORS['slate'])
ax.set_title('Account Balance: Subscribers vs Non-Subscribers', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])
ax.plot([], [], 'D', color=COLORS['light_green'], markersize=8, label='Mean')
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('05_balance_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 6: Balance by Loan Status (Interaction)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

loan_order = ['No Loans', 'Housing Only', 'Personal Only', 'Both Loans']
balance_by_loan = [df[df['loan_status'] == status]['balance'].median() for status in loan_order]
sub_rate_by_loan = [df[df['loan_status'] == status]['y_encoded'].mean() * 100 for status in loan_order]

ax2 = ax.twinx()

bars = ax.bar(loan_order, balance_by_loan, color=COLORS['medium_blue'], alpha=0.7, 
              edgecolor='white', linewidth=1.5, label='Median Balance')
line = ax2.plot(loan_order, sub_rate_by_loan, color=COLORS['dark_green'], linewidth=3, 
                marker='o', markersize=10, label='Subscription Rate')

ax.set_ylabel('Median Balance (€)', fontsize=12, color=COLORS['medium_blue'])
ax2.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['dark_green'])
ax.set_title('Balance & Subscription Rate by Loan Status', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.tick_params(axis='y', colors=COLORS['medium_blue'])
ax2.tick_params(axis='y', colors=COLORS['dark_green'])

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('06_balance_loan_interaction.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 7: Heatmap - Subscription Rate by Balance & Loan Status
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Create balance bins for heatmap
df['balance_bin'] = pd.cut(df['balance'], 
                           bins=[-float('inf'), 0, 1000, 3000, float('inf')],
                           labels=['Negative/Zero', 'Low (€0-1K)', 'Medium (€1-3K)', 'High (€3K+)'])

heatmap_data = df.pivot_table(values='y_encoded', 
                               index='balance_bin', 
                               columns='loan_status', 
                               aggfunc='mean') * 100

heatmap_data = heatmap_data[['No Loans', 'Housing Only', 'Personal Only', 'Both Loans']]

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='BuGn', ax=ax,
            cbar_kws={'label': 'Subscription Rate (%)'}, linewidths=1, linecolor='white',
            annot_kws={'size': 12, 'fontweight': 'bold'})

ax.set_xlabel('Loan Status', fontsize=12, color=COLORS['slate'])
ax.set_ylabel('Balance Category', fontsize=12, color=COLORS['slate'])
ax.set_title('Subscription Rate: Balance × Loan Status Interaction', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)

plt.tight_layout()
plt.savefig('07_balance_loan_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 8: Financial Profile Summary - Who Subscribes?
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate key metrics
profiles = {
    'High Balance\n+ No Loans': df[(df['balance'] > 3000) & (df['loan_status'] == 'No Loans')]['y_encoded'].mean() * 100,
    'High Balance\n+ Has Loans': df[(df['balance'] > 3000) & (df['loan_status'] != 'No Loans')]['y_encoded'].mean() * 100,
    'Low Balance\n+ No Loans': df[(df['balance'] <= 1000) & (df['loan_status'] == 'No Loans')]['y_encoded'].mean() * 100,
    'Low Balance\n+ Has Loans': df[(df['balance'] <= 1000) & (df['loan_status'] != 'No Loans')]['y_encoded'].mean() * 100,
}

colors = [COLORS['dark_green'], COLORS['medium_green'], COLORS['medium_blue'], COLORS['dark_blue']]
bars = ax.bar(profiles.keys(), profiles.values(), color=colors, edgecolor='white', linewidth=1.5)

for bar, rate in zip(bars, profiles.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{rate:.1f}%', ha='center', fontsize=13, fontweight='bold', color=COLORS['slate'])

ax.axhline(y=df['y_encoded'].mean() * 100, color=COLORS['slate'], linestyle='--', 
           linewidth=2, label=f'Overall Rate: {df["y_encoded"].mean()*100:.1f}%')

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_title('Financial Profile Summary: Who Subscribes to Term Deposits?', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('08_financial_profile_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# MODEL COMPARISON CHARTS (4 total)
# =============================================================================

# -----------------------------------------------------------------------------
# FIGURE 9: ROC Curves Comparison
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

for (name, res), color in zip(results.items(), model_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], color=COLORS['slate'], linestyle='--', lw=1.5, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, color=COLORS['slate'])
ax.set_ylabel('True Positive Rate', fontsize=12, color=COLORS['slate'])
ax.set_title('ROC Curve: Financial Profile Features Only', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])
ax.set_facecolor('#f8fafa')

plt.tight_layout()
plt.savefig('09_roc_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 10: Model Performance Metrics
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(metrics))
width = 0.25

for i, ((name, res), color) in enumerate(zip(results.items(), model_colors)):
    values = [res['accuracy'], res['precision'], res['recall'], res['f1']]
    bars = ax.bar(x + i*width, values, width, label=name, color=color, 
                  edgecolor='white', linewidth=1)
    for j, val in enumerate(values):
        ax.text(x[j] + i*width, val + 0.02, f'{val:.2f}', ha='center', fontsize=8, color=COLORS['slate'])

ax.set_ylabel('Score', fontsize=12, color=COLORS['slate'])
ax.set_title('Model Performance (Financial Profile Features)', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.set_ylim(0, 1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('10_model_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 11: Feature Importance (Financial Profile Features)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

rf_model = results['Random Forest']['model']
feature_labels = ['Account Balance', 'Housing Loan', 'Personal Loan']

importance_df = pd.DataFrame({
    'feature': feature_labels,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

colors = [COLORS['medium_blue'], COLORS['muted_teal'], COLORS['dark_green']]
bars = ax.barh(importance_df['feature'], importance_df['importance'], 
               color=colors, edgecolor='white', linewidth=1.5, height=0.5)

for bar, imp in zip(bars, importance_df['importance']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{imp:.3f}', va='center', fontsize=12, fontweight='bold', color=COLORS['slate'])

ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12, color=COLORS['slate'])
ax.set_title('Which Financial Factor Matters Most?', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_xlim(0, max(importance_df['importance']) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('11_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 12: Decision Tree Visualization
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

dt_model = results['Decision Tree']['model']

plot_tree(dt_model, feature_names=feature_labels, class_names=['No', 'Yes'],
          filled=True, rounded=True, ax=ax, fontsize=10, max_depth=3,
          proportion=True, impurity=False)
ax.set_title('Decision Tree: How Financial Profile Predicts Subscription', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)

plt.tight_layout()
plt.savefig('12_decision_tree.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ All 12 charts generated successfully!")
print("\n" + "="*60)
print("Question 1: Financial Profile Impact Analysis")
print("="*60)
print("\nData Visualization Charts (8):")
print("  01. Subscription by Combined Loan Status")
print("  02. Housing Loan Impact")
print("  03. Personal Loan Impact")
print("  04. Subscription by Balance Category")
print("  05. Balance Distribution (Subscribers vs Non)")
print("  06. Balance & Loan Status Interaction")
print("  07. Heatmap: Balance × Loan Status")
print("  08. Financial Profile Summary")
print("\nModel Comparison Charts (4):")
print("  09. ROC Curves")
print("  10. Model Performance Metrics")
print("  11. Feature Importance")
print("  12. Decision Tree Visualization")