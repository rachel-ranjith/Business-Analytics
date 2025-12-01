"""
Bank Marketing Analytics: Campaign Effectiveness Analysis
Does date, duration, and contact type affect success rate?
Using Logistic Regression, Random Forest, and K-Means Clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score,
                             accuracy_score, f1_score, precision_score, recall_score,
                             silhouette_score)
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
model_colors = [COLORS['dark_blue'], COLORS['dark_green'], COLORS['muted_teal']]

# Load and prepare data
df = pd.read_excel('Bank Marketing Data.xlsx')
df['y_encoded'] = (df['y'] == 'yes').astype(int)

# Create useful date/time features
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
df['month_num'] = df['month'].apply(lambda x: month_order.index(x) + 1)

# Quarter
df['quarter'] = pd.cut(df['month_num'], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Day of month categories
df['day_category'] = pd.cut(df['day'], bins=[0, 10, 20, 31], labels=['Early (1-10)', 'Mid (11-20)', 'Late (21-31)'])

# Duration categories (in minutes)
df['duration_mins'] = df['duration'] / 60
df['duration_category'] = pd.cut(df['duration_mins'], 
                                  bins=[0, 2, 5, 10, float('inf')],
                                  labels=['Short (<2 min)', 'Medium (2-5 min)', 
                                         'Long (5-10 min)', 'Very Long (10+ min)'])

# Campaign intensity
df['campaign_category'] = pd.cut(df['campaign'], 
                                  bins=[0, 1, 3, 6, float('inf')],
                                  labels=['Single Contact', '2-3 Contacts', 
                                         '4-6 Contacts', '7+ Contacts'])

# Encode categorical variables
le_contact = LabelEncoder()
le_month = LabelEncoder()
le_poutcome = LabelEncoder()

df['contact_encoded'] = le_contact.fit_transform(df['contact'])
df['month_encoded'] = le_month.fit_transform(df['month'])
df['poutcome_encoded'] = le_poutcome.fit_transform(df['poutcome'])

# Features for campaign analysis
campaign_features = ['duration', 'campaign', 'day', 'month_num', 'contact_encoded', 
                     'pdays', 'previous', 'poutcome_encoded']

X = df[campaign_features]
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
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
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

# Clustering on successful campaigns
successful = df[df['y_encoded'] == 1][['duration_mins', 'campaign', 'day', 'month_num']].copy()
scaler_cluster = StandardScaler()
successful_scaled = scaler_cluster.fit_transform(successful)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
successful['cluster'] = kmeans.fit_predict(successful_scaled)


# -----------------------------------------------------------------------------
# FIGURE 1: Success Rate by Contact Type
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

contact_sub = df.groupby('contact')['y_encoded'].agg(['sum', 'count'])
contact_sub['rate'] = contact_sub['sum'] / contact_sub['count'] * 100
contact_sub = contact_sub.sort_values('rate', ascending=True)

bars = ax.barh(contact_sub.index, contact_sub['rate'], 
               color=[COLORS['dark_blue'], COLORS['medium_green'], COLORS['dark_green']],
               edgecolor='white', linewidth=1.5, height=0.5)

for bar, rate, count in zip(bars, contact_sub['rate'], contact_sub['count']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
            f'{rate:.1f}% (n={count})', va='center', fontsize=12, fontweight='bold', color=COLORS['slate'])

ax.set_xlabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_title('Success Rate by Contact Method', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_xlim(0, max(contact_sub['rate']) * 1.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('01_success_by_contact_type.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 2: Success Rate by Month
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

month_sub = df.groupby('month')['y_encoded'].agg(['sum', 'count'])
month_sub['rate'] = month_sub['sum'] / month_sub['count'] * 100
month_sub = month_sub.reindex(month_order)

# Color gradient based on success rate
colors_month = [plt.cm.BuGn(0.3 + 0.6 * (r - month_sub['rate'].min()) / 
                            (month_sub['rate'].max() - month_sub['rate'].min())) 
                for r in month_sub['rate']]

bars = ax.bar(range(len(month_sub)), month_sub['rate'], color=colors_month, 
              edgecolor='white', linewidth=1.5)
ax.set_xticks(range(len(month_sub)))
ax.set_xticklabels([m.capitalize() for m in month_sub.index], fontsize=10)

for bar, rate in zip(bars, month_sub['rate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['slate'])

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_xlabel('Month', fontsize=12, color=COLORS['slate'])
ax.set_title('Campaign Success Rate by Month', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('02_success_by_month.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 3: Success Rate by Call Duration
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

dur_sub = df.groupby('duration_category', observed=True)['y_encoded'].agg(['sum', 'count'])
dur_sub['rate'] = dur_sub['sum'] / dur_sub['count'] * 100

gradient_colors = [COLORS['dark_blue'], COLORS['medium_blue'], COLORS['medium_green'], COLORS['dark_green']]

bars = ax.bar(range(len(dur_sub)), dur_sub['rate'], color=gradient_colors, 
              edgecolor='white', linewidth=1.5)
ax.set_xticks(range(len(dur_sub)))
ax.set_xticklabels(dur_sub.index, fontsize=10)

for bar, rate, count in zip(bars, dur_sub['rate'], dur_sub['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold', color=COLORS['slate'])
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
            f'n={count}', ha='center', fontsize=10, color='white', fontweight='bold')

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_xlabel('Call Duration', fontsize=12, color=COLORS['slate'])
ax.set_title('Longer Calls = Higher Success Rate', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('03_success_by_duration.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 4: Success Rate by Number of Contacts
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

camp_sub = df.groupby('campaign_category', observed=True)['y_encoded'].agg(['sum', 'count'])
camp_sub['rate'] = camp_sub['sum'] / camp_sub['count'] * 100

bars = ax.bar(range(len(camp_sub)), camp_sub['rate'], 
              color=[COLORS['dark_green'], COLORS['medium_green'], COLORS['medium_blue'], COLORS['dark_blue']],
              edgecolor='white', linewidth=1.5)
ax.set_xticks(range(len(camp_sub)))
ax.set_xticklabels(camp_sub.index, fontsize=10)

for bar, rate, count in zip(bars, camp_sub['rate'], camp_sub['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold', color=COLORS['slate'])
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
            f'n={count}', ha='center', fontsize=10, color='white', fontweight='bold')

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_xlabel('Number of Contacts in Campaign', fontsize=12, color=COLORS['slate'])
ax.set_title('Fewer Contacts = Higher Conversion', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('04_success_by_contacts.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 5: Success Rate by Day of Month
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

day_sub = df.groupby('day_category', observed=True)['y_encoded'].agg(['sum', 'count'])
day_sub['rate'] = day_sub['sum'] / day_sub['count'] * 100

bars = ax.bar(range(len(day_sub)), day_sub['rate'], 
              color=[COLORS['dark_blue'], COLORS['muted_teal'], COLORS['dark_green']],
              edgecolor='white', linewidth=1.5, width=0.6)
ax.set_xticks(range(len(day_sub)))
ax.set_xticklabels(day_sub.index, fontsize=11)

for bar, rate, count in zip(bars, day_sub['rate'], day_sub['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
            f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold', color=COLORS['slate'])
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
            f'n={count}', ha='center', fontsize=10, color='white', fontweight='bold')

ax.set_ylabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_xlabel('Day of Month', fontsize=12, color=COLORS['slate'])
ax.set_title('Success Rate by Day of Month', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('05_success_by_day.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 6: ROC Curves
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

for (name, res), color in zip(results.items(), model_colors[:2]):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], color=COLORS['slate'], linestyle='--', lw=1.5, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, color=COLORS['slate'])
ax.set_ylabel('True Positive Rate', fontsize=12, color=COLORS['slate'])
ax.set_title('ROC Curve: Campaign Features Only', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])
ax.set_facecolor('#f8fafa')

plt.tight_layout()
plt.savefig('06_roc_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 7: Precision-Recall Curves
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

for (name, res), color in zip(results.items(), model_colors[:2]):
    precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
    ap = average_precision_score(y_test, res['y_prob'])
    ax.plot(recall, precision, color=color, lw=2.5, label=f'{name} (AP = {ap:.3f})')

no_skill = len(y_test[y_test==1]) / len(y_test)
ax.axhline(y=no_skill, color=COLORS['slate'], linestyle='--', lw=1.5, alpha=0.5, label=f'Baseline ({no_skill:.2f})')

ax.set_xlabel('Recall', fontsize=12, color=COLORS['slate'])
ax.set_ylabel('Precision', fontsize=12, color=COLORS['slate'])
ax.set_title('Precision-Recall Curve: Campaign Features', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.legend(loc='upper right', fontsize=11, frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])
ax.set_facecolor('#f8fafa')

plt.tight_layout()
plt.savefig('07_precision_recall_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 8: Confusion Matrices
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (name, res), color in zip(axes, results.items(), model_colors[:2]):
    cm = confusion_matrix(y_test, res['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn', ax=ax,
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
plt.savefig('08_confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 9: Campaign Feature Importance
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

rf_model = results['Random Forest']['model']
importance_df = pd.DataFrame({
    'feature': campaign_features,
    'importance': rf_model.feature_importances_
})

feature_labels = {
    'duration': 'Call Duration',
    'campaign': '# of Contacts',
    'day': 'Day of Month',
    'month_num': 'Month',
    'contact_encoded': 'Contact Type',
    'pdays': 'Days Since Last Contact',
    'previous': 'Previous Contacts',
    'poutcome_encoded': 'Previous Outcome'
}
importance_df['label'] = importance_df['feature'].map(feature_labels)
importance_df = importance_df.sort_values('importance', ascending=True)

# Color by importance level
colors_imp = [plt.cm.BuGn(0.3 + 0.6 * i / len(importance_df)) for i in range(len(importance_df))]

bars = ax.barh(importance_df['label'], importance_df['importance'], 
               color=colors_imp, edgecolor='white', linewidth=1.5, height=0.7)

for bar, imp in zip(bars, importance_df['importance']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{imp:.3f}', va='center', fontsize=11, fontweight='bold', color=COLORS['slate'])

ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12, color=COLORS['slate'])
ax.set_title('Which Campaign Factors Matter Most?', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_xlim(0, max(importance_df['importance']) * 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('09_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 10: Successful Campaign Clusters
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

cluster_colors = [COLORS['dark_blue'], COLORS['dark_green'], COLORS['muted_teal']]
cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 3']

for cluster, color, name in zip([0, 1, 2], cluster_colors, cluster_names):
    subset = successful[successful['cluster'] == cluster]
    ax.scatter(subset['duration_mins'], subset['campaign'], 
               c=color, s=60, alpha=0.6, label=name, edgecolor='white', linewidth=0.5)

ax.set_xlabel('Call Duration (minutes)', fontsize=12, color=COLORS['slate'])
ax.set_ylabel('Number of Contacts', fontsize=12, color=COLORS['slate'])
ax.set_title('Clusters of Successful Campaigns\n(Duration vs Contact Frequency)', 
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])
ax.set_facecolor('#f8fafa')

plt.tight_layout()
plt.savefig('10_successful_clusters.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 11: Cluster Profiles
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))

cluster_profiles = successful.groupby('cluster').agg({
    'duration_mins': 'mean',
    'campaign': 'mean',
    'day': 'mean',
    'month_num': 'mean'
}).round(2)

cluster_profiles['count'] = successful.groupby('cluster').size()
cluster_profiles['pct'] = (cluster_profiles['count'] / cluster_profiles['count'].sum() * 100).round(1)

# Normalize for radar-like bar comparison
metrics = ['Avg Duration\n(mins)', 'Avg Contacts', 'Avg Day\nof Month', 'Avg Month']
x = np.arange(len(metrics))
width = 0.25

for i, (cluster, color) in enumerate(zip([0, 1, 2], cluster_colors)):
    values = [cluster_profiles.loc[cluster, 'duration_mins'],
              cluster_profiles.loc[cluster, 'campaign'],
              cluster_profiles.loc[cluster, 'day'],
              cluster_profiles.loc[cluster, 'month_num']]
    bars = ax.bar(x + i*width, values, width, label=f'Cluster {cluster+1} ({cluster_profiles.loc[cluster, "pct"]:.0f}%)', 
                  color=color, edgecolor='white', linewidth=1)

ax.set_ylabel('Value', fontsize=12, color=COLORS['slate'])
ax.set_title('Profile of Successful Campaign Clusters', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=10)
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['light_blue'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('11_cluster_profiles.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 12: Previous Outcome Impact
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

pout_sub = df.groupby('poutcome')['y_encoded'].agg(['sum', 'count'])
pout_sub['rate'] = pout_sub['sum'] / pout_sub['count'] * 100
pout_sub = pout_sub.sort_values('rate', ascending=True)

bars = ax.barh(pout_sub.index, pout_sub['rate'], 
               color=[COLORS['dark_blue'], COLORS['medium_blue'], COLORS['medium_green'], COLORS['dark_green']],
               edgecolor='white', linewidth=1.5, height=0.5)

for bar, rate, count in zip(bars, pout_sub['rate'], pout_sub['count']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{rate:.1f}% (n={count})', va='center', fontsize=11, fontweight='bold', color=COLORS['slate'])

ax.set_xlabel('Subscription Rate (%)', fontsize=12, color=COLORS['slate'])
ax.set_title('Previous Campaign Outcome → Current Success', fontsize=14, fontweight='bold', color=COLORS['dark_blue'], pad=15)
ax.set_xlim(0, max(pout_sub['rate']) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['slate'])

plt.tight_layout()
plt.savefig('12_previous_outcome_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 13: Key Insights Summary
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'KEY FINDINGS: Campaign Effectiveness', 
        ha='center', fontsize=16, fontweight='bold', color=COLORS['dark_blue'])

# Calculate key stats
best_month = month_sub['rate'].idxmax()
best_month_rate = month_sub['rate'].max()
worst_month = month_sub['rate'].idxmin()
cellular_rate = contact_sub.loc['cellular', 'rate'] if 'cellular' in contact_sub.index else 0
long_call_rate = dur_sub.loc['Very Long (10+ min)', 'rate'] if 'Very Long (10+ min)' in dur_sub.index else 0
single_contact_rate = camp_sub.loc['Single Contact', 'rate'] if 'Single Contact' in camp_sub.index else 0
success_prev = pout_sub.loc['success', 'rate'] if 'success' in pout_sub.index else 0

findings = [
    ('⏱️', 'Call Duration is #1 Predictor', 
     f'Calls 10+ min have {long_call_rate:.1f}% success vs {dur_sub["rate"].min():.1f}% for short calls', COLORS['dark_green']),
    ('📅', f'Best Month: {best_month.capitalize()}', 
     f'{best_month_rate:.1f}% conversion rate - worst is {worst_month.capitalize()}', COLORS['medium_green']),
    ('📞', 'Cellular Outperforms Other Contact Methods', 
     f'Cellular: {cellular_rate:.1f}% success rate', COLORS['dark_blue']),
    ('🔄', 'Previous Success = Future Success', 
     f'Customers who converted before: {success_prev:.1f}% conversion', COLORS['medium_blue']),
    ('📉', 'Less is More with Contact Frequency', 
     f'Single contact: {single_contact_rate:.1f}% vs diminishing returns with more calls', COLORS['muted_teal'])
]

for i, (icon, title, desc, color) in enumerate(findings):
    y_pos = 7.5 - i * 1.5
    
    rect = plt.Rectangle((0.5, y_pos - 0.5), 9, 1.2, facecolor=color, alpha=0.15, 
                          edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    
    ax.text(1.2, y_pos + 0.1, icon, fontsize=22, ha='center', va='center')
    ax.text(2, y_pos + 0.2, title, fontsize=12, fontweight='bold', color=color, va='center')
    ax.text(2, y_pos - 0.2, desc, fontsize=10, color=COLORS['slate'], va='center')

# Recommendation
rect = plt.Rectangle((0.5, 0.2), 9, 1.0, facecolor=COLORS['dark_green'], alpha=0.2,
                      edgecolor=COLORS['dark_green'], linewidth=2)
ax.add_patch(rect)
ax.text(5, 0.7, '💡 OPTIMIZE: Focus on quality (longer calls) over quantity, use cellular, target in peak months',
        ha='center', fontsize=11, fontweight='bold', color=COLORS['dark_green'])

plt.tight_layout()
plt.savefig('13_key_insights.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()