import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Create folders if they don't exist
os.makedirs('data/v2/linear/csv_files', exist_ok=True)
os.makedirs('data/v2/linear/images', exist_ok=True)

# Load existing data
timelogs = pd.read_csv('data/v2/user_timelogs_v2.csv')
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Define months for consistency
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Validate input data
print("Timelogs sample:")
print(timelogs.head())

# --- Historical Analysis ---

# 1. User-level: Hours logged per user, per month, per year
user_monthly_summary = timelogs.groupby(['year', 'month', 'username'])['timelog'].sum().reset_index()
user_monthly_summary['month'] = pd.Categorical(user_monthly_summary['month'], categories=months, ordered=True)

# Find top user per month and year
top_users = user_monthly_summary.loc[user_monthly_summary.groupby(['year', 'month'], observed=True)['timelog'].idxmax()]
top_users = top_users.sort_values(['year', 'month'])
top_users.to_csv('data/v2/linear/csv_files/v2_linear_top_users_historical.csv', index=False)

# 2. Project-level: Hours logged per project, per month, per year
project_monthly_summary = timelogs.groupby(['year', 'month', 'projectname'])['timelog'].sum().reset_index()
project_monthly_summary['month'] = pd.Categorical(project_monthly_summary['month'], categories=months, ordered=True)

# --- Predictions for 2025 using Linear Regression ---

def month_to_numeric(month):
    month_str = str(month)[:3]
    if month_str not in months:
        raise ValueError(f"Invalid month value: {month_str}")
    return months.index(month_str) + 1

def add_cyclical_features(df):
    if 'month' not in df.columns:
        raise KeyError("Column 'month' not found in DataFrame")
    df['month_num'] = df['month'].apply(month_to_numeric)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    return df

# 1. User-level predictions
user_predictions_2025 = []
model = LinearRegression()

for username in tqdm(timelogs['username'].unique(), desc="Predicting Users"):
    user_data = timelogs[timelogs['username'] == username]
    user_monthly = user_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
    user_monthly['month'] = user_monthly['month'].astype(str).str[:3]
    
    if len(user_monthly) < 3:
        user_predictions_2025.extend([(username, month, 0) for month in months])
        continue
    
    user_monthly = add_cyclical_features(user_monthly)
    X = user_monthly[['year', 'month_sin', 'month_cos']]
    y = user_monthly['timelog']
    
    model.fit(X, y)
    X_2025 = pd.DataFrame({'year': [2025] * 12, 'month': months})
    X_2025 = add_cyclical_features(X_2025)
    forecast = np.maximum(model.predict(X_2025[['year', 'month_sin', 'month_cos']]), 0)
    
    user_predictions_2025.extend([(username, month, hours) for month, hours in zip(months, forecast)])

user_pred_df = pd.DataFrame(user_predictions_2025, columns=['username', 'month', 'predicted_hours'])
user_pred_df['year'] = 2025
user_pred_df['month'] = pd.Categorical(user_pred_df['month'], categories=months, ordered=True)
top_users_2025 = user_pred_df.loc[user_pred_df.groupby('month', observed=True)['predicted_hours'].idxmax()]
top_users_2025.to_csv('data/v2/linear/csv_files/v2_linear_top_users_2025_prediction.csv', index=False)

# 2. Project-level predictions (all projects, Jan-Dec 2025)
project_predictions_2025 = []
model = LinearRegression()

for projectname in tqdm(timelogs['projectname'].unique(), desc="Predicting Projects"):
    project_data = timelogs[timelogs['projectname'] == projectname]
    project_monthly = project_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
    project_monthly['month'] = project_monthly['month'].astype(str).str[:3]
    
    if len(project_monthly) < 3:
        project_predictions_2025.extend([(projectname, month, 0) for month in months])
        continue
    
    project_monthly = add_cyclical_features(project_monthly)
    X = project_monthly[['year', 'month_sin', 'month_cos']]
    y = project_monthly['timelog']
    
    model.fit(X, y)
    X_2025 = pd.DataFrame({'year': [2025] * 12, 'month': months})
    X_2025 = add_cyclical_features(X_2025)
    forecast = np.maximum(model.predict(X_2025[['year', 'month_sin', 'month_cos']]), 0)
    
    project_predictions_2025.extend([(projectname, month, hours) for month, hours in zip(months, forecast)])

project_pred_df = pd.DataFrame(project_predictions_2025, columns=['projectname', 'month', 'predicted_hours'])
project_pred_df['year'] = 2025
project_pred_df['month'] = pd.Categorical(project_pred_df['month'], categories=months, ordered=True)
project_pred_df.to_csv('data/v2/linear/csv_files/v2_linear_all_projects_2025_prediction.csv', index=False)

# --- Graphing Section ---

# 1. Historical Top Users Graph (2010-2017 and 2018-2024)
for period, years in [('2010-2017', range(2010, 2018)), ('2018-2024', range(2018, 2025))]:
    plt.figure(figsize=(12, 6))
    for year in years:
        year_data = top_users[top_users['year'] == year]
        plt.plot(year_data['month'], year_data['timelog'], label=str(year), marker='o')
    plt.title(f'Historical Top User Hours by Month ({period})')
    plt.xlabel('Month')
    plt.ylabel('Hours Logged by Top User')
    plt.legend(title='Year')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'data/v2/linear/images/v2_linear_historical_top_users_graph_{period}.png')
    plt.close()

# 2. Predicted Top Users 2025 Graph
plt.figure(figsize=(10, 5))
plt.plot(top_users_2025['month'], top_users_2025['predicted_hours'], marker='o', color='purple')
plt.title('Predicted Top User Hours for 2025 by Month')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/v2/linear/images/v2_linear_2025_top_users_prediction_graph.png')
plt.close()

# 3. Historical Project Hours Graph (Monthly trends like users)
for period, years in [('2010-2017', range(2010, 2018)), ('2018-2024', range(2018, 2025))]:
    plt.figure(figsize=(12, 6))
    for year in years:
        year_data = project_monthly_summary[project_monthly_summary['year'] == year]
        total_hours = year_data.groupby('month')['timelog'].sum()
        plt.plot(total_hours.index, total_hours.values, label=str(year), marker='o')
    plt.title(f'Historical Total Project Hours by Month ({period})')
    plt.xlabel('Month')
    plt.ylabel('Total Hours Logged')
    plt.legend(title='Year')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'data/v2/linear/images/v2_linear_historical_project_hours_graph_{period}.png')
    plt.close()

# 4. Predicted All Projects 2025 Graph
plt.figure(figsize=(12, 6))
for projectname in project_pred_df['projectname'].unique():
    project_data = project_pred_df[project_pred_df['projectname'] == projectname]
    plt.plot(project_data['month'], project_data['predicted_hours'], label=projectname, marker='o')
plt.title('Predicted Hours for All Projects in 2025 by Month')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.legend(title='Project', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/v2/linear/images/v2_linear_2025_all_projects_prediction_graph.png')
plt.close()

print("\nCSVs saved:")
print("  - data/v2/linear/csv_files/v2_linear_top_users_historical.csv")
print("  - data/v2/linear/csv_files/v2_linear_top_users_2025_prediction.csv")
print("  - data/v2/linear/csv_files/v2_linear_all_projects_2025_prediction.csv")
print("Images saved:")
print("  - data/v2/linear/images/v2_linear_historical_top_users_graph_2010-2017.png")
print("  - data/v2/linear/images/v2_linear_historical_top_users_graph_2018-2024.png")
print("  - data/v2/linear/images/v2_linear_2025_top_users_prediction_graph.png")
print("  - data/v2/linear/images/v2_linear_historical_project_hours_graph_2010-2017.png")
print("  - data/v2/linear/images/v2_linear_historical_project_hours_graph_2018-2024.png")
print("  - data/v2/linear/images/v2_linear_2025_all_projects_prediction_graph.png")