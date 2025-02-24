import pandas as pd
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Create folders if they don't exist
os.makedirs('data/sarimax/csv_files', exist_ok=True)
os.makedirs('data/sarimax/images', exist_ok=True)

# Load existing data
timelogs = pd.read_csv('data/user_timelogs.csv')
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Define months for consistency
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# --- Historical Analysis ---

# 1. User-level: Hours logged per user, per month, per year
user_monthly_summary = timelogs.groupby(['year', 'month', 'username'])['timelog'].sum().reset_index()
user_monthly_summary['month'] = pd.Categorical(user_monthly_summary['month'], categories=months, ordered=True)

# Find top user per month and year
top_users = user_monthly_summary.loc[user_monthly_summary.groupby(['year', 'month'], observed=False)['timelog'].idxmax()]
top_users = top_users.sort_values(['year', 'month'])

print("Top User per Month and Year (2016-2024):")
print(top_users)
top_users.to_csv(os.path.join('data/sarimax/csv_files', 'sarimax_top_users_historical.csv'), index=False)

# 2. Project-level: Hours logged per project, per user
project_user_summary = timelogs.groupby(['projectname', 'username'])['timelog'].sum().reset_index()
top_users_per_project = project_user_summary.loc[project_user_summary.groupby('projectname')['timelog'].idxmax()]

print("\nTop User per Project (2016-2024 Total Hours):")
print(top_users_per_project)
top_users_per_project.to_csv(os.path.join('data/sarimax/csv_files', 'sarimax_top_users_per_project_historical.csv'), index=False)

# --- Predictions for 2025 ---

# 1. User-level predictions with error handling
user_predictions_2025 = []
for username in timelogs['username'].unique():
    try:
        user_data = timelogs[timelogs['username'] == username]
        user_monthly = user_data.groupby(['year', 'month'])['timelog'].sum().reindex(
            pd.MultiIndex.from_product([range(2016, 2025), months], names=['year', 'month']),
            fill_value=0
        ).values
        # Simplified SARIMA model to improve convergence
        model = SARIMAX(user_monthly, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12))
        fitted_model = model.fit(disp=False, maxiter=50)  # Limit iterations
        forecast = fitted_model.forecast(steps=12)
        user_predictions_2025.extend([(username, month, hours) for month, hours in zip(months, forecast)])
    except Exception as e:
        print(f"Warning: Prediction failed for user {username}: {str(e)}")
        user_predictions_2025.extend([(username, month, 0) for month in months])  # Default to 0 if fails

user_pred_df = pd.DataFrame(user_predictions_2025, columns=['username', 'month', 'predicted_hours'])
user_pred_df['year'] = 2025
user_pred_df['month'] = pd.Categorical(user_pred_df['month'], categories=months, ordered=True)

# Find top user per month in 2025
top_users_2025 = user_pred_df.loc[user_pred_df.groupby('month', observed=False)['predicted_hours'].idxmax()]
top_users_2025 = top_users_2025.sort_values('month')

print("\nPredicted Top User per Month in 2025:")
print(top_users_2025)
top_users_2025.to_csv(os.path.join('data/sarimax/csv_files', 'sarimax_top_users_2025_prediction.csv'), index=False)

# 2. Project-level predictions with error handling
project_predictions_2025 = []
for projectname in timelogs['projectname'].unique():
    try:
        project_data = timelogs[timelogs['projectname'] == projectname]
        project_monthly = project_data.groupby(['year', 'month'])['timelog'].sum().reindex(
            pd.MultiIndex.from_product([range(2016, 2025), months], names=['year', 'month']),
            fill_value=0
        ).values
        # Simplified SARIMA model
        model = SARIMAX(project_monthly, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12))
        fitted_model = model.fit(disp=False, maxiter=50)
        forecast = fitted_model.forecast(steps=12)
        project_predictions_2025.extend([(projectname, month, hours) for month, hours in zip(months, forecast)])
    except Exception as e:
        print(f"Warning: Prediction failed for project {projectname}: {str(e)}")
        project_predictions_2025.extend([(projectname, month, 0) for month in months])  # Default to 0 if fails

project_pred_df = pd.DataFrame(project_predictions_2025, columns=['projectname', 'month', 'predicted_hours'])
project_pred_df['year'] = 2025
project_pred_df['month'] = pd.Categorical(project_pred_df['month'], categories=months, ordered=True)

# Find top project per month in 2025
top_projects_2025 = project_pred_df.loc[project_pred_df.groupby('month', observed=False)['predicted_hours'].idxmax()]
top_projects_2025 = top_projects_2025.sort_values('month')

print("\nPredicted Top Project per Month in 2025:")
print(top_projects_2025)
top_projects_2025.to_csv(os.path.join('data/sarimax/csv_files', 'sarimax_top_projects_2025_prediction.csv'), index=False)

# --- Graphing Section ---

# 1. Historical Top Users Graph
plt.figure(figsize=(12, 6))
for year in range(2016, 2025):
    year_data = top_users[top_users['year'] == year]
    plt.plot(year_data['month'], year_data['timelog'], label=str(year), marker='o')
plt.title('Historical Top User Hours by Month (2016-2024)')
plt.xlabel('Month')
plt.ylabel('Hours Logged by Top User')
plt.legend(title='Year')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('data/sarimax/images', 'sarimax_historical_top_users_graph.png'))
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
plt.savefig(os.path.join('data/sarimax/images', 'sarimax_2025_top_users_prediction_graph.png'))
plt.close()

# 3. Historical Project Hours Graph (total per project)
project_totals = timelogs.groupby('projectname')['timelog'].sum()
plt.figure(figsize=(10, 5))
project_totals.plot(kind='bar', color='skyblue')
plt.title('Total Hours Logged per Project (2016-2024)')
plt.xlabel('Project')
plt.ylabel('Total Hours')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('data/sarimax/images', 'sarimax_historical_project_totals_graph.png'))
plt.close()

# 4. Predicted Top Projects 2025 Graph
plt.figure(figsize=(10, 5))
plt.plot(top_projects_2025['month'], top_projects_2025['predicted_hours'], marker='o', color='green')
plt.title('Predicted Top Project Hours for 2025 by Month')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('data/sarimax/images', 'sarimax_2025_top_projects_prediction_graph.png'))
plt.close()

print("\nCSVs saved in 'data/sarimax/csv_files' folder:")
print("  - sarimax_top_users_historical.csv")
print("  - sarimax_top_users_per_project_historical.csv")
print("  - sarimax_top_users_2025_prediction.csv")
print("  - sarimax_top_projects_2025_prediction.csv")
print("Images saved in 'data/sarimax/images' folder:")
print("  - sarimax_historical_top_users_graph.png")
print("  - sarimax_2025_top_users_prediction_graph.png")
print("  - sarimax_historical_project_totals_graph.png")
print("  - sarimax_2025_top_projects_prediction_graph.png")