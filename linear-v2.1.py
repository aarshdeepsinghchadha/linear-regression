import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Create folder for last 5 years analysis
output_folder = 'data/v2/last_5_years_linear'
os.makedirs(f'{output_folder}/csv_files', exist_ok=True)
os.makedirs(f'{output_folder}/images', exist_ok=True)

# Load existing data
timelogs = pd.read_csv('data/v2/user_timelogs_v2.csv')
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Filter for last 5 years (2020-2024)
timelogs = timelogs[timelogs['year'].between(2020, 2024)]
print("Timelogs sample (2020-2024):")
print(timelogs.head())

# Define months for consistency
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# --- Historical Analysis (2020-2024) ---

# 1. User-level: Hours logged per user, per month, per year
user_monthly_summary = timelogs.groupby(['year', 'month', 'username'])['timelog'].sum().reset_index()
user_monthly_summary['month'] = pd.Categorical(user_monthly_summary['month'], categories=months, ordered=True)
top_users = user_monthly_summary.loc[user_monthly_summary.groupby(['year', 'month'], observed=True)['timelog'].idxmax()]
top_users = top_users.sort_values(['year', 'month'])
top_users.to_csv(f'{output_folder}/csv_files/last_5_years_top_users_historical.csv', index=False)

# 2. Project-level: Hours logged per project, per month, per year
project_monthly_summary = timelogs.groupby(['year', 'month', 'projectname'])['timelog'].sum().reset_index()
project_monthly_summary['month'] = pd.Categorical(project_monthly_summary['month'], categories=months, ordered=True)
project_monthly_summary.to_csv(f'{output_folder}/csv_files/last_5_years_project_historical.csv', index=False)

# --- Prediction Functions ---

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

# --- Prediction for 2024 (using 2020-2023) and 2025 (using 2020-2024) ---

def predict_year(training_data, predict_year, entity_col, desc):
    predictions = []
    model = LinearRegression()
    for entity in tqdm(training_data[entity_col].unique(), desc=desc):
        entity_data = training_data[training_data[entity_col] == entity]
        entity_monthly = entity_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
        entity_monthly['month'] = entity_monthly['month'].astype(str).str[:3]
        
        if len(entity_monthly) < 3:
            predictions.extend([(entity, month, 0) for month in months])
            continue
        
        entity_monthly = add_cyclical_features(entity_monthly)
        X = entity_monthly[['year', 'month_sin', 'month_cos']]
        y = entity_monthly['timelog']
        
        model.fit(X, y)
        X_pred = pd.DataFrame({'year': [predict_year] * 12, 'month': months})
        X_pred = add_cyclical_features(X_pred)
        forecast = np.maximum(model.predict(X_pred[['year', 'month_sin', 'month_cos']]), 0)
        
        predictions.extend([(entity, month, hours) for month, hours in zip(months, forecast)])
    return predictions

# Predict 2024 using 2020-2023 data
training_data_2024 = timelogs[timelogs['year'].between(2020, 2023)]

# User predictions for 2024
user_pred_2024 = predict_year(training_data_2024, 2024, 'username', "Predicting Users for 2024")
user_pred_df_2024 = pd.DataFrame(user_pred_2024, columns=['username', 'month', 'predicted_hours'])
user_pred_df_2024['year'] = 2024
user_pred_df_2024['month'] = pd.Categorical(user_pred_df_2024['month'], categories=months, ordered=True)
top_users_2024_pred = user_pred_df_2024.loc[user_pred_df_2024.groupby('month', observed=True)['predicted_hours'].idxmax()]
top_users_2024_pred.to_csv(f'{output_folder}/csv_files/last_5_years_top_users_2024_prediction.csv', index=False)

# Project predictions for 2024
project_pred_2024 = predict_year(training_data_2024, 2024, 'projectname', "Predicting Projects for 2024")
project_pred_df_2024 = pd.DataFrame(project_pred_2024, columns=['projectname', 'month', 'predicted_hours'])
project_pred_df_2024['year'] = 2024
project_pred_df_2024['month'] = pd.Categorical(project_pred_df_2024['month'], categories=months, ordered=True)
project_pred_df_2024.to_csv(f'{output_folder}/csv_files/last_5_years_all_projects_2024_prediction.csv', index=False)

# Predict 2025 using 2020-2024 data
training_data_2025 = timelogs[timelogs['year'].between(2020, 2024)]

# User predictions for 2025
user_pred_2025 = predict_year(training_data_2025, 2025, 'username', "Predicting Users for 2025")
user_pred_df_2025 = pd.DataFrame(user_pred_2025, columns=['username', 'month', 'predicted_hours'])
user_pred_df_2025['year'] = 2025
user_pred_df_2025['month'] = pd.Categorical(user_pred_df_2025['month'], categories=months, ordered=True)
top_users_2025_pred = user_pred_df_2025.loc[user_pred_df_2025.groupby('month', observed=True)['predicted_hours'].idxmax()]
top_users_2025_pred.to_csv(f'{output_folder}/csv_files/last_5_years_top_users_2025_prediction.csv', index=False)

# Project predictions for 2025
project_pred_2025 = predict_year(training_data_2025, 2025, 'projectname', "Predicting Projects for 2025")
project_pred_df_2025 = pd.DataFrame(project_pred_2025, columns=['projectname', 'month', 'predicted_hours'])
project_pred_df_2025['year'] = 2025
project_pred_df_2025['month'] = pd.Categorical(project_pred_df_2025['month'], categories=months, ordered=True)
project_pred_df_2025.to_csv(f'{output_folder}/csv_files/last_5_years_all_projects_2025_prediction.csv', index=False)

# --- Graphing Section ---

# 1. Historical Top Users (2020-2024) + Predicted 2025
plt.figure(figsize=(12, 6))
for year in range(2020, 2025):
    year_data = top_users[top_users['year'] == year]
    plt.plot(year_data['month'], year_data['timelog'], label=str(year), marker='o')
plt.plot(top_users_2025_pred['month'], top_users_2025_pred['predicted_hours'], label='Predicted 2025', marker='o', linestyle='--', color='purple')
plt.title('Top User Hours by Month (2020-2024 Historical + 2025 Prediction)')
plt.xlabel('Month')
plt.ylabel('Hours Logged by Top User')
plt.legend(title='Year')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/last_5_years_historical_top_users_with_2025_graph.png')
plt.close()

# 2. Predicted vs Actual Top Users 2024
actual_2024_users = top_users[top_users['year'] == 2024]
plt.figure(figsize=(12, 6))
plt.plot(actual_2024_users['month'], actual_2024_users['timelog'], label='Actual 2024', marker='o', color='blue')
plt.plot(top_users_2024_pred['month'], top_users_2024_pred['predicted_hours'], label='Predicted 2024', marker='o', color='orange')
plt.title('Top User Hours: Predicted vs Actual (2024)')
plt.xlabel('Month')
plt.ylabel('Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/last_5_years_2024_top_users_pred_vs_actual_graph.png')
plt.close()

# 3. Historical Project Hours (2020-2024) + Predicted 2025
plt.figure(figsize=(12, 6))
for year in range(2020, 2025):
    year_data = project_monthly_summary[project_monthly_summary['year'] == year]
    total_hours = year_data.groupby('month')['timelog'].sum()
    plt.plot(total_hours.index, total_hours.values, label=str(year), marker='o')
pred_2025_projects = project_pred_df_2025.groupby('month')['predicted_hours'].sum()
plt.plot(pred_2025_projects.index, pred_2025_projects.values, label='Predicted 2025', marker='o', linestyle='--', color='purple')
plt.title('Total Project Hours by Month (2020-2024 Historical + 2025 Prediction)')
plt.xlabel('Month')
plt.ylabel('Total Hours Logged')
plt.legend(title='Year')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/last_5_years_historical_project_hours_with_2025_graph.png')
plt.close()

# 4. Predicted vs Actual Project Hours 2024 (Total Hours)
actual_2024_projects = project_monthly_summary[project_monthly_summary['year'] == 2024].groupby('month')['timelog'].sum()
pred_2024_projects = project_pred_df_2024.groupby('month')['predicted_hours'].sum()
plt.figure(figsize=(12, 6))
plt.plot(actual_2024_projects.index, actual_2024_projects.values, label='Actual 2024', marker='o', color='blue')
plt.plot(pred_2024_projects.index, pred_2024_projects.values, label='Predicted 2024', marker='o', color='orange')
plt.title('Total Project Hours: Predicted vs Actual (2024)')
plt.xlabel('Month')
plt.ylabel('Total Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/last_5_years_2024_project_hours_pred_vs_actual_graph.png')
plt.close()

# 5. Predicted Top Users 2025 (standalone graph)
plt.figure(figsize=(12, 6))
plt.plot(top_users_2025_pred['month'], top_users_2025_pred['predicted_hours'], label='Predicted 2025', marker='o', color='purple')
plt.title('Predicted Top User Hours by Month (2025)')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/last_5_years_2025_top_users_prediction_graph.png')
plt.close()

# 6. Predicted Project Hours 2025 (standalone graph)
plt.figure(figsize=(12, 6))
plt.plot(pred_2025_projects.index, pred_2025_projects.values, label='Predicted 2025', marker='o', color='purple')
plt.title('Predicted Total Project Hours by Month (2025)')
plt.xlabel('Month')
plt.ylabel('Predicted Total Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/last_5_years_2025_project_hours_prediction_graph.png')
plt.close()

print(f"\nCSVs saved in '{output_folder}/csv_files':")
print("  - last_5_years_top_users_historical.csv")
print("  - last_5_years_project_historical.csv")
print("  - last_5_years_top_users_2024_prediction.csv")
print("  - last_5_years_all_projects_2024_prediction.csv")
print("  - last_5_years_top_users_2025_prediction.csv")
print("  - last_5_years_all_projects_2025_prediction.csv")
print(f"Images saved in '{output_folder}/images':")
print("  - last_5_years_historical_top_users_with_2025_graph.png")
print("  - last_5_years_2024_top_users_pred_vs_actual_graph.png")
print("  - last_5_years_historical_project_hours_with_2025_graph.png")
print("  - last_5_years_2024_project_hours_pred_vs_actual_graph.png")
print("  - last_5_years_2025_top_users_prediction_graph.png")
print("  - last_5_years_2025_project_hours_prediction_graph.png")