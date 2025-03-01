import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Create new folder structure
output_folder = 'data/v3/forecast_2025'
os.makedirs(f'{output_folder}/csv_files', exist_ok=True)
os.makedirs(f'{output_folder}/images', exist_ok=True)

# Load data
timelogs = pd.read_csv('data/v2/user_timelogs_v2.csv')
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Filter for last 5 years (2020-2024)
timelogs = timelogs[timelogs['year'].between(2020, 2024)]
print("Timelogs sample (2020-2024):")
print(timelogs.head())

# Define months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# --- Historical Analysis ---

# Projects by month and project
project_monthly_summary = timelogs.groupby(['year', 'month', 'projectname'])['timelog'].sum().reset_index()
project_monthly_summary['month'] = pd.Categorical(project_monthly_summary['month'], categories=months, ordered=True)
project_monthly_summary.to_csv(f'{output_folder}/csv_files/historical_projects_2020_2024.csv', index=False)

# Users by month (top users)
user_monthly_summary = timelogs.groupby(['year', 'month', 'username'])['timelog'].sum().reset_index()
user_monthly_summary['month'] = pd.Categorical(user_monthly_summary['month'], categories=months, ordered=True)
top_users_historical = user_monthly_summary.loc[user_monthly_summary.groupby(['year', 'month'], observed=True)['timelog'].idxmax()]
top_users_historical = top_users_historical.sort_values(['year', 'month'])
top_users_historical.to_csv(f'{output_folder}/csv_files/historical_top_users_2020_2024.csv', index=False)

# --- Prediction Functions ---

def month_to_numeric(month):
    month_str = str(month)[:3]
    if month_str not in months:
        raise ValueError(f"Invalid month value: {month_str}")
    return months.index(month_str) + 1

def add_cyclical_features(df):
    df['month_num'] = df['month'].apply(month_to_numeric)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    return df

def predict_year(training_data, predict_year, entity_col, desc, entity_type='user'):
    predictions = []
    model = LinearRegression()
    overall_mean = training_data.groupby(['year', 'month'])['timelog'].sum().mean() / training_data[entity_col].nunique()  # Fallback mean per entity
    
    for entity in tqdm(training_data[entity_col].unique(), desc=desc):
        entity_data = training_data[training_data[entity_col] == entity]
        entity_monthly = entity_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
        entity_monthly['month'] = entity_monthly['month'].astype(str).str[:3]
        
        # If too little data, use a smoothed average instead of 0
        if len(entity_monthly) < 6:  # Adjusted threshold
            entity_avg = entity_monthly['timelog'].mean() if len(entity_monthly) > 0 else overall_mean
            predictions.extend([(entity, month, entity_avg * (1 + (predict_year - 2024) * 0.05)) for month in months])  # Slight growth assumption
            continue
        
        entity_monthly = add_cyclical_features(entity_monthly)
        # Add a trend feature: years since start
        entity_monthly['years_since_start'] = entity_monthly['year'] - 2020
        X = entity_monthly[['years_since_start', 'month_sin', 'month_cos']]
        y = entity_monthly['timelog']
        
        model.fit(X, y)
        X_pred = pd.DataFrame({'year': [predict_year] * 12, 'month': months})
        X_pred = add_cyclical_features(X_pred)
        X_pred['years_since_start'] = predict_year - 2020
        forecast = np.maximum(model.predict(X_pred[['years_since_start', 'month_sin', 'month_cos']]), 0)
        
        predictions.extend([(entity, month, hours) for month, hours in zip(months, forecast)])
    
    # Save all predictions for inspection
    pred_df = pd.DataFrame(predictions, columns=[entity_col, 'month', 'predicted_hours'])
    pred_df['year'] = predict_year
    pred_df.to_csv(f'{output_folder}/csv_files/debug_all_{entity_type}_predictions_2025.csv', index=False)
    return predictions

# --- Predictions for 2025 ---

training_data = timelogs[timelogs['year'].between(2020, 2024)]

# Predict users for 2025
user_pred_2025 = predict_year(training_data, 2025, 'username', "Predicting Users for 2025", entity_type='user')
user_pred_df_2025 = pd.DataFrame(user_pred_2025, columns=['username', 'month', 'predicted_hours'])
user_pred_df_2025['year'] = 2025
user_pred_df_2025['month'] = pd.Categorical(user_pred_df_2025['month'], categories=months, ordered=True)
top_users_2025_pred = user_pred_df_2025.loc[user_pred_df_2025.groupby('month', observed=True)['predicted_hours'].idxmax()]
top_users_2025_pred.to_csv(f'{output_folder}/csv_files/predicted_top_users_2025.csv', index=False)

# Predict projects for 2025 (unchanged for now)
project_pred_2025 = predict_year(training_data, 2025, 'projectname', "Predicting Projects for 2025", entity_type='project')
project_pred_df_2025 = pd.DataFrame(project_pred_2025, columns=['projectname', 'month', 'predicted_hours'])
project_pred_df_2025['year'] = 2025
project_pred_df_2025['month'] = pd.Categorical(project_pred_df_2025['month'], categories=months, ordered=True)
project_pred_df_2025.to_csv(f'{output_folder}/csv_files/predicted_projects_2025.csv', index=False)

# --- Graphing Section ---

# Historical Top Users + Predicted 2025
plt.figure(figsize=(12, 6))
for year in range(2020, 2025):
    year_data = top_users_historical[top_users_historical['year'] == year]
    plt.plot(year_data['month'], year_data['timelog'], label=str(year), marker='o')
plt.plot(top_users_2025_pred['month'], top_users_2025_pred['predicted_hours'], label='Predicted 2025', marker='o', linestyle='--', color='purple')
plt.title('Top User Hours by Month (2020-2024 Historical + 2025 Prediction)')
plt.xlabel('Month')
plt.ylabel('Hours Logged by Top User')
plt.legend(title='Year')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/historical_top_users_with_2025.png')
plt.close()

# Historical Project Hours + Predicted 2025
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
plt.savefig(f'{output_folder}/images/historical_project_hours_with_2025.png')
plt.close()

# Predicted Top Users 2025
plt.figure(figsize=(12, 6))
plt.plot(top_users_2025_pred['month'], top_users_2025_pred['predicted_hours'], label='Predicted 2025', marker='o', color='purple')
plt.title('Predicted Top User Hours by Month (2025)')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/predicted_top_users_2025.png')
plt.close()

# Predicted Project Hours 2025
plt.figure(figsize=(12, 6))
plt.plot(pred_2025_projects.index, pred_2025_projects.values, label='Predicted 2025', marker='o', color='purple')
plt.title('Predicted Total Project Hours by Month (2025)')
plt.xlabel('Month')
plt.ylabel('Predicted Total Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/predicted_project_hours_2025.png')
plt.close()

print(f"\nCSVs saved in '{output_folder}/csv_files':")
print("  - historical_projects_2020_2024.csv")
print("  - historical_top_users_2020_2024.csv")
print("  - predicted_top_users_2025.csv")
print("  - predicted_projects_2025.csv")
print("  - debug_all_user_predictions_2025.csv (for validation)")
print("  - debug_all_project_predictions_2025.csv (for validation)")
print(f"Images saved in '{output_folder}/images':")
print("  - historical_top_users_with_2025.png")
print("  - historical_project_hours_with_2025.png")
print("  - predicted_top_users_2025.png")
print("  - predicted_project_hours_2025.png")