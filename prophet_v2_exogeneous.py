import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from tqdm import tqdm
import os

# Set matplotlib style
plt.style.use('seaborn-v0_8')

# Define months for consistency
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create new folder structure
output_folder = 'v7/forecasts'
os.makedirs(f'{output_folder}/csv_files', exist_ok=True)
os.makedirs(f'{output_folder}/images', exist_ok=True)

# Load data
timelogs = pd.read_csv('data/v4/user_timelogs_v4.csv')
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Use full dataset (2010-2024)
print("Timelogs sample (2010-2024):")
print(timelogs.head())

# Aggregate by month for users and projects, including projectid for users
monthly_user = timelogs.groupby(['username', 'year', 'month'])['timelog'].sum().reset_index()
monthly_user_project = timelogs.groupby(['username', 'year', 'month'])['projectid'].mean().reset_index()  # Average projectid per month
monthly_user = monthly_user.merge(monthly_user_project, on=['username', 'year', 'month'])
monthly_project = timelogs.groupby(['projectname', 'year', 'month'])['timelog'].sum().reset_index()

# Convert numeric month to string representation
monthly_user['month_str'] = monthly_user['month'].apply(lambda x: months[x-1])
monthly_user['month'] = pd.Categorical(monthly_user['month_str'], categories=months, ordered=True)
monthly_project['month_str'] = monthly_project['month'].apply(lambda x: months[x-1])
monthly_project['month'] = pd.Categorical(monthly_project['month_str'], categories=months, ordered=True)

# Check for NaN
print("\nChecking for NaN in monthly_user:")
print(monthly_user.isnull().sum())
print("\nChecking for NaN in monthly_project:")
print(monthly_project.isnull().sum())

# Function to prepare Prophet DataFrame
def prepare_prophet_df(df, group_col, value_col, regressor_cols=None):
    df = df.copy()
    df = df.dropna(subset=['year', 'month_str'])
    df['month_num'] = df['month_str'].apply(lambda x: months.index(x) + 1)
    df['ds'] = pd.to_datetime({'year': df['year'], 'month': df['month_num'], 'day': 1}, errors='coerce')
    if df['ds'].isnull().any():
        print(f"NaN found in 'ds' for {group_col}:")
        print(df[df['ds'].isnull()])
        raise ValueError("NaN detected in 'ds'")
    cols = [group_col, 'ds', value_col] + (regressor_cols or [])
    return df[cols].rename(columns={value_col: 'y'})

# Prepare data
user_data = prepare_prophet_df(monthly_user, 'username', 'timelog', ['projectid'])
project_data = prepare_prophet_df(monthly_project, 'projectname', 'timelog')

# Function to train and predict with Prophet
def forecast_2025(df, group_col, periods=12, regressor_cols=None):
    forecasts = {}
    for group in tqdm(df[group_col].unique(), desc=f"Forecasting {group_col}"):
        group_df = df[df[group_col] == group][['ds', 'y'] + (regressor_cols or [])]
        if len(group_df) < 2:
            print(f"Skipping {group} due to insufficient data points ({len(group_df)})")
            continue
        model = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
        # Add regressors if provided
        for col in (regressor_cols or []):
            model.add_regressor(col)
        model.fit(group_df)
        future = model.make_future_dataframe(periods=periods, freq='MS')
        # Extend regressors to future (simple mean here; could improve)
        for col in (regressor_cols or []):
            future[col] = group_df[col].mean()
        forecast = model.predict(future)
        forecast_2025 = forecast[forecast['ds'].dt.year == 2025][['ds', 'yhat']]
        forecast_2025[group_col] = group
        forecasts[group] = forecast_2025
    if not forecasts:
        raise ValueError(f"No forecasts generated for {group_col}")
    return pd.concat(forecasts.values(), ignore_index=True)

# Forecast
user_forecasts = forecast_2025(user_data, 'username', regressor_cols=['projectid'])
project_forecasts = forecast_2025(project_data, 'projectname')

# Save forecasts
user_forecasts.to_csv(f'{output_folder}/csv_files/user_forecasts_2025_v7.csv', index=False)
project_forecasts.to_csv(f'{output_folder}/csv_files/project_forecasts_2025_v7.csv', index=False)

# Historical and Predicted Analysis
top_users_historical = monthly_user.loc[monthly_user.groupby(['year', 'month'], observed=True)['timelog'].idxmax()]
top_users_historical = top_users_historical.sort_values(['year', 'month'])
top_users_historical.to_csv(f'{output_folder}/csv_files/historical_top_users_2010_2024_v7.csv', index=False)

user_forecasts['month'] = user_forecasts['ds'].dt.strftime('%b')
user_forecasts['month'] = pd.Categorical(user_forecasts['month'], categories=months, ordered=True)
user_forecasts['year'] = user_forecasts['ds'].dt.year
top_users_2025_pred = user_forecasts.loc[user_forecasts.groupby('month', observed=True)['yhat'].idxmax()]
top_users_2025_pred.to_csv(f'{output_folder}/csv_files/predicted_top_users_2025_v7.csv', index=False)

# Graphing Section
# 1. Historical Top Users (2010-2024)
plt.figure(figsize=(12, 6))
for year in range(2010, 2025):
    year_data = top_users_historical[top_users_historical['year'] == year]
    plt.plot(year_data['month'], year_data['timelog'], label=str(year), marker='o')
plt.title('Historical Top User Hours by Month (2010-2024)')
plt.xlabel('Month')
plt.ylabel('Hours Logged by Top User')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/historical_top_users_2010_2024_v7.png')
plt.close()

# 2. Historical Total Project Hours (2010-2024)
project_totals = monthly_project.groupby(['year', 'month'])['timelog'].sum().reset_index()
plt.figure(figsize=(12, 6))
for year in range(2010, 2025):
    year_data = project_totals[project_totals['year'] == year]
    plt.plot(year_data['month'], year_data['timelog'], label=str(year), marker='o')
plt.title('Historical Total Project Hours by Month (2010-2024)')
plt.xlabel('Month')
plt.ylabel('Total Hours Logged')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/historical_project_hours_2010_2024_v7.png')
plt.close()

# 3. Predicted Top Users 2025
plt.figure(figsize=(12, 6))
plt.plot(top_users_2025_pred['month'], top_users_2025_pred['yhat'], 
         label='Predicted 2025', marker='o', linestyle='--', color='purple')
plt.title('Predicted Top User Hours by Month (2025)')
plt.xlabel('Month')
plt.ylabel('Predicted Hours by Top User')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/predicted_top_users_2025_v7.png')
plt.close()

# 4. Predicted Total Project Hours 2025
project_forecasts['month'] = project_forecasts['ds'].dt.strftime('%b')
project_forecasts['month'] = pd.Categorical(project_forecasts['month'], categories=months, ordered=True)
project_forecasts_totals = project_forecasts.groupby('month')['yhat'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(project_forecasts_totals['month'], project_forecasts_totals['yhat'], 
         label='Predicted 2025', marker='o', linestyle='--', color='purple')
plt.title('Predicted Total Project Hours by Month (2025)')
plt.xlabel('Month')
plt.ylabel('Predicted Total Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/images/predicted_project_hours_2025_v7.png')
plt.close()

# Output Summary
print(f"\nCSVs saved in '{output_folder}/csv_files':")
print("  - user_forecasts_2025_v7.csv")
print("  - project_forecasts_2025_v7.csv")
print("  - historical_top_users_2010_2024_v7.csv")
print("  - predicted_top_users_2025_v7.csv")
print(f"Images saved in '{output_folder}/images':")
print("  - historical_top_users_2010_2024_v7.png")
print("  - historical_project_hours_2010_2024_v7.png")
print("  - predicted_top_users_2025_v7.png")
print("  - predicted_project_hours_2025_v7.png")