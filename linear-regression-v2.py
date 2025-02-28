import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars

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
print("Unique months in timelogs:", timelogs['date'].dt.strftime('%b').unique())

# --- Historical Analysis ---

# 1. User-level: Hours logged per user, per month, per year
user_monthly_summary = timelogs.groupby(['year', 'month', 'username'])['timelog'].sum().reset_index()
user_monthly_summary['month'] = pd.Categorical(user_monthly_summary['month'], categories=months, ordered=True)

# Find top user per month and year
top_users = user_monthly_summary.loc[user_monthly_summary.groupby(['year', 'month'], observed=True)['timelog'].idxmax()]
top_users = top_users.sort_values(['year', 'month'])

print("Top User per Month and Year (2010-2024):")
print(top_users)
top_users.to_csv('data/v2/linear/csv_files/v2_linear_top_users_historical.csv', index=False)

# 2. Project-level: Hours logged per project, per user
project_user_summary = timelogs.groupby(['projectname', 'username'])['timelog'].sum().reset_index()
top_users_per_project = project_user_summary.loc[project_user_summary.groupby('projectname')['timelog'].idxmax()]

print("\nTop User per Project (2010-2024 Total Hours):")
print(top_users_per_project)
top_users_per_project.to_csv('data/v2/linear/csv_files/v2_linear_top_users_per_project_historical.csv', index=False)

# --- Predictions for 2025 using Linear Regression ---

# Function to convert month to numeric (1-12)
def month_to_numeric(month):
    month_str = str(month)[:3]  # Handle 'January' -> 'Jan' or similar
    if month_str not in months:
        raise ValueError(f"Invalid month value: {month_str}")
    return months.index(month_str) + 1

# Improved feature engineering: Add month as cyclical feature
def add_cyclical_features(df):
    if 'month' not in df.columns:
        raise KeyError("Column 'month' not found in DataFrame")
    try:
        df['month_num'] = df['month'].apply(month_to_numeric)
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    except Exception as e:
        raise ValueError(f"Error processing cyclical features: {str(e)}")
    return df

# 1. User-level predictions
user_predictions_2025 = []
model = LinearRegression()

# Wrap the loop with tqdm for user predictions
for username in tqdm(timelogs['username'].unique(), desc="Predicting Users"):
    try:
        # Prepare data for this user
        user_data = timelogs[timelogs['username'] == username]
        user_monthly = user_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
        
        # Ensure 'month' is in the correct format
        if 'month' not in user_monthly.columns or user_monthly['month'].isnull().any():
            raise ValueError("Missing or invalid 'month' column")
        user_monthly['month'] = user_monthly['month'].astype(str).str[:3]  # Normalize to 'Jan', 'Feb', etc.
        
        # Skip if insufficient data
        if len(user_monthly) < 3:  # Need at least 3 points for 3 features
            print(f"Skipping user {username}: Insufficient data ({len(user_monthly)} rows)")
            user_predictions_2025.extend([(username, month, 0) for month in months])
            continue
        
        user_monthly = add_cyclical_features(user_monthly)
        
        # Features (X) and target (y)
        X = user_monthly[['year', 'month_sin', 'month_cos']]
        y = user_monthly['timelog']
        
        # Debug: Check data before fitting
        print(f"Debugging user {username}: X shape {X.shape}, y shape {y.shape}")
        
        # Fit the model
        model.fit(X, y)
        
        # Predict for 2025
        X_2025 = pd.DataFrame({
            'year': [2025] * 12,
            'month': months  # Directly use month names
        })
        X_2025 = add_cyclical_features(X_2025)
        forecast = model.predict(X_2025[['year', 'month_sin', 'month_cos']])
        
        # Ensure non-negative predictions
        forecast = np.maximum(forecast, 0)
        
        user_predictions_2025.extend([(username, month, hours) for month, hours in zip(months, forecast)])
    except Exception as e:
        print(f"Warning: Prediction failed for user {username}: {str(e)}")
        user_predictions_2025.extend([(username, month, 0) for month in months])

user_pred_df = pd.DataFrame(user_predictions_2025, columns=['username', 'month', 'predicted_hours'])
user_pred_df['year'] = 2025
user_pred_df['month'] = pd.Categorical(user_pred_df['month'], categories=months, ordered=True)

top_users_2025 = user_pred_df.loc[user_pred_df.groupby('month', observed=True)['predicted_hours'].idxmax()]
top_users_2025 = top_users_2025.sort_values('month')

print("\nPredicted Top User per Month in 2025:")
print(top_users_2025)
top_users_2025.to_csv('data/v2/linear/csv_files/v2_linear_top_users_2025_prediction.csv', index=False)

# 2. Project-level predictions
project_predictions_2025 = []
model = LinearRegression()

# Wrap the loop with tqdm for project predictions
for projectname in tqdm(timelogs['projectname'].unique(), desc="Predicting Projects"):
    try:
        # Prepare data for this project
        project_data = timelogs[timelogs['projectname'] == projectname]
        project_monthly = project_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
        
        # Ensure 'month' is in the correct format
        if 'month' not in project_monthly.columns or project_monthly['month'].isnull().any():
            raise ValueError("Missing or invalid 'month' column")
        project_monthly['month'] = project_monthly['month'].astype(str).str[:3]
        
        # Skip if insufficient data
        if len(project_monthly) < 3:
            print(f"Skipping project {projectname}: Insufficient data ({len(project_monthly)} rows)")
            project_predictions_2025.extend([(projectname, month, 0) for month in months])
            continue
        
        project_monthly = add_cyclical_features(project_monthly)
        
        # Features (X) and target (y)
        X = project_monthly[['year', 'month_sin', 'month_cos']]
        y = project_monthly['timelog']
        
        # Debug: Check data before fitting
        print(f"Debugging project {projectname}: X shape {X.shape}, y shape {y.shape}")
        
        # Fit the model
        model.fit(X, y)
        
        # Predict for 2025
        X_2025 = pd.DataFrame({
            'year': [2025] * 12,
            'month': months
        })
        X_2025 = add_cyclical_features(X_2025)
        forecast = model.predict(X_2025[['year', 'month_sin', 'month_cos']])
        
        # Ensure non-negative predictions
        forecast = np.maximum(forecast, 0)
        
        project_predictions_2025.extend([(projectname, month, hours) for month, hours in zip(months, forecast)])
    except Exception as e:
        print(f"Warning: Prediction failed for project {projectname}: {str(e)}")
        project_predictions_2025.extend([(projectname, month, 0) for month in months])

project_pred_df = pd.DataFrame(project_predictions_2025, columns=['projectname', 'month', 'predicted_hours'])
project_pred_df['year'] = 2025
project_pred_df['month'] = pd.Categorical(project_pred_df['month'], categories=months, ordered=True)

top_projects_2025 = project_pred_df.loc[project_pred_df.groupby('month', observed=True)['predicted_hours'].idxmax()]
top_projects_2025 = top_projects_2025.sort_values('month')

print("\nPredicted Top Project per Month in 2025:")
print(top_projects_2025)
top_projects_2025.to_csv('data/v2/linear/csv_files/v2_linear_top_projects_2025_prediction.csv', index=False)

# --- Graphing Section ---

# 1. Historical Top Users Graph (split into two periods for clarity)
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
plt.title('Predicted Top User Hours for 2025 by Month (Linear Regression)')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/v2/linear/images/v2_linear_2025_top_users_prediction_graph.png')
plt.close()

# 3. Historical Project Hours Graph
project_totals = timelogs.groupby('projectname')['timelog'].sum()
plt.figure(figsize=(12, 6))
project_totals.plot(kind='bar', color='skyblue')
plt.title('Total Hours Logged per Project (2010-2024)')
plt.xlabel('Project')
plt.ylabel('Total Hours')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/v2/linear/images/v2_linear_historical_project_totals_graph.png')
plt.close()

# 4. Predicted Top Projects 2025 Graph
plt.figure(figsize=(10, 5))
plt.plot(top_projects_2025['month'], top_projects_2025['predicted_hours'], marker='o', color='green')
plt.title('Predicted Top Project Hours for 2025 by Month (Linear Regression)')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/v2/linear/images/v2_linear_2025_top_projects_prediction_graph.png')
plt.close()

print("\nCSVs saved in 'data/v2/linear/csv_files':")
print("  - v2_linear_top_users_historical.csv")
print("  - v2_linear_top_users_per_project_historical.csv")
print("  - v2_linear_top_users_2025_prediction.csv")
print("  - v2_linear_top_projects_2025_prediction.csv")
print("Images saved in 'data/v2/linear/images':")
print("  - v2_linear_historical_top_users_graph_2010-2017.png")
print("  - v2_linear_historical_top_users_graph_2018-2024.png")
print("  - v2_linear_2025_top_users_prediction_graph.png")
print("  - v2_linear_historical_project_totals_graph.png")
print("  - v2_linear_2025_top_projects_prediction_graph.png")