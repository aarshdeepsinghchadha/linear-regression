import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Create folders if they don't exist
os.makedirs('data/tensorflow/csv_files', exist_ok=True)
os.makedirs('data/tensorflow/images', exist_ok=True)

# Load existing data
timelogs = pd.read_csv('data/user_timelogs.csv')
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Define months for consistency
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# --- Historical Analysis ---
user_monthly_summary = timelogs.groupby(['year', 'month', 'username'])['timelog'].sum().reset_index()
user_monthly_summary['month'] = pd.Categorical(user_monthly_summary['month'], categories=months, ordered=True)
top_users = user_monthly_summary.loc[user_monthly_summary.groupby(['year', 'month'], observed=False)['timelog'].idxmax()]
top_users = top_users.sort_values(['year', 'month'])

print("Top User per Month and Year (2016-2024):")
print(top_users)
top_users.to_csv(os.path.join('data/tensorflow/csv_files', 'top_users_historical.csv'), index=False)

project_user_summary = timelogs.groupby(['projectname', 'username'])['timelog'].sum().reset_index()
top_users_per_project = project_user_summary.loc[project_user_summary.groupby('projectname')['timelog'].idxmax()]

print("\nTop User per Project (2016-2024 Total Hours):")
print(top_users_per_project)
top_users_per_project.to_csv(os.path.join('data/tensorflow/csv_files', 'tensor_top_users_per_project_historical.csv'), index=False)

# --- Predictions for 2025 using TensorFlow Linear Regression ---
def month_to_numeric(month):
    return months.index(month) + 1

user_predictions_2025 = []
for username in timelogs['username'].unique():
    try:
        user_data = timelogs[timelogs['username'] == username]
        user_monthly = user_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
        user_monthly['month_num'] = user_monthly['month'].apply(month_to_numeric)
        
        X = user_monthly[['year', 'month_num']].values.astype(np.float32)
        y = user_monthly['timelog'].values.astype(np.float32)
        
        # Define and train TensorFlow linear regression model
        model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[2])])
        model.compile(optimizer='sgd', loss='mse')
        model.fit(X, y, epochs=50, verbose=0)
        
        # Predict for 2025
        X_2025 = np.array([[2025, m] for m in range(1, 13)], dtype=np.float32)
        forecast = model.predict(X_2025, verbose=0).flatten()
        forecast = np.maximum(forecast, 0)  # Ensure non-negative
        
        user_predictions_2025.extend([(username, month, hours) for month, hours in zip(months, forecast)])
    except Exception as e:
        print(f"Warning: Prediction failed for user {username}: {str(e)}")
        user_predictions_2025.extend([(username, month, 0) for month in months])

user_pred_df = pd.DataFrame(user_predictions_2025, columns=['username', 'month', 'predicted_hours'])
user_pred_df['year'] = 2025
user_pred_df['month'] = pd.Categorical(user_pred_df['month'], categories=months, ordered=True)
top_users_2025 = user_pred_df.loc[user_pred_df.groupby('month', observed=False)['predicted_hours'].idxmax()]
top_users_2025 = top_users_2025.sort_values('month')

print("\nPredicted Top User per Month in 2025:")
print(top_users_2025)
top_users_2025.to_csv(os.path.join('data/tensorflow/csv_files', 'ttensor_op_users_2025_prediction.csv'), index=False)

# Project-level predictions
project_predictions_2025 = []
for projectname in timelogs['projectname'].unique():
    try:
        project_data = timelogs[timelogs['projectname'] == projectname]
        project_monthly = project_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
        project_monthly['month_num'] = project_monthly['month'].apply(month_to_numeric)
        
        X = project_monthly[['year', 'month_num']].values.astype(np.float32)
        y = project_monthly['timelog'].values.astype(np.float32)
        
        model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[2])])
        model.compile(optimizer='sgd', loss='mse')
        model.fit(X, y, epochs=50, verbose=0)
        
        X_2025 = np.array([[2025, m] for m in range(1, 13)], dtype=np.float32)
        forecast = model.predict(X_2025, verbose=0).flatten()
        forecast = np.maximum(forecast, 0)
        
        project_predictions_2025.extend([(projectname, month, hours) for month, hours in zip(months, forecast)])
    except Exception as e:
        print(f"Warning: Prediction failed for project {projectname}: {str(e)}")
        project_predictions_2025.extend([(projectname, month, 0) for month in months])

project_pred_df = pd.DataFrame(project_predictions_2025, columns=['projectname', 'month', 'predicted_hours'])
project_pred_df['year'] = 2025
project_pred_df['month'] = pd.Categorical(project_pred_df['month'], categories=months, ordered=True)
top_projects_2025 = project_pred_df.loc[project_pred_df.groupby('month', observed=False)['predicted_hours'].idxmax()]
top_projects_2025 = top_projects_2025.sort_values('month')

print("\nPredicted Top Project per Month in 2025:")
print(top_projects_2025)
top_projects_2025.to_csv(os.path.join('data/tensorflow/csv_files', 'tensor_top_projects_2025_prediction.csv'), index=False)

# --- Graphing Section ---
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
plt.savefig(os.path.join('data/tensorflow/images', 'tensor_historical_top_users_graph.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(top_users_2025['month'], top_users_2025['predicted_hours'], marker='o', color='purple')
plt.title('Predicted Top User Hours for 2025 by Month (TensorFlow Linear Regression)')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('data/tensorflow/images', 'tensor_2025_top_users_prediction_graph.png'))
plt.close()

project_totals = timelogs.groupby('projectname')['timelog'].sum()
plt.figure(figsize=(10, 5))
project_totals.plot(kind='bar', color='skyblue')
plt.title('Total Hours Logged per Project (2016-2024)')
plt.xlabel('Project')
plt.ylabel('Total Hours')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('data/tensorflow/images', 'tensor_historical_project_totals_graph.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(top_projects_2025['month'], top_projects_2025['predicted_hours'], marker='o', color='green')
plt.title('Predicted Top Project Hours for 2025 by Month (TensorFlow Linear Regression)')
plt.xlabel('Month')
plt.ylabel('Predicted Hours')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('data/tensorflow/images', 'tensor_2025_top_projects_prediction_graph.png'))
plt.close()

print("\nCSVs saved in 'data/tensorflow/csv_files' folder:")
print("  - tensor_top_users_historical.csv")
print("  - tensor_top_users_per_project_historical.csv")
print("  - tensor_top_users_2025_prediction.csv")
print("  - tensor_top_projects_2025_prediction.csv")
print("Images saved in 'data/tensorflow/images' folder:")
print("  - tensor_historical_top_users_graph.png")
print("  - tensor_2025_top_users_prediction_graph.png")
print("  - tensor_historical_project_totals_graph.png")
print("  - tensor_2025_top_projects_prediction_graph.png")