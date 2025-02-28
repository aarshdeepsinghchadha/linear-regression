import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm  

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create folders if they don't exist
os.makedirs('data/v2/tensorflow/csv_files', exist_ok=True)
os.makedirs('data/v2/tensorflow/images', exist_ok=True)

# Load existing data
logging.info("Script started: Loading data...")
timelogs = pd.read_csv('data/v2/user_timelogs_v2.csv')
timelogs['date'] = pd.to_datetime(timelogs['date'])
logging.info("Data loaded successfully.")

# Define months for consistency
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Validate input data
logging.info("Validating input data...")
print(timelogs.head())
logging.info("Unique months in timelogs: %s", timelogs['date'].dt.strftime('%b').unique())

# --- Historical Analysis ---
def historical_analysis():
    logging.info("Starting historical analysis...")
    user_monthly_summary = timelogs.groupby(['year', 'month', 'username'])['timelog'].sum().reset_index()
    user_monthly_summary['month'] = pd.Categorical(user_monthly_summary['month'], categories=months, ordered=True)
    top_users = user_monthly_summary.loc[user_monthly_summary.groupby(['year', 'month'], observed=True)['timelog'].idxmax()]
    top_users = top_users.sort_values(['year', 'month'])
    
    logging.info("Top User per Month and Year (2010-2024):")
    print(top_users)
    top_users.to_csv('data/v2/tensorflow/csv_files/v2_tensor_top_users_historical.csv', index=False)
    
    project_user_summary = timelogs.groupby(['projectname', 'username'])['timelog'].sum().reset_index()
    top_users_per_project = project_user_summary.loc[project_user_summary.groupby('projectname')['timelog'].idxmax()]
    
    logging.info("\nTop User per Project (2010-2024 Total Hours):")
    print(top_users_per_project)
    top_users_per_project.to_csv('data/v2/tensorflow/csv_files/v2_tensor_top_users_per_project_historical.csv', index=False)
    logging.info("Historical analysis completed.")

# --- Prediction Model ---
def create_model(input_shape=(3,)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std

def month_to_numeric(month):
    month_str = str(month)[:3]
    if month_str not in months:
        raise ValueError(f"Invalid month value: {month_str}")
    return months.index(month_str) + 1

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

# User Predictions for 2025
def predict_users_2025(model):
    logging.info("Starting user predictions for 2025...")
    user_predictions_2025 = []
    total_users = len(timelogs['username'].unique())
    
    for username in tqdm(timelogs['username'].unique(), desc="Predicting Users", total=total_users):
        try:
            user_data = timelogs[timelogs['username'] == username]
            user_monthly = user_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
            
            user_monthly['month'] = user_monthly['month'].astype(str).str[:3]
            if len(user_monthly) < 3:
                logging.info(f"Skipping user {username}: Insufficient data ({len(user_monthly)} rows)")
                user_predictions_2025.extend([(username, month, 0) for month in months])
                continue
            
            user_monthly = add_cyclical_features(user_monthly)
            X = user_monthly[['year', 'month_sin', 'month_cos']].values.astype(np.float32)
            y = user_monthly['timelog'].values.astype(np.float32)
            X_norm, mean, std = normalize_data(X)
            
            model.fit(X_norm, y, epochs=100, verbose=0)
            
            X_2025 = pd.DataFrame({'year': [2025] * 12, 'month': months})
            X_2025 = add_cyclical_features(X_2025)
            X_2025_norm = (X_2025[['year', 'month_sin', 'month_cos']].values - mean) / std
            forecast = model.predict(X_2025_norm, verbose=0).flatten()
            forecast = np.maximum(forecast, 0)
            forecast = np.nan_to_num(forecast, nan=0.0)
            
            user_predictions_2025.extend([(username, month, hours) for month, hours in zip(months, forecast)])
        except Exception as e:
            logging.warning(f"Prediction failed for user {username}: {str(e)}")
            user_predictions_2025.extend([(username, month, 0) for month in months])
    
    user_pred_df = pd.DataFrame(user_predictions_2025, columns=['username', 'month', 'predicted_hours'])
    user_pred_df['year'] = 2025
    user_pred_df['month'] = pd.Categorical(user_pred_df['month'], categories=months, ordered=True)
    user_pred_df['predicted_hours'] = pd.to_numeric(user_pred_df['predicted_hours'], errors='coerce').fillna(0)
    
    logging.info("user_pred_df sample:")
    print(user_pred_df.head())
    
    top_users_2025 = user_pred_df.loc[user_pred_df.groupby('month', observed=True)['predicted_hours'].idxmax()]
    top_users_2025 = top_users_2025.sort_values('month')
    
    logging.info("\nPredicted Top User per Month in 2025:")
    print(top_users_2025)
    top_users_2025.to_csv('data/v2/tensorflow/csv_files/v2_tensor_top_users_2025_prediction.csv', index=False)
    logging.info("User predictions for 2025 completed.")

# Project Predictions for 2025
def predict_projects_2025(model):
    logging.info("Starting project predictions for 2025...")
    project_predictions_2025 = []
    total_projects = len(timelogs['projectname'].unique())
    
    for projectname in tqdm(timelogs['projectname'].unique(), desc="Predicting Projects", total=total_projects):
        try:
            project_data = timelogs[timelogs['projectname'] == projectname]
            project_monthly = project_data.groupby(['year', 'month'])['timelog'].sum().reset_index()
            
            project_monthly['month'] = project_monthly['month'].astype(str).str[:3]
            if len(project_monthly) < 3:
                logging.info(f"Skipping project {projectname}: Insufficient data ({len(project_monthly)} rows)")
                project_predictions_2025.extend([(projectname, month, 0) for month in months])
                continue
            
            project_monthly = add_cyclical_features(project_monthly)
            X = project_monthly[['year', 'month_sin', 'month_cos']].values.astype(np.float32)
            y = project_monthly['timelog'].values.astype(np.float32)
            X_norm, mean, std = normalize_data(X)
            
            model.fit(X_norm, y, epochs=100, verbose=0)
            
            X_2025 = pd.DataFrame({'year': [2025] * 12, 'month': months})
            X_2025 = add_cyclical_features(X_2025)
            X_2025_norm = (X_2025[['year', 'month_sin', 'month_cos']].values - mean) / std
            forecast = model.predict(X_2025_norm, verbose=0).flatten()
            forecast = np.maximum(forecast, 0)
            forecast = np.nan_to_num(forecast, nan=0.0)
            
            project_predictions_2025.extend([(projectname, month, hours) for month, hours in zip(months, forecast)])
        except Exception as e:
            logging.warning(f"Prediction failed for project {projectname}: {str(e)}")
            project_predictions_2025.extend([(projectname, month, 0) for month in months])
    
    project_pred_df = pd.DataFrame(project_predictions_2025, columns=['projectname', 'month', 'predicted_hours'])
    project_pred_df['year'] = 2025
    project_pred_df['month'] = pd.Categorical(project_pred_df['month'], categories=months, ordered=True)
    project_pred_df['predicted_hours'] = pd.to_numeric(project_pred_df['predicted_hours'], errors='coerce').fillna(0)
    
    logging.info("project_pred_df sample:")
    print(project_pred_df.head())
    
    top_projects_2025 = project_pred_df.loc[project_pred_df.groupby('month', observed=True)['predicted_hours'].idxmax()]
    top_projects_2025 = top_projects_2025.sort_values('month')
    
    logging.info("\nPredicted Top Project per Month in 2025:")
    print(top_projects_2025)
    top_projects_2025.to_csv('data/v2/tensorflow/csv_files/v2_tensor_top_projects_2025_prediction.csv', index=False)
    logging.info("Project predictions for 2025 completed.")

# --- Graphing Section ---
def generate_graphs():
    logging.info("Starting graph generation...")
    top_users = pd.read_csv('data/v2/tensorflow/csv_files/v2_tensor_top_users_historical.csv')
    top_users_2025 = pd.read_csv('data/v2/tensorflow/csv_files/v2_tensor_top_users_2025_prediction.csv')
    top_projects_2025 = pd.read_csv('data/v2/tensorflow/csv_files/v2_tensor_top_projects_2025_prediction.csv')
    
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
        plt.savefig(f'data/v2/tensorflow/images/v2_tensor_historical_top_users_graph_{period}.png')
        plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(top_users_2025['month'], top_users_2025['predicted_hours'], marker='o', color='purple')
    plt.title('Predicted Top User Hours for 2025 by Month (TensorFlow)')
    plt.xlabel('Month')
    plt.ylabel('Predicted Hours')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/v2/tensorflow/images/v2_tensor_2025_top_users_prediction_graph.png')
    plt.close()
    
    project_totals = timelogs.groupby('projectname')['timelog'].sum()
    plt.figure(figsize=(12, 6))
    project_totals.plot(kind='bar', color='skyblue')
    plt.title('Total Hours Logged per Project (2010-2024)')
    plt.xlabel('Project')
    plt.ylabel('Total Hours')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('data/v2/tensorflow/images/v2_tensor_historical_project_totals_graph.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(top_projects_2025['month'], top_projects_2025['predicted_hours'], marker='o', color='green')
    plt.title('Predicted Top Project Hours for 2025 by Month (TensorFlow)')
    plt.xlabel('Month')
    plt.ylabel('Predicted Hours')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/v2/tensorflow/images/v2_tensor_2025_top_projects_prediction_graph.png')
    plt.close()
    logging.info("Graph generation completed.")

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Main execution started...")
    historical_analysis()
    model = create_model()
    predict_users_2025(model)
    predict_projects_2025(model)
    generate_graphs()
    
    logging.info("\nCSVs saved in 'data/v2/tensorflow/csv_files':")
    logging.info("  - v2_tensor_top_users_historical.csv")
    logging.info("  - v2_tensor_top_users_per_project_historical.csv")
    logging.info("  - v2_tensor_top_users_2025_prediction.csv")
    logging.info("  - v2_tensor_top_projects_2025_prediction.csv")
    logging.info("Images saved in 'data/v2/tensorflow/images':")
    logging.info("  - v2_tensor_historical_top_users_graph_2010-2017.png")
    logging.info("  - v2_tensor_historical_top_users_graph_2018-2024.png")
    logging.info("  - v2_tensor_2025_top_users_prediction_graph.png")
    logging.info("  - v2_tensor_historical_project_totals_graph.png")
    logging.info("  - v2_tensor_2025_top_projects_prediction_graph.png")
    logging.info("Script execution completed.")