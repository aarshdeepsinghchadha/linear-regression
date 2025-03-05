# create_data.py
import pandas as pd
import random
from datetime import datetime
import os
from tqdm import tqdm

# Create 'data/v3' folder
if not os.path.exists('data/v4'):
    os.makedirs('data/v4')

# Projects (10 projects)
projects = pd.DataFrame({
    'projectid': range(1, 11),
    'projectname': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 
                    'Project Epsilon', 'Project Zeta', 'Project Eta', 'Project Theta',
                    'Project Iota', 'Project Kappa']
})

# Users (50 users)
users = [f'user{i}' for i in range(1, 51)]

# Days in each month (simplified)
days_in_month = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

# Generate data for 5 years (2020-2024)
data = []
for year in tqdm(range(2020, 2025), desc="Generating years"):
    for month in range(1, 13):
        days = days_in_month[month]
        for day in range(1, days + 1):
            for username in users:
                # 70% chance user logs hours
                if random.random() < 0.7:
                    projectid = random.randint(1, 10)
                    # Varied timelog distribution
                    timelog = random.choice([
                        random.randint(10, 50),   # Short
                        random.randint(50, 150),  # Medium
                        random.randint(150, 300)  # Long
                    ]) * random.uniform(0.8, 1.2)  # Slight variation
                    date_str = f'{year}-{month:02d}-{day:02d}'
                    data.append([username, date_str, year, month, projectid, 
                                 projects.loc[projectid-1, 'projectname'], round(timelog)])

# Create DataFrame
timelogs = pd.DataFrame(data, columns=['username', 'date', 'year', 'month', 
                                       'projectid', 'projectname', 'timelog'])
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Save files
timelogs.to_csv('data/v4/user_timelogs_v4.csv', index=False)
projects.to_csv('data/v4/projects_v4.csv', index=False)

# Verify
print("Total hours by project:")
print(timelogs.groupby('projectname')['timelog'].sum())
print("\nTotal hours by user (first 5):")
print(timelogs.groupby('username')['timelog'].sum().head())