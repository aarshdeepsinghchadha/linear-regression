# create_data.py
import pandas as pd
import numpy as np
import random
from datetime import datetime
import os
from tqdm import tqdm

# Create 'data/v8' folder
if not os.path.exists('data/v8'):
    os.makedirs('data/v8')

# Projects (10 projects)
projects = pd.DataFrame({
    'projectid': range(1, 11),
    'projectname': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 
                    'Project Epsilon', 'Project Zeta', 'Project Eta', 'Project Theta',
                    'Project Iota', 'Project Kappa']
})

# Users (50 users)
users = [f'user{i}' for i in range(1, 51)]

# Days in each month (simplified, ignoring leap years)
days_in_month = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

# Generate data for 15 years (2010-2024)
data = []
for year in tqdm(range(2010, 2025), desc="Generating years"):
    for month in range(1, 13):
        days = days_in_month[month]
        for username in users:
            # Randomly decide user behavior for the month
            behavior = random.choices(
                ['full_time', 'part_time', 'no_log'], 
                weights=[0.5, 0.3, 0.2],  # 50% full time, 30% part time, 20% no log
                k=1
            )[0]

            if behavior == 'no_log':
                # User logs no hours this month
                continue

            elif behavior == 'full_time':
                # User logs exactly 160 hours
                total_hours = 160
                num_projects = random.randint(2, 5)  # More projects for full-time
            else:  # part_time
                # User logs less than 160 hours
                total_hours = random.randint(20, 159)  # Some reasonable range
                num_projects = random.randint(1, 3)  # Fewer projects for part-time

            # Select random projects
            selected_project_ids = random.sample(range(1, 11), num_projects)
            
            # Distribute total_hours across selected projects
            hours_per_project = np.random.dirichlet(np.ones(num_projects)) * total_hours
            hours_per_project = [round(h) for h in hours_per_project]
            # Adjust to ensure exact total matches intended total_hours
            diff = total_hours - sum(hours_per_project)
            if diff != 0:
                hours_per_project[0] += diff
            
            # Assign hours to days randomly
            for proj_idx, projectid in enumerate(selected_project_ids):
                hours_left = hours_per_project[proj_idx]
                # Randomly pick days to log hours (1 to 10 days)
                num_days = min(random.randint(1, 10), days)
                days_selected = random.sample(range(1, days + 1), num_days)
                
                # Distribute hours across selected days
                daily_hours = np.random.dirichlet(np.ones(num_days)) * hours_left
                daily_hours = [round(h) for h in daily_hours]
                diff = hours_left - sum(daily_hours)
                if diff != 0:
                    daily_hours[0] += diff
                
                # Add to data
                for day_idx, day in enumerate(days_selected):
                    date_str = f'{year}-{month:02d}-{day:02d}'
                    data.append([username, date_str, year, month, projectid, 
                                 projects.loc[projectid-1, 'projectname'], daily_hours[day_idx]])

# Create DataFrame
timelogs = pd.DataFrame(data, columns=['username', 'date', 'year', 'month', 
                                       'projectid', 'projectname', 'timelog'])
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Save files
timelogs.to_csv('data/v8/user_timelogs_v8.csv', index=False)
projects.to_csv('data/v8/projects_v8.csv', index=False)

# Verify
print("Total hours by project:")
print(timelogs.groupby('projectname')['timelog'].sum())
print("\nTotal hours by user per month (first 10 entries):")
monthly_hours = timelogs.groupby(['username', 'year', 'month'])['timelog'].sum().reset_index()
print(monthly_hours.head(10))