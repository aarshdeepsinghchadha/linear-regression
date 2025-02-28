import pandas as pd
import random
from datetime import datetime
import os

# Create 'data/v2' folder if it doesn't exist
if not os.path.exists('data/v2'):
    os.makedirs('data/v2')  # makedirs creates parent directories as needed

# Projects (10 projects)
projects = pd.DataFrame({
    'projectid': range(1, 11),
    'projectname': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 
                   'Project Epsilon', 'Project Zeta', 'Project Eta', 'Project Theta',
                   'Project Iota', 'Project Kappa']
})

# Users (50 users)
users = [f'user{i}' for i in range(1, 51)]  # user1, user2, ..., user50

# Days in each month (simplified, not accounting for leap years)
days_in_month = {
    'Jan': 31, 'Feb': 28, 'Mar': 31, 'Apr': 30, 'May': 31, 'Jun': 30,
    'Jul': 31, 'Aug': 31, 'Sep': 30, 'Oct': 31, 'Nov': 30, 'Dec': 31
}

data = []
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Generate data for 15 years (2010-2024)
for year in range(2010, 2025):
    for month_idx, month in enumerate(months, 1):
        days = days_in_month[month]
        for day in range(1, days + 1):
            for userid, username in enumerate(users, 1):
                # Randomly decide if user works on a project that day (70% chance)
                if random.random() < 0.7:
                    projectid = random.randint(1, 10)
                    # More varied timelog range with different distributions
                    timelog = random.choice([
                        random.randint(10, 50),    # Short tasks
                        random.randint(50, 150),   # Medium tasks
                        random.randint(150, 300)   # Long tasks
                    ]) * random.uniform(0.5, 1.5)  # Add some variation
                    data.append([userid, username, f'{year}-{month_idx:02d}-{day:02d}', 
                                month, year, projectid, 
                                projects.loc[projectid-1, 'projectname'], 
                                round(timelog)])

# Create DataFrame
timelogs = pd.DataFrame(data, columns=['userid', 'username', 'date', 'month', 'year',
                                      'projectid', 'projectname', 'timelog'])
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Save to CSV in 'data/v2' folder
timelogs.to_csv('data/v2/user_timelogs_v2.csv', index=False)
projects.to_csv('data/v2/projects_v2.csv', index=False)

# Optional: Print some basic stats to verify variation
print("Total hours by project:")
print(timelogs.groupby('projectname')['timelog'].sum())
print("\nTotal hours by user (first 5):")
print(timelogs.groupby('username')['timelog'].sum().head())