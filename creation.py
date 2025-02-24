import pandas as pd
import random
from datetime import datetime
import os

# Create 'data' folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Projects (5 projects)
projects = pd.DataFrame({
    'projectid': [1, 2, 3, 4, 5],
    'projectname': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 'Project Epsilon']
})

# Users (20 users)
users = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack',
         'Kelly', 'Liam', 'Mia', 'Noah', 'Olivia', 'Peter', 'Quinn', 'Rose', 'Sam', 'Tara']

# Days in each month (simplified, not accounting for leap years)
days_in_month = {
    'Jan': 31, 'Feb': 28, 'Mar': 31, 'Apr': 30, 'May': 31, 'Jun': 30,
    'Jul': 31, 'Aug': 31, 'Sep': 30, 'Oct': 31, 'Nov': 30, 'Dec': 31
}

data = []
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    for month_idx, month in enumerate(months, 1):
        days = days_in_month[month]
        for day in range(1, days + 1):
            for userid, username in enumerate(users, 1):
                projectid = random.choice([1, 2, 3, 4, 5])
                timelog = random.randint(50, 200)
                data.append([userid, username, f'{year}-{month_idx:02d}-{day:02d}', month, year,
                            projectid, projects.loc[projectid-1, 'projectname'], timelog])

# Create DataFrame
timelogs = pd.DataFrame(data, columns=['userid', 'username', 'date', 'month', 'year',
                                      'projectid', 'projectname', 'timelog'])
timelogs['date'] = pd.to_datetime(timelogs['date'])

# Save to CSV in 'data' folder
timelogs.to_csv('data/user_timelogs.csv', index=False)
projects.to_csv('data/projects.csv', index=False)