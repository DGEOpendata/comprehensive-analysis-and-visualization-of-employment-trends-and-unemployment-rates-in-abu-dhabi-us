python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
employment_data = pd.read_csv('Employment_Trends_Abu_Dhabi_2022.csv')

# Data Cleaning and Preparation
employment_data.dropna(inplace=True)  # Drop missing values
employment_data['Unemployment_Rate'] = employment_data['Unemployed'] / employment_data['Labor_Force']

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.barplot(data=employment_data, x='Sector', y='Employment_Rate', hue='Gender')
plt.title('Employment Rate by Sector and Gender')
plt.xticks(rotation=45)
plt.show()

# Predictive Modeling
X = employment_data[['Age_Group', 'Education_Level', 'Gender_Encoded', 'Sector_Encoded']]
y = employment_data['Unemployment_Rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
