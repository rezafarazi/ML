import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create sample data
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
data = {
    'age': np.random.normal(35, 10, n_samples),
    'experience': np.random.normal(10, 5, n_samples),
    'education_years': np.random.normal(16, 3, n_samples),
    'salary': np.zeros(n_samples)
}


# Create a DataFrame
df = pd.DataFrame(data)

# Create a salary formula based on features
df['salary'] = (
    50000 +
    1500 * df['experience'] +
    2000 * df['education_years'] +
    100 * df['age'] +
    np.random.normal(0, 5000, n_samples)
)


# Basic data exploration
print("\nDataset Overview:")
print(df.head())
print("\nBasic Statistics:")
print(df.describe())

# # Data visualization
plt.figure(figsize=(12, 6))

# Create subplots
plt.subplot(1, 2, 1)
plt.scatter(df['experience'], df['salary'], alpha=0.5)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')

plt.subplot(1, 2, 2)
plt.scatter(df['education_years'], df['salary'], alpha=0.5)
plt.xlabel('Years of Education')
plt.ylabel('Salary')
plt.title('Education vs Salary')

plt.tight_layout()
plt.savefig('salary_analysis.png')


# Prepare data for modeling
X = df[['age', 'experience', 'education_years']]
y = df['salary']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print(f"\nModel R² Score: {model.score(X_test, y_test):.4f}")

# Example prediction
example_person = pd.DataFrame({
    'age': [30],
    'experience': [5],
    'education_years': [16]
})

predicted_salary = model.predict(example_person)
print(f"\nPredicted salary for a 30-year-old person with 5 years experience and 16 years of education:")
print(f"${predicted_salary[0]:,.2f}")

# Save the DataFrame to a CSV file
df.to_csv('salary_data.csv', index=False)
print("\nData has been saved to 'salary_data.csv'") 