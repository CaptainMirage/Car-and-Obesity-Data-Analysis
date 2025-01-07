import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')  # Suppress warnings

# URLs for the CSV files
url_metric_cars = "https://raw.githubusercontent.com/datawookie/data-diaspora/master/cars-metric.csv"
url_obesity_data = "https://raw.githubusercontent.com/datawookie/data-diaspora/master/obesity/death-rate-from-obesity.csv"

# Load the csv.metric-cars dataset from the URL
try:
    response_metric_cars = requests.get(url_metric_cars)
    response_metric_cars.raise_for_status()
    metric_cars = pd.read_csv(StringIO(response_metric_cars.text))
except Exception as e:
    print(f"Failed to read the 'cars-metric' file. Please make sure the file is available.")
    raise

# Load the csv.obesity-from-rate-death dataset from the URL
try:
    response_obesity_data = requests.get(url_obesity_data)
    response_obesity_data.raise_for_status()
    obesity_data = pd.read_csv(StringIO(response_obesity_data.text))
except Exception as e:
    print(f"\nFailed to read the 'death-rate-from-obesity' file. Please make sure the file is available.")
    raise

# Task 1: Using a for loop, create a dictionary with keys as factories and values as lists of different models for each factory
factory_dict = {}
for factory, model in zip(metric_cars['mfr'], metric_cars['mod']):
    if factory not in factory_dict:
        factory_dict[factory] = [model]
    else:
        factory_dict[factory].append(model)

# Task 1.1: Output formatting for Task 1
print("\nTask 1: Dictionary with keys as factories and values as lists of different models")
for factory, models in factory_dict.items():
    print(f"{factory}: {models}")

# Task 2: For the 'small' type from the 'type' column, display the average length and weight of American and non-American cars
small_cars = metric_cars[metric_cars['type'] == 'Small']
american_small_cars = small_cars[small_cars['org'] == 'USA']
non_american_small_cars = small_cars[small_cars['org'] != 'USA']

# Task 2.1: Output formatting for Task 2
print("\nTask 2: Average length and weight of Small Cars")
print(f"American Small Cars - Length: {american_small_cars['len'].mean()}, Weight: {american_small_cars['mass'].mean()}")
print(f"Non-American Small Cars - Length: {non_american_small_cars['len'].mean()}, Weight: {non_american_small_cars['mass'].mean()}")

# Task 2.2: Display histograms for the average length and weight of American and non-American 'small' type cars
fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Adjust the figure size here

# Plot American cars with specified colors and thin bars
axs[0, 0].hist(american_small_cars['len'], bins=10, alpha=0.7, label='American', color='#588B8B', edgecolor='black', linewidth=0.3)
axs[0, 0].axvline(american_small_cars['len'].mean(), color='#588B8B', linestyle='dashed', linewidth=2)  # Mean line
axs[0, 0].set_title('Average Length of American Small Cars')
axs[0, 0].legend()

axs[0, 1].hist(american_small_cars['mass'], bins=10, alpha=0.7, label='American', color='#002E2C', edgecolor='black', linewidth=0.3)
axs[0, 1].axvline(american_small_cars['mass'].mean(), color='#002E2C', linestyle='dashed', linewidth=2)  # Mean line
axs[0, 1].set_title('Average Weight of American Small Cars')
axs[0, 1].legend()

# Plot Non-American cars with specified colors and thin bars
axs[1, 0].hist(non_american_small_cars['len'], bins=10, alpha=0.7, label='Non-American', color='#B84A62', edgecolor='black', linewidth=0.3)
axs[1, 0].axvline(non_american_small_cars['len'].mean(), color='#B84A62', linestyle='dashed', linewidth=2)  # Mean line
axs[1, 0].set_title('Average Length of Non-American Small Cars')
axs[1, 0].legend()

axs[1, 1].hist(non_american_small_cars['mass'], bins=10, alpha=0.7, label='Non-American', color='#4C243B', edgecolor='black', linewidth=0.3)
axs[1, 1].axvline(non_american_small_cars['mass'].mean(), color='#4C243B', linestyle='dashed', linewidth=2)  # Mean line
axs[1, 1].set_title('Average Weight of Non-American Small Cars')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Task 2.3: Add a column to the dataset with values as the ratio of weight to length and remove the 'org' column
metric_cars['weight_to_length_ratio'] = metric_cars['mass'] / metric_cars['len']
metric_cars = metric_cars.drop(columns=['org'])

# Task 3: Download the csv.obesity-from-rate-death dataset and perform linear regression
try:
    obesity_data = pd.read_csv(url_obesity_data)
except Exception as e:
    print(f"\nFailed to read the 'death-rate-from-obesity' file. Please make sure the file is available.")
    raise

# Task 4: Scatter plots and linear regression for America and China for the 'Entity' column
print("\nTask 4: Scatter plots and linear regression for America and China")
countries = ['America', 'China']
for country in countries:
    # Use the existing column 'Entity' for country filtering
    country_data = obesity_data[obesity_data['Entity'] == country]

    # Check if there are samples for the country
    if country_data.shape[0] > 0:
        X = country_data[['Year']]
        y = country_data['Deaths - High body-mass index - Sex: Both - Age: Age-standardized (Rate) (deaths per 100,000)']

        # Check if there are enough samples to split
        if X.shape[0] > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Convert X_train to NumPy array
            X_train_np = X_train.values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X_train_np, y_train)

            plt.scatter(X_test, y_test, color='black')
            plt.plot(X_test, model.predict(X_test.values.reshape(-1, 1)), color='blue', linewidth=3)
            plt.title(f"Linear Regression for {country}")
            plt.xlabel("Year")
            plt.ylabel("Obesity Rate")
            plt.show()
        else:
            print(f"Not enough samples for {country} to perform linear regression.")
    else:
        print(f"No data found for {country} in the dataset.")

# Task (what number are we on?): Predict mortality rate due to obesity for a user-specified year using linear regression
print("\nTask 5: Predict mortality rate due to obesity for a user-specified year")
user_year = int(input("Enter a year between 2025 and 2045: "))
if 2025 <= user_year <= 2045:
    for country in countries:
        country_data = obesity_data[obesity_data['Entity'] == country]  # Update to 'Entity'
        X_train = country_data[['Year']]
        y_train = country_data['Deaths - High body-mass index - Sex: Both - Age: Age-standardized (Rate) (deaths per 100,000)']

        # Check if there are enough samples to fit the model
        if X_train.shape[0] > 1:
            model = LinearRegression()
            model.fit(X_train, y_train)

            predicted_rate = model.predict([[user_year]])
            print(f"Predicted Obesity Rate for {country} in {user_year}: {predicted_rate[0]}")
        else:
            print(f"Not enough samples for {country} to make a prediction.")
else:
    print("Invalid year. Please enter a year between 2025 and 2045.")
