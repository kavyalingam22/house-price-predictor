import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Area": [1000, 1500, 2000, 2500, 3000],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Price": [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

# Features & target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Model
model = LinearRegression()
model.fit(X, y)

# Input
area = int(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))

# Prediction
price = model.predict([[area, bedrooms]])

print(f"Estimated Price: {price[0]:.2f}")