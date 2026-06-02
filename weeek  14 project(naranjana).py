import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
data = pd.read_csv("creditcard.csv")

# Display Dataset Information
print("First 5 Records:")
print(data.head())

# Fraud and Normal Transaction Count
print("\nFraud vs Normal Transactions:")
print(data["Class"].value_counts())

# Independent Variable
X = data[["Time"]]

# Dependent Variable
y = data["Amount"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate Model
score = r2_score(y_test, y_pred)

print("\nR2 Score:", score)

# Risk Analysis
def risk_level(amount):
    if amount < 100:
        return "Low Risk"
    elif amount < 1000:
        return "Medium Risk"
    else:
        return "High Risk"

data["Risk_Level"] = data["Amount"].apply(risk_level)

print("\nRisk Analysis:")
print(data["Risk_Level"].value_counts())