from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Define the ticker symbol for the S&P 500 index
tickerSymbol = '^GSPC'

# Get data for the ticker symbol from Yahoo Finance
sp500 = yf.Ticker(tickerSymbol)

# Get the historical data for the past year
data = sp500.history(period='1y')

# Clean the data by removing missing values
data.dropna(inplace=True)

# Define the features and target variable
features = ['Open', 'High', 'Low', 'Volume']
target = ['Close']

# Split the data into training and testing sets
train_data = data[:-100]
test_data = data[-100:]

# Train a linear regression model on the training data
lr_model = LinearRegression()
lr_model.fit(train_data[features], train_data[target])

# Define a function to calculate the accuracy of the model
def calculate_accuracy(price):
	# Create a DataFrame with the input price
	df = pd.DataFrame({'Open': [price], 'High': [price], 'Low': [price], 'Volume': [1]})

	# Use the model to make a prediction
	prediction = lr_model.predict(df[features])[0][0]

	# Calculate the actual value of the S&P 500 index on the same day
	actual = test_data.iloc[0]['Close']

	# Calculate the mean squared error between the prediction and the actual value
	mse = mean_squared_error([actual], [prediction])

	# Return the accuracy as a percentage
	return round((1 - mse / actual) * 100, 2)

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
	# Get the latest data from Yahoo Finance
	latest_data = sp500.history(period='1d')

	# Use the model to make a prediction for the latest data
	latest_data['Prediction'] = lr_model.predict(latest_data[features])

	# Render the template with the latest data and model metrics
	if request.method == 'POST':
		price = float(request.form['price'])
		accuracy = calculate_accuracy(price)
		return render_template('dashboard.html', data=latest_data.tail(10).to_dict('records'), mse=mean_squared_error(test_data[target], lr_model.predict(test_data[features])), accuracy=accuracy)
	else:
		return render_template('dashboard.html', data=latest_data.tail(10).to_dict('records'), mse=mean_squared_error(test_data[target], lr_model.predict(test_data[features])))

if __name__ == '__main__':
	app.run(debug=True)
