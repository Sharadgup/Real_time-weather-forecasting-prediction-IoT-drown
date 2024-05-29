import pandas as pd
from pymongo import MongoClient
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, request, jsonify
import requests

# MongoDB setup
client = MongoClient('mongodb+srv://shardgupta65:Typer%401345@cluster0.sp87qsr.mongodb.net/chatgpt')
db = client['chatgpt']
collection = db['weather_data']

# Load Kaggle dataset
historical_data = pd.read_csv('weather.csv')

# Store each row of the DataFrame in MongoDB
def store_historical_data(dataframe):
    records = dataframe.to_dict(orient='records')
    collection.insert_many(records)

# Store the Kaggle dataset into MongoDB
store_historical_data(historical_data)

def fetch_historical_data():
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))
    return data

# Fetch historical data from MongoDB to confirm
historical_data_from_db = fetch_historical_data()

# Remove MongoDB ObjectId and unnecessary columns
if '_id' in historical_data_from_db.columns:
    historical_data_from_db = historical_data_from_db.drop(columns=['_id'])

# Normalize the data
scaler = MinMaxScaler()
historical_data_scaled = scaler.fit_transform(historical_data_from_db)

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, 0])  # Predicting the temperature
    return np.array(X), np.array(Y)

# Prepare the dataset
time_step = 10
X, Y = create_dataset(historical_data_scaled, time_step)

# Split into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[test_size:]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Define and train the model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))  # Predicting temperature
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model()
model.fit(X_train, Y_train, batch_size=1, epochs=10)

# Save the initial model
model.save('weather_model.h5')

app = Flask(__name__)

API_KEY = '277ec9e16aef455c82951906242905'
BASE_URL = " http://api.weatherapi.com/v1/current.json"

def get_weather_data(city):
    url = f"{BASE_URL}q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return data

def weather_to_dataframe(data):
    weather_dict = {
        'temperature': [data['main']['temp']],
        'humidity': [data['main']['humidity']],
        'wind_speed': [data['wind']['speed']],
    }
    return pd.DataFrame(weather_dict)

def store_weather_data(data):
    collection.insert_one(data)

@app.route('/predict', methods=['GET'])
def predict():
    city = request.args.get('city')
    weather_data = get_weather_data(city)
    weather_df = weather_to_dataframe(weather_data)
    
    # Store real-time weather data in MongoDB
    store_weather_data(weather_df.to_dict('records')[0])
    
    # Fetch historical data from MongoDB
    combined_data = fetch_historical_data().append(weather_df, ignore_index=True)
    combined_data = combined_data.drop(columns=['_id'])  # Drop MongoDB ObjectId
    combined_data_scaled = scaler.transform(combined_data)
    
    # Prepare the dataset for prediction
    X_new, _ = create_dataset(combined_data_scaled, time_step)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], X_new.shape[2])
    
    # Load the model
    model = tf.keras.models.load_model('weather_model.h5')
    
    # Make prediction
    prediction = model.predict(X_new[-1].reshape(1, time_step, X_new.shape[2]))
    prediction_inverted = scaler.inverse_transform(np.concatenate((prediction, np.zeros((prediction.shape[0], X_new.shape[2] - 1))), axis=1))[:, 0]
    
    return jsonify({'city': city, 'predicted_temperature': prediction_inverted[0]})

@app.route('/retrain', methods=['POST'])
def retrain():
    # Fetch historical data from MongoDB
    historical_data_from_db = fetch_historical_data()
    
    # Remove MongoDB ObjectId and unnecessary columns
    if '_id' in historical_data_from_db.columns:
        historical_data_from_db = historical_data_from_db.drop(columns=['_id'])

    # Normalize the data
    historical_data_scaled = scaler.fit_transform(historical_data_from_db)

    # Prepare the dataset
    X, Y = create_dataset(historical_data_scaled, time_step)

    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Build and retrain the model
    model = build_model()
    model.fit(X_train, Y_train, batch_size=1, epochs=10)

    # Save the updated model
    model.save('weather_model.h5')
    
    return jsonify({'message': 'Model retrained successfully'})

if __name__ == '__main__':
    app.run(debug=True)
