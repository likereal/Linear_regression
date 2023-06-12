import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

data = pd.read_csv('Google_Demo.csv')
data=data[:52]

data['Time'] = pd.to_datetime(data['Time'], format='%b-%y')
data['Time'] = data['Time'].apply(lambda x: x.timestamp())

train_data=data[:43]
test_data=data[43:52]

#test_data = train_test_split(data, test_size=0.2)

X_train = train_data['Time'].values.reshape(-1, 1)
y_train = train_data['Sales'].values.reshape(-1, 1)
X_test = test_data['Time'].values.reshape(-1, 1)
y_test = test_data['Sales'].values.reshape(-1, 1)
print(train_data)



model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss='mse')

history = model.fit(X_train, y_train, epochs=800)


last_date = pd.to_datetime(train_data['Time'].max(), unit='s')
next_dates = pd.date_range(start=last_date, periods=18, freq='MS')
next_dates_unix = next_dates.map(lambda x: x.timestamp())
y_new = model.predict(next_dates_unix.values.reshape(-1, 1))
test_data_upd=np.append(test_data['Sales'],([0]*9))
predictions = pd.DataFrame({'Time': next_dates,'Sales':test_data_upd ,'Predict': y_new.flatten()//1})
print(predictions)

predictions.set_index('Time',inplace=True)
predictions.to_csv('name.csv')