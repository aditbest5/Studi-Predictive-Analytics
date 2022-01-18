# -*- coding: utf-8 -*-
"""Submission : Studi Predictive Analytics

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-_7Lbj0iTiEmmkRRnGRmWIQE2l6s7g1t

# **Studi Predictive Analytics : Microsoft Stock dengan pendekatan Time Series Forecasting dan LSTM**

Model yang dibuat menggunakan Time Series dengan LSTM. Kita akan memprediksi harga stock Microsoft dimasa yang akan datang menggunakan pendekatan Time Series Analysis. 


## **Time Series**

Time series dapat dipahami sebagai kumpulan nilai yang tersusun secara runtut dalam rentang waktu tertentu.

## **Dataset**

Dataset yang digunakan menggunakan dataset Microsoft_Stock.csv yang bisa ditemukan di kaggle.com. Dataset ini memiliki 1511 baris dan 5 kolom. Kolom yang terdapat pada dataset ini yakni Date (tanggal), Open(Harga pembukaan), High(Harga tertinggi), Low(Harga terendah), Close(Harga closing/penutup), dan Volume(volum ketersediaan stok).
"""

# import library yang dibutuhkan
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""## **Membaca data menggunakan Pandas**

Disini kita akan membaca data Microsoft_Stock.csv menggunakan pandas lalu kita akan parse_dates bagian kolom date dan index_col pada bagian Date. parse_dates disini bertujuan agar tanggal dapat dikenali dalam bahasa python dan index_col bertujuan agar kolom Date menjadi sebuah index dalam data.


"""

df = pd.read_csv('Microsoft_Stock.csv', parse_dates=['Date'], index_col='Date')
df

"""## **Melihat tipe data**

Tipe data yang ada diinfo terdapat 1511 non-null data yang artinya tidak ada data kosong/missing value. Kolom Open, High, Low, Close bertipe float sedangkan volume bertipe integer.
"""

df.info()

"""## **Mengecek nilai menggunakan deksripsi statistika**
Deskripsi statisika digunakan untuk menggambarkan/summary data. Seperti : nilai mean, median, modus, variance, standar deviasi, quartile, range dan sejenisnya. perintah yang digunakan DataFrame.describe() Dapat dilihat bahwa pada nilai min/minimum tidak terdapat angka 0 yang mengindikasikan tidak ada missing value.


"""

df.describe()

"""## **Mengecek nilai missing value pada setiap kolom lalu handling missing value**

Hal ini bertujuan untuk melihat apakah terdapat nilai nol atau tidak pada data. 
Dan ternyata tidak ada data yang null atau bernilai 0 dan tidak perlu melakukan handling missing value.


"""

df.isnull().sum()

"""## **Melihat outlier yang ada didata**
Nilai outliers (atau yang biasa disebut dengan nilai pencilan) merupakan suatu nilai yang tidak normal. Dalam kata lain, nilai tersebut bernilai jauh sekali dari pusat data. Nilai pencilan ini dapat menyebabkan distorsi terhadap nilai yang asli.

Dampaknya jika tidak serius menanggulangi nilai outliers akan berdampak pada ketidaksesuaian data analisa dan ketidakakuratan model machine learning yang dikerjakan
"""

sns.boxplot(x=df['High'])

sns.boxplot(x=df['Low'])

sns.boxplot(x=df['Close'])

"""## **Menangani Outlier Menggunakan Interquartile**
Dengan menggunakan interquartil, nilai pencilan yang melebihi nilai kuartil1(Q1) akan diubah menjadi nilai Q1. Sementara itu, nilai pencilan yang kurang dari nilai kuartil3(Q3) akan diubah nilainya menjadi nilai Q3. Sehingga, tidak ada lagi nilai pencilan yang tentunya akan meningkatkan efektifitas model.
"""

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
 
# Cek ukuran dataset setelah kita drop outliers
df.shape

"""## **Membuat Grafik menggunakan matplotlib**

Setelah kita melakukan data cleaning dengan proses EDA. Kita akan melihat grafik dari data microsoft_stock.csv. Libarary yang digunakan menggunakan matplotlib.pyplot. Sumbu-x adalah index penanggalan dan sumbu-y hanya menggunakan harga close, open, high, dan low. Tetapi, harga keempatnya hampir sama dan terlihat berdekatan hasil grafiknya.


"""

import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history',color='b')
plt.plot(df['Open'], label='Open Price history',color='r')
plt.plot(df['High'], label='High Price history',color='g')
plt.plot(df['Low'], label='Low Price history',color='b')

plt.xlabel('Date',size=20)
plt.ylabel('Stock Price',size=20)
plt.title('Stock Price of Microsoft Over the Years',size=25);
plt.legend()
plt.show()

data = df['Close'].values
dates = df.index.values

"""# **Membagi Dataset**

Pada proses ini data akan dibagi dengan porsi 80:20 atau 80% data train dan 20% data test dengan random state brnilai False dan shuffle bernilai False
"""

# Membagi data set 80:20
x_train, x_test, y_train, y_test = train_test_split(dates, data, test_size=0.2, random_state=False, shuffle=False)
y_train

"""## **Normalize Data menggunakan MinMaxScaler**

selanjutnya, data akan dinormalisasi menggunakan MinMaxScaler yang dimana data train menggunakan fit_transform dan data test menggunakan transform()
"""

from sklearn.preprocessing import StandardScaler
y_train = y_train.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
y_train = scaler.fit_transform(y_train)

y_test = y_test.reshape(-1, 1)
y_test = scaler.transform(y_test)
print(len(y_train))
print(len(y_test))

"""## **Windowing Data**"""

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  series = tf.expand_dims(series, axis=-1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds= ds.window(window_size + 1, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda w: w.batch(window_size +1))
  ds = ds.shuffle(shuffle_buffer)
  ds = ds.map(lambda w: (w[:-1], w[-1:]))
  return ds.batch(batch_size).prefetch(1)

early = EarlyStopping(monitor='val_loss',patience=5)
reduce = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, mil_lr=0.001)

"""## **Model LSTM**"""

train_set = windowed_dataset(y_train, window_size=10, batch_size=32, shuffle_buffer=1000)
val_set = windowed_dataset(y_test, window_size=10, batch_size=32, shuffle_buffer=1000)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(10, 1)),
    tf.keras.layers.RepeatVector(10),
    tf.keras.layers.LSTM(200, activation='relu', return_sequences=True),
    tf.keras.layers.Dense(10)
    ])

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])

history = model.fit(train_set, epochs=epochs, batch_size=144, verbose=1,
         validation_data=val_set, callbacks=[early,reduce])

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend(['Train','Test']);

"""## **Model Forecasting**"""

data2 =df.reset_index()

from statsmodels.base.transform import BoxCox
bc=BoxCox()
data2['Close'], lmbda=bc.transform_boxcox(data2['Close'])

new_data=data2[["Date","Close"]]
new_data.columns=["ds","y"]

model_params={
    "daily_seasonality":False,
    "weekly_seasonality": False,
    "yearly_seasonality": True,
    "seasonality_mode": "multiplicative",
    "growth": "logistic"
}

from fbprophet import Prophet
model=Prophet(**model_params)

new_data["cap"]=new_data['y'].max() + new_data['y'].std()*0.05

model.fit(new_data)
future=model.make_future_dataframe(periods=365)
future["cap"]=new_data["cap"].max()
forecast=model.predict(future)

model.plot_components(forecast)
model.plot(forecast)