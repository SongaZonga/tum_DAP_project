import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# loss function of smape
def smape_loss(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    numerator = tf.reduce_sum(tf.abs(y_true - y_pred))
    denominator = tf.reduce_sum(y_true + y_pred)

    return float(numerator / denominator)

# Function to create lag window
def slicewindow(data, step):
    X, y = [], []
    for i in range(0, len(data) - step, 1):
        end = i + step
        oneX, oney = data[i:end, :], data[end, :]
        X.append(oneX)
        y.append(oney)
    return np.array(X), np.array(y)

# function to create training and testing datasets
def datasplit(dataset, step):
    datasetX, datasetY = slicewindow(dataset, step)
    train_size = int(len(datasetX) * 0.80)
    X_train, y_train = datasetX[0:train_size, :], datasetY[0:train_size, :]
    X_test, y_test = datasetX[train_size:len(datasetX), :], datasetY[train_size:len(datasetX), :]
    X_train = X_train.reshape(X_train.shape[0], step, -1)
    X_test = X_test.reshape(X_test.shape[0], step, -1)
    return X_train, X_test, y_train, y_test

# function to create training and testing datasets
def datasplit_3618(dataset, step):
    datasetX, datasetY = slicewindow(dataset, step)
    train_size = int(0.80 * (len(datasetX) - 18))
    X_train, y_train = datasetX[0:train_size, :], datasetY[0:train_size, :]
    X_test, y_test = datasetX[train_size:len(datasetX) - 18, :], datasetY[train_size:len(datasetX)-18, :]
    future_y = datasetY[len(datasetX)-18:len(datasetX), :]
    X_train = X_train.reshape(X_train.shape[0], step, -1)
    X_test = X_test.reshape(X_test.shape[0], step, -1)
    return X_train, X_test, y_train, y_test, future_y

# Plot the trend
def plotting(name, result, period, hyper=False, stock=False):
    actual = pd.DataFrame(result[4].reset_index(drop=True))
    predict = pd.DataFrame(result[3])
    predict.index = list(range(len(actual) - len(predict), len(actual)))
    future = pd.DataFrame(result[2])
    future.index = list(range(len(actual), len(actual) + period))
    plt.figure(figsize=(12, 6))
    plt.plot(actual['demand'], label="Actual Demand")
    plt.plot(predict[0], label="Predicted Demand")
    plt.plot(future[0], label="Future Demand")
    plt.xlabel("Months")
    plt.ylabel("Demand")
    percent_string = '%'
    plt.title("Actual vs Predicted Demand: %s, test accuracy: %s%s" % (
        name, round(100.00 * float(result[0]), 2), percent_string))
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    if hyper:
        kurs = "LSTM_plot/hyper/%s.png" % name
        if stock:
            kurs = "LSTM_plot/stock/hyper/%s.png" % name
    else:
        kurs = "LSTM_plot/%s.png" % name
        if stock:
            kurs = "LSTM_plot/stock/%s.png" % name
    plt.savefig(kurs, format='png')
#  result[name] = [accuracy, smape, future_demand, predicted_demand[:, 0], real_demand]  
# Plot the trend for 18 months validation
def plotting_3618(name, result, period, hyper=False, stock=False):
    actual = pd.DataFrame(result[4].reset_index(drop=True))
    predict = pd.DataFrame(result[3])
    predict.index = list(range(len(actual)- 18 - len(predict), len(actual)-18))
    future = pd.DataFrame(result[2])
    future.index = list(range(len(actual)-18, len(actual)))
    future_accuracy = 1 - smape_loss(actual.iloc[len(actual)-18:len(actual),:], future)
    plt.figure(figsize=(12, 6))
    plt.plot(actual['demand'], label="Actual Demand")
    plt.plot(predict[0], label="Predicted Demand")
    plt.plot(future[0], label="Future Demand")
    plt.xlabel("Months")
    plt.ylabel("Demand")
    percent_string = '%'
    plt.title("Actual vs Predicted Demand: %s, test accuracy: %s%s, future accuracy: %s%s" % (
        name, round(100.00 * float(result[0]), 2), percent_string, round(100.00 * float(future_accuracy), 2), percent_string))
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    if hyper:
        kurs = "LSTM_plot_3618/hyper/%s.png" % name
        if stock:
            kurs = "LSTM_plot_3618/stock/hyper/%s.png" % name
    else:
        kurs = "LSTM_plot_3618/%s.png" % name
        if stock:
            kurs = "LSTM_plot_3618/stock/%s.png" % name
    plt.savefig(kurs, format='png')

