import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import itertools
from keras import backend as K
from model_preparation import smape_loss, datasplit
# SEED = 666
# os.environ['PYTHONHASHSEED']=str(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
# rn.seed(SEED)
# function of LSTM model
def seq2seqModel(X, step):
    K.clear_session()
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(step, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(X.shape[2]))
    model.compile(optimizer='adam', loss=smape_loss)
    return model

# function to create future demand
def predictfuture(dataset, step, model, period):
    for _ in range(period):
        datasetX = dataset[-step:]
        X_train = datasetX.reshape(1, step, datasetX.shape[1])
        next_prediction = model.predict(X_train)
        dataset = np.vstack([dataset, next_prediction])
    return dataset[-period:]

#Calculate the error indicator
def calculate_smape(df_forcalculate, step):
    # calculate the correlation between each feature and demand and choose the first 8 features
    corr_matrix = df_forcalculate.corr()
    demand_corr = corr_matrix.sort_values(by=['demand'], ascending=False)
    index_list = demand_corr.index.tolist()
    df = df_forcalculate[index_list[0:8]]
    real_demand = df['demand']
    # Normalize the data
    scaler = StandardScaler()
    df_sc = scaler.fit_transform(df.reset_index(drop=True))
    dataset = np.array(df_sc)

    X_train, X_test, y_train, y_test = datasplit(dataset, step)
    model = seq2seqModel(X_train, step)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    model.fit(X_train, y_train, epochs=50, verbose=0, validation_split=0.2, batch_size=64, callbacks=[es])
    y_pred = model.predict(X_test)
    predicted_demand = scaler.inverse_transform(y_pred)
    y_true = scaler.inverse_transform(y_test)
    smape = smape_loss(y_true[:, 0], predicted_demand[:, 0])

    return dataset, model, X_train, X_test, y_train, y_test, predicted_demand, smape, scaler, real_demand

#Function og hyperparameter tunning
def LSTM_HyperParameter_Tuning(config, x_train, y_train, x_test, y_test, sc, accuracy):
    n_neurons, n_batch_size, dropout = config
    possible_combinations = list(itertools.product(n_neurons, n_batch_size, dropout))

    hist = []
    best_accracy = accuracy
    best_smape = 1 - best_accracy
    indicator = False
    for i in range(0, len(possible_combinations)):

        print(f'{i + 1}th combination: \n')
        print('--------------------------------------------------------------------')

        n_neurons, n_batch_size, dropout = possible_combinations[i]

        # instantiating the model in the strategy scope creates the model on the TPU
        # with tpu_strategy.scope():
        K.clear_session()
        regressor = Sequential()
        regressor.add(LSTM(units=n_neurons, activation='relu', return_sequences=True,
                           input_shape=(x_train.shape[1], x_train.shape[2])))
        regressor.add(Dropout(dropout))

#         if first_additional_layer:
#             regressor.add(LSTM(units=n_neurons, activation='relu', return_sequences=True,
#                                input_shape=(x_train.shape[1], x_train.shape[2])))
#             regressor.add(Dropout(dropout))

        # if second_additional_layer:
        #     regressor.add(LSTM(units=n_neurons))
        #     regressor.add(Dropout(dropout))
        #
        # if third_additional_layer:
        #     regressor.add(GRU(units=n_neurons))
        #     regressor.add(Dropout(dropout))

        regressor.add(LSTM(units=n_neurons, activation='relu'))
        regressor.add(Dropout(dropout))
        regressor.add(Dense(units=x_train.shape[2]))
        regressor.compile(optimizer='adam', loss=smape_loss)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        '''''
        From the mentioned article above --> If a validation dataset is specified to the fit() function via the validation_data or v
        alidation_split arguments,then the loss on the validation dataset will be made available via the name “val_loss.”
        '''''

        file_path = 'temp/best_model.h5'

        mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        '''''
        cb = Callback(...)  # First, callbacks must be instantiated.
        cb_list = [cb, ...]  # Then, one or more callbacks that you intend to use must be added to a Python list.
        model.fit(..., callbacks=cb_list)  # Finally, the list of callbacks is provided to the callback argument when fitting the model.
        '''''

        regressor.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=n_batch_size, callbacks=[es, mc],
                      verbose=0)

        # load the best model
        # regressor = load_model('best_model.h5')

        y_pred = regressor.predict(x_test)
        predicted_demand = sc.inverse_transform(y_pred)
        y_true = sc.inverse_transform(y_test)
        smape = smape_loss(y_true[:, 0], predicted_demand[:, 0])
        if 0.00 < smape < 1.00:
            new_accuracy = 1.00 - smape
            if new_accuracy > best_accracy:
                indicator = True
                best_accracy = new_accuracy
                best_smape = smape
                best_regressor = regressor
                best_predicted_demand = predicted_demand
                hist.append(list(
                    (n_neurons, n_batch_size,
                     dropout,
                     best_accracy)))
                if best_accracy > 0.7999999:
                    break
            else:
                if indicator:
                    pass
                else:
                    best_regressor, best_predicted_demand = [], []
                hist.append(list((n_neurons, n_batch_size, dropout, 'not working')))
        else:
            if indicator:
                pass
            else:
                best_regressor, best_predicted_demand = [], []
            hist.append(list((n_neurons, n_batch_size, dropout, 'not working')))
    return best_accracy, best_smape, best_regressor, best_predicted_demand, hist

#Calculate the error indicator
def calculate_smape_RF(df_forcalculate, step):
    # calculate the correlation between each feature and demand and choose the first 8 features
    corr_matrix = df_forcalculate.corr()
    demand_corr = corr_matrix.sort_values(by=['demand'], ascending=False)
    index_list = demand_corr.index.tolist()
    df = df_forcalculate[index_list[0:8]]
    real_demand = df['demand']
    # Normalize the data
    scaler = StandardScaler()
    df_sc = scaler.fit_transform(df.reset_index(drop=True))
    dataset = np.array(df_sc)

    X_train, X_test, y_train, y_test = datasplit(dataset, step)
    model = seq2seqModel(X_train, step)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    model.fit(X_train, y_train, epochs=50, verbose=0, validation_split=0.2, batch_size=64, callbacks=[es])
    y_pred = model.predict(X_test)
    predicted_demand = scaler.inverse_transform(y_pred)
    y_true = scaler.inverse_transform(y_test)
    smape = smape_loss(y_true[:, 0], predicted_demand[:, 0])

    return dataset, model, X_train, X_test, y_train, y_test, predicted_demand, smape, scaler, real_demand