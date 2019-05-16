import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


class NetModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.dims = X_train.shape

    def build_net(self, conv_window=(6, 3), pooling_window=(10, 1), n_filters=(64, 32, 16)):

        input_img = Input(shape=self.dims[1:])  # adapt this if using `channels_first` image data format
        print("shape of input", K.int_shape(input_img))
        conv_1 = Conv2D(n_filters[0], conv_window, activation='relu', padding='same')(input_img)
        print("shape after first conv", K.int_shape(conv_1))
        pool_1 = MaxPooling2D(pooling_window, padding='same')(conv_1)
        print("shape after first pooling", K.int_shape(pool_1))
        conv_2 = Conv2D(n_filters[1], conv_window, activation='relu', padding='same')(pool_1)
        print("shape after second conv", K.int_shape(conv_2))

        pool_2 = MaxPooling2D(pooling_window, padding='same')(conv_2)
        print("shape after second pooling", K.int_shape(pool_2))

        conv_3 = Conv2D(n_filters[2], conv_window, activation='relu', padding='same')(pool_2)
        print("shape after third conv", K.int_shape(conv_3))

        encoded = MaxPooling2D(pooling_window, padding='same')(conv_3)
        print("shape of encoded", K.int_shape(encoded))

        up_3 = UpSampling2D(pooling_window)(encoded)
        print("shape after upsample third pooling", K.int_shape(up_3))

        conv_neg_3 = Conv2D(n_filters[2], conv_window, activation='relu', padding='same')(up_3)
        print("shape after decode third conv", K.int_shape(conv_neg_3))

        up_2 = UpSampling2D(pooling_window)(conv_neg_3)
        print("shape after upsample second pooling", K.int_shape(up_2))

        conv_neg_2 = Conv2D(n_filters[1], conv_window, activation='relu', padding='same')(up_2)
        print("shape after decode second conv", K.int_shape(conv_neg_2))
        up_1 = UpSampling2D(pooling_window)(conv_neg_2)
        print("shape after upsample first pooling", K.int_shape(up_1))
        conv_neg_3 = Conv2D(n_filters[0], conv_window, activation='relu', padding='same')(up_1)
        print("shape after decode first conv", K.int_shape(conv_neg_3))
        decoded = Conv2D(1, conv_window, activation='linear', padding='same')(conv_neg_3)
        print("shape after decode to input", K.int_shape(decoded))

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        self.encoder_model = Model(self.autoencoder.input, self.autoencoder.layers[6].output)

    def train_encoder(self, n_epochs=100, batch_size=64):
        self.autoencoder.fit(self.X_train, self.X_train, epochs=n_epochs,
                             batch_size=batch_size, shuffle=True)

    def get_encoded_series(self):
        self.reconstructed_train = self.autoencoder.predict(self.X_train)
        self.reconstructed_test = self.autoencoder.predict(self.X_test)
        self.lf_train = self.flatten_arr(self.encoder_model.predict(self.X_train))
        self.lf_test = self.flatten_arr(self.encoder_model.predict(self.X_test))
        self.train_features = self.merge_features(self.reconstructed_train, self.X_train, self.lf_train)
        self.test_features = self.merge_features(self.reconstructed_test, self.X_test, self.lf_test)

    @staticmethod
    def merge_features(X_, X, lf):
        recon_loss = [mean_squared_error(X_[i][:, :, 0], X[i][:, :, 0]) for i in range(len(X))]
        keys = [f'feature_{i}' for i in range(lf.shape[1])]
        vals = lf.T
        df = pd.DataFrame(dict(list(zip(keys, vals))))
        df['recon_loss'] = recon_loss
        return df

    @staticmethod
    def flatten_arr(arr):
        flat = []
        for a in arr:
            flat.append(a.reshape(-1,))
        return np.array(flat)

    def train_classifier(self, model='xgb', n_search=10):
        if model == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            param_grid = {"max_depth": [10, 20, 40, None],
                          "max_features": sp_randint(1, 20),
                          "min_samples_split": sp_randint(5, 50),
                          "min_samples_leaf": sp_randint(5, 50),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
            clf = RandomForestClassifier(verbose=0, n_estimators=100)
        elif model == 'xgb':
            import xgboost as xgb
            param_grid = {'silent': [True],
                          'max_depth': [5, 10, 20],
                          'learning_rate': [0.001, 0.01],
                          'subsample': [0.2, 0.3, 0.5, 0.6, 0.9, 1.0],
                          'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                          'gamma': [0, 0.25, 0.5, 1.0],
                          'reg_lambda': [0.1, 1.0, 50.0, 100.0, 200.0],
                          'n_estimators': [100],
                          'max_features': [3, 10, None]}
            clf = xgb.XGBClassifier()

        clf_grid = RandomizedSearchCV(clf, param_distributions=param_grid,
                                      n_iter=n_seach, cv=3, iid=False)
        clf_grid.fit(self.train_features.values, self.y_train)
        self.train_acc = clf_grid.score(self.train_features.values, self.y_train)
        print(f'training acc: {self.train_acc}')

        y_pred = clf_grid.predict(self.test_features.values)
        print(classification_report(self.y_test, y_pred))

        self.clf = clf_grid.best_estimator_
