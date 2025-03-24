from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import pandas as pd


class Encoder:
    def __init__(self):
        pass

    @staticmethod
    def booleanOneColumnEncoder(df_feature, false_feature_parameter):
        df_feature_encoded = df_feature.apply(lambda x: 1 if x == false_feature_parameter else 0)
        return df_feature_encoded.astype('uint8')

    @staticmethod
    def labelEncoder(df_feature):
        encoders = LabelEncoder()
        df_feature_encoded = encoders.fit_transform(df_feature)
        return df_feature_encoded, encoders

    @staticmethod
    def oneHotEncoder(X_train, X_test, columns_for_encoding):
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        encoders = {}
        for column in columns_for_encoding:
            encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
            X_train_encoded = pd.DataFrame(
                encoder.fit_transform(X_train_copy[[column]]),
                columns=encoder.get_feature_names_out([column]),
                index=X_train_copy.index
            )
            X_test_encoded = pd.DataFrame(
                encoder.transform(X_test_copy[[column]]),
                columns=encoder.get_feature_names_out([column]),
                index=X_test_copy.index
            )
            X_train_copy = X_train_copy.drop(column, axis=1).join(X_train_encoded)
            X_test_copy = X_test_copy.drop(column, axis=1).join(X_test_encoded)
            encoders[column] = encoder
        return X_train_copy, X_test_copy, encoders

    @staticmethod
    def binaryEncoder(X_train, X_test, columns_for_encoding):
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        encoders = {}
        for column in columns_for_encoding:
            encoder = ce.BinaryEncoder(cols=[column])
            X_train_encoded = encoder.fit_transform(X_train_copy[[column]])
            X_test_encoded = encoder.transform(X_test_copy[[column]])
            X_train_copy = X_train_copy.drop(column, axis=1).join(X_train_encoded)
            X_test_copy = X_test_copy.drop(column, axis=1).join(X_test_encoded)
            encoders[column] = encoder
        return X_train_copy, X_test_copy, encoders

    @staticmethod
    def helmertEncoder(X_train, X_test, columns_for_encoding):
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        encoders = {}
        for column in columns_for_encoding:
            encoder = ce.HelmertEncoder(cols=[column])
            X_train_encoded = encoder.fit_transform(X_train_copy[[column]])
            X_test_encoded = encoder.transform(X_test_copy[[column]])
            X_train_copy = X_train_copy.drop(column, axis=1).join(X_train_encoded)
            X_test_copy = X_test_copy.drop(column, axis=1).join(X_test_encoded)
            encoders[column] = encoder
        return X_train_copy, X_test_copy, encoders

    @staticmethod
    def backwardDifferenceEncoder(X_train, X_test, columns_for_encoding):
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        encoders = {}
        for column in columns_for_encoding:
            encoder = ce.BackwardDifferenceEncoder(cols=[column])
            X_train_encoded = encoder.fit_transform(X_train_copy[[column]])
            X_test_encoded = encoder.transform(X_test_copy[[column]])
            X_train_copy = X_train_copy.drop(column, axis=1).join(X_train_encoded)
            X_test_copy = X_test_copy.drop(column, axis=1).join(X_test_encoded)
            encoders[column] = encoder
        return X_train_copy, X_test_copy, encoders

    @staticmethod
    def hashingEncoder(X_train, X_test, columns_for_encoding, n_components=8):
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        encoders = {}
        for column in columns_for_encoding:
            encoder = ce.HashingEncoder(n_components=n_components)
            X_train_encoded = encoder.fit_transform(X_train_copy[[column]])
            X_test_encoded = encoder.transform(X_test_copy[[column]])
            X_train_copy = X_train_copy.drop(column, axis=1).join(X_train_encoded)
            X_test_copy = X_test_copy.drop(column, axis=1).join(X_test_encoded)
            encoders[column] = encoder
        return X_train_copy, X_test_copy, encoders

    @staticmethod
    def targetEncoder(X_train, X_test, y_train, columns_for_encoding):
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        encoders = {}
        for column in columns_for_encoding:
            encoder = ce.TargetEncoder()
            X_train_encoded[column] = encoder.fit_transform(X_train[column], y_train)
            X_test_encoded[column] = encoder.transform(X_test[column])
            encoders[column] = encoder
        return X_train_encoded, X_test_encoded, encoders

    @staticmethod
    def leaveOneOutEncoder(X_train, X_test, y_train, columns_for_encoding):
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        encoders = {}
        for column in columns_for_encoding:
            encoder = ce.LeaveOneOutEncoder()
            X_train_encoded[column] = encoder.fit_transform(X_train[column], y_train)
            X_test_encoded[column] = encoder.transform(X_test[column])
            encoders[column] = encoder
        return X_train_encoded, X_test_encoded, encoders

    @staticmethod
    def jamesSteinEncoder(X_train, X_test, y_train, columns_for_encoding):
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        encoders = {}
        for column in columns_for_encoding:
            encoder = ce.JamesSteinEncoder()
            X_train_encoded[column] = encoder.fit_transform(X_train[column], y_train)
            X_test_encoded[column] = encoder.transform(X_test[column])
            encoders[column] = encoder
        return X_train_encoded, X_test_encoded, encoders


