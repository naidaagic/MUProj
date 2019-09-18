import numpy as np

from michiko.utils.metrics import accuracy
from michiko.model import NeuralNetwork

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs


def magic_data():
    data, labels = make_blobs(
        n_samples=5000,
        centers=3,
        n_features=3,
        random_state=0
    )
    x_train, x_val, y_train, y_val = train_test_split(data, labels, stratify=labels, random_state=0)
    return [x_train, x_val, y_train, y_val]


def one_hot_encoder(y_train, y_val):

    enc = OneHotEncoder(categories="auto")
    y_one_hit_train = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
    y_one_hit_val = enc.fit_transform(np.expand_dims(y_val, 1)).toarray()
    return [y_one_hit_train, y_one_hit_val]


def train():
    [x_train, x_val, y_train, y_val] = magic_data()
    [y_train_enc, _] = one_hot_encoder(y_train, y_val)

    model = NeuralNetwork()

    model.fit(
        x_train,
        y_train_enc
    )

    y_prediction_train = model.predict(x_train)
    y_prediction_train = np.argmax(y_prediction_train, 1)
    y_prediction_val = model.predict(x_val)
    y_prediction_val = np.argmax(y_prediction_val, 1)

    accuracy_train = accuracy(y_train, y_prediction_train)
    accuracy_validation = accuracy(y_val, y_prediction_val)

    print(f"acc: {round(accuracy_train, 2)}")
    print(f"val acc: {round(accuracy_validation, 2)}")


if __name__ == "__main__":
    train()