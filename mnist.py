import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0

    model = tf.keras.models.Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
        metrics=["acc"])
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    model.evaluate(x_test, y_test)
