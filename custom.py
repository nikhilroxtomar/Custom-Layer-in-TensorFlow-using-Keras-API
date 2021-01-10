import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, num_units, activation="relu"):
        super(CustomDense, self).__init__()

        self.num_units = num_units
        self.activation = Activation(activation)

    def build(self, input_shape):
        ## (32, 784) * (784, 10) + (10)

        self.weight = self.add_weight(shape=[input_shape[-1], self.num_units])
        self.bias = self.add_weight(shape=[self.num_units])

    def call(self, input):
        y = tf.matmul(input, self.weight) + self.bias
        y = self.activation(y)
        return y

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0

    model = tf.keras.models.Sequential([
        Flatten(input_shape=(28, 28)),
        CustomDense(128, activation="relu"),
        Dropout(0.3),
        CustomDense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
        metrics=["acc"])
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    model.evaluate(x_test, y_test)
