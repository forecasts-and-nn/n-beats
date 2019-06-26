import numpy as np
import tensorflow as tf


# tf.enable_eager_execution()

def mse(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred))


def smape(y_true, y_pred):
    return tf.reduce_mean(2 * tf.abs(y_true - y_pred) / (tf.abs(y_true) + tf.abs(y_pred)))


def mase(y_true, y_pred, m=12):  # m = period.
    return tf.reduce_mean(
        tf.abs(y_true - y_pred) / tf.reduce_mean(y_true[m + 1:] - y_true[0:tf.shape(y_true)[0] - (m + 1)]))


def owa(y_true, y_pred, m=12):
    return 0.5 * smape(y_true, y_pred) + 0.5 * mase(y_true, y_pred, m)


# https://machinelearningmastery.com/time-series-seasonality-with-python/

def trend_model(thetas, p, length, is_forecast=True):
    if is_forecast:
        t = tf.linspace(0, length - 1, length)
    else:
        t = tf.linspace(-length, 0, length)
    t_p = tf.stack([t ** i for i in range(p)])
    return t_p * thetas


def block(x_input, units=64, block_type='generic', backcast_length=10, forecast_length=5):
    x = x_input
    for _ in range(4):
        x = tf.layers.Dense(units, activation='relu')(x)
    theta_b = tf.layers.Dense(units)(x)
    theta_f = tf.layers.Dense(units)(x)

    if block_type == 'generic':
        backcast = tf.layers.Dense(backcast_length)(theta_b)  # generic.
        forecast = tf.layers.Dense(forecast_length)(theta_f)  # generic.
    elif block_type == 'trend':
        backcast = trend_model(theta_b, tf.shape(theta_b)[-1], backcast_length, is_forecast=False)
        forecast = trend_model(theta_f, tf.shape(theta_f)[-1], forecast_length, is_forecast=True)
    elif block_type == 'seasonality':
        backcast = 1
        forecast = 1
    else:
        raise Exception('Unknown block_type.')

    backcast = x_input - backcast
    return backcast, forecast


def net(x, nb_layers=3, nb_blocks=4, block_types=['generic'] * 3, backcast_length=10, forecast_length=5):
    forecasts = []
    for j in range(nb_layers):
        skip_connections = []
        for i in range(nb_blocks):
            x, f = block(x, 3, block_types[j], backcast_length, forecast_length)
            skip_connections.append(f)
        y = tf.add_n(skip_connections)
        forecasts.append(y)
    y = tf.add_n(forecasts)
    return x, y


def train():
    backcast_length = 10
    forecast_length = 5
    # sig = np.random.standard_normal(size=(1, 100))
    # x = tf.constant(sig, dtype=tf.float32)

    sess = tf.Session()

    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, backcast_length))
    tf_y = tf.placeholder(dtype=tf.float32, shape=(None, forecast_length))
    res, y = net(tf_x, backcast_length=backcast_length, forecast_length=forecast_length)
    loss = mse(tf_y, y)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    sess.run(tf.global_variables_initializer())
    for step in range(100000):
        x = np.arange(0, 10, 10 / (backcast_length + forecast_length))
        x = np.expand_dims(x, axis=0)
        y = x[:, backcast_length:]
        x = x[:, :backcast_length]
        feed_dict = {tf_x: x, tf_y: y}
        print(step, sess.run([loss, train_op], feed_dict))


if __name__ == '__main__':
    train()
