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


def linear_space(length, fwd_looking=True):
    if fwd_looking:
        t = tf.linspace(0.0, tf.cast(length, tf.float32) - 1, tf.cast(length, tf.int32))
    else:
        t = tf.linspace(-tf.cast(length, tf.float32), 0.0, tf.cast(length, tf.int32))
    return t


def linear_space_divided(length, fwd_looking=True):
    t1 = linear_space(length / 2, fwd_looking=fwd_looking)
    if t1.get_shape().as_list()[-1] * 2 != length:
        t2 = linear_space(length / 2 + 1, fwd_looking=fwd_looking)  # odd.
    else:
        t2 = t1  # even.
    return t1, t2


def trend_model(thetas, length, is_forecast=True):
    p = thetas.get_shape().as_list()[-1]
    t = linear_space(length, fwd_looking=is_forecast)
    T = tf.stack([t ** i for i in range(p)], axis=0)
    return tf.matmul(thetas, T)


def seasonality_model(thetas, length, is_forecast=True):
    p = thetas.get_shape().as_list()[-1]
    t1, t2 = linear_space_divided(length, is_forecast)
    s1 = tf.stack([tf.cos(2 * np.pi * i * t1) for i in range(p)], axis=0)
    s2 = tf.stack([tf.sin(2 * np.pi * i * t2) for i in range(p)], axis=0)
    # s1 = tf.map_fn(lambda z: tf.cos(2 * np.pi * z), t)
    # s2 = tf.map_fn(lambda z: tf.sin(2 * np.pi * z), t)
    S = tf.concat([s1, s2], axis=-1)
    return tf.matmul(thetas, S)


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
        backcast = trend_model(theta_b, backcast_length, is_forecast=False)
        forecast = trend_model(theta_f, forecast_length, is_forecast=True)
    elif block_type == 'seasonality':
        backcast = seasonality_model(theta_b, backcast_length, is_forecast=False)
        forecast = seasonality_model(theta_f, forecast_length, is_forecast=True)
    else:
        raise Exception('Unknown block_type.')

    backcast = x_input - backcast
    return backcast, forecast


def net(x, nb_layers=3, nb_thetas=3, nb_blocks=4, block_types=['seasonality'] * 3,
        backcast_length=10, forecast_length=5):
    forecasts = []
    for j in range(nb_layers):
        skip_connections = []
        for i in range(nb_blocks):
            x, f = block(x, nb_thetas, block_types[j], backcast_length, forecast_length)
            skip_connections.append(f)
        y = tf.add_n(skip_connections)
        forecasts.append(y)
    y = tf.add_n(forecasts)
    return x, y


def train():
    backcast_length = 20
    forecast_length = 5

    signal_type = 'seasonality'
    block_types = ['seasonality'] * 3

    sess = tf.Session()

    x_inputs = tf.placeholder(dtype=tf.float32, shape=(None, backcast_length))
    y_true = tf.placeholder(dtype=tf.float32, shape=(None, forecast_length))
    res, output = net(x_inputs,
                      nb_thetas=10,
                      block_types=block_types,
                      backcast_length=backcast_length,
                      forecast_length=forecast_length)
    loss = mse(y_true, output)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    sess.run(tf.global_variables_initializer())
    for step in range(100000):
        offset = np.random.rand() * 5
        if signal_type in ['trend', 'generic']:
            x = np.arange(0, 1, 1 / (backcast_length + forecast_length)) + offset
        elif signal_type == 'seasonality':
            x = np.cos(2 * np.pi * np.arange(0, 1, 1 / (backcast_length + forecast_length))) + offset
            # import matplotlib.pyplot as plt
            # plt.plot(x)
            # plt.show()
            # exit(1)
        else:
            raise Exception('Unknown signal type.')
        x = np.expand_dims(x, axis=0)
        y = x[:, backcast_length:]
        x = x[:, :backcast_length]
        feed_dict = {x_inputs: x, y_true: y}
        sess.run(train_op, feed_dict)
        if step % 1000 == 0:
            print(step, sess.run(loss, feed_dict))


if __name__ == '__main__':
    train()
