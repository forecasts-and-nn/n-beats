import collections

import numpy as np
import tensorflow as tf

EAGER_EXECUTION = False  # used for debugging.
if EAGER_EXECUTION:
    tf.enable_eager_execution()


def mae(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true - y_pred))


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
    t = t / tf.cast(tf.cast(length, tf.int32), tf.float32)  # normalise.
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


def seasonality_model(thetas, h, is_forecast=True):
    t = linear_space(h, is_forecast)
    if h % 2 == 0:
        max_p2 = max_p1 = h // 2
    else:
        max_p1 = h // 2
        max_p2 = h - max_p1
    # range(1, 3) => [1, 2].
    s1 = tf.stack([tf.cos(2 * np.pi * i * t) for i in range(max_p1)], axis=0)  # H/2-1
    s2 = tf.stack([tf.sin(2 * np.pi * i * t) for i in range(1, max_p2)], axis=0)
    S = tf.concat([s1, s2], axis=0)
    return tf.matmul(thetas, S)


def block(x, units=256, nb_thetas=64, block_type='generic', backcast_length=10, forecast_length=5):
    for _ in range(4):
        x = tf.layers.Dense(units, activation='relu')(x)

    # 3.1 Basic block. Phi_theta^f : R^{dim(x)} -> theta_f.
    # 3.1 Basic block. Phi_theta^b : R^{dim(x)} -> theta_b.
    if block_type == 'generic':
        theta_b = tf.layers.Dense(nb_thetas, activation='linear')(x)
        theta_f = tf.layers.Dense(nb_thetas, activation='linear')(x)
        backcast = tf.layers.Dense(backcast_length, activation='linear')(theta_b)  # generic. 3.3.
        forecast = tf.layers.Dense(forecast_length, activation='linear')(theta_f)  # generic. 3.3.
    elif block_type == 'trend':
        theta_b = tf.layers.Dense(nb_thetas, activation='linear')(x)
        theta_f = tf.layers.Dense(nb_thetas, activation='linear')(x)
        backcast = trend_model(theta_b, backcast_length, is_forecast=False)  # 3.3 g_f = g_b
        forecast = trend_model(theta_f, forecast_length, is_forecast=True)
    elif block_type == 'seasonality':
        # length(theta) is pre-defined here.
        theta_b = tf.layers.Dense(backcast_length - 1, activation='linear')(x)
        theta_f = tf.layers.Dense(forecast_length - 1, activation='linear')(x)
        backcast = seasonality_model(theta_b, backcast_length, is_forecast=False)  # 3.3 g_f = g_b
        forecast = seasonality_model(theta_f, forecast_length, is_forecast=True)
    else:
        raise Exception('Unknown block_type.')

    return backcast, forecast


def net(x, units=256, nb_layers=2, nb_thetas=10, nb_blocks=3,
        block_types=['seasonality'] * 2, backcast_length=10, forecast_length=5):
    assert len(block_types) == nb_layers
    forecasts = []
    for j in range(nb_layers):
        skip_connections = []
        for i in range(nb_blocks):
            b, f = block(x, units, nb_thetas, block_types[j], backcast_length, forecast_length)
            x = x - b
            skip_connections.append(f)
        y = tf.add_n(skip_connections)
        forecasts.append(y)
    y = tf.add_n(forecasts)
    return x, y


def get_data(length, test_starts_at, signal_type='generic', random=False):
    if random:
        offset = np.random.standard_normal() * 0.1
    else:
        offset = 1
    if signal_type in ['trend', 'generic']:
        x = np.arange(0, 1, 1 / length) + offset
    elif signal_type == 'seasonality':
        random_period_coefficient = np.random.randint(low=6, high=10)
        random_period_coefficient_2 = np.random.randint(low=2, high=6)
        x = np.cos(random_period_coefficient * np.pi * np.arange(0, 1, 1 / length)) + np.sign(offset) * np.arange(0, 1,
                                                                                                                  1 / length) + offset
        x += np.cos(random_period_coefficient_2 * np.pi * np.arange(0, 1, 1 / length))
        # import matplotlib.pyplot as plt
        # plt.plot(x)
        # plt.show()
    else:
        raise Exception('Unknown signal type.')
    x = np.expand_dims(x, axis=0)
    y = x[:, test_starts_at:]
    x = x[:, :test_starts_at]
    return x, y


def train():
    forecast_length = 20
    backcast_length = 3 * forecast_length  # 4H in [2H, 7H].

    signal_type = 'seasonality'
    block_types = ['trend', 'seasonality']
    # block_types = ['trend', 'seasonality']

    sess = tf.Session()

    if EAGER_EXECUTION:
        x, y = get_data(length=backcast_length + forecast_length,
                        test_starts_at=backcast_length,
                        signal_type=signal_type)
        x_inputs = tf.constant(dtype=tf.float32, value=x)
        y_true = tf.constant(dtype=tf.float32, value=y)
    else:
        x_inputs = tf.placeholder(dtype=tf.float32, shape=(None, backcast_length))
        y_true = tf.placeholder(dtype=tf.float32, shape=(None, forecast_length))
    res, output = net(x_inputs,
                      units=256,
                      nb_layers=len(block_types),
                      nb_thetas=2,
                      nb_blocks=3,
                      block_types=block_types,
                      backcast_length=backcast_length,
                      forecast_length=forecast_length)

    if EAGER_EXECUTION:
        exit(1)  # stop here. eager used for debugging.

    loss = mae(y_true, output)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    sess.run(tf.global_variables_initializer())
    running_loss = collections.deque(maxlen=100)
    for step in range(1000000):
        x, y = get_data(length=backcast_length + forecast_length,
                        test_starts_at=backcast_length,
                        signal_type=signal_type,
                        random=True)
        feed_dict = {x_inputs: x, y_true: y}
        current_loss, _ = sess.run([loss, train_op], feed_dict)
        running_loss.append(current_loss)
        if step % 100 == 0:
            if step % 2000 == 0:
                predictions = sess.run(output, feed_dict)
                import matplotlib.pyplot as plt
                plt.grid(True)
                x_y = np.concatenate([x, y], axis=-1).flatten()
                plt.plot(list(range(backcast_length)), x.flatten(), color='b')
                plt.plot(list(range(len(x_y) - forecast_length, len(x_y))), y.flatten(), color='g')
                plt.plot(list(range(len(x_y) - forecast_length, len(x_y))), predictions.flatten(), color='r')
                plt.scatter(range(len(x_y)), x_y.flatten(), color=['b'] * backcast_length + ['g'] * forecast_length)
                plt.scatter(list(range(len(x_y) - forecast_length, len(x_y))), predictions.flatten(),
                            color=['r'] * forecast_length)
                plt.legend(['backtest', 'forecast', 'predictions of forecast'])
                # plt.plot(np.concatenate([x, predictions], axis=-1).flatten(), c='red')
                plt.show()
            print(step, running_loss[-1], np.mean(running_loss))


if __name__ == '__main__':
    train()
