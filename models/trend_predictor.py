import tensorflow as tf

FEATURE_COLS = ['skill_idx', 'time_idx', 'year', 'quarter']


def build_trend_model(input_dim: int = 4) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
