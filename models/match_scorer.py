import tensorflow as tf

FEATURE_COLS = [
    'skill_overlap', 'missing_skills', 'extra_skills',
    'exp_ratio', 'edu_match', 'req_exp', 'resume_exp', 'role_idx',
]


def build_match_model(input_dim: int = 8) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
