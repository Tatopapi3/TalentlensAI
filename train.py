import os
import json
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.generate_data import generate_match_data, generate_trend_data, SKILL_TRENDS, ROLES, SKILLS, EDUCATION
from models.match_scorer import build_match_model, FEATURE_COLS as MATCH_FEATURES
from models.trend_predictor import build_trend_model, FEATURE_COLS as TREND_FEATURES
from core.drift import save_training_stats

os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)


def train_match_scorer():
    print("\n── Match Scorer ──────────────────────────────")
    df = generate_match_data(6000)
    df.to_csv('data/match_data.csv', index=False)

    X = df[MATCH_FEATURES].values
    y = df['match_score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    save_training_stats(df[MATCH_FEATURES], MATCH_FEATURES)

    model = build_match_model(X_train.shape[1])
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30, batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    model.save('model/match_model.h5')
    with open('model/match_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    _, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Match Scorer saved  |  MAE: {mae:.3f}")


def train_trend_predictor():
    print("\n── Trend Predictor ───────────────────────────")
    df = generate_trend_data()
    df.to_csv('data/trend_data.csv', index=False)

    X = df[TREND_FEATURES].values
    y = df['demand_score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = build_trend_model(X_train.shape[1])
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30, batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    model.save('model/trend_model.h5')
    with open('model/trend_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    _, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Trend Predictor saved  |  MAE: {mae:.3f}")


if __name__ == '__main__':
    meta = {
        'skills':     list(SKILL_TRENDS.keys()),
        'roles':      ROLES,
        'all_skills': SKILLS,
        'education':  EDUCATION,
    }
    with open('data/meta.json', 'w') as f:
        json.dump(meta, f)

    train_match_scorer()
    train_trend_predictor()
    print("\nAll models saved to model/")
