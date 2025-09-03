from src.data_loader import load_data
from src.preprocessing import scale_data
from src.sequence_preparation import create_sequences
from src.split_data import split_data
from src.modeling import build_lstm, build_rnn, build_gru
from src.evaluation import evaluate_model
import joblib

if __name__ == "__main__":
    # Load dataset
    df = load_data("data/processed_stock_data.csv")
    scaled_df, scaler = scale_data(df[["Close"]], "MinMaxScaler")

    X, y = create_sequences(scaled_df.values, seq_length=60)
    X_train, y_train, X_test, y_test = split_data(X, y)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    models = {
        "lstm": build_lstm(input_shape=(X_train.shape[1], 1)),
        "rnn": build_rnn(input_shape=(X_train.shape[1], 1)),
        "gru": build_gru(input_shape=(X_train.shape[1], 1)),
    }

    for name, model in models.items():
        print(f"\n🔹 Training {name.upper()} model...")
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
        model.save(f"models/{name}_model.h5")
        print(f"✅ {name.upper()} model saved.")

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved.")
