import argparse
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler




def preprocess_data(data, scaler):
    selected_features = ['EK', 'Skewness']  # Actualiza con las características seleccionadas
    preprocessed_data = scaler.transform(data[selected_features])
    return preprocessed_data


def predict(model, preprocessed_data):
    predictions = model.predict(preprocessed_data)
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Pulsar Prediction')
    parser.add_argument('data_file', type=str, help='Archivo CSV con los datos de entrada')
    parser.add_argument('model_file', type=str, help='Archivo del modelo entrenado (.h5)')
    parser.add_argument('scaler_file', type=str, help='Archivo del normalizador (.pkl)')
    args = parser.parse_args()

    # Cargar el modelo
    model = tf.keras.models.load_model(args.model_file)

    # Cargar el escalador
    scaler = joblib.load(args.scaler_file)

    data = pd.read_csv(args.data_file)

    preprocessed_data = preprocess_data(data, scaler)

    predictions = predict(model, preprocessed_data)

    summary = pd.DataFrame({'Neuronas': np.arange(1, len(predictions) + 1), 'Prediccion': predictions.flatten()})
    print(summary)


if __name__ == '__main__':
    main()
