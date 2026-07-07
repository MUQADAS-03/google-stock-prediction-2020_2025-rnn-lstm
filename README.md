# Google Stock Price Prediction — RNN, LSTM & GRU

## Project Overview
A multi-interface deep learning application that predicts Google stock prices using three sequence-modeling architectures — **RNN**, **LSTM**, and **GRU** — trained on historical time-series data. The project goes beyond model training to include a full comparative analysis of the three architectures' forecasting performance, plus three separate deployment interfaces (Streamlit, Gradio, and Flask) for interactive visualization and next-day price forecasting.

## Key Features
- Historical stock price data cleaning and exploratory analysis
- Three trained deep learning models (RNN, GRU, LSTM) for time-series forecasting
- Side-by-side model comparison to evaluate forecasting accuracy across architectures
- Interactive dashboards for visualizing historical trends alongside model predictions
- Three independent deployment options: **Streamlit** dashboard, **Gradio** interface, and **Flask** API
- Persisted models and scaler for efficient, repeatable inference without retraining

## Tech Stack
- **Deep Learning:** TensorFlow / Keras (RNN, LSTM, GRU)
- **Data Preprocessing:** pandas, scikit-learn (MinMaxScaler)
- **Deployment:** Streamlit, Gradio, Flask
- **Model Persistence:** HDF5 (`.h5`) for models, pickle (`.pkl`) for the scaler

## Repository Structure
```
google-stock-price-prediction/
├── data_cleaning_eda.ipynb        # Data cleaning and exploratory analysis
├── lstm_model_training.ipynb      # Model training and comparison (RNN/LSTM/GRU)
├── google_stock_cleaned.csv       # Cleaned historical stock price dataset
├── lstm_model.h5                  # Trained LSTM model
├── rnn_model.h5                   # Trained RNN model
├── gru_model.h5                   # Trained GRU model
├── scaler.pkl                     # Fitted MinMaxScaler for inference
├── app_streamlit.py               # Streamlit dashboard interface
├── app_flask.py                   # Flask API interface
├── gradio_app.py                  # Gradio interactive interface
├── README.md
└── requirements.txt
```

## Dataset Information
| Attribute | Detail |
|---|---|
| Subject | Google (GOOGL) historical stock price data |
| File | `google_stock_cleaned.csv` |
| Preprocessing | Cleaned and normalized using `MinMaxScaler` before model training |

## Environment Setup
```bash
git clone https://github.com/MUQADAS-03/google-stock-price-prediction.git
cd google-stock-price-prediction
pip install -r requirements.txt
```

## Running the Project

**1. Data cleaning & EDA**
```bash
jupyter notebook data_cleaning_eda.ipynb
```

**2. Model training & comparison**
```bash
jupyter notebook lstm_model_training.ipynb
```

**3. Run an interface (pick one):**
```bash
# Streamlit dashboard
streamlit run app_streamlit.py

# Gradio interface
python gradio_app.py

# Flask API
python app_flask.py
```

## Modeling Approach
- Historical price sequences were normalized using `MinMaxScaler` to bring values into a consistent range suitable for neural network training
- Three architectures — **RNN**, **LSTM**, and **GRU** — were built and trained using TensorFlow/Keras to capture temporal dependencies in the stock price sequence
- Each model was evaluated and compared on its ability to forecast next-day prices, allowing a direct comparison of how well simple RNNs, LSTM's long-term memory gating, and GRU's simplified gating mechanism each handle financial time-series data
- Trained models (`lstm_model.h5`, `rnn_model.h5`, `gru_model.h5`) and the fitted scaler (`scaler.pkl`) were persisted separately, so inference doesn't require retraining

## Deployment Interfaces
Three separate front-ends were built around the same trained models, each serving a different use case:
- **Streamlit** — a full interactive dashboard for visualizing historical trends against model predictions
- **Gradio** — a lightweight interface for quick, shareable demos
- **Flask** — a REST API for programmatic access to predictions

## Key Learnings
This project reinforced practical experience in sequence modeling, hyperparameter tuning for time-series data, and deploying the same trained model through multiple, framework-different web interfaces.

## Author
**Muqadas Yasin**
