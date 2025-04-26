# LSTM Stock Price Predictor for $TSLA

[![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Keras/TensorFlow-FF6F00?logo=tensorflow)](https://www.tensorflow.org/api_docs/python/tf/keras)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Add License if applicable -->

![Stock Prediction Plot](placeholder_for_plot_image.png) <!-- Optional: Add a screenshot of your final plot -->

**Predicting future stock prices for Tesla ($TSLA) using historical data and a Long Short-Term Memory (LSTM) neural network.**

This project demonstrates how to build, train, and evaluate an LSTM model using Keras (TensorFlow backend) to forecast stock closing prices based on the previous 100 days of data.

## 📚 Table of Contents

*   [Overview](#-overview)
*   [Features](#-features)
*   [Methodology](#-methodology)
*   [Tech Stack](#-tech-stack)
*   [Getting Started](#-getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation-)
*   [Usage](#-usage)
*   [Model](#-model)
*   [Results](#-results)
*   [Contributing](#-contributing)
*   [License](#-license)
*   [Acknowledgments](#-acknowledgments)

## 📗 Overview

Stock market prediction is a classic time-series forecasting problem. This project tackles it by leveraging LSTMs, a type of recurrent neural network (RNN) well-suited for learning patterns in sequential data like stock prices. We use historical data for Tesla ($TSLA) obtained via the `yfinance` library, preprocess it, and train an LSTM model to predict the next day's closing price based on a window of the preceding 100 days.

## ✨ Features

*   **Data Acquisition:** Fetches historical stock data directly using the `yfinance` API.
*   **Data Preprocessing:** Cleans and scales data using `pandas` and `scikit-learn`'s `MinMaxScaler`.
*   **LSTM Model:** Implements a multi-layered LSTM network with Dropout for regularization using Keras.
*   **Training & Evaluation:** Splits data into training and testing sets, trains the model, and evaluates performance by comparing predicted vs. actual prices.
*   **Visualization:** Uses `matplotlib` to plot historical prices, moving averages, and the comparison between predicted and actual closing prices.
*   **Saved Model:** Includes a pre-trained Keras model (`modelo_keras.h5`) for quick use or further analysis.

## 🔬 Methodology

1.  **Data Collection:** Download historical $TSLA stock data (Open, High, Low, Close, Volume) from 2010 to the present.
2.  **Feature Selection:** Focus on the 'Close' price for prediction.
3.  **Data Splitting:** Divide the dataset into training (70%) and testing (30%) sets chronologically.
4.  **Scaling:** Apply `MinMaxScaler` to scale the 'Close' prices between 0 and 1, which helps LSTM convergence.
5.  **Sequence Creation:** Transform the data into sequences. For each day's price (y), the input (X) consists of the scaled prices of the previous 100 days.
6.  **Model Building:** Construct a Sequential Keras model with multiple LSTM layers (using ReLU activation) and Dropout layers to prevent overfitting. A final Dense layer outputs the prediction.
7.  **Training:** Train the model on the prepared training sequences using the Adam optimizer and Mean Squared Error loss function.
8.  **Prediction:** Prepare the test sequences (ensuring the first sequence uses the last 100 days of the *training* data) and generate predictions using the trained model.
9.  **Evaluation:** Inverse transform the scaled predictions and actual test values back to their original price scale and visualize the results.

## 🛠️ Tech Stack

*   **Language:** Python 3.7+
*   **Data Handling:** `pandas`, `numpy`
*   **Data Source:** `yfinance`
*   **Machine Learning:** `tensorflow` (with `keras` API), `scikit-learn`
*   **Visualization:** `matplotlib`, `seaborn`
*   **Environment:** Jupyter Notebook / Google Colab (or standard Python environment)

## 🚀 Getting Started

### Prerequisites

*   Python 3.7 or later
*   pip (Python package installer)
*   Git (Optional, for cloning)

### Installation 🔧

1.  **Clone the repository (Optional):**
    ```bash
    git clone https://github.com/YourUsername/Stock_price_prediction.git # Replace YourUsername
    cd Stock_price_prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```
    *(Alternatively, if a `requirements.txt` is provided: `pip install -r requirements.txt`)*

## ✍️ Usage

1.  Open the `modelo_prediccion.ipynb` notebook in a Jupyter environment (like Jupyter Lab, Jupyter Notebook, Google Colab, or VS Code).
2.  Run the cells sequentially from top to bottom.
3.  The notebook will:
    *   Install `yfinance`.
    *   Download the latest $TSLA data.
    *   Preprocess the data.
    *   Build and train the LSTM model (or load the existing one if you uncomment the load section).
    *   Generate predictions on the test set.
    *   Display plots showing the data, moving averages, and the comparison of actual vs. predicted prices.

*   **Note:** Training the model (`model.fit`) can take some time depending on your hardware and the number of epochs. The provided `modelo_keras.h5` allows skipping the training step if desired.

## 🧠 Model

The core of the prediction is the `modelo_keras.h5` file, which contains the trained weights of the LSTM network.

*   **Architecture:** Sequential model with 4 LSTM layers (50, 60, 80, 120 units respectively) using ReLU activation and Dropout layers (0.2, 0.3, 0.4, 0.5) after each LSTM layer, followed by a Dense output layer with 1 unit.
*   **Input Shape:** The model expects input sequences of shape `(100, 1)`, representing 100 days of single-feature (scaled closing price) data.

You can load this pre-trained model using:
```python
from keras.models import load_model
model = load_model('modelo_keras.h5')
```

## 📊 Results

The final output includes a plot comparing the actual closing prices (blue line) with the prices predicted by the LSTM model (green line) for the test period. This visualization helps assess the model's performance in capturing the trend and fluctuations of the stock price.

*(Consider adding the final plot image here or linking to it)*

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE.txt` for more information. (***Note:** Add a LICENSE file if you intend to use one*).

## 🙏 Acknowledgments

*   [yfinance](https://github.com/ranaroussi/yfinance) for providing easy access to stock data.
*   [TensorFlow/Keras](https://www.tensorflow.org/guide/keras) for the deep learning framework.
*   [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/) for data manipulation and visualization.
