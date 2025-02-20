# Crypto Price Prediction

## 📌 Project Overview
This project focuses on predicting cryptocurrency prices using Python, Jupyter Notebook, and various data science libraries. The model leverages historical price data and machine learning techniques to forecast future prices, providing insights for traders and investors.

## 🚀 Features
- Fetch real-time and historical crypto data
- Data preprocessing with Pandas & NumPy
- Exploratory Data Analysis (EDA) using Matplotlib & Seaborn
- Machine Learning models for price prediction (e.g., Linear Regression, LSTM, Random Forest)
- Model evaluation with performance metrics
- Interactive visualization of predictions

## 🛠 Tech Stack
- **Programming Language:** Python
- **IDE:** Jupyter Notebook
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, TensorFlow/Keras (for deep learning models)
- **APIs for Data Retrieval:** CoinGecko API, Binance API

## 📊 Dataset
The dataset consists of historical price data for various cryptocurrencies (e.g., Bitcoin, Ethereum). It includes features like:
- Open, High, Low, Close (OHLC) prices
- Volume traded
- Market capitalization
- Technical indicators (moving averages, RSI, MACD, etc.)

Data is fetched from APIs such as Binance, CoinGecko, or Kaggle datasets.

## 🧠 Model Training & Prediction
The model training process involves:
1. **Data Collection & Cleaning**
2. **Feature Engineering** (e.g., Moving Averages, RSI, MACD)
3. **Training Machine Learning Models** (Linear Regression, LSTMs, Random Forest, etc.)
4. **Hyperparameter Tuning & Optimization**
5. **Model Evaluation & Validation**
6. **Visualization of Predictions**

## ⚡ Installation
### Clone the repository
```sh
git clone https://github.com/mehdidinaru/crypto-price-predict.git
cd crypto-price-predict
```
### Create a virtual environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
### Install dependencies
```sh
pip install -r requirements.txt
```

## 🔍 Usage
1. Open Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
2. Run the notebooks in the `notebooks/` directory to fetch data, preprocess, train, and evaluate the model.
3. Modify parameters and retrain the model as needed.

## 📈 Results & Performance
- The model is evaluated using metrics such as:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
- Comparison of different models' performance
- Visualization of predicted vs. actual prices

## 📌 Future Improvements
- Integrating more advanced models like Transformers
- Implementing Reinforcement Learning for trading strategies
- Deploying the model via Flask/Django API
- Creating a real-time prediction dashboard

## 🤝 Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References
- [CoinGecko API](https://www.coingecko.com/en/api)
- [Binance API](https://binance-docs.github.io/apidocs/)
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
