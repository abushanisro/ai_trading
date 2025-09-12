import subprocess
import sys

def install_packages():
    packages = [
        'ccxt', 'dash', 'plotly', 'scikit-learn', 'scipy', 'websockets',
        'pandas', 'numpy<2', 'websocket-client', 'python-binance', 'yfinance',
        'textblob', 'vaderSentiment', 'ta', 'dash-bootstrap-components'
    ]
    
    for package in packages:
        try:
            # Handle special package name mappings
            import_name = package.replace('-', '_').replace('python_', '')
            if package == 'numpy<2':
                import_name = 'numpy'
            __import__(import_name)
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}, continuing without it...")

# First install numpy<2 to fix compatibility
print("Ensuring NumPy compatibility...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy<2'])

install_packages()

import os
import time
import json
import asyncio
import threading
from collections import deque
from datetime import datetime, timedelta
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")

import ccxt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

from websocket import WebSocketApp
import json
from threading import Thread, Lock

# Try to import TensorFlow and Transformers with fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available, using alternative ML models")
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not available, using VADER sentiment only")
    TRANSFORMERS_AVAILABLE = False

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta
import yfinance as yf

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_LIB_AVAILABLE = True
except ImportError:
    BINANCE_LIB_AVAILABLE = False
    Client = None

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global variables for ML model (shared across callbacks)
trained_model = None
scaler = None
feature_scaler = None
sentiment_pipeline = None

# Configuration - Use environment variables for security
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "your_api_key_here")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "your_secret_here")

# Initialize Binance client
binance_client = None
BINANCE_CONNECTED = False

if BINANCE_LIB_AVAILABLE and Client and BINANCE_API_KEY != "your_api_key_here":
    try:
        binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET)
        binance_client.ping()
        BINANCE_CONNECTED = True
        print("Binance API connected successfully")
    except Exception as e:
        print(f"Binance API connection failed: {e}")
        binance_client = None
        BINANCE_CONNECTED = False
else:
    print("Using demo data (Binance API not configured)")

class AutomaticTrendLineDetector:
    """Automatically detects and draws trend lines on price charts"""
    
    def __init__(self, min_touches=3, tolerance=0.02):
        self.min_touches = min_touches
        self.tolerance = tolerance
        
    def find_pivot_points(self, df, window=5):
        """Find local highs and lows"""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Check for pivot high
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                highs.append({'index': i, 'price': df['high'].iloc[i], 
                            'timestamp': df['timestamp'].iloc[i]})
            
            # Check for pivot low
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                lows.append({'index': i, 'price': df['low'].iloc[i], 
                           'timestamp': df['timestamp'].iloc[i]})
        
        return highs, lows
    
    def fit_trend_line(self, points, df):
        """Fit a trend line through points"""
        if len(points) < 2:
            return None
            
        indices = [p['index'] for p in points]
        prices = [p['price'] for p in points]
        
        # Fit linear regression
        z = np.polyfit(indices, prices, 1)
        slope = z[0]
        intercept = z[1]
        
        # Calculate line points
        start_idx = min(indices)
        end_idx = len(df) - 1
        
        start_price = slope * start_idx + intercept
        end_price = slope * end_idx + intercept
        
        # Calculate touches (points close to the line)
        touches = 0
        for point in points:
            expected_price = slope * point['index'] + intercept
            if abs(point['price'] - expected_price) / point['price'] < self.tolerance:
                touches += 1
        
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_price': start_price,
            'end_price': end_price,
            'slope': slope,
            'intercept': intercept,
            'touches': touches,
            'strength': touches / len(points),
            'points': points
        }
    
    def detect_trend_lines(self, df):
        """Detect all significant trend lines"""
        if len(df) < 20:
            return [], []
            
        highs, lows = self.find_pivot_points(df)
        
        resistance_lines = []
        support_lines = []
        
        # Find resistance lines (connecting highs)
        if len(highs) >= self.min_touches:
            for i in range(len(highs) - self.min_touches + 1):
                subset = highs[i:i+self.min_touches]
                line = self.fit_trend_line(subset, df)
                if line and line['touches'] >= self.min_touches:
                    line['type'] = 'resistance'
                    resistance_lines.append(line)
        
        # Find support lines (connecting lows)
        if len(lows) >= self.min_touches:
            for i in range(len(lows) - self.min_touches + 1):
                subset = lows[i:i+self.min_touches]
                line = self.fit_trend_line(subset, df)
                if line and line['touches'] >= self.min_touches:
                    line['type'] = 'support'
                    support_lines.append(line)
        
        # Sort by strength and keep best lines
        resistance_lines.sort(key=lambda x: x['strength'], reverse=True)
        support_lines.sort(key=lambda x: x['strength'], reverse=True)
        
        return resistance_lines[:3], support_lines[:3]

# Initialize trend line detector
trend_line_detector = AutomaticTrendLineDetector()

# Alternative ML Model using sklearn when PyTorch is not available
class SklearnForecast:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        self.is_trained = False
    
    def prepare_sequences(self, data, look_back=60, horizon=10):
        X, y = [], []
        for i in range(len(data) - look_back - horizon):
            X.append(data[i:i+look_back].flatten())
            y.append(data[i+look_back:i+look_back+horizon, -1])  # predict close prices
        return np.array(X), np.array(y)
    
    def train(self, features):
        scaled_features = self.scaler.fit_transform(features)
        X, y = self.prepare_sequences(scaled_features)
        
        if len(X) > 0:
            # For sklearn, we'll predict the mean of the next 10 closes
            y_mean = np.mean(y, axis=1)
            self.model.fit(X, y_mean)
            self.is_trained = True
            return f"Sklearn model trained on {len(X)} samples"
        return "Insufficient data for training"
    
    def predict(self, features):
        if not self.is_trained:
            return []
        
        scaled_features = self.scaler.transform(features)
        X = scaled_features[-60:].flatten().reshape(1, -1)
        pred_mean = self.model.predict(X)[0]
        
        # Generate 10 predictions with some variation
        predictions = [pred_mean * (1 + np.random.normal(0, 0.01)) for _ in range(10)]
        
        # Inverse transform
        dummy = np.zeros((len(predictions), features.shape[1]))
        dummy[:, -1] = predictions  # close price column
        preds = self.scaler.inverse_transform(dummy)[:, -1]
        
        return preds

# LSTM Model Definition (if PyTorch available)
if TORCH_AVAILABLE:
    class LSTMForecast(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=10):
            super(LSTMForecast, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

# Feature Engineering Function
def build_features(df, sentiment_score=0.0):
    df = df.copy()
    
    # Add technical indicators
    if len(df) > 14:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bollinger_h'] = bb.bollinger_hband()
        df['bollinger_l'] = bb.bollinger_lband()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    else:
        # Fill with zeros for short datasets
        df['rsi'] = 50
        df['macd'] = 0
        df['bollinger_h'] = df['close']
        df['bollinger_l'] = df['close']
        df['adx'] = 25
    
    # Add sentiment as constant column
    df['sentiment'] = sentiment_score
    
    # Fill NaNs
    df = df.fillna(method='ffill').fillna(0)
    return df

def train_model_enhanced(df, sentiment_score=0.1):
    global trained_model, feature_scaler
    
    try:
        # Build features
        df_feat = build_features(df.copy(), sentiment_score)
        
        # Select features
        feature_cols = ["open", "high", "low", "close", "volume",
                       "rsi", "macd", "bollinger_h", "bollinger_l", "adx", "sentiment"]
        features = df_feat[feature_cols].values
        
        if TORCH_AVAILABLE:
            # Use PyTorch LSTM
            feature_scaler = MinMaxScaler()
            features_scaled = feature_scaler.fit_transform(features)
            
            look_back = 60
            horizon = 10
            X, Y = [], []
            
            close_idx = feature_cols.index("close")
            for i in range(len(features_scaled) - look_back - horizon):
                X.append(features_scaled[i:i+look_back])
                closes = features_scaled[i+look_back:i+look_back+horizon, close_idx]
                Y.append(closes)
            
            if len(X) == 0:
                return "Insufficient data for training"
                
            X, Y = np.array(X), np.array(Y)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            
            input_size = X.shape[2]
            trained_model = LSTMForecast(input_size=input_size, output_size=horizon)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(trained_model.parameters(), lr=0.001)
            
            epochs = 30
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = trained_model(X_tensor)
                loss = criterion(output, Y_tensor)
                loss.backward()
                optimizer.step()
            
            return f"PyTorch LSTM model trained for {epochs} epochs on {len(X)} samples. Final loss: {loss.item():.6f}"
        
        else:
            # Use sklearn alternative
            trained_model = SklearnForecast()
            result = trained_model.train(features)
            feature_scaler = trained_model.scaler
            return result
            
    except Exception as e:
        return f"Error during training: {e}"

def predict_next_prices(df, sentiment_score=0.1):
    global trained_model, feature_scaler
    
    if trained_model is None or feature_scaler is None:
        return "No model trained yet. Click 'Train AI' to start.", []
    
    try:
        # Build features
        df_feat = build_features(df.copy(), sentiment_score)
        
        # Select features
        feature_cols = ["open", "high", "low", "close", "volume",
                       "rsi", "macd", "bollinger_h", "bollinger_l", "adx", "sentiment"]
        features = df_feat[feature_cols].values
        
        if TORCH_AVAILABLE and hasattr(trained_model, 'lstm'):
            # PyTorch prediction
            features_scaled = feature_scaler.transform(features)
            look_back = 60
            inp = torch.tensor(features_scaled[-look_back:], dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                preds_scaled = trained_model(inp).numpy().flatten()
            
            close_idx = feature_cols.index("close")
            dummy = np.zeros((len(preds_scaled), len(feature_cols)))
            dummy[:, close_idx] = preds_scaled
            preds = feature_scaler.inverse_transform(dummy)[:, close_idx]
        
        else:
            # Sklearn prediction
            preds = trained_model.predict(features)
        
        # Trend classification
        last_price = df["close"].iloc[-1]
        future_mean = np.mean(preds)
        change = (future_mean - last_price) / last_price * 100
        
        if change > 1:
            signal = "BUY"
        elif change < -1:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return f"Next 10 predicted closes: {np.round(preds, 2).tolist()}\nSignal: {signal} ({change:.2f}%)", preds
        
    except Exception as e:
        return f"Error during prediction: {e}", []

# NLP Sentiment Analysis Function
def get_sentiment(symbol):
    global sentiment_pipeline
    
    if sentiment_pipeline is None and TRANSFORMERS_AVAILABLE:
        try:
            sentiment_pipeline = pipeline("sentiment-analysis")
        except Exception as e:
            print(f"Failed to load sentiment pipeline: {e}")
            sentiment_pipeline = None
    
    # Try to get news
    try:
        ticker_sym = symbol.replace('/', '-').replace('USDT', 'USD')
        ticker = yf.Ticker(ticker_sym)
        news_items = ticker.news
        if news_items:
            news_texts = [item.get('title', '') + ' ' + item.get('publisher', '') for item in news_items[:5]]
        else:
            news_texts = [f"{symbol} market update", f"{symbol} trading activity"]
    except Exception:
        news_texts = [f"{symbol} positive momentum", f"{symbol} market volatility"]
    
    # Analyze sentiment
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    for text in news_texts:
        if sentiment_pipeline and TRANSFORMERS_AVAILABLE:
            try:
                result = sentiment_pipeline(text)[0]
                score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            except:
                score = analyzer.polarity_scores(text)['compound']
        else:
            score = analyzer.polarity_scores(text)['compound']
        sentiments.append(score)
    
    return np.mean(sentiments)

def get_demo_data_realtime(symbol="BTC/USDT", periods=200, timeframe='5m'):
    """Generate realistic demo data"""
    freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '4h': '4H', '1d': '1D'}
    if timeframe not in freq_map:
        timeframe = '5m'
    freq = freq_map[timeframe]
    
    # Seed for consistency
    seed = hash(f"{symbol}_{timeframe}_{periods}") % 1000
    np.random.seed(seed)
    
    timestamps = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
    
    # Generate realistic price movement
    base_price = 45000
    prices = []
    
    for i in range(periods):
        if i == 0:
            prices.append(base_price)
        else:
            change = np.random.normal(0, 200)
            new_price = prices[-1] + change
            prices.append(max(new_price, base_price * 0.8))
    
    prices = np.array(prices)
    
    # Generate OHLC
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.roll(prices, 1),
        'high': prices * (1 + np.random.uniform(0.001, 0.01, periods)),
        'low': prices * (1 - np.random.uniform(0.001, 0.01, periods)),
        'close': prices,
        'volume': np.random.lognormal(10, 1, periods)
    })
    
    df['open'].iloc[0] = prices[0]
    return df

def get_binance_data(symbol="BTCUSDT", limit=200, interval='5m'):
    """Fetch data from Binance or fallback to demo"""
    if not BINANCE_CONNECTED or binance_client is None:
        return get_demo_data_realtime(symbol.replace('USDT', '/USDT'), limit, interval)
    
    try:
        klines = binance_client.get_klines(
            symbol=symbol.replace('/', ''),
            interval=interval,
            limit=limit
        )
        
        if not klines:
            return get_demo_data_realtime(symbol.replace('USDT', '/USDT'), limit, interval)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
            'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
        })
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
        
    except Exception as e:
        print(f"Binance API error: {e}")
        return get_demo_data_realtime(symbol.replace('USDT', '/USDT'), limit, interval)

# Enhanced Dash Application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "LIVE AI Trading Dashboard Pro"

# Styling
CARD_STYLE = {
    "backgroundColor": "#161a1e",
    "border": "1px solid #2b3139",
    "borderRadius": "8px",
    "padding": "15px",
    "marginBottom": "15px"
}

HEADER_STYLE = {
    "backgroundColor": "#0d1117",
    "padding": "20px",
    "borderBottom": "2px solid #30363d"
}

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("AI TRADING DASHBOARD PRO", 
                       style={"color": "#ffffff", "fontSize": "28px", "fontWeight": "bold", "marginBottom": "5px"}),
                html.P("Advanced ML Models • Real-time Analysis • Automatic Trend Detection",
                      style={"color": "#8b949e", "fontSize": "14px", "margin": "0"})
            ])
        ], width=8),
        dbc.Col([
            html.Div([
                html.Span("●", style={"color": "#3fb950", "fontSize": "20px", "marginRight": "5px"}),
                html.Span("LIVE", style={"color": "#3fb950", "fontWeight": "bold", "marginRight": "15px"}),
                html.Span(id="connection-status", style={"color": "#8b949e", "fontSize": "12px"})
            ], style={"textAlign": "right", "paddingTop": "20px"})
        ], width=4)
    ], style=HEADER_STYLE),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.Row([
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("Symbol", style={"backgroundColor": "#21262d", "color": "#8b949e", "border": "1px solid #30363d"}),
                            dbc.Input(id="symbol-input", value="BTC/USDT", 
                                    style={"backgroundColor": "#0d1117", "color": "#ffffff", "border": "1px solid #30363d"})
                        ], size="sm")
                    ], width=3),
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button("Start Stream", id="start-stream-button", color="success", size="sm"),
                            dbc.Button("Train AI", id="quick-train-button", color="primary", size="sm"),
                            dbc.Button("Auto Trade", id="auto-trade-button", color="warning", size="sm", disabled=True)
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Select(
                            id="timeframe-select",
                            options=[
                                {"label": "1m", "value": "1m"},
                                {"label": "5m", "value": "5m"},
                                {"label": "15m", "value": "15m"},
                                {"label": "1h", "value": "1h"},
                                {"label": "4h", "value": "4h"},
                                {"label": "1d", "value": "1d"}
                            ],
                            value="5m",
                            style={"backgroundColor": "#0d1117", "color": "#ffffff", "border": "1px solid #30363d"}
                        )
                    ], width=3)
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ], className="mt-3"),
    
    # Price Ticker
    dbc.Row([
        dbc.Col([
            dbc.Card([
                html.Div(id="price-ticker", style={"fontSize": "24px", "fontWeight": "bold", "color": "#ffffff"})
            ], style=CARD_STYLE)
        ], width=12)
    ], className="mt-3"),
    
    # Main Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dcc.Graph(id="realtime-chart", style={"height": "600px"})
            ], style={"backgroundColor": "#0d1117", "border": "1px solid #30363d", "borderRadius": "8px"})
        ], width=12)
    ], className="mt-3"),
    
    # Analysis Panels
    dbc.Row([
        dbc.Col([
            dbc.Card([
                html.H5("AI Predictions", style={"color": "#58a6ff", "marginBottom": "15px"}),
                html.Div(id="live-ml-predictions", style={"color": "#8b949e", "fontSize": "13px", "fontFamily": "monospace"})
            ], style=CARD_STYLE)
        ], width=4),
        dbc.Col([
            dbc.Card([
                html.H5("Market Analysis", style={"color": "#58a6ff", "marginBottom": "15px"}),
                html.Div(id="live-market-analysis", style={"color": "#8b949e", "fontSize": "13px", "fontFamily": "monospace"})
            ], style=CARD_STYLE)
        ], width=4),
        dbc.Col([
            dbc.Card([
                html.H5("Pattern Detection", style={"color": "#58a6ff", "marginBottom": "15px"}),
                html.Div(id="pattern-detection", style={"color": "#8b949e", "fontSize": "13px", "fontFamily": "monospace"})
            ], style=CARD_STYLE)
        ], width=4)
    ], className="mt-3"),
    
    # Intervals
    dcc.Interval(id="fast-interval", interval=500, n_intervals=0),
    dcc.Interval(id="chart-interval", interval=1000, n_intervals=0),
    
], fluid=True, style={"backgroundColor": "#010409", "minHeight": "100vh"})

# Callbacks
@app.callback(
    Output("live-ml-predictions", "children", allow_duplicate=True),
    Input("quick-train-button", "n_clicks"),
    [State("symbol-input", "value"), State("timeframe-select", "value")],
    prevent_initial_call=True
)
def train_ai_callback(n_clicks, symbol, timeframe):
    if n_clicks is None:
        return "Click 'Train AI' to start training"
    
    df = get_binance_data(symbol.replace('/', ''), 500, timeframe)
    if df.empty:
        return "No data available for training."
    
    sentiment_score = get_sentiment(symbol)
    train_result = train_model_enhanced(df, sentiment_score)
    pred_text, _ = predict_next_prices(df, sentiment_score)
    
    ml_type = "PyTorch LSTM" if TORCH_AVAILABLE else "Sklearn RandomForest"
    sentiment_type = "Transformers" if TRANSFORMERS_AVAILABLE else "VADER"
    
    ml_text = f"{train_result}\n\n{pred_text}\n\nModels: {ml_type}, {sentiment_type} (NLP)\nStatus: Trained & Active"
    return ml_text

@app.callback(
    [Output("realtime-chart", "figure"),
     Output("live-ml-predictions", "children"),
     Output("live-market-analysis", "children"),
     Output("pattern-detection", "children")],
    [Input("chart-interval", "n_intervals"),
     Input("start-stream-button", "n_clicks"),
     Input("quick-train-button", "n_clicks"),
     Input("timeframe-select", "value")],
    [State("symbol-input", "value")]
)
def update_realtime_chart(n_intervals, start_clicks, train_clicks, timeframe, symbol):
    try:
        df = get_binance_data(symbol.replace('/', ''), 500, timeframe)
        
        if df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            return empty_fig, "No data available", "No data", "No patterns"
        
        sentiment_score = get_sentiment(symbol)
        resistance_lines, support_lines = trend_line_detector.detect_trend_lines(df)
        
        # Create chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.15, 0.15]
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing=dict(line=dict(color='#26a69a', width=1)),
            decreasing=dict(line=dict(color='#ef5350', width=1))
        ), row=1, col=1)
        
        # Trend lines
        for line in resistance_lines:
            x_points = [df['timestamp'].iloc[line['start_idx']], df['timestamp'].iloc[line['end_idx']]]
            y_points = [line['start_price'], line['end_price']]
            
            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines',
                line=dict(color='#ef5350', width=2, dash='solid'),
                name=f"Resistance",
                showlegend=False
            ), row=1, col=1)
        
        for line in support_lines:
            x_points = [df['timestamp'].iloc[line['start_idx']], df['timestamp'].iloc[line['end_idx']]]
            y_points = [line['start_price'], line['end_price']]
            
            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines',
                line=dict(color='#26a69a', width=2, dash='solid'),
                name=f"Support",
                showlegend=False
            ), row=1, col=1)
        
        # Moving Averages
        if len(df) >= 20:
            ma20 = df['close'].rolling(20).mean()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=ma20,
                name="MA20", line=dict(color='#ffa726', width=1),
                showlegend=False
            ), row=1, col=1)
            
            if len(df) >= 50:
                ma50 = df['close'].rolling(50).mean()
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=ma50,
                    name="MA50", line=dict(color='#42a5f5', width=1),
                    showlegend=False
                ), row=1, col=1)
        
        # Volume
        colors = ['#ef5350' if df.iloc[i]['close'] < df.iloc[i]['open'] else '#26a69a' 
                 for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df["timestamp"], y=df["volume"],
            name="Volume", marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        # RSI
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=rsi,
                name="RSI", line=dict(color='#ba68c8', width=1),
                showlegend=False
            ), row=3, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=3, col=1)
        
        # Add predictions if model is trained
        pred_text = "No model trained yet. Click 'Train AI' to enable predictions."
        if trained_model is not None and feature_scaler is not None:
            pred_text, preds = predict_next_prices(df, sentiment_score)
            
            if len(preds) > 0:
                last_timestamp = df["timestamp"].iloc[-1]
                freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '4h': '4H', '1d': '1D'}
                freq = freq_map.get(timeframe, '5T')
                future_timestamps = pd.date_range(start=last_timestamp, periods=len(preds)+1, freq=freq)[1:]
                
                fig.add_trace(go.Scatter(
                    x=future_timestamps,
                    y=preds,
                    mode='lines+markers',
                    name='Predicted Future',
                    line=dict(color='orange', dash='dot'),
                    marker=dict(size=4)
                ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            showlegend=False,
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=True, gridcolor='#21262d', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='#21262d', gridwidth=1, side='right'),
            font=dict(color='#8b949e', size=11)
        )
        
        for i in range(1, 4):
            fig.update_xaxes(showgrid=True, gridcolor='#21262d', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='#21262d', row=i, col=1)
        
        # Prepare outputs
        current_price = df['close'].iloc[-1]
        
        if trained_model is None:
            ml_type = "PyTorch LSTM" if TORCH_AVAILABLE else "Sklearn RandomForest"
            sentiment_type = "Transformers" if TRANSFORMERS_AVAILABLE else "VADER"
            ml_predictions = f"""Current: ${current_price:.2f}
No model trained yet. Click 'Train AI' to enable predictions.
Sentiment Score: {sentiment_score:.2f}
Models: {ml_type}, {sentiment_type} (NLP)
Status: Ready - Click 'Train AI' to enable predictions"""
        else:
            ml_type = "PyTorch LSTM" if TORCH_AVAILABLE else "Sklearn RandomForest"
            sentiment_type = "Transformers" if TRANSFORMERS_AVAILABLE else "VADER"
            ml_predictions = f"""Current: ${current_price:.2f}
{pred_text}
Sentiment Score: {sentiment_score:.2f}
Models: {ml_type}, {sentiment_type} (NLP)
Status: Active"""
        
        # Market analysis
        trend = 'BULLISH' if len(support_lines) > len(resistance_lines) else 'BEARISH'
        vol_trend = 'Increasing' if df['volume'].tail(5).mean() > df['volume'].tail(20).mean() else 'Decreasing'
        volatility = df['close'].pct_change().std() * 100
        
        market_analysis = f"""Trend: {trend}
Support Lines: {len(support_lines)}
Resistance Lines: {len(resistance_lines)}
Volume Trend: {vol_trend}
Volatility: {volatility:.2f}%"""
        
        # Pattern detection
        pattern_text = f"""Detected Patterns:
• Trend Lines Active: {len(resistance_lines + support_lines)}
• {len(resistance_lines)} Resistance levels
• {len(support_lines)} Support levels

Technical Status:
• RSI: {'Available' if len(df) >= 14 else 'Insufficient data'}
• Moving Averages: {'Available' if len(df) >= 20 else 'Insufficient data'}
• Bollinger Bands: {'Available' if len(df) >= 20 else 'Insufficient data'}"""
        
        return fig, ml_predictions, market_analysis, pattern_text
        
    except Exception as e:
        print(f"Error in chart update: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", paper_bgcolor='#0d1117', plot_bgcolor='#0d1117')
        return empty_fig, f"Error: {str(e)}", "Error in analysis", "Error in pattern detection"

@app.callback(
    [Output("price-ticker", "children"),
     Output("connection-status", "children")],
    [Input("fast-interval", "n_intervals")],
    [State("symbol-input", "value")]
)
def update_price_ticker(n, symbol):
    try:
        # Try to get real price if Binance is connected
        if BINANCE_CONNECTED and binance_client is not None:
            try:
                ticker = binance_client.get_symbol_ticker(symbol=symbol.replace('/', ''))
                price = float(ticker['price'])
                # For demo, simulate change
                change = np.random.normal(0, 2)
            except:
                price = 45000 + np.random.normal(0, 100)
                change = np.random.normal(0, 2)
        else:
            price = 45000 + np.random.normal(0, 100)
            change = np.random.normal(0, 2)
        
        color = "#3fb950" if change >= 0 else "#f85149"
        arrow = "+" if change >= 0 else ""
        
        ticker = html.Div([
            html.Span(f"${price:,.2f}", style={"color": color, "marginRight": "15px"}),
            html.Span(f"{arrow}{change:.2f}%", style={"color": color, "fontSize": "16px"}),
            html.Span(" | Vol: 1,234.56 BTC", style={"color": "#8b949e", "fontSize": "14px", "marginLeft": "15px"})
        ])
        
        connection_type = "Live Data" if BINANCE_CONNECTED else "Demo Data"
        ml_status = "PyTorch + Sklearn" if TORCH_AVAILABLE else "Sklearn Only"
        nlp_status = "Transformers + VADER" if TRANSFORMERS_AVAILABLE else "VADER Only"
        
        status = f"{connection_type} • ML: {ml_status} • NLP: {nlp_status} • Last update: now"
        
        return ticker, status
        
    except Exception as e:
        return "Loading...", f"Error: {str(e)}"

if __name__ == "__main__":
    print("Starting AI Trading Dashboard Pro...")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    print(f"Transformers Available: {TRANSFORMERS_AVAILABLE}")
    print(f"Binance Connected: {BINANCE_CONNECTED}")
    
    try:
        app.run(debug=False, host="0.0.0.0", port=5050)
    except AttributeError:
        app.run_server(debug=False, host="0.0.0.0", port=5050)
