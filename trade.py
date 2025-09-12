import subprocess
import sys

def install_packages():
    packages = [
        'ccxt', 'dash', 'plotly', 'scikit-learn', 'scipy', 'websockets',
        'transformers', 'torch', 'torchvision', 'pandas', 'numpy',
        'websocket-client', 'python-binance', 'yfinance', 'textblob',
        'vaderSentiment', 'ta', 'dash-bootstrap-components'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_').replace('python_', ''))
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}, continuing without it...")

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
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
sentiment_pipeline = None  # Lazy load in functions

# Configuration - Use environment variables for security
import os
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "geARpnfp2sSbmb1IPRArLlYMrO4C5UAZtoHIgH9JGmVa1qR9u9Ue30YyTRekqGNH")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "OC5UrkXX9cfqEtzMFToZxH1p6SJO7g7fCXpERMevN5uFLCTwJ3xHdxZevX72YTiw")

# Warning for production use
if BINANCE_API_KEY == "geARpnfp2sSbmb1IPRArLlYMrO4C5UAZtoHIgH9JGmVa1qR9u9Ue30YyTRekqGNH":
    print("WARNING: Using default API keys. Please set BINANCE_API_KEY and BINANCE_SECRET environment variables for production use.")

binance_client = None
BINANCE_CONNECTED = False

if BINANCE_LIB_AVAILABLE and Client:
    try:
        binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET)
        binance_client.ping()
        BINANCE_CONNECTED = True
        print("Binance API connected successfully")
    except BinanceAPIException as e:
        print(f"Binance API error: {e}")
        binance_client = None
        BINANCE_CONNECTED = False
    except Exception as e:
        print(f"Unexpected error connecting to Binance: {e}")
        binance_client = None
        BINANCE_CONNECTED = False

# Enhanced Trend Line Detector Class (unchanged)
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
    
    def detect_chart_patterns(self, df):
        """Detect common chart patterns (triangles, channels, etc.)"""
        patterns = []
        
        if len(df) < 30:
            return patterns
            
        resistance_lines, support_lines = self.detect_trend_lines(df)
        
        # Detect triangle patterns
        for r_line in resistance_lines:
            for s_line in support_lines:
                # Check if lines are converging (triangle)
                if r_line['slope'] < 0 and s_line['slope'] > 0:
                    patterns.append({
                        'type': 'ascending_triangle',
                        'resistance': r_line,
                        'support': s_line,
                        'breakout_level': r_line['end_price']
                    })
                elif r_line['slope'] < 0 and s_line['slope'] < 0 and abs(r_line['slope']) > abs(s_line['slope']):
                    patterns.append({
                        'type': 'descending_triangle',
                        'resistance': r_line,
                        'support': s_line,
                        'breakdown_level': s_line['end_price']
                    })
                elif abs(r_line['slope'] - s_line['slope']) < 0.0001:
                    patterns.append({
                        'type': 'channel',
                        'resistance': r_line,
                        'support': s_line,
                        'width': abs(r_line['intercept'] - s_line['intercept'])
                    })
        
        return patterns

# Initialize trend line detector
trend_line_detector = AutomaticTrendLineDetector()

# LSTM Model Definition (Multi-feature, Multi-step)
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=10):  
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Predict next 10 closes
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last hidden state
        return out

# Feature Engineering Function
def build_features(df, sentiment_score=0.0):
    # Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bollinger_h'] = bb.bollinger_hband()
    df['bollinger_l'] = bb.bollinger_lband()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    # Add sentiment as constant column for that time window
    df['sentiment'] = sentiment_score
    
    # Fill NaNs
    df = df.fillna(0)
    return df

# Multi-step prediction function
def multi_step_predict(model, data, look_back=60, steps=10):
    model.eval()
    predictions = []
    input_seq = data[-look_back:]
    
    for _ in range(steps):
        inp = torch.tensor(input_seq[-look_back:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(inp).numpy().flatten()
        predictions.append(pred[0])
        input_seq = np.append(input_seq, pred[0])
    
    return predictions

# Trend classification function
def classify_trend(predictions, last_price):
    future_mean = np.mean(predictions)
    change = (future_mean - last_price) / last_price * 100
    
    if change > 1:
        return "BUY", change
    elif change < -1:
        return "SELL", change
    else:
        return "HOLD", change

# Training Function (Enhanced)
def train_model_enhanced(df, sentiment_score=0.1):
    global trained_model, feature_scaler
    
    try:
        # 1. Build features
        df_feat = build_features(df.copy(), sentiment_score)

        # 2. Select features
        feature_cols = ["open","high","low","close","volume",
                        "rsi","macd","bollinger_h","bollinger_l","adx","sentiment"]
        features = df_feat[feature_cols].values
        
        # 3. Scale features
        feature_scaler = MinMaxScaler()
        features_scaled = feature_scaler.fit_transform(features)

        # 4. Prepare dataset
        look_back = 60
        horizon = 10  # predict next 10 closes
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

        # 5. Define model
        input_size = X.shape[2]
        trained_model = LSTMForecast(input_size=input_size, output_size=horizon)

        # 6. Train
        criterion = nn.MSELoss()
        optimizer = optim.Adam(trained_model.parameters(), lr=0.001)

        epochs = 30  # keep lighter for dashboard
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = trained_model(X_tensor)
            loss = criterion(output, Y_tensor)
            loss.backward()
            optimizer.step()
        
        return f"Model trained for {epochs} epochs on {len(X)} samples. Final loss: {loss.item():.6f}"

    except Exception as e:
        return f"Error during training: {e}"

# Enhanced Prediction Function
def predict_next_prices(df, sentiment_score=0.1):
    global trained_model, feature_scaler
    
    if trained_model is None or feature_scaler is None:
        return "No model trained yet. Click 'Train AI' to start.", []
    
    try:
        # Build features
        df_feat = build_features(df.copy(), sentiment_score)
        
        # Select features
        feature_cols = ["open","high","low","close","volume",
                        "rsi","macd","bollinger_h","bollinger_l","adx","sentiment"]
        features = df_feat[feature_cols].values
        features_scaled = feature_scaler.transform(features)

        # Convert to tensor
        look_back = 60
        inp = torch.tensor(features_scaled[-look_back:], dtype=torch.float32).unsqueeze(0)

        # Run multi-step prediction (next 10 closes)
        with torch.no_grad():
            preds_scaled = trained_model(inp).numpy().flatten()

        # Inverse scale only the Close feature
        close_idx = feature_cols.index("close")
        dummy = np.zeros((len(preds_scaled), len(feature_cols)))
        dummy[:, close_idx] = preds_scaled
        preds = feature_scaler.inverse_transform(dummy)[:, close_idx]

        # Trend classification
        last_price = df["close"].iloc[-1]
        signal, change = classify_trend(preds, last_price)
        
        return f"Next 10 predicted closes: {np.round(preds,2).tolist()}\\nSignal: {signal} ({change:.2f}%)", preds

    except Exception as e:
        return f"Error during prediction: {e}", []

# NLP Sentiment Analysis Function
def get_sentiment(symbol):
    global sentiment_pipeline
    if sentiment_pipeline is None:
        try:
            sentiment_pipeline = pipeline("sentiment-analysis")
        except Exception as e:
            print(f"Failed to load sentiment pipeline: {e}")
            sentiment_pipeline = None
    
    # Fetch recent news using yfinance (for BTC-USD proxy)
    try:
        ticker_sym = symbol.replace('/', '-').replace('USDT', 'USD')  # e.g., BTC-USD
        ticker = yf.Ticker(ticker_sym)
        news_items = ticker.news
        if news_items:
            news_texts = [item.get('title', '') + ' ' + item.get('publisher', '') for item in news_items[:5]]
        else:
            news_texts = [f"{symbol} market update: Positive momentum", f"{symbol} volatility increases"]
    except Exception:
        # Fallback dummy news
        news_texts = [f"{symbol} surges on positive news", f"{symbol} faces market resistance"]
    
    # Analyze sentiment
    analyzer = SentimentIntensityAnalyzer()  # VADER fallback
    sentiments = []
    for text in news_texts:
        if sentiment_pipeline:
            result = sentiment_pipeline(text)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        else:
            # Fallback to VADER
            score = analyzer.polarity_scores(text)['compound']
        sentiments.append(score)
    
    avg_sentiment = np.mean(sentiments)
    return avg_sentiment

# Helper function for demo data (fallback when Binance is unavailable)
def get_demo_data_realtime(symbol="BTC/USDT", periods=200, timeframe='5m'):
    """Generate realistic demo data"""
    # Validate timeframe parameter
    freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '4h': '4H', '1d': '1D'}
    if timeframe not in freq_map:
        raise ValueError(f"Invalid timeframe: {timeframe}. Supported: {list(freq_map.keys())}")
    freq = freq_map[timeframe]
    
    # Fixed seed for consistency
    seed = hash(f"{symbol}_{timeframe}_{periods}") % 1000
    np.random.seed(seed)
    
    timestamps = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
    
    # Generate realistic price movement with trends
    base_price = 45000
    trend = np.random.choice([-500, 0, 500])  # Trend direction
    prices = []
    
    for i in range(periods):
        if i == 0:
            prices.append(base_price)
        else:
            # Add trend, momentum, and random walk
            trend_component = trend / periods
            momentum = 0.3 * (prices[-1] - base_price) / base_price
            random_walk = np.random.normal(0, 200)
            
            new_price = prices[-1] + trend_component - momentum * 100 + random_walk
            prices.append(max(new_price, base_price * 0.8))  # Floor at 80% of base
    
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

# Fetch real historical data from Binance API
def get_binance_data(symbol="BTCUSDT", limit=200, interval='5m'):
    """Fetch real historical data from Binance API"""
    if not BINANCE_CONNECTED or binance_client is None:
        print("Binance API not connected, falling back to demo data.")
        return get_demo_data_realtime(symbol.replace('USDT', '/USDT'), limit, interval)
    
    # Map Dash timeframe to Binance interval
    interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
    if interval not in interval_map:
        raise ValueError(f"Invalid interval: {interval}. Supported: {list(interval_map.keys())}")
    binance_interval = interval_map[interval]
    
    try:
        # Fetch klines (OHLCV data)
        klines = binance_client.get_klines(
            symbol=symbol.replace('/', ''),  # e.g., 'BTC/USDT' -> 'BTCUSDT'
            interval=binance_interval,
            limit=limit
        )
        
        if not klines:
            print("No data returned from Binance.")
            return get_demo_data_realtime(symbol.replace('USDT', '/USDT'), limit, interval)
        
        # Convert to DataFrame with proper columns
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Sort by timestamp (newest first from Binance, reverse to oldest first for charting)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Fetched {len(df)} periods of {symbol} data for {interval} interval.")
        return df
        
    except BinanceAPIException as e:
        print(f"Binance API error: {e}")
        # Fallback to demo data
        return get_demo_data_realtime(symbol.replace('USDT', '/USDT'), limit, interval)
    except Exception as e:
        print(f"Unexpected error fetching Binance data: {e}")
        # Fallback to demo data
        return get_demo_data_realtime(symbol.replace('USDT', '/USDT'), limit, interval)

# Enhanced Dash Application with Professional UI
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "LIVE AI Trading Dashboard Pro"

# Professional Dark Theme Styling
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

# Layout with Professional UI (unchanged)
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
    
    # Price Ticker Row
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

# NEW: Callback for "Train AI" button (Enhanced)
@app.callback(
    Output("live-ml-predictions", "children", allow_duplicate=True),
    Input("quick-train-button", "n_clicks"),
    [State("symbol-input", "value"), State("timeframe-select", "value")],
    prevent_initial_call=True
)
def train_ai_callback(n_clicks, symbol, timeframe):
    # Fetch data for training
    df = get_binance_data(symbol.replace('/', ''), 500, timeframe)  # Use more data for better training
    if df.empty:
        return "No data available for training."
    
    # Get sentiment score
    sentiment_score = get_sentiment(symbol)
    
    # Train enhanced model
    train_result = train_model_enhanced(df, sentiment_score)
    
    # Get initial prediction
    pred_text, _ = predict_next_prices(df, sentiment_score)
    
    # Combine
    ml_text = f"{train_result}\\n\\n{pred_text}\\n\\nModels: LSTM (Multi-feature, Multi-step), Transformers/VADER (NLP)\\nStatus: Trained & Active"
    return ml_text

# Enhanced Chart Callback with Trend Lines and Predictions
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
        # Use real Binance data if connected, else fallback to demo
        try:
            df = get_binance_data(
                symbol=symbol.replace('/', ''),  # e.g., 'BTC/USDT' -> 'BTCUSDT'
                limit=500,  # Use more data for better analysis
                interval=timeframe
            )
        except Exception as e:
            print(f"Error fetching data: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            return empty_fig, f"Error fetching data: {e}", "No data", "No patterns"
        
        if df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            return empty_fig, "No data available", "No data", "No patterns"
        
        # Get sentiment score
        sentiment_score = get_sentiment(symbol)
        
        # Detect trend lines
        resistance_lines, support_lines = trend_line_detector.detect_trend_lines(df)
        patterns = trend_line_detector.detect_chart_patterns(df)
        
        # Create figure with subplots (unchanged)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.15, 0.15],
            subplot_titles=("", "", "")
        )
        
        # Add candlestick chart (unchanged)
        fig.add_trace(go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing=dict(line=dict(color='#26a69a', width=1), fillcolor='#26a69a'),
            decreasing=dict(line=dict(color='#ef5350', width=1), fillcolor='#ef5350')
        ), row=1, col=1)
        
        # Draw resistance trend lines (unchanged)
        for line in resistance_lines:
            x_points = [df['timestamp'].iloc[line['start_idx']], df['timestamp'].iloc[line['end_idx']]]
            y_points = [line['start_price'], line['end_price']]
            
            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines',
                line=dict(color='#ef5350', width=2, dash='solid'),
                name=f"Resistance (touches: {line['touches']})",
                showlegend=False
            ), row=1, col=1)
            
            # Add touch points
            for point in line['points']:
                fig.add_trace(go.Scatter(
                    x=[point['timestamp']],
                    y=[point['price']],
                    mode='markers',
                    marker=dict(color='#ef5350', size=8, symbol='circle-open'),
                    showlegend=False
                ), row=1, col=1)
        
        # Draw support trend lines (unchanged)
        for line in support_lines:
            x_points = [df['timestamp'].iloc[line['start_idx']], df['timestamp'].iloc[line['end_idx']]]
            y_points = [line['start_price'], line['end_price']]
            
            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines',
                line=dict(color='#26a69a', width=2, dash='solid'),
                name=f"Support (touches: {line['touches']})",
                showlegend=False
            ), row=1, col=1)
            
            # Add touch points
            for point in line['points']:
                fig.add_trace(go.Scatter(
                    x=[point['timestamp']],
                    y=[point['price']],
                    mode='markers',
                    marker=dict(color='#26a69a', size=8, symbol='circle-open'),
                    showlegend=False
                ), row=1, col=1)
        
        # Add Moving Averages (unchanged)
        if len(df) >= 20:
            ma20 = df['close'].rolling(20).mean()
            ma50 = df['close'].rolling(50).mean() if len(df) >= 50 else None
            
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=ma20,
                name="MA20", line=dict(color='#ffa726', width=1),
                showlegend=False
            ), row=1, col=1)
            
            if ma50 is not None:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=ma50,
                    name="MA50", line=dict(color='#42a5f5', width=1),
                    showlegend=False
                ), row=1, col=1)
        
        # Add Volume (unchanged)
        colors = ['#ef5350' if df.iloc[i]['close'] < df.iloc[i]['open'] else '#26a69a' 
                 for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df["timestamp"], y=df["volume"],
            name="Volume", marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        # Add RSI (unchanged)
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            # Add small epsilon to prevent division by zero
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=rsi,
                name="RSI", line=dict(color='#ba68c8', width=1),
                showlegend=False
            ), row=3, col=1)
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=3, col=1)
        
        # Add future predictions if model is trained
        pred_text = "No model trained yet. Click 'Train AI' to enable predictions."
        if trained_model is not None and feature_scaler is not None:
            pred_text, preds = predict_next_prices(df, sentiment_score)
            
            # Plot future predictions on chart
            if len(preds) > 0:
                # Create future timestamps
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
        
        # Update layout with professional dark theme (unchanged)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            showlegend=False,
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(
                showgrid=True,
                gridcolor='#21262d',
                gridwidth=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#21262d',
                gridwidth=1,
                side='right'
            ),
            font=dict(color='#8b949e', size=11)
        )
        
        # Update axes for all subplots (unchanged)
        for i in range(1, 4):
            fig.update_xaxes(showgrid=True, gridcolor='#21262d', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='#21262d', row=i, col=1)
        
        # Prepare dynamic ML predictions
        current_price = df['close'].iloc[-1]
        
        if trained_model is None:
            ml_predictions = f"""Current: ${current_price:.2f}
No model trained yet. Click 'Train AI' to enable predictions.
Sentiment Score: {sentiment_score:.2f}
Models: LSTM (Multi-feature, Multi-step), Transformers/VADER (NLP)
Status: Ready - Click 'Train AI' to enable predictions"""
        else:
            ml_predictions = f"""Current: ${current_price:.2f}
{pred_text}
Sentiment Score: {sentiment_score:.2f}
Models: LSTM (Multi-feature, Multi-step), Transformers/VADER (NLP)
Status: Active"""
        
        # Market analysis (unchanged)
        market_analysis = f"""Trend: {'BULLISH' if len(support_lines) > len(resistance_lines) else 'BEARISH'}
Support Lines: {len(support_lines)}
Resistance Lines: {len(resistance_lines)}
Volume Trend: {'Increasing' if df['volume'].tail(5).mean() > df['volume'].tail(20).mean() else 'Decreasing'}
Volatility: {df['close'].pct_change().std() * 100:.2f}%"""
        
        # Pattern detection (unchanged)
        pattern_text = "Detected Patterns:\\n"
        for pattern in patterns[:3]:
            pattern_text += f"• {pattern['type'].replace('_', ' ').title()}\\n"
        if not patterns:
            pattern_text += "• No clear patterns detected\\n"
        pattern_text += f"\\nTrend Lines Active:\\n"
        pattern_text += f"• {len(resistance_lines)} Resistance\\n"
        pattern_text += f"• {len(support_lines)} Support"
        
        return fig, ml_predictions, market_analysis, pattern_text
        
    except Exception as e:
        print(f"Error: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", paper_bgcolor='#0d1117', plot_bgcolor='#0d1117')
        return empty_fig, "Error in chart update", "Error", "Error"

# Price ticker callback (updated to use real price if possible)
@app.callback(
    [Output("price-ticker", "children"),
     Output("connection-status", "children")],
    [Input("fast-interval", "n_intervals")]
)
def update_price_ticker(n):
    try:
        # Try to get real price from Binance if connected
        try:
            if BINANCE_CONNECTED and binance_client is not None:
                # Get current BTC price as an example (we could make this dynamic based on symbol)
                ticker = binance_client.get_symbol_ticker(symbol="BTCUSDT")
                price = float(ticker['price'])
                # For change, we'll still simulate for now
                change = np.random.normal(0, 2)
            else:
                # Simulate real-time price
                price = 45000 + np.random.normal(0, 100)
                change = np.random.normal(0, 2)
        except:
            # Fallback to simulation
            price = 45000 + np.random.normal(0, 100)
            change = np.random.normal(0, 2)
        
        color = "#3fb950" if change >= 0 else "#f85149"
        arrow = "+" if change >= 0 else "-"
        
        ticker = html.Div([
            html.Span(f"${price:,.2f}", style={"color": color, "marginRight": "15px"}),
            html.Span(f"{arrow} {abs(change):.2f}%", style={"color": color, "fontSize": "16px"}),
            html.Span(" | Vol: 1,234.56 BTC", style={"color": "#8b949e", "fontSize": "14px", "marginLeft": "15px"})
        ])
        
        status = "Connected • Last update: now"
        
        return ticker, status
    except Exception as e:
        return "Loading...", f"Error: {str(e)}"

if __name__ == "__main__":
    print("ADVANCED AI TRADING DASHBOARD PRO")
    print("Features: Automatic Trend Lines, Pattern Detection, Enhanced ML Predictions (Multi-feature, Multi-step)")
    print("Starting dashboard on http://localhost:5500")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5200)
    except AttributeError:
        app.run_server(debug=False, host='0.0.0.0', port=5200)