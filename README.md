![AI Trading Dashboard](https://raw.githubusercontent.com/abushanisro/ai_trading/c95b0e66d684615e09f27b044ac78f697ed74e11/Advance.svg?token=ATLI6NXV7YOZH2SQEFPUTPTIYRVDA)


# Professional Trading Dashboard

This is a professional trading dashboard that fetches real market data from Binance and displays it with advanced technical analysis features and enhanced AI predictions.

## Features

1. **Real Binance Data**: Fetches actual historical OHLCV data from Binance API
2. **Automatic Trend Line Detection**: Identifies support and resistance levels
3. **Chart Pattern Recognition**: Detects common patterns like triangles and channels
4. **Technical Indicators**: Displays RSI, Moving Averages, and Volume
5. **Enhanced AI Predictions**: 
   - Multi-feature LSTM model for price prediction (OHLCV + indicators + sentiment)
   - Multi-step forecasting (next 10 candles)
   - Trend classification (Buy/Sell/Hold) with confidence scoring
   - NLP sentiment analysis using Hugging Face transformers
   - VADER/TextBlob fallback for sentiment analysis
6. **Professional Dark Theme UI**: Modern, dark-themed interface with responsive design
7. **Fallback to Demo Data**: If Binance API is unavailable, falls back to synthetic data

## Implementation Details

### Data Consistency
- Uses real Binance klines (candlestick) data for consistency
- Data remains stable across updates (no constant random changes)
- Fallback to consistent demo data when Binance is unavailable

### Technical Analysis
- Automatic trend line detection with touch point validation
- Chart pattern recognition (triangles, channels)
- RSI calculation with division-by-zero protection
- Moving averages (MA20, MA50)
- MACD, Bollinger Bands, ADX indicators

### Enhanced AI Predictions
- **Multi-feature LSTM Model**: Deep learning model using OHLCV + technical indicators + sentiment
- **Multi-step Forecasting**: Predicts next 10 candle closes instead of just 1
- **Trend Classification**: Classifies signals as Buy/Sell/Hold with confidence percentage
- **Sentiment Analysis**: NLP analysis of recent news using Hugging Face transformers pipeline
- **Real-time Visualization**: Plots future predictions directly on the chart
- **On-demand Training**: Training via the "Train AI" button using current data

### Security
- API keys handled through environment variables
- Warning when using default/fallback keys
- Secure error handling for API failures

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Binance API keys as environment variables (optional but recommended):
   ```bash
   export BINANCE_API_KEY="your_api_key_here"
   export BINANCE_SECRET="your_secret_here"
   ```

3. Run the dashboard:
   ```bash
   python trade.py
   ```

4. Access the dashboard at `http://localhost:8053`

## Usage

1. Select a trading pair (e.g., "BTC/USDT")
2. Choose a timeframe (1m, 5m, 15m, 1h, 4h, 1d)
3. View real-time chart with technical analysis
4. Click "Train AI" to train the enhanced LSTM model and enable predictions
5. Monitor AI predictions and market analysis panels
6. View future price predictions plotted directly on the chart

## AI Predictions Explained

### Multi-feature LSTM Model
- Uses OHLCV data + technical indicators (RSI, MACD, Bollinger Bands, ADX) + sentiment score
- Predicts next 10 candle closes for short-term trend analysis
- Quick training (30 epochs) for demonstration purposes
- Loss-based confidence scoring

### Trend Classification
- Classifies trend as Buy (>1% expected gain), Sell (<-1% expected loss), or Hold (sideways)
- Provides percentage change prediction for confidence assessment
- Combines technical analysis with sentiment for better accuracy

### NLP Sentiment Analysis
- Fetches recent news for the selected cryptocurrency using yfinance
- Analyzes sentiment using Hugging Face transformers pipeline
- Fallback to VADER/TextBlob if transformers fail
- Incorporates sentiment as a feature in the LSTM model

## Example Output

When you click "Train AI", you'll see output like:
```
Model trained for 30 epochs on 430 samples. Final loss: 0.000452

Next 10 predicted closes: [27050.25, 27120.33, 27210.45, ...]
Signal: BUY (+2.35%)
Sentiment Score: 0.15

Models: LSTM (Multi-feature, Multi-step), Transformers/VADER (NLP)
Status: Active
```

On the chart, you'll see a dotted orange line extending into the future showing the predicted prices.

## Future Enhancements

1. Real-time WebSocket streaming for live updates
2. Additional technical indicators
3. Trading signal generation
4. Backtesting capabilities
5. Export functionality for charts and analysis
6. Enhanced LSTM model with attention mechanisms
7. Improved sentiment analysis with more sources
8. Risk management and position sizing recommendations# ai_trading
