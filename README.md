**Hybrid Stock Trading Dashboard — Supervised Learning + DQN**
A Streamlit-based experimental trading dashboard that combines a Supervised Learning (SL) model with a Deep Q-Network (DQN) Reinforcement Learning (RL) agent to generate and evaluate Buy/Hold/Sell trading actions.
The system downloads market data with yfinance, engineers technical indicators, trains a RandomForestClassifier to estimate the probability of a future upward move, and feeds that signal into a custom Gymnasium trading environment used to train a DQN agent.
Important: This project is an educational and research prototype. It is not financial advice and should not be used as an automated live-trading system without extensive additional validation, realistic transaction-cost modeling, risk controls, and out-of-sample testing.

Features
- Historical market-data retrieval using yfinance
- Automatic ticker selection from the Streamlit sidebar
- Technical-indicator feature engineering
- Supervised Learning using RandomForestClassifier
- Time-series-aware train/test splitting with TimeSeriesSplit
- Feature standardization with StandardScaler
- Buy/Hold/Sell RL action space
- Custom Gymnasium trading environment
- DQN training with Stable-Baselines3
- Streamlit interactive dashboard
- Interactive price and return visualizations with Plotly
- Supervised model evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Classification report
  - Confusion matrix
- Trading-signal visualization
- RL evaluation:
  - Cumulative Reward
  - Final Portfolio Value
  - Sharpe Ratio
  - Maximum Drawdown
  - Portfolio-value curve
- Model persistence using joblib
- TensorBoard logging for DQN training
System Architecture
                    ┌─────────────────────┐
                    │   User enters       │
                    │   Stock Ticker      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Yahoo Finance     │
                    │     yfinance        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │                     │
                    │ SMA / EMA           │
                    │ RSI                 │
                    │ MACD                │
                    │ Bollinger Bands     │
                    │ Returns             │
                    │ Volatility          │
                    │ Lagged Returns      │
                    │ OBV                 │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Target Construction │
                    │                     │
                    │ 5-day future return │
                    │ Threshold = 1%      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Time-Series Split   │
                    │ + StandardScaler    │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Random Forest Classifier   │
                 │ Supervised Learning (SL)   │
                 └─────────────┬──────────────┘
                               │
                     Up-move probability
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Custom SLTradingEnv        │
                 │ Gymnasium Environment      │
                 └─────────────┬──────────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Deep Q-Network (DQN)       │
                 │ Reinforcement Learning     │
                 └─────────────┬──────────────┘
                               │
                     Buy / Hold / Sell
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Streamlit Dashboard        │
                 │                            │
                 │ Metrics / Charts / Signals │
                 └────────────────────────────┘
Machine Learning Pipeline
1. Market Data
The application retrieves historical market data using:
yf.download(
    ticker,
    period="2y",
    interval="1d"
)
The default configuration uses:
- Period: 2y
- Interval: 1d
- Maximum retries: 3
The application also handles MultiIndex columns returned by some yfinance configurations.
2. Feature Engineering
The feature-engineering pipeline creates several technical indicators from the closing price and volume.
Moving Averages
- SMA 10
- SMA 20
- EMA 10
- EMA 20
RSI
A 14-period Relative Strength Index is calculated from price changes.
Returns
Daily percentage return:
Return = Close(t) / Close(t-1) - 1
A 10-period rolling standard deviation is used as the volatility feature.
MACD
The project calculates:
- EMA 12
- EMA 26
- MACD
- 9-period Signal Line
Bollinger Bands
The implementation creates:
- Middle Band
- Upper Band
- Lower Band
using a 20-period rolling window.
Lagged Returns
The following lag features are generated:
Return_lag1
Return_lag2
Return_lag3
Return_lag5
On-Balance Volume
OBV is calculated using price-direction changes and trading volume.
3. Target Definition
The supervised learning target is based on the future return over a configurable look-forward period.
Default configuration:
lookforward = 5
threshold = 0.01
The target is defined as:
future_return = Close(t+5) / Close(t) - 1
Then:
Target = 1  if future_return > 1%
Target = 0  otherwise
Therefore, the Random Forest model is primarily learning a future upward-move classification problem, rather than directly predicting a future stock price.
4. Time-Series Validation
The project uses:
TimeSeriesSplit(n_splits=3)
The final split is selected for model evaluation.
This preserves temporal ordering and avoids randomly mixing earlier and later observations.
The features are then standardized using:
StandardScaler()
The scaler is fitted only on the training data before transforming the test data.
Supervised Learning Model
The SL component uses:
RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
The Random Forest produces:
predict_proba(...)
The probability of the positive class is then used inside the RL environment.
The trained model is saved as:
signal_model_<TICKER>.pkl
The feature list is saved as:
feature_columns_<TICKER>.pkl
Reinforcement Learning Model
The reinforcement-learning component uses:
- Gymnasium
- Stable-Baselines3
- DQN
- MLP policy
The custom environment is:
SLTradingEnv
Action Space
The environment has three discrete actions:
0 → Sell
1 → Hold
2 → Buy
Observation Space
The agent receives three values:
1. Supervised-learning up-move probability
2. Current position
3. Current profit percentage
The observation is represented as:
[SL probability, current position, current profit %]
Trading Environment
The environment starts with:
initial_balance = 10000
The portfolio tracks:
- Cash balance
- Shares held
- Current position
- Portfolio value
- Current profit percentage
Buy
The current available balance is used to purchase shares.
Sell
Held shares are converted back into cash.
Hold
No transaction is performed.
The environment calculates the portfolio value after each action.
DQN Configuration
The DQN agent uses:
DQN(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=64,
    exploration_fraction=0.7,
    exploration_final_eps=0.05,
    verbose=1
)
The demonstration configuration trains for:
10,000 timesteps
Training logs are written to:
./tensorboard_logs/<TICKER>/
The trained agent is saved as:
dqn_sl_trader_<TICKER>.zip
Dashboard
The Streamlit dashboard provides several sections.
1. Market Data
The dashboard retrieves market data for the selected ticker.
Example:
AAPL
The ticker can be changed from the sidebar.
2. Price Visualization
The dashboard displays the historical closing price using Plotly.
3. Daily Return Distribution
A histogram displays the distribution of daily returns.
4. Supervised Learning Evaluation
The dashboard reports:
Accuracy
Precision
Recall
F1-Score
It also displays:
- Classification report
- Confusion matrix
5. Trading Signal Visualization
The Random Forest probability is converted into signals using:
Probability > 0.60 → Buy
Probability < 0.40 → Sell
Otherwise           → Hold
The resulting signals are plotted against the closing price.
6. RL Recommendation
The trained DQN agent generates one of:
Sell
Hold
Buy
The dashboard also displays the supervised-learning probability of an upward move.
7. RL Evaluation
Clicking:
Evaluate RL Agent
runs an evaluation episode and displays:
Cumulative Reward
The average cumulative reward calculated by the evaluation routine.
Final Portfolio Value
The portfolio value at the end of the evaluation episode.
Sharpe Ratio
The implementation estimates the Sharpe ratio from portfolio returns using an annualization factor of 252.
Maximum Drawdown
Maximum observed decline from a running portfolio peak.
Portfolio Curve
A chart displays the portfolio value throughout the evaluation episode.
Project Structure
A recommended repository structure is:
hybrid-trading-dashboard/
│
├── trading_dashboard_app_fixed.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── signal_model_<TICKER>.pkl
│   ├── feature_columns_<TICKER>.pkl
│   └── dqn_sl_trader_<TICKER>.zip
│
├── data/
│   └── ...
│
├── tensorboard_logs/
│   └── <TICKER>/
│
└── .gitignore
The current implementation saves generated model files in the working directory. Moving them into a dedicated models/ directory would make the repository easier to maintain.
Installation
1. Clone the Repository
git clone https://github.com/your-username/hybrid-trading-dashboard.git
cd hybrid-trading-dashboard
Replace the repository URL with your actual GitHub repository.
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Create a requirements.txt file containing:
numpy
pandas
matplotlib
scikit-learn
joblib
plotly
streamlit
yfinance
gymnasium
stable-baselines3
torch
Then install:
pip install -r requirements.txt
Running the Application
Start the Streamlit application with:
streamlit run trading_dashboard_app_fixed.py
The Streamlit interface will open in your browser.
Enter a supported stock ticker such as:
AAPL
The application then:
1. Downloads market data.
2. Creates technical indicators.
3. Builds the classification target.
4. Performs time-series splitting.
5. Trains or loads the Random Forest model.
6. Creates the RL dataset.
7. Trains or loads the DQN agent.
8. Displays market charts.
9. Shows SL metrics.
10. Generates trading signals.
11. Allows RL-agent evaluation.
Example Workflow
Enter Ticker
     │
     ▼
Download Historical Data
     │
     ▼
Create Technical Indicators
     │
     ▼
Create Future-Return Target
     │
     ▼
Time-Series Train/Test Split
     │
     ▼
Random Forest
     │
     ├──────────────► Classification Metrics
     │
     ▼
Up-Move Probability
     │
     ▼
Custom Trading Environment
     │
     ▼
DQN Agent
     │
     ├──────────────► Buy
     ├──────────────► Hold
     └──────────────► Sell
     │
     ▼
Portfolio Evaluation
     │
     ├── Final Portfolio Value
     ├── Sharpe Ratio
     ├── Maximum Drawdown
     └── Portfolio Curve
Model Artifacts
The application generates several serialized files.
File	Purpose
signal_model_<TICKER>.pkl	Trained Random Forest classifier
feature_columns_<TICKER>.pkl	Feature names used by the SL model
formatted_data_for_rl_<TICKER>.pkl	Processed data supplied to the RL environment
dqn_sl_trader_<TICKER>.zip	Trained Stable-Baselines3 DQN model


Technologies Used
Category	Technologies
Programming	Python
Data	Pandas, NumPy
Market Data	yfinance
Machine Learning	Scikit-learn
Supervised Model	Random Forest
Reinforcement Learning	DQN
RL Framework	Stable-Baselines3
Environment	Gymnasium
Visualization	Matplotlib, Plotly
Dashboard	Streamlit
Model Persistence	Joblib
Experiment Logging	TensorBoard


Evaluation Metrics
Supervised Learning
The dashboard calculates:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Classification Report
These metrics describe the classification performance of the Random Forest on the selected time-series test split.
Reinforcement Learning
The dashboard calculates:
Cumulative Reward
Final Portfolio Value
Sharpe Ratio
Maximum Drawdown
Portfolio Curve
These metrics are intended to provide a basic view of the RL agent's simulated portfolio behavior.
Important Limitations
This project should be considered an experimental trading research prototype.
Several aspects should be improved before considering realistic backtesting or deployment.
1. Transaction Costs
The current environment does not explicitly model:
- Brokerage fees
- Bid/ask spread
- Slippage
- Taxes
- Market impact
Therefore, reported portfolio performance may not represent realistic trading conditions.
2. Position Sizing
The current implementation can use the available balance to purchase shares. More realistic systems should incorporate:
- Position-size limits
- Risk-per-trade
- Stop-loss rules
- Maximum exposure
- Portfolio diversification
3. Reward Design
The environment currently calculates reward from the difference between portfolio value and the initial balance.
A more robust reward design could consider:
- Step-wise returns
- Risk-adjusted returns
- Drawdown penalties
- Transaction costs
- Volatility
4. Out-of-Sample Testing
A production-grade trading system should use strict chronological:
Training
    ↓
Validation
    ↓
Test
    ↓
Walk-forward evaluation
and avoid repeatedly tuning the system against the same test period.
5. Data Leakage
Financial ML systems require careful prevention of look-ahead bias and leakage.
Any future version should audit:
- Feature construction
- Scaling
- Target generation
- RL environment state
- Hyperparameter tuning
- Model-selection procedures
6. RL Evaluation
The current RL evaluation uses a single evaluation episode by default.
More robust experiments should evaluate across:
- Multiple random seeds
- Multiple market periods
- Bull markets
- Bear markets
- High-volatility periods
- Different assets
Future Improvements
Potential extensions include:
- Walk-forward backtesting
- Transaction-cost modeling
- Slippage simulation
- Risk-adjusted reward functions
- Portfolio-level RL
- Multiple-stock trading
- Position sizing
- Stop-loss and take-profit mechanisms
- Benchmark comparison against Buy-and-Hold
- Sharpe/Sortino/Calmar analysis
- Hyperparameter optimization
- Experiment tracking
- MLflow integration
- Docker deployment
- FastAPI model-serving layer
- Automated model retraining
- Model monitoring
- Feature-drift monitoring
- Paper-trading integration
- Unit and integration tests
- CI/CD pipeline
Research Direction
This project can be extended into a broader financial AI architecture by combining:
Market Data
     +
Technical Indicators
     +
Financial News
     +
NLP Embeddings
     +
Supervised Learning
     +
Reinforcement Learning
     +
Explainable AI
Possible future models include:
- XGBoost
- LightGBM
- LSTM
- BiLSTM
- Transformer-based time-series models
- Graph Neural Networks
- Graph Attention Networks
- Multi-agent reinforcement learning
Disclaimer
This software is intended for education, experimentation, and research.
It does not constitute financial, investment, or trading advice.
Historical or simulated performance does not guarantee future results. Any real-world trading system should undergo extensive independent validation, realistic transaction-cost analysis, risk management, security review, and regulatory/compliance assessment before deployment.
Author
Advaith S
AI/ML Engineer | Data Scientist | FinTech
- LinkedIn: https://linkedin.com/in/advaith-s-a301b6260
- GitHub: https://github.com/advaith-s
