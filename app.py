# -------------------------
# trading_dashboard_app_fixed.py
# -------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from time import sleep
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import plotly.express as px
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# -------------------------
# Module 1: Fetch data
# -------------------------
def fetch_data(ticker, period="2y", interval="1d", max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            sleep(2)

# -------------------------
# Module 2: Feature engineering (enhanced)
# -------------------------
def create_features(df):
    df = df.copy()

    # --- Moving averages ---
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # --- RSI ---
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- Returns & volatility ---
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(10).std()

    # --- MACD & Signal Line ---
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands ---
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(20).std()

    # --- Lagged Returns ---
    for lag in [1, 2, 3, 5]:
        df[f'Return_lag{lag}'] = df['Return'].shift(lag)

    # --- Volume-based indicator: On-Balance Volume (OBV) ---
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    df.dropna(inplace=True)
    return df

# -------------------------
# Module 3: Process data (threshold tuned)
# -------------------------
def process_data(df, lookforward=5, threshold=0.01):  # lowered threshold for better balance
    df = df.copy()
    df['future_return'] = df['Close'].pct_change(lookforward).shift(-lookforward)
    df['Target'] = (df['future_return'] > threshold).astype(int)

    # Select only engineered features (avoid raw OHLC leakage)
    feature_columns = [col for col in df.columns if col not in
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return', 'Target']]

    X = df[feature_columns]
    y = df['Target']

    # Time series split (use last split)
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, df

# -------------------------
# Module 4: Train SL Model (tuned RF)
# -------------------------
def train_sl_model(X_train, X_test, y_train, y_test, ticker, feature_columns):
    model = RandomForestClassifier(
        n_estimators=400,      # more trees for stability
        max_depth=None,        # let forest grow fully
        min_samples_split=5,   # prevent overfitting
        min_samples_leaf=3,    # smoother splits
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"SL Model performance for {ticker}:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, f'signal_model_{ticker}.pkl')
    joblib.dump(feature_columns, f'feature_columns_{ticker}.pkl')
    return model

# -------------------------
# Module 5: RL Environment (unchanged)
# -------------------------
class SLTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, ticker, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.ticker = ticker
        self.initial_balance = initial_balance

        self.current_step = 0
        self.balance = float(initial_balance)
        self.shares_held = 0.0
        self.current_position = 0
        self.current_profit_pct = 0.0
        self.portfolio_value = float(initial_balance)

        self.sl_model = joblib.load(f'signal_model_{ticker}.pkl')
        self.feature_columns = joblib.load(f'feature_columns_{ticker}.pkl')

        self.action_space = spaces.Discrete(3)
        low = np.array([0.0, -1.0, -np.finfo(np.float32).max], dtype=np.float32)
        high = np.array([1.0, 1.0, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_state(self):
        row = self.df.loc[self.current_step, self.feature_columns]
        features = np.asarray(row.values, dtype=np.float32).reshape(1, -1)
        sl_prob = float(self.sl_model.predict_proba(features)[0][1])
        return np.array([sl_prob, float(self.current_position), float(self.current_profit_pct)], dtype=np.float32)

    def _take_action(self, action: int):
        price = float(self.df.loc[self.current_step, "Close"])
        if action == 2:  # Buy
            if self.balance > 0:
                self.shares_held = self.balance / price
                self.balance = 0.0
                self.current_position = 1
        elif action == 0:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * price
                self.shares_held = 0.0
                self.current_position = -1
        self.portfolio_value = self.balance + self.shares_held * price
        self.current_profit_pct = 0.0

    def step(self, action):
        self._take_action(int(action))
        reward = float(self.portfolio_value - self.initial_balance)
        self.current_step += 1
        terminated = bool(self.current_step >= len(self.df) - 1)
        truncated = False
        obs = self._get_state()
        info = {"portfolio_value": self.portfolio_value}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.current_position = 0
        self.current_profit_pct = 0.0
        self.portfolio_value = float(self.initial_balance)
        return self._get_state(), {}

    def render(self):
        print(f"Step {self.current_step} | Balance {self.balance:.2f} | Shares {self.shares_held:.4f} | Portfolio {self.portfolio_value:.2f}")

# -------------------------
# Module 6: Train RL Agent
# -------------------------
def train_rl_agent(ticker, total_timesteps=10000):
    df = joblib.load(f'formatted_data_for_rl_{ticker}.pkl')
    vec_env = DummyVecEnv([lambda: SLTradingEnv(df, ticker)])
    model = DQN(
        'MlpPolicy',
        vec_env,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=64,
        exploration_fraction=0.7,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{ticker}/"
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(f"dqn_sl_trader_{ticker}")
    return model

# -------------------------
# Module 7: Streamlit Dashboard
# -------------------------
st.set_page_config(page_title="SL+RL Trading Dashboard (fixed)", layout="wide")
st.title("Trading Dashboard: SL + RL- Optimized Strategy")

st.sidebar.header("Ticker Configuration")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()

if not ticker:
    st.error("Enter ticker symbol in the sidebar.")
    st.stop()

st.info(f"Fetching data for {ticker}...")
data = fetch_data(ticker)
data_feat = create_features(data)

X_train, X_test, y_train, y_test, scaler, feature_columns, processed_data = process_data(data_feat)

if not os.path.exists(f"signal_model_{ticker}.pkl"):
    st.info("Training SL model (RandomForest)...")
    sl_model = train_sl_model(X_train, X_test, y_train, y_test, ticker, feature_columns)
else:
    sl_model = joblib.load(f"signal_model_{ticker}.pkl")
    feature_columns = joblib.load(f"feature_columns_{ticker}.pkl")

joblib.dump(processed_data, f"formatted_data_for_rl_{ticker}.pkl")

if not os.path.exists(f"dqn_sl_trader_{ticker}.zip"):
    st.info("Training RL agent (this can take time) — using 10k timesteps for demo.")
    rl_agent = train_rl_agent(ticker, total_timesteps=10000)
else:
    rl_agent = DQN.load(f"dqn_sl_trader_{ticker}")

# -------------------------
# Module 8: Visualization
# -------------------------
st.header(f"Evaluation Metrics & Charts for {ticker}")
df_eval = joblib.load(f"formatted_data_for_rl_{ticker}.pkl")
fig_close = px.line(df_eval, x=df_eval.index, y='Close', title=f"{ticker} Close Price Over Time")
st.plotly_chart(fig_close, use_container_width=True)

df_eval['Return'] = df_eval['Close'].pct_change()
fig_hist = px.histogram(df_eval, x='Return', nbins=50, title=f"{ticker} Daily Returns Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------
# Module 9: RL-Optimized Recommendation
# -------------------------
st.header("RL-Optimized Recommendation")
last_row = df_eval.iloc[-1]
last_features = last_row[feature_columns].values.reshape(1, -1).astype(np.float32)
sl_prob = float(sl_model.predict_proba(last_features)[0][1])
state = np.array([sl_prob, 0.0, 0.0], dtype=np.float32)
action, _ = rl_agent.predict(state, deterministic=True)
recommendation_map = {0: "Sell 🔴", 1: "Hold 🟡", 2: "Buy 🟢"}
recommendation = recommendation_map.get(int(action), "Hold 🟡")

st.subheader(f"RL action: {recommendation}")
st.write(f"SL model probability of up-move: {sl_prob:.3f}")

# -------------------------
# Module 10: SL Model Evaluation
# -------------------------
st.header("SL Model Evaluation")
y_pred = sl_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.write("### Metrics")
st.json({
    "Accuracy": round(acc, 3),
    "Precision": round(prec, 3),
    "Recall": round(rec, 3),
    "F1-Score": round(f1, 3),
})
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred, zero_division=0))
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_estimator(sl_model, X_test, y_test, ax=ax_cm)
st.pyplot(fig_cm)

# -------------------------
# Module 11: Signal Visualization
# -------------------------
st.header("Trading Signal Visualization")
df_eval["SL_Prob"] = sl_model.predict_proba(df_eval[feature_columns])[:, 1]
df_eval["Signal"] = np.where(df_eval["SL_Prob"] > 0.6, "Buy",
                      np.where(df_eval["SL_Prob"] < 0.4, "Sell", "Hold"))
fig_sig, ax_sig = plt.subplots(figsize=(12, 6))
ax_sig.plot(df_eval.index, df_eval["Close"], label="Close Price", color="blue")
buy_signals = df_eval[df_eval["Signal"] == "Buy"]
sell_signals = df_eval[df_eval["Signal"] == "Sell"]
ax_sig.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="green", label="Buy")
ax_sig.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="red", label="Sell")
ax_sig.set_title(f"{ticker} SL Model Trading Signals")
ax_sig.legend()
st.pyplot(fig_sig)

# -----------------------------
# (Replace from here) After printing data.head()/tail()
# -----------------------------

# Prepare vectorized env (used for training/loading) and non-vectorized eval env
vec_env = DummyVecEnv([lambda: SLTradingEnv(processed_data, ticker)])
eval_env = SLTradingEnv(processed_data, ticker)

# -----------------------------
# Make sure we have an RL agent (use rl_agent from earlier code if available)
# -----------------------------
agent = None
if 'rl_agent' in globals():
    agent = rl_agent
else:
    model_path = f"dqn_sl_trader_{ticker}"
    if os.path.exists(model_path + ".zip"):
        # load without setting env (we'll use eval_env for evaluation)
        agent = DQN.load(model_path)
        st.success(f"Loaded RL agent from {model_path}.zip")
    else:
        st.error("No RL agent found. Please train the RL agent first (or ensure the model file exists).")
        st.stop()

# -----------------------------
# Module 12: Evaluation of RL Agent Performance (robust)
# -----------------------------
def evaluate_agent(agent, env, episodes=1):
    """
    Evaluate agent on a non-vectorized environment (SLTradingEnv).
    Returns metrics and the portfolio curve for the first episode.
    """
    all_rewards = []
    portfolio_values_list = []

    for ep in range(episodes):
        obs, _ = env.reset()              # SLTradingEnv.reset -> (obs, {})
        done = False
        ep_rewards = []
        ep_portfolio = []

        while not done:
            # agent.predict works with a single observation (shape (obs_dim,))
            action, _ = agent.predict(obs, deterministic=True)

            # action may be a numpy scalar/array — convert to int
            if isinstance(action, np.ndarray):
                if action.shape == ():  # scalar array
                    action = int(action)
                else:
                    action = int(action[0])
            else:
                action = int(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            done = bool(terminated or truncated)

            ep_rewards.append(reward)
            ep_portfolio.append(info.get("portfolio_value", env.portfolio_value))

        all_rewards.append(np.sum(ep_rewards))
        portfolio_values_list.append(ep_portfolio)

    # Use first episode portfolio for metrics/plot
    portfolio_array = np.array(portfolio_values_list[0])
    rewards_array = np.array(all_rewards)

    cumulative_reward = rewards_array.mean()
    final_portfolio_value = float(portfolio_array[-1])

    # compute returns for Sharpe: daily returns of portfolio
    if len(portfolio_array) >= 2 and np.std(np.diff(portfolio_array) / portfolio_array[:-1]) != 0:
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # max drawdown
    running_max = np.maximum.accumulate(portfolio_array)
    drawdowns = (portfolio_array - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0

    return {
        "Cumulative Reward": float(cumulative_reward),
        "Final Portfolio Value": final_portfolio_value,
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
        "Portfolio Curve": portfolio_array
    }

# -----------------------------
# Streamlit evaluation button
# -----------------------------
if st.button("Evaluate RL Agent"):
    st.subheader("RL Agent Evaluation")

    eval_results = evaluate_agent(agent, eval_env, episodes=1)

    st.write("📊 **Evaluation Metrics:**")
    st.write(f"- Cumulative Reward: {eval_results['Cumulative Reward']:.2f}")
    st.write(f"- Final Portfolio Value: ${eval_results['Final Portfolio Value']:.2f}")
    st.write(f"- Sharpe Ratio: {eval_results['Sharpe Ratio']:.2f}")
    st.write(f"- Max Drawdown: {eval_results['Max Drawdown']:.2%}")

    # Plot portfolio curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_results["Portfolio Curve"], label="Portfolio Value")
    ax.set_title("Portfolio Value Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# (End replacement)
# -----------------------------
