import backtrader as bt
import pandas as pd

# from google.colab import drive
import matplotlib.pyplot as plt
import gspread

# from oauth2client.service_account import ServiceAccountCredentials
import datetime
import numpy as np
from itertools import product
import time
import csv


def load_data():
    # Step 1: Mount Google Drive
    # drive.mount('/content/drive')

    # Step 2: Specify the path to the CSV file on Google Drive
    file_path = "binance_ethusdt_15m.csv"  # Modify this with the correct file path

    # Step 3: Load and preprocess the CSV file from Google Drive
    df = pd.read_csv(file_path)

    # Convert timestamps to datetime
    df["datetime"] = pd.to_datetime(df["Open Time"], unit="ms")
    df = df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

    # Remove duplicated rows
    df = df.drop_duplicates()

    # Filter data by date range (example: from '2021-01-01' to '2021-12-31')
    start_date = "2024-01-01"
    end_date = "2025-12-31"

    # Convert the datetime column to datetime objects for filtering
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter the dataframe
    df_filtered = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

    # Drop any NaN or infinite values
    df_filtered = df_filtered.dropna()

    # Set datetime as index and sort
    df_filtered.set_index("datetime", inplace=True)
    df_filtered = df_filtered.sort_index()

    # Step 4: Load Data into Backtrader
    class PandasData(bt.feeds.PandasData):
        lines = ("datetime",)
        params = (
            ("datetime", None),
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("close", "Close"),
            ("volume", "Volume"),
            ("openinterest", None),
        )

    # Convert to Backtrader data feed
    data = PandasData(dataname=df_filtered)
    return data


# Step 5: Define Strategy
class TradingViewStrategy(bt.Strategy):
    params = (
        ("ema_short", 12),
        ("ema_medium", 50),
        ("ema_long", 200),
        ("atr_period", 14),
        ("atr_multiplier", 2),
        ("rsi_period", 14),
        ("rsi_long_lower", 55),
        ("rsi_long_upper", 70),
        ("rsi_short_lower", 30),
        ("rsi_short_upper", 45),
        ("length", 100),  # Length for percentile calculation
        ("percentile_low", 20),  # Low percentile
        ("percentile_high", 80),  # High percentile
        ("risk_percent", 10),
        ("risk_multiplier", 1.3),
        ("max_risk_percent", 30),
        ("reward_risk_ratio", 2),
    )

    def __init__(self):
        # Indicators
        self.ema_short = bt.indicators.EMA(
            self.data.close, period=self.params.ema_short
        )
        self.ema_medium = bt.indicators.EMA(
            self.data.close, period=self.params.ema_medium
        )
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.params.ema_long)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # Signals
        self.cross_over = bt.indicators.CrossOver(self.ema_short, self.ema_medium)
        self.cross_under = bt.indicators.CrossOver(self.ema_medium, self.ema_short)

        # Store the previous loss
        self.previous_loss = 0

    def notify_trade(self, trade):
        if not trade.isclosed:
            return  # Only handle trades that are closed
        self.previous_loss = abs(round(trade.pnl, 0) if trade.pnl < 0 else 0)
        # print("*" * 20)
        # print(f"TRADE CLOSED: {trade.data._name}")
        # print(f"  Size: {trade.size}")
        # print(f"  Entry Price: {trade.price:.2f}")
        # print(f"  Exit Price: {trade.price:.2f}")
        # print(f"  Gross Profit/Loss: {trade.pnl:.2f}")
        # print(f"  Net Profit/Loss: {trade.pnlcomm:.2f}")  # pnlcomm includes commission costs
        # print("-" * 20)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order is active
        self.order = None

    def next(self):
        # Percentile Calculation
        length = self.params.length
        percentile_low = self.params.percentile_low
        percentile_high = self.params.percentile_high

        # Create a sorted list of the 'close' prices over the defined length
        sorted_prices = sorted(self.data.close.get(size=length))

        # Calculate the index for the low and high percentiles
        low_index = int(percentile_low / 100 * length)
        high_index = int(percentile_high / 100 * length)

        # Get the low and high percentile values
        low_percentile = sorted_prices[low_index]
        high_percentile = sorted_prices[high_index]

        # Long and Short signals based on percentiles
        long_percentile_signal = self.data.close[0] > high_percentile
        short_percentile_signal = self.data.close[0] < low_percentile

        # Risk management calculations
        available_balance = round(self.broker.cash, 0)
        risk_amount = round(available_balance * (self.params.risk_percent / 100), 0)
        atr_value = self.atr[0] * self.params.atr_multiplier

        # If there was a previous loss, use it to adjust position size
        if self.previous_loss > 0:
            risk_amount = (
                self.previous_loss * self.params.risk_multiplier
            )  # Increase the risk by 1.5x the previous loss
            risk_amount = (
                risk_amount
                if risk_amount
                <= round(available_balance * (self.params.max_risk_percent / 100), 0)
                else round(available_balance * (20 / 100), 0)
            )

        # Long signals
        if (
            self.cross_over > 0
            and self.data.close[0] > self.ema_long[0]
            and self.params.rsi_long_lower <= self.rsi[0] <= self.params.rsi_long_upper
            and long_percentile_signal
        ):
            if not self.position:  # Only enter if no active position
                stop_loss = round(self.data.close[0] - atr_value, 2)
                take_profit = round(
                    self.data.close[0] + (atr_value * self.params.reward_risk_ratio), 2
                )
                quantity = round(risk_amount / abs(self.data.close[0] - stop_loss), 3)

                # Bracket order for long position
                self.buy_bracket(
                    size=quantity,
                    stopprice=stop_loss,  # Stop-loss price
                    limitprice=take_profit,  # Take-profit price
                )

        # Short signals
        if (
            self.cross_under > 0
            and self.data.close[0] < self.ema_long[0]
            and self.params.rsi_short_lower
            <= self.rsi[0]
            <= self.params.rsi_short_upper
            and short_percentile_signal
        ):
            if not self.position:  # Only enter if no active position
                stop_loss = round(self.data.close[0] + atr_value, 2)
                take_profit = round(
                    self.data.close[0] - (atr_value * self.params.reward_risk_ratio), 2
                )
                quantity = round(risk_amount / abs(self.data.close[0] - stop_loss), 3)

                # Bracket order for short position
                self.sell_bracket(
                    size=quantity,
                    stopprice=stop_loss,  # Stop-loss price
                    limitprice=take_profit,  # Take-profit price
                )


# Function to write data to csv
def write_to_csv(data):
    csvFile = "backtrader-optimization-ethusdt.csv"
    with open(csvFile, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(data)


# Function to write results to Google Sheets
def clear_google_sheet(sheet_id):
    # Define scope and authenticate
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = "kmm-scan-eth-blocks-dc989520c58e.json"
    # creds = ServiceAccountCredentials.from_json_keyfile_name("path_to_your_service_account.json", scope)
    # client = gspread.authorize(creds)
    client = gspread.auth.service_account(creds, scope)

    # Open the spreadsheet and select the sheet
    sheet = client.open_by_key(sheet_id).worksheet(
        "Sheet1"
    )  # Assuming single sheet; adjust if needed
    sheet.clear()


def write_header():
    analysis_results_headers = [
        "date_time",
        # Parameters
        "ema_short",
        "ema_medium",
        "ema_long",
        "atr_period",
        "atr_multiplier",
        "rsi_period",
        "rsi_long_lower",
        "rsi_long_upper",
        "rsi_short_lower",
        "rsi_short_upper",
        "length",
        "percentile_low",
        "percentile_high",
        "risk_percent",
        "risk_multiplier",
        "max_risk_percent",
        "reward_risk_ratio",
        # Sharpe Ratio
        "Sharpe Ratio",
        # Drawdown Analysis
        "Current Drawdown (%)",
        "Money Down",
        "Max Drawdown (%)",
        "Max Money Down",
        "Max Drawdown Length (bars)",
        # Trade Analysis
        "Total Trades",
        "Open Trades",
        "Closed Trades",
        "Winning Streak",
        "Longest Winning Streak",
        "Losing Streak",
        "Longest Losing Streak",
        # Profit and Loss (PnL)
        "Gross PnL",
        "Average PnL",
        "Net PnL",
        # Winning Trades
        "Total Wins",
        "Total Win PnL",
        "Average Win PnL",
        "Maximum Win",
        # Losing Trades
        "Total Losses",
        "Total Loss PnL",
        "Average Loss PnL",
        "Maximum Loss",
    ]
    sheet_id = "1PowXxg89fAhbKlEdKacm-qDWyLH99U8Y2QDTLuCHiBk"
    clear_google_sheet(sheet_id)
    write_to_google_sheet(sheet_id, analysis_results_headers)


# Function to write results to Google Sheets
def write_to_google_sheet(sheet_id, data):
    counter = 0
    max_retry = 5
    wait_time = 5
    write_success = False

    while not write_success:
        try:
            # Define scope and authenticate
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = "kmm-scan-eth-blocks-dc989520c58e.json"
            # creds = ServiceAccountCredentials.from_json_keyfile_name("path_to_your_service_account.json", scope)
            # client = gspread.authorize(creds)
            client = gspread.auth.service_account(creds, scope)

            # Open the spreadsheet and select the sheet
            sheet = client.open_by_key(sheet_id).worksheet(
                "Sheet1"
            )  # Assuming single sheet; adjust if needed
            # Append data
            sheet.append_row(data)
            write_success = True
        except:
            if counter <= max_retry:
                counter = counter + 1
                time.sleep(wait_time)
                wait_time = wait_time * 2
            else:
                write_success = True
    write_to_csv(data)


def main(**param_dict):
    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Add Strategy
    if param_dict:
        cerebro.addstrategy(TradingViewStrategy, **param_dict)
    else:
        cerebro.addstrategy(TradingViewStrategy)

    # Add Data
    cerebro.adddata(load_data())

    # Set Broker Parameters
    cerebro.broker.set_cash(10000)  # Starting balance
    cerebro.broker.setcommission(leverage=125)

    # Add Analyzers
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")  # PyFolio analyzer
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")  # Sharpe Ratio
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")  # Drawdown Analysis
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")  # Trade Analysis

    # Display Starting Portfolio Value
    print("Starting Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))

    # Run Backtest
    results = cerebro.run()
    strategy = results[0]  # Access the first strategy in the results
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    trade_analysis = strategy.analyzers.trades.get_analysis()

    if param_dict:
        # Display Starting Portfolio Value
        print(f"Running with {param_dict}")
        print("Starting Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))

        # Create data list
        data = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
            param_dict["ema_short"],
            param_dict["ema_medium"],
            param_dict["ema_long"],
            param_dict["atr_period"],
            param_dict["atr_multiplier"],
            param_dict["rsi_period"],
            param_dict["rsi_long_lower"],
            param_dict["rsi_long_upper"],
            param_dict["rsi_short_lower"],
            param_dict["rsi_short_upper"],
            param_dict["length"],
            param_dict["percentile_low"],
            param_dict["percentile_high"],
            param_dict["risk_percent"],
            param_dict["risk_multiplier"],
            param_dict["max_risk_percent"],
            param_dict["reward_risk_ratio"],
            # Sharpe Ratio
            sharpe_analysis.get("sharperatio", "N/A"),
            # Drawdown Analysis
            drawdown_analysis["drawdown"],
            drawdown_analysis["moneydown"],
            drawdown_analysis["max"]["drawdown"],
            drawdown_analysis["max"]["moneydown"],
            drawdown_analysis["max"]["len"],
            # Trade Analysis
            trade_analysis.total.total,
            trade_analysis.total.open,
            trade_analysis.total.closed,
            trade_analysis.streak.won.current,
            trade_analysis.streak.won.longest,
            trade_analysis.streak.lost.current,
            trade_analysis.streak.lost.longest,
            # Profit and Loss (PnL)
            trade_analysis.pnl.gross.total,
            trade_analysis.pnl.gross.average,
            trade_analysis.pnl.net.total,
            # Winning Trades
            trade_analysis.won.total,
            trade_analysis.won.pnl.total,
            trade_analysis.won.pnl.average,
            trade_analysis.won.pnl.max,
            # Losing Trades
            trade_analysis.lost.total,
            trade_analysis.lost.pnl.total,
            trade_analysis.lost.pnl.average,
            trade_analysis.lost.pnl.max,
        ]

        # Write to Google Sheets (or other destination)
        sheet_id = "1PowXxg89fAhbKlEdKacm-qDWyLH99U8Y2QDTLuCHiBk"
        write_to_google_sheet(sheet_id, data)

    else:
        # Display Ending Portfolio Value
        print("Ending Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))

        # Analyze and Display Results
        print(f"\n--- Analysis Results ---\n")

        # Sharpe Ratio

        print("### Sharpe Ratio")
        print(f"  Sharpe Ratio: {sharpe_analysis.get('sharperatio', 'N/A')}")

        # Drawdown Analysis

        print("\n### Drawdown Analysis")
        print(f"  Current Drawdown: {drawdown_analysis['drawdown']:.2f}%")
        print(f"  Money Down: ${drawdown_analysis['moneydown']:.2f}")
        print(f"  Max Drawdown: {drawdown_analysis['max']['drawdown']:.2f}%")
        print(f"  Max Money Down: ${drawdown_analysis['max']['moneydown']:.2f}")
        print(f"  Max Drawdown Length: {drawdown_analysis['max']['len']} bars")

        # Trade Analysis

        print("\n### Trade Analysis")
        print("- Total Trades: {}".format(trade_analysis.total.total))
        print("  - Open Trades: {}".format(trade_analysis.total.open))
        print("  - Closed Trades: {}".format(trade_analysis.total.closed))
        print(
            "- Winning Streak: {} (Longest: {})".format(
                trade_analysis.streak.won.current, trade_analysis.streak.won.longest
            )
        )
        print(
            "- Losing Streak: {} (Longest: {})".format(
                trade_analysis.streak.lost.current, trade_analysis.streak.lost.longest
            )
        )

        # Profit and Loss
        print("\n#### Profit and Loss (PnL)")
        print(f"- Gross PnL: ${trade_analysis.pnl.gross.total:.2f}")
        print(f"- Average PnL: ${trade_analysis.pnl.gross.average:.2f}")
        print(f"- Net PnL: ${trade_analysis.pnl.net.total:.2f}")

        # Winning Trades
        print("\n#### Winning Trades")
        print(f"- Total Wins: {trade_analysis.won.total}")
        print(f"  - Total PnL: ${trade_analysis.won.pnl.total:.2f}")
        print(f"  - Average PnL: ${trade_analysis.won.pnl.average:.2f}")
        print(f"  - Maximum Win: ${trade_analysis.won.pnl.max:.2f}")

        # Losing Trades
        print("\n#### Losing Trades")
        print(f"- Total Losses: {trade_analysis.lost.total}")
        print(f"  - Total PnL: ${trade_analysis.lost.pnl.total:.2f}")
        print(f"  - Average PnL: ${trade_analysis.lost.pnl.average:.2f}")
        print(f"  - Maximum Loss: ${trade_analysis.lost.pnl.max:.2f}")

    # Clear strategy for the next loop
    cerebro.strats = []
    results = None
    strategy = None
    sharpe_analysis = None
    drawdown_analysis = None
    trade_analysis = None


def loopMain():
    # Original parameter values
    original_params = {
        "ema_short": 12,
        "ema_medium": 50,
        "ema_long": 200,
        "atr_period": 14,
        "rsi_period": 14,
        "rsi_long_lower": 55,
        "rsi_long_upper": 70,
        "rsi_short_lower": 30,
        "rsi_short_upper": 45,
        "length": 100,
        "percentile_low": 20,
        "percentile_high": 80,
        "risk_percent": 10,
        "risk_multiplier": 1.3,
        "max_risk_percent": 30,
        "reward_risk_ratio": 2,
    }

    # Optimizing parameter ranges
    parameter_ranges = {
        "rsi_long_lower": range(51, 55, 1),
        "atr_multiplier": np.arange(1.5, 3.0, 0.1),
        "rsi_long_upper": range(70, 75, 1),
        "rsi_short_lower": range(25, 30, 1),
        "rsi_short_upper": range(45, 49, 1),
        "percentile_low": range(20, 30, 1),
        "percentile_high": range(70, 80, 1),
        "risk_percent": range(1, 10, 1),
        "risk_multiplier": np.arange(1.1, 2, 0.1),
        "max_risk_percent": range(20, 40, 1),
        "reward_risk_ratio": np.arange(1.5, 4, 0.1),
    }

    # Create all combinations of optimizing parameters
    param_combinations = product(*parameter_ranges.values())

    for params in param_combinations:
        # Combine the original parameters with the current optimized parameters
        param_dict = {
            **original_params,
            **{key: val for key, val in zip(parameter_ranges.keys(), params)},
        }
        main(**param_dict)


# Function to loop through parameters and run backtest
def loopThroughParameters():
    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Original parameter values
    original_params = {
        "ema_short": 12,
        "ema_medium": 50,
        "ema_long": 200,
        "atr_period": 14,
        "rsi_period": 14,
        "rsi_long_lower": 55,
        "rsi_long_upper": 70,
        "rsi_short_lower": 30,
        "rsi_short_upper": 45,
        "length": 100,
        "percentile_low": 20,
        "percentile_high": 80,
        "risk_percent": 10,
        "risk_multiplier": 1.3,
        "max_risk_percent": 30,
        "reward_risk_ratio": 2,
    }

    # Optimizing parameter ranges
    parameter_ranges = {
        "rsi_long_lower": range(51, 55, 1),
        "atr_multiplier": np.arange(1.5, 3.0, 0.1),
        "rsi_long_upper": range(70, 75, 1),
        "rsi_short_lower": range(25, 30, 1),
        "rsi_short_upper": range(45, 49, 1),
        "percentile_low": range(20, 30, 1),
        "percentile_high": range(70, 80, 1),
        "risk_percent": range(1, 10, 1),
        "risk_multiplier": np.arange(1.1, 2, 0.1),
        "max_risk_percent": range(20, 40, 1),
        "reward_risk_ratio": np.arange(1.5, 4, 0.1),
    }

    # Create all combinations of optimizing parameters
    param_combinations = product(*parameter_ranges.values())

    for params in param_combinations:
        # Combine the original parameters with the current optimized parameters
        param_dict = {
            **original_params,
            **{key: val for key, val in zip(parameter_ranges.keys(), params)},
        }

        # Add the strategy with the current set of parameters
        cerebro.addstrategy(TradingViewStrategy, **param_dict)

        # Add Data (you need to define the data variable)
        cerebro.adddata(load_data())

        # Set Broker Parameters
        cerebro.broker.set_cash(10000)
        cerebro.broker.setcommission(leverage=125)

        # Add Analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        # Display Starting Portfolio Value
        print(f"Running with {param_dict}")
        print("Starting Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))

        # Run Backtest
        results = cerebro.run()
        strategy = results[0]  # Access the first strategy in the results

        # Collect the analysis results
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        trade_analysis = strategy.analyzers.trades.get_analysis()

        # Create data list
        data = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
            param_dict["ema_short"],
            param_dict["ema_medium"],
            param_dict["ema_long"],
            param_dict["atr_period"],
            param_dict["atr_multiplier"],
            param_dict["rsi_period"],
            param_dict["rsi_long_lower"],
            param_dict["rsi_long_upper"],
            param_dict["rsi_short_lower"],
            param_dict["rsi_short_upper"],
            param_dict["length"],
            param_dict["percentile_low"],
            param_dict["percentile_high"],
            param_dict["risk_percent"],
            param_dict["risk_multiplier"],
            param_dict["max_risk_percent"],
            param_dict["reward_risk_ratio"],
            # Sharpe Ratio
            sharpe_analysis.get("sharperatio", "N/A"),
            # Drawdown Analysis
            drawdown_analysis["drawdown"],
            drawdown_analysis["moneydown"],
            drawdown_analysis["max"]["drawdown"],
            drawdown_analysis["max"]["moneydown"],
            drawdown_analysis["max"]["len"],
            # Trade Analysis
            trade_analysis.total.total,
            trade_analysis.total.open,
            trade_analysis.total.closed,
            trade_analysis.streak.won.current,
            trade_analysis.streak.won.longest,
            trade_analysis.streak.lost.current,
            trade_analysis.streak.lost.longest,
            # Profit and Loss (PnL)
            trade_analysis.pnl.gross.total,
            trade_analysis.pnl.gross.average,
            trade_analysis.pnl.net.total,
            # Winning Trades
            trade_analysis.won.total,
            trade_analysis.won.pnl.total,
            trade_analysis.won.pnl.average,
            trade_analysis.won.pnl.max,
            # Losing Trades
            trade_analysis.lost.total,
            trade_analysis.lost.pnl.total,
            trade_analysis.lost.pnl.average,
            trade_analysis.lost.pnl.max,
        ]

        # Write to Google Sheets (or other destination)
        sheet_id = "1PowXxg89fAhbKlEdKacm-qDWyLH99U8Y2QDTLuCHiBk"
        write_to_google_sheet(sheet_id, data)

        # Clear strategy for the next loop
        cerebro.strats = []


# main()
write_header()
# loopThroughParameters()
loopMain()
