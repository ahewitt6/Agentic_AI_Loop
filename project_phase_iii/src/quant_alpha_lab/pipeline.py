#import relevant packages
import time
import numpy as np
import pandas as pd

from quant_alpha_lab.features import build_features
import os

from quant_alpha_lab.candidates import (
    generate_candidate_library,
    generate_signal_definition_library
)

from quant_alpha_lab.evaluation import (
    evaluate_all_candidates,
    build_evaluation_report
)

from quant_alpha_lab.validation import (
    temporal_split,
    filter_accepted_signals,
    validate_accepted_signals,
    build_validation_report
)


#extract raw data
#pull data from financial modeling prep

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import requests

import pandas as pd
import numpy as np

API_KEY = os.environ["FMP_API_KEY"]

'''def prepare_phase_iii_data():
    

    BASE_URL = "https://financialmodelingprep.com/stable"



    tickers = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # NVIDIA
        "AMZN",   # Amazon
        "GOOGL",  # Alphabet
        "META",   # Meta
        "JPM",    # JPMorgan Chase
        "XOM",    # Exxon Mobil
        "JNJ",    # Johnson & Johnson
        "WMT",    # Walmart
    ]

    tickers = [
        # Technology
        "AAPL", "MSFT", "NVDA", "AMD", "INTC",
        "CSCO",

        # Consumer discretionary
        "AMZN", "TSLA", "LOW", "MCD",
        "SBUX", "NKE", "GM", "F", "TGT",

        # Consumer staples
        "WMT", "COST", "PG", "KO", "PEP",
        "PM", "MO", "CL", "MDLZ",

        # Financials
        "JPM", "BAC", "WFC", "GS", "MS",
        "C", "AXP", "SCHW", "USB", "PNC",
        "V", "MA", "BK", "AIG", "MET",

        # Healthcare
        "JNJ", "PFE", "MRK", "ABBV", "ABT",
        "TMO", "MDT", "AMGN", "GILD", "CVS",
        "CI", "BMY", "SYK",

        # Industrials
        "CAT", "DE", "HON", "UPS", "LMT",
        "NOC", "UNP", "FDX", "BA", "MMM",
        "GE", "EMR", "ITW", "GD", "CSX",

        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG",
        "MPC", "OXY",

        # Utilities
        "NEE", "SO", "DUK", "AEP", "EXC",

        # Materials
        "LIN", "APD", "SHW", "FCX", "NEM",

        # Communication / diversified
        "DIS", "CMCSA", "T", "VZ", "ADP"
    ]

    def get_fundamental_data(
        ticker,
        daily_limit=1500,
        quarterly_limit=24,
        annual_limit=10
    ):
        import numpy as np
        import pandas as pd
        import requests

        v3_base_url = "https://financialmodelingprep.com/api/v3"

        # =========================================================
        # Helper: GET request
        # =========================================================
        def get_data(url, params):
            response = requests.get(url, params=params)

            if response.status_code != 200:
                return []

            try:
                data = response.json()
            except ValueError:
                return []

            # FMP sometimes returns API errors as JSON
            if isinstance(data, dict):
                error_text = str(data).lower()

                if any(
                    word in error_text
                    for word in [
                        "upgrade",
                        "premium",
                        "subscription",
                        "not available",
                        "invalid api key"
                    ]
                ):
                    return []

            return data


        # =========================================================
        # DAILY PRICE DATA
        # =========================================================

        price_response = get_data(
            f"{BASE_URL}/historical-price-eod/full",
            {
                "symbol": ticker,
                "apikey": API_KEY
            }
        )

        # Fallback endpoint
        if not price_response:
            price_response = get_data(
                f"{BASE_URL}/historical-price-eod/light",
                {
                    "symbol": ticker,
                    "apikey": API_KEY
                }
            )

        prices = pd.DataFrame(price_response)

        # Some endpoints use "price" instead of "close"
        if "price" in prices.columns and "close" not in prices.columns:
            prices = prices.rename(columns={"price": "close"})

        if not {"date", "close"}.issubset(prices.columns):
            raise ValueError(
                f"Could not retrieve daily price data for {ticker}."
            )

        # Volume may not exist on every endpoint/tier
        if "volume" not in prices.columns:
            prices["volume"] = np.nan

        prices = prices[
            [
                "date",
                "close",
                "volume"
            ]
        ].copy()

        prices["date"] = pd.to_datetime(prices["date"])

        prices = (
            prices
            .sort_values("date")
            .tail(daily_limit)
            .reset_index(drop=True)
        )


        # =========================================================
        # DAILY FEATURES
        # =========================================================

        # 1-day return
        prices["return_1d"] = (
            prices["close"].pct_change()
        )

        # 5-day momentum
        prices["return_5d"] = (
            prices["close"]
            / prices["close"].shift(5)
            - 1
        )

        # 20-day momentum
        prices["return_20d"] = (
            prices["close"]
            / prices["close"].shift(20)
            - 1
        )

        # 20-day realized volatility
        prices["volatility_20d"] = (
            prices["return_1d"]
            .rolling(20)
            .std()
        )

        # Relative volume:
        # compare today's volume with PREVIOUS 20-day average
        prices["avg_volume_20d"] = (
            prices["volume"]
            .shift(1)
            .rolling(20)
            .mean()
        )

        prices["relative_volume"] = (
            prices["volume"]
            / prices["avg_volume_20d"]
        )


        # =========================================================
        # QUARTERLY FUNDAMENTALS
        # =========================================================

        quarterly_response = get_data(
            f"{v3_base_url}/income-statement/{ticker}",
            {
                "period": "quarter",
                "limit": quarterly_limit,
                "apikey": API_KEY
            }
        )

        quarterly = pd.DataFrame(quarterly_response)

        required_quarterly = {
            "date",
            "revenue",
            "netIncome"
        }

        if required_quarterly.issubset(quarterly.columns):

            keep_cols = [
                "date",
                "revenue",
                "netIncome"
            ]

            # Keep publication timestamps when available
            for col in ["fillingDate", "acceptedDate"]:
                if col in quarterly.columns:
                    keep_cols.append(col)

            quarterly = quarterly[keep_cols].copy()

            quarterly["date"] = pd.to_datetime(
                quarterly["date"]
            )

            # -----------------------------------------
            # IMPORTANT:
            # use the date investors could know the data
            # -----------------------------------------
            if "acceptedDate" in quarterly.columns:

                quarterly["available_date"] = pd.to_datetime(
                    quarterly["acceptedDate"],
                    errors="coerce"
                )

            elif "fillingDate" in quarterly.columns:

                quarterly["available_date"] = pd.to_datetime(
                    quarterly["fillingDate"],
                    errors="coerce"
                )

            else:
                # Fallback only.
                # Less desirable because period-end date can create leakage.
                quarterly["available_date"] = quarterly["date"]

            quarterly = (
                quarterly
                .sort_values("date")
                .reset_index(drop=True)
            )

            # YoY quarterly revenue growth
            quarterly["quarterly_revenue_growth"] = (
                quarterly["revenue"]
                / quarterly["revenue"].shift(4)
                - 1
            )

            # Net profit margin
            quarterly["quarterly_net_margin"] = (
                quarterly["netIncome"]
                / quarterly["revenue"]
            )

            quarterly = quarterly[
                [
                    "available_date",
                    "quarterly_revenue_growth",
                    "quarterly_net_margin"
                ]
            ].dropna(
                subset=["available_date"]
            )

            quarterly = quarterly.sort_values(
                "available_date"
            )

        else:
            quarterly = pd.DataFrame()


        # =========================================================
        # ANNUAL FUNDAMENTALS
        # =========================================================

        annual_income_response = get_data(
            f"{v3_base_url}/income-statement/{ticker}",
            {
                "period": "annual",
                "limit": annual_limit,
                "apikey": API_KEY
            }
        )

        annual_balance_response = get_data(
            f"{v3_base_url}/balance-sheet-statement/{ticker}",
            {
                "period": "annual",
                "limit": annual_limit,
                "apikey": API_KEY
            }
        )

        annual_income = pd.DataFrame(
            annual_income_response
        )

        annual_balance = pd.DataFrame(
            annual_balance_response
        )

        required_income = {"date", "eps"}
        required_balance = {
            "date",
            "totalAssets",
            "totalDebt"
        }

        if (
            required_income.issubset(annual_income.columns)
            and
            required_balance.issubset(annual_balance.columns)
        ):

            income_cols = ["date", "eps"]

            for col in ["fillingDate", "acceptedDate"]:
                if col in annual_income.columns:
                    income_cols.append(col)

            annual_income = annual_income[
                income_cols
            ].copy()

            annual_balance = annual_balance[
                [
                    "date",
                    "totalAssets",
                    "totalDebt"
                ]
            ].copy()

            annual = annual_income.merge(
                annual_balance,
                on="date",
                how="inner"
            )

            annual["date"] = pd.to_datetime(
                annual["date"]
            )

            if "acceptedDate" in annual.columns:

                annual["available_date"] = pd.to_datetime(
                    annual["acceptedDate"],
                    errors="coerce"
                )

            elif "fillingDate" in annual.columns:

                annual["available_date"] = pd.to_datetime(
                    annual["fillingDate"],
                    errors="coerce"
                )

            else:
                annual["available_date"] = annual["date"]

            annual = (
                annual
                .sort_values("date")
                .reset_index(drop=True)
            )

            # Annual EPS growth
            annual["annual_eps_growth"] = (
                annual["eps"].pct_change()
            )

            # Debt / assets
            annual["annual_debt_to_assets"] = (
                annual["totalDebt"]
                / annual["totalAssets"]
            )

            annual = annual[
                [
                    "available_date",
                    "annual_eps_growth",
                    "annual_debt_to_assets"
                ]
            ].dropna(
                subset=["available_date"]
            )

            annual = annual.sort_values(
                "available_date"
            )

        else:
            annual = pd.DataFrame()


        # =========================================================
        # BUILD DAILY PANEL
        # =========================================================

        df = prices[
            [
                "date",
                "close",
                "volume",
                "return_1d",
                "return_5d",
                "return_20d",
                "volatility_20d",
                "relative_volume"
            ]
        ].copy()


        # =========================================================
        # MERGE QUARTERLY DATA
        # =========================================================

        if not quarterly.empty:

            df = pd.merge_asof(
                df.sort_values("date"),
                quarterly.sort_values("available_date"),
                left_on="date",
                right_on="available_date",
                direction="backward"
            )

            df = df.drop(
                columns=["available_date"]
            )

        else:

            df["quarterly_revenue_growth"] = np.nan
            df["quarterly_net_margin"] = np.nan


        # =========================================================
        # MERGE ANNUAL DATA
        # =========================================================

        if not annual.empty:

            df = pd.merge_asof(
                df.sort_values("date"),
                annual.sort_values("available_date"),
                left_on="date",
                right_on="available_date",
                direction="backward"
            )

            df = df.drop(
                columns=["available_date"]
            )

        else:

            df["annual_eps_growth"] = np.nan
            df["annual_debt_to_assets"] = np.nan


        # =========================================================
        # IDENTIFIER
        # =========================================================

        df["ticker"] = ticker


        # =========================================================
        # FINAL OUTPUT
        # =========================================================

        columns = [
            "date",
            "ticker",

            # Raw market data
            "close",
            "volume",

            # Market features
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "relative_volume",

            # Quarterly fundamental features
            "quarterly_revenue_growth",
            "quarterly_net_margin",

            # Annual fundamental features
            "annual_eps_growth",
            "annual_debt_to_assets"
        ]

        return (
            df[columns]
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

    # 1. Fetch each ticker
    frames = []

    for ticker in tickers:
        frames.append(
            get_fundamental_data(ticker)
        )

    # 2. Build panel
    raw_df = pd.concat(
        frames,
        ignore_index=True
    )

    raw_df = raw_df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    # 3. Create prediction target
    raw_df["future_5d_return"] = (
        raw_df.groupby("ticker")["close"].shift(-5)
        / raw_df["close"]
        - 1
    )

    # 4. Inspect
    print(raw_df.shape)
    print(raw_df.columns)
    raw_df.head()

    #clean data

    # =========================================================
    # 1. CLEAN / SORT RAW DATA
    # =========================================================

    raw_df = raw_df.copy()

    raw_df["date"] = pd.to_datetime(raw_df["date"])

    raw_df = (
        raw_df
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    print("RAW DATA")
    print("Shape:", raw_df.shape)
    print("Tickers:", raw_df["ticker"].nunique())
    print("Date range:", raw_df["date"].min(), "to", raw_df["date"].max())

    #starting dataframe
    df = raw_df.dropna(axis=1, how="all")
    df_mod = df.copy()
    df_mod["date"] = pd.to_datetime(df_mod["date"])
    df_mod = df_mod.sort_values(["date","ticker"])
    df_mod = df_mod.reset_index(drop=True)
    df_mod["return_1d"] = df_mod.groupby("ticker")["close"].transform(lambda x: x/x.shift(1)-1)

    #add features
    feature_df = build_features(df_mod,window=20)
    feature_df = feature_df = (feature_df.sort_values(["date", "ticker"]).reset_index(drop=True))

    #inspect it
    print(feature_df.columns)
    print(feature_df.shape)
    #feature_df


    #make target
    feature_df["future_5d_return"] = feature_df.groupby("ticker")["close"].shift(-5)/feature_df["close"] - 1

    #temporal splits
    train_end = pd.to_datetime("2022-12-31")
    discovery_end = pd.to_datetime("2023-12-31")
    validation_end = pd.to_datetime("2024-12-31")
    train_df, discovery_df, validation_df, test_df = temporal_split(feature_df, train_end, discovery_end, validation_end)

    print("training: ", train_df.shape, train_df["date"].min(), train_df["date"].max())
    print("discovery: ", discovery_df.shape, discovery_df["date"].min(), discovery_df["date"].max())
    print("validation: ", validation_df.shape, validation_df["date"].min(), validation_df["date"].max())
    print("test: ", test_df.shape, test_df["date"].min(), test_df["date"].max())

    #create feature columns that we will use to generate candidates
    feature_columns = ["momentum_20d", "volatility_20d", "relative_volume", "vol_rolling_avg"]

    #check whether all of these features are in the feature dataframe
    missing_columns = []
    for feature in feature_columns:
        if feature not in feature_df.columns:
            missing_columns.append(feature)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are not in feature_df")
    print("train_df, discovery_df, validation_df, test_df created...")

    feature_columns = [

        "return_5d",

        "momentum_20d",

        "volatility_20d",

        "relative_volume"

    ]

    required_columns = [
        "date",
        "ticker",
        "future_5d_return"
    ] + feature_columns

    train_df = train_df[required_columns].copy()
    discovery_df = discovery_df[required_columns].copy()
    validation_df = validation_df[required_columns].copy()
    test_df = test_df[required_columns].copy()

    return train_df, discovery_df, validation_df, test_df'''



import os
import numpy as np
import pandas as pd
import requests

from quant_alpha_lab.features import build_features
#from quant_alpha_lab.splits import temporal_split

API_KEY = os.environ["FMP_API_KEY"]
def prepare_phase_iii_data():

    # =========================================================
    # CONFIG
    # =========================================================

    # API_KEY = os.environ["FMP_API_KEY"]
    API_KEY = os.environ["FMP_API_KEY"]
    BASE_URL = "https://financialmodelingprep.com/stable"

    DAILY_LIMIT = 1500

    tickers = [
        # Technology
        "AAPL", "MSFT", "NVDA", "AMD", "INTC",
        "CSCO", "IBM", "ORCL", "ADBE", "CRM",
        "TXN", "AMAT", "MU", "HPQ", "DELL"
    ]
    '''
    # Communication / Media
    "GOOGL", "GOOG", "META", "NFLX", "DIS",
    "CMCSA", "T", "VZ", "TMUS"

    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "LOW", "MCD",
    "SBUX", "NKE", "GM", "F", "TGT",
    "TJX", "ROST", "MAR", "HLT", "BKNG",

    # Consumer Staples
    "WMT", "COST", "PG", "KO", "PEP",
    "PM", "MO", "CL", "MDLZ", "KMB",
    "GIS", "K", "KHC", "HSY", "SJM",

    # Financials
    "JPM", "BAC", "WFC", "GS", "MS",
    "C", "AXP", "SCHW", "USB", "PNC",
    "V", "MA", "BK", "AIG", "MET",
    "PRU", "ALL", "CB", "CME", "ICE",

    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "ABT",
    "TMO", "MDT", "AMGN", "GILD", "CVS",
    "CI", "BMY", "SYK", "BDX", "ZTS",
    "HUM", "REGN", "VRTX", "ISRG", "EW",

    # Industrials
    "CAT", "DE", "HON", "UPS", "LMT",
    "NOC", "UNP", "FDX", "BA", "MMM",
    "GE", "EMR", "ITW", "GD", "CSX",
    "NSC", "ETN", "PH", "CMI", "ROK",

    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    "MPC", "OXY", "PSX", "VLO", "HAL",
    "BKR", "DVN", "FANG",

    # Utilities
    "NEE", "SO", "DUK", "AEP", "EXC",
    "D", "SRE", "XEL", "ED", "PEG",

    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM",
    "DOW", "NUE", "VMC", "MLM", "PPG",

    # Real Estate
    "PLD", "AMT", "O", "SPG", "PSA",
    "CCI", "WELL", "DLR", "VICI",

    # Misc large caps
    "ADP", "PAYX", "ACN", "CTAS", "FAST",
    "GWW", "RSG", "WM", "EA", "EBAY"
    ]'''

    # =========================================================
    # HELPER: SAFE REQUEST
    # =========================================================

    def get_data(url, params):

        try:
            response = requests.get(
                url,
                params=params,
                timeout=30
            )
        except requests.RequestException:
            return []

        if response.status_code != 200:
            return []

        try:
            data = response.json()
        except ValueError:
            return []

        # FMP sometimes returns an error message as JSON
        if isinstance(data, dict):

            error_text = str(data).lower()

            error_words = [
                "upgrade",
                "premium",
                "subscription",
                "not available",
                "invalid api key",
                "limit reached"
            ]

            if any(word in error_text for word in error_words):
                return []

        return data


    # =========================================================
    # FETCH ONE TICKER
    # =========================================================

    def get_market_data(ticker):

        # Try full endpoint first
        data = get_data(
            f"{BASE_URL}/historical-price-eod/full",
            {
                "symbol": ticker,
                "apikey": API_KEY
            }
        )

        # Fall back to light endpoint
        if not data:
            data = get_data(
                f"{BASE_URL}/historical-price-eod/light",
                {
                    "symbol": ticker,
                    "apikey": API_KEY
                }
            )

        if not data:
            raise ValueError(
                f"Could not retrieve price data for {ticker}"
            )

        prices = pd.DataFrame(data)

        # Some FMP responses call closing price "price"
        if "price" in prices.columns and "close" not in prices.columns:
            prices = prices.rename(
                columns={"price": "close"}
            )

        if not {"date", "close"}.issubset(prices.columns):
            raise ValueError(
                f"Missing date/close data for {ticker}"
            )

        # Relative volume requires volume.
        # Don't silently create NaN volume and pretend the ticker is usable.
        if "volume" not in prices.columns:
            raise ValueError(
                f"No volume data available for {ticker}"
            )

        prices = prices[
            ["date", "close", "volume"]
        ].copy()

        prices["date"] = pd.to_datetime(
            prices["date"]
        )

        prices = (
            prices
            .sort_values("date")
            .tail(DAILY_LIMIT)
            .reset_index(drop=True)
        )

        prices["ticker"] = ticker

        return prices


    # =========================================================
    # FETCH UNIVERSE
    # =========================================================

    frames = []

    successful_tickers = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, start=1):

        print(
            f"Fetching {ticker} "
            f"({i}/{len(tickers)})..."
        )

        try:

            ticker_df = get_market_data(ticker)

            frames.append(ticker_df)
            successful_tickers.append(ticker)

        except Exception as e:

            failed_tickers.append(
                (ticker, str(e))
            )

            print(
                f"  Skipping {ticker}: {e}"
            )


    if not frames:
        raise ValueError(
            "No ticker data could be retrieved."
        )


    # =========================================================
    # BUILD RAW PANEL
    # =========================================================

    raw_df = pd.concat(
        frames,
        ignore_index=True
    )

    raw_df = (
        raw_df
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    print("\n==============================")
    print("DATA RETRIEVAL SUMMARY")
    print("==============================")

    print(
        f"Successful tickers: "
        f"{len(successful_tickers)}"
    )

    print(
        f"Failed tickers: "
        f"{len(failed_tickers)}"
    )

    if failed_tickers:

        print("\nFailed:")
        for ticker, error in failed_tickers:
            print(f"{ticker}: {error}")

    print("\nRAW DATA")
    print("Shape:", raw_df.shape)
    print(
        "Tickers:",
        raw_df["ticker"].nunique()
    )
    print(
        "Date range:",
        raw_df["date"].min(),
        "to",
        raw_df["date"].max()
    )


    # =========================================================
    # BASE RETURNS
    # =========================================================

    raw_df["return_1d"] = (
        raw_df
        .groupby("ticker")["close"]
        .pct_change()
    )

    raw_df["return_5d"] = (

        raw_df.groupby("ticker")["close"]

        .transform(lambda x: x / x.shift(5) - 1)

    )

    # =========================================================
    # ENGINEER FEATURES
    # =========================================================

    feature_df = build_features(
        raw_df,
        window=20
    )

    feature_df = (
        feature_df
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


    # =========================================================
    # TARGET
    # =========================================================

    feature_df["future_5d_return"] = (
        feature_df
        .groupby("ticker")["close"]
        .shift(-5)
        / feature_df["close"]
        - 1
    )


    # =========================================================
    # TEMPORAL SPLITS
    # =========================================================

    train_end = pd.to_datetime(
        "2022-12-31"
    )

    discovery_end = pd.to_datetime(
        "2023-12-31"
    )

    validation_end = pd.to_datetime(
        "2024-12-31"
    )

    (
        train_df,
        discovery_df,
        validation_df,
        test_df
    ) = temporal_split(
        feature_df,
        train_end,
        discovery_end,
        validation_end
    )


    # =========================================================
    # FEATURES AVAILABLE TO AGENTS
    # =========================================================

    feature_columns = [
        "return_5d",
        "momentum_20d",
        "volatility_20d",
        "relative_volume"
    ]

    required_columns = [
        "date",
        "ticker",
        "future_5d_return"
    ] + feature_columns


    # Verify engineered features exist
    missing_columns = [
        col
        for col in required_columns
        if col not in feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )


    # =========================================================
    # REMOVE ROWS THAT CANNOT BE EVALUATED
    # =========================================================

    train_df = (
        train_df[required_columns]
        .dropna()
        .copy()
    )

    discovery_df = (
        discovery_df[required_columns]
        .dropna()
        .copy()
    )

    validation_df = (
        validation_df[required_columns]
        .dropna()
        .copy()
    )

    test_df = (
        test_df[required_columns]
        .dropna()
        .copy()
    )


    # =========================================================
    # FINAL INSPECTION
    # =========================================================

    print("\n==============================")
    print("TEMPORAL SPLITS")
    print("==============================")

    for name, df in [
        ("training", train_df),
        ("discovery", discovery_df),
        ("validation", validation_df),
        ("test", test_df)
    ]:

        print(
            name,
            df.shape,
            df["date"].min(),
            df["date"].max(),
            "tickers:",
            df["ticker"].nunique()
        )


    print(
        "\ntrain_df, discovery_df, "
        "validation_df, test_df created..."
    )


    # SAME OUTPUT AS BEFORE
    return (
        train_df,
        discovery_df,
        validation_df,
        test_df
    )


