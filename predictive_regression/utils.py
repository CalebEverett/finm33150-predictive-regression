import gzip
import hashlib
import os
from typing import Dict, List
from urllib.request import urlretrieve

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from canvasapi import Canvas
import numpy as np
import numpy as np
import pandas as pd
from patsy import dmatrices
import plotly.graph_objects as go
import plotly.express as px
from plotly import colors
from plotly.subplots import make_subplots
import quandl
import requests
from scipy import stats
import statsmodels.api as sm
from tqdm.notebook import tqdm

# =============================================================================
# Credentials
# =============================================================================

quandl.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")


# =============================================================================
# Canvas
# =============================================================================


def download_files(filename_frag: str):
    """Downloads files from Canvas with `filename_frag` in filename."""

    url = os.getenv("CANVAS_URL")
    token = os.getenv("CANVAS_TOKEN")

    course_id = 33395
    canvas = Canvas(url, token)
    course = canvas.get_course(course_id)

    for f in course.get_files():
        if filename_frag in f.filename:
            print(f.filename, f.id)
            file = course.get_file(f.id)
            file.download(file.filename)


# =============================================================================
# Reading Data
# =============================================================================


def get_trade_data(pair: str, year: str, path: str = "accumulation_opportunity/data"):
    """Reads local gzipped trade data file and return dataframe."""

    dtypes = {
        "PriceMillionths": int,
        "Side": int,
        "SizeBillionths": int,
        "timestamp_utc_nanoseconds": int,
    }

    filename = f"trades_narrow_{pair}_{year}.delim.gz"
    delimiter = {"2018": "|", "2021": "\t"}[year]

    with gzip.open(f"{path}/{filename}") as f:
        df = pd.read_csv(f, delimiter=delimiter, usecols=dtypes.keys(), dtype=dtypes)

    df.timestamp_utc_nanoseconds = pd.to_datetime(df.timestamp_utc_nanoseconds)

    return df.set_index("timestamp_utc_nanoseconds")


# =============================================================================
# Price Data
# =============================================================================


def get_table(dataset_code: str, database_code: str = "ZACKS"):
    """Downloads Zacks fundamental table from export api to local zip file."""

    url = (
        f"https://www.quandl.com/api/v3/datatables/{database_code}/{dataset_code}.json"
    )
    r = requests.get(
        url, params={"api_key": os.getenv("QUANDL_API_KEY"), "qopts.export": "true"}
    )
    data = r.json()
    urlretrieve(
        data["datatable_bulk_download"]["file"]["link"],
        f"zacks_{dataset_code.lower()}.zip",
    )


def load_table_files(table_filenames: Dict):
    """Loads Zacks fundamentals tables from csv files."""

    dfs = []
    for v in tqdm(table_filenames.values()):
        dfs.append(pd.read_csv(v, low_memory=False))

    return dfs


def get_hash(string: str) -> str:
    """Returns md5 hash of string."""

    return hashlib.md5(str(string).encode()).hexdigest()


def fetch_ticker(
    dataset_code: str, query_params: Dict = None, database_code: str = "EOD"
):
    """Fetches price data for a single ticker."""

    url = f"https://www.quandl.com/api/v3/datasets/{database_code}/{dataset_code}.json"

    params = dict(api_key=os.getenv("QUANDL_API_KEY"))
    if query_params is not None:
        params = dict(**params, **query_params)

    r = requests.get(url, params=params)

    dataset = r.json()["dataset"]
    df = pd.DataFrame(
        dataset["data"], columns=[c.lower() for c in dataset["column_names"]]
    )
    df["ticker"] = dataset["dataset_code"]

    return df.sort_values("date")


def fetch_all_tickers(tickers: List, query_params: Dict) -> pd.DataFrame:
    """Fetches price data from Quandl for each ticker in provide list and
    returns a dataframe of them concatenated together.
    """

    df_prices = pd.DataFrame()
    for t in tqdm(tickers):
        try:
            df = fetch_ticker(t, query_params)
            df_prices = pd.concat([df_prices, df])
        except:
            print(f"Couldn't get prices for {t}.")

    not_missing_data = (
        df_prices.set_index(["ticker", "date"])[["adj_close"]]
        .unstack("date")
        .isna()
        .sum(axis=1)
        == 0
    )

    df_prices = df_prices[
        df_prices.ticker.isin(not_missing_data[not_missing_data].index)
    ]

    return df_prices.set_index(["ticker", "date"])


# =============================================================================
# Download Preprocessed Files from S3
# =============================================================================


def upload_s3_file(filename: str):
    """Uploads file to S3. Requires credentials with write permissions to exist
    as environment variables.
    """

    client = boto3.client("s3")
    client.upload_file(filename, "finm33150", filename)


def download_s3_file(filename: str):
    """Downloads file from read only S3 bucket."""

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    client.download_file("finm33150", filename, filename)


# =============================================================================
# Prepare Data
# =============================================================================


def get_data(date_range: pd.date_range) -> pd.DataFrame:
    """
    Loads spreads and prices, calculates spreads and returns dataframe.
    """

    # load spreads data
    df_spreads = (
        pd.read_csv("Liq5YCDS.delim", sep="\t", index_col=0, parse_dates=["date"])
        .set_index(["ticker", "date"])
        .unstack("ticker")
        .loc[date_range, ["spread5y"]]
    )
    df_returns = np.log(df_spreads / df_spreads.shift())
    df_returns.columns = df_returns.columns.set_levels(["r_spread"], level=0)
    df_spreads = pd.concat([df_spreads, df_returns], axis=1)

    # load prices
    df_prices = (
        pd.read_csv("df_prices_new.csv", parse_dates=["date"])
        .set_index(["ticker", "date"])[["adj_close"]]
        .unstack("ticker")
        .reindex(date_range)
        .ffill()
    )
    assert len(df_prices) == len(df_spreads)

    df_prices.columns = df_prices.columns.set_names(["series", "ticker"])
    df_returns = np.log(df_prices / df_prices.shift())
    df_returns.columns = df_returns.columns.set_levels(["r_equity"], level=0)
    df_prices = pd.concat([df_prices, df_returns], axis=1)
    df_data = pd.concat([df_prices, df_spreads], axis=1).iloc[1:]
    df_data = df_data.stack("ticker")
    df_data.index = df_data.index.set_names(["date", "ticker"])

    # Excludes return of subject security
    df_data["r_index"] = (
        df_data.groupby("date")["r_spread"].transform("sum") - df_data["r_spread"]
    ) / (df_data.groupby("date")["r_spread"].transform("count") - 1)

    df_data["r_spy"] = df_data.index.get_level_values("date").map(
        df_data[df_data.index.get_level_values("ticker") == "SPY"]
        .reset_index()
        .set_index("date")
        .r_equity
    )

    df_data = df_data[df_data.index.get_level_values("ticker") != "SPY"]
    df_data.index = df_data.index.remove_unused_levels()

    return df_data


# =============================================================================
# Regression
# =============================================================================


def get_errors(
    df_data: pd.DataFrame,
    model: str = "OLS",
    penalty: str = "Huber",
    weeks_in: int = 16,
    weeks_out: int = 1,
    formula: str = "r_equity ~ r_spy + 1",
):

    errors_list = []
    for day in df_data.index.levels[0][: -(weeks_in + weeks_out - 1)]:
        df_in = df_data.loc[day : day + pd.DateOffset(weeks=weeks_in - 1)]
        df_out = df_data.loc[
            day
            + pd.DateOffset(weeks=weeks_in) : day
            + pd.DateOffset(weeks=weeks_in + weeks_out - 1)
        ]

        for ticker in df_in.index.levels[1]:
            y_in, X_in = dmatrices(
                formula,
                data=df_in.loc[df_in.index.get_level_values("ticker") == ticker],
                return_type="dataframe",
            )

            if model == "OLS":
                res = sm.OLS(y_in, X_in).fit()
            else:
                if penalty == "Tukey":
                    res = sm.RLM(y_in, X_in, M=sm.robust.norms.TukeyBiweight()).fit()
                else:
                    res = sm.RLM(y_in, X_in).fit()

            y_out, X_out = dmatrices(
                formula,
                data=df_out.loc[df_out.index.get_level_values("ticker") == ticker],
                return_type="dataframe",
            )
            df_e = (res.predict(X_out) - y_out[formula.split(" ~ ")[0]]).to_frame(
                name="resid"
            )
            df_e["scale"] = res.scale
            df_e["sresid"] = df_e.resid / df_e.scale
            df_e["model_date"] = df_in.reset_index().date.min()

            df_e["distance"] = df_e.groupby(["date"]).ngroup() + 1
            errors_list.append(df_e)

    return pd.concat(errors_list), res.summary()


# =============================================================================
# Charts
# =============================================================================


COLORS = colors.qualitative.T10


def make_histograms(dfs_error):

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Residuals", "Scaled Residuals"],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    c = 0
    for name, df_e in dfs_error.items():
        c += 1
        fig.add_trace(
            go.Histogram(
                x=df_e.resid,
                histnorm="percent",
                name=name,
                marker_color=COLORS[c],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=df_e.sresid,
                histnorm="percent",
                name=name,
                showlegend=False,
                marker_color=COLORS[c],
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title_text=("Residuals Histograms"),
        height=800,
        font=dict(size=10),
        margin=dict(l=50, r=10, b=40, t=90),
        barmode="overlay",
        yaxis_title="% of all total obs",
        yaxis2_title="% of all total obs",
        xaxis_title="error",
        xaxis2_title="scaled error",
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    return fig


def make_cum_error_chart(errors: Dict, scaled: bool = False):

    fig = go.Figure()

    for model, df in errors.items():
        df_cum = get_cum_errors(df, scaled=scaled)
        fig.add_trace(
            go.Scatter(
                x=df_cum.index,
                y=df_cum.cummean,
                line=dict(width=2),
                name=model,
            ),
        )

    fig.update_layout(
        title_text=("Cumulative Mean Error by Quantile"),
        showlegend=True,
        font=dict(size=10),
        margin=dict(l=50, r=10, b=40, t=90),
        xaxis_title=f"{'scaled ' if scaled else ''}cumulative error quantile",
        yaxis_title=f"{'scaled ' if scaled else ''}error",
        hovermode="x",
    )

    return fig


def make_efficiency_chart(errors: Dict, scaled: bool = False):

    efficiency = (
        get_cum_errors(list(errors.values())[0], scaled=scaled).cummean
        / get_cum_errors(list(errors.values())[1], scaled=scaled).cummean
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=efficiency.index,
            y=efficiency.values,
            line=dict(width=2),
            name=(":").join(errors.keys()),
        ),
    )

    fig.update_layout(
        title_text=("Relative Efficiency by Error Quantile"),
        showlegend=True,
        font=dict(size=10),
        margin=dict(l=50, r=10, b=40, t=90),
        xaxis_title=f"{'scaled ' if scaled else ''}cumulative error quantile",
        yaxis_title=f"{'scaled ' if scaled else ''}error",
    )

    return fig


def make_residual_chart(errors: Dict, scaled: bool = False):

    resid_type = "sresid" if scaled else "resid"

    df_e0 = list(errors.values())[0]
    df_e1 = list(errors.values())[1]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_e0[resid_type].sort_values(),
            y=df_e1[resid_type].sort_values(),
            name=(":").join(errors.keys()),
            mode="markers",
        ),
    )

    fig.update_layout(
        title_text=("Residual-Residual Plot"),
        showlegend=True,
        font=dict(size=10),
        margin=dict(l=50, r=10, b=40, t=90),
        xaxis_title=f"{'scaled ' if scaled else ''}error: {list(errors.keys())[0]}",
        yaxis_title=f"{'scaled ' if scaled else ''}error {list(errors.keys())[1]}",
    )

    return fig
