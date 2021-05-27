from ast import Sub
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


def get_daily_data():
    """
    Loads spreads and prices, calculates spreads and returns dataframe on daily
    basis with no re-indexing.
    """

    df_spreads = (
        pd.read_csv("Liq5YCDS.delim", sep="\t", index_col=0, parse_dates=["date"])
        .set_index(["ticker", "date"])
        .unstack("ticker")[["spread5y"]]
    )
    df_returns = np.log(df_spreads / df_spreads.shift())
    df_returns.columns = df_returns.columns.set_levels(["r_spread"], level=0)
    df_spreads = pd.concat([df_spreads, df_returns], axis=1)

    df_prices = (
        pd.read_csv("df_prices_new.csv", parse_dates=["date"])
        .set_index(["ticker", "date"])[["adj_close"]]
        .unstack("ticker")
        .ffill()
    )

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
    df_data = df_data.loc["2018-01-03":"2020-04-20"]

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


def get_contemp_resids(df_data: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function to return contemporaneous equity and cds
    residuals.
    """

    df_eq_errors, _ = get_errors(df_data, model="OLS")
    df_cds_errors, _ = get_errors(
        df_data, model="RLM", formula="r_spread ~ r_equity + r_index + 1"
    )

    df_resid = pd.concat(
        [df_cds_errors.resid, df_eq_errors.resid.groupby("ticker").shift()], axis=1
    )

    df_resid.columns = ["cds_resid", "eq_resid"]

    return df_resid.dropna()


def get_predicted_resids(df_resid: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates regression coefficients and predicted residuals using
    discounted least squares for exponentially weighted and boxcar
    windows of 4 to 60 weeks.
    """

    df_resid = df_resid.unstack("ticker")
    win_lengths = np.arange(6, 47, 2)

    col_list = []
    for w in win_lengths:
        for s in df_resid.columns.levels[1]:
            df_pair = df_resid.loc[:, [("cds_resid", s), ("eq_resid", s)]]
            df_pair.columns = ["cds_resid", "eq_resid"]
            for win_type, df_win in {
                "exp_wm": df_pair.ewm(alpha=1 / w),
                "boxcar": df_pair.rolling(window=2 * w, min_periods=4),
            }.items():
                df_cov = df_win.cov()
                df_mean = df_win.mean()

                s_var = df_cov["eq_resid"].xs("eq_resid", level=1)
                s_var.name = ("var_x", win_type, f"t_{w:02d}", s)

                s_cov = df_cov["eq_resid"].xs("cds_resid", level=1)
                s_cov.name = ("cov_xy", win_type, f"t_{w:02d}", s)

                s_beta_1 = s_cov / s_var
                s_beta_1.name = ("beta_1", win_type, f"t_{w:02d}", s)

                s_beta_0 = s_beta_1 * df_mean.eq_resid - df_mean.cds_resid
                s_beta_0.name = ("beta_0", win_type, f"t_{w:02d}", s)

                s_resid_sq = (
                    s_beta_0 + df_pair.eq_resid * s_beta_1 - df_pair.cds_resid
                ).pow(2)
                s_resid_sq.name = ("resid_sq", win_type, f"t_{w:02d}", s)

                s_error_sq = (
                    df_pair.cds_resid.loc[~s_resid_sq.isna()]
                    - df_pair.cds_resid.loc[~s_resid_sq.isna()].mean()
                ).pow(2)
                s_error_sq.name = ("error_sq", win_type, f"t_{w:02d}", s)

                col_list.extend(
                    [s_resid_sq, s_error_sq, s_beta_0, s_beta_1, s_var, s_cov]
                )

    df_betas = pd.concat(col_list, axis=1)
    df_betas.columns.names = ["stat", "win_type", "win_length", "ticker"]
    df_betas = df_betas.stack("ticker")
    df_betas = df_betas.swaplevel("stat", "win_type", axis=1)

    return df_betas.sort_index(axis=1)


def get_ticker_rsq(df_resid_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates R-squared for each ticker for each combination of window
    methodology and length.
    """

    ticker_rsq = []
    for t in df_resid_pred.index.levels[1]:
        df = df_resid_pred.loc[df_resid_pred.index.get_level_values("ticker") == t]
        for w in df_resid_pred.columns.levels[0]:
            rsq = 1 - df[w].resid_sq.sum() / df[w].error_sq.sum()
            rsq.name = (t, w)
            ticker_rsq.append(rsq)

    df_ticker_rsq = pd.concat(ticker_rsq, axis=1).sort_index(axis=1)
    df_ticker_rsq.columns.names = ["ticker", "win_type"]

    return df_ticker_rsq


# =============================================================================
# Charts
# =============================================================================


COLORS = colors.qualitative.T10


def make_lines_chart(
    series: pd.Series,
    title: str = "5-Year CDS Spread",
) -> go.Figure:
    fig = go.Figure()

    for t in series.index.levels[1]:
        fig.add_trace(
            go.Scatter(
                x=series.index.levels[0],
                y=series.loc[series.index.get_level_values("ticker") == t],
                name=t,
                line=dict(width=1),
            )
        )

    fig.update(layout_title=title)

    return fig


IS_labels = [
    ("obs", lambda x: f"{x:>7d}"),
    ("min:max", lambda x: f"{x[0]:>0.4f}:{x[1]:>0.3f}"),
    ("mean", lambda x: f"{x:>7.4f}"),
    ("std", lambda x: f"{x:>7.4f}"),
    ("skewness", lambda x: f"{x:>7.4f}"),
    ("kurtosis", lambda x: f"{x:>7.4f}"),
]


def get_moments_annotation(
    s: pd.Series,
    xref: str,
    yref: str,
    x: float,
    y: float,
    xanchor: str,
    title: str,
    labels: List,
) -> go.layout.Annotation:
    """Calculates summary statistics for a series and returns and
    Annotation object.
    """
    moments = list(stats.describe(s.to_numpy()))
    moments[3] = np.sqrt(moments[3])

    sharpe = s.mean() / s.std()

    return go.layout.Annotation(
        text=(
            ("<br>").join(
                [f"{k[0]:<9}{k[1](moments[i])}" for i, k in enumerate(labels)]
            )
        ),
        align="left",
        showarrow=False,
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        bordercolor="black",
        borderwidth=0.5,
        borderpad=2,
        bgcolor="white",
        xanchor=xanchor,
        yanchor="top",
    )


def make_overview_chart(
    series: pd.DataFrame, title: str, subtitle_base: str = "Residuals"
) -> go.Figure:

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{subtitle_base} Distribution",
            f"Q/Q Plot",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    # Returns Distribution
    series_cuts = pd.cut(series, 100).value_counts().sort_index()
    midpoints = series_cuts.index.map(lambda interval: interval.right).to_numpy()
    norm_dist = stats.norm.pdf(midpoints, loc=series.mean(), scale=series.std())

    fig.add_trace(
        go.Bar(
            x=[interval.mid for interval in series_cuts.index],
            y=series_cuts / series_cuts.sum(),
            name="pct. of returns",
            marker=dict(color=COLORS[0]),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[interval.mid for interval in series_cuts.index],
            y=norm_dist / norm_dist.sum(),
            name="normal",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=1,
        col=1,
    )

    # Q/Q Data
    returns_norm = ((series - series.mean()) / series.std()).sort_values()
    norm_dist = pd.Series(
        list(map(stats.norm.ppf, np.linspace(0.001, 0.999, len(series)))),
        name="normal",
    )

    fig.append_trace(
        go.Scatter(
            x=norm_dist,
            y=returns_norm,
            name="resid. norm.",
            mode="markers",
            marker=dict(color=COLORS[0], size=3),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=norm_dist,
            y=norm_dist,
            name="norm.",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=1,
        col=2,
    )

    fig.add_annotation(
        get_moments_annotation(
            series.dropna(),
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.45,
            xanchor="right",
            title="Returns",
            labels=IS_labels,
        ),
        font=dict(size=6, family="Courier New, monospace"),
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    fig.update_layout(
        title_text=(
            f"{title}<br>"
            f"{series.index.levels[0].min().strftime('%Y-%m-%d')}"
            f" - {series.index.levels[0].max().strftime('%Y-%m-%d')}"
        ),
        showlegend=False,
        height=400,
        width=800,
        font=dict(size=10),
        margin=dict(l=50, r=50, b=50, t=100),
        yaxis=dict(tickformat="0.3f"),
        yaxis2=dict(tickformat="0.1f"),
        xaxis2=dict(tickformat="0.1f"),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    fig.update_annotations(font=dict(size=10))

    return fig


def make_residual_scatter(df_resid: pd.DataFrame, n_trend_obs: int = 200) -> go.Figure:
    df_resid.eq_resid = df_resid.eq_resid.shift()
    df_resid = df_resid.dropna()

    ols_result = stats.linregress(df_resid.eq_resid, df_resid.cds_resid)
    # print(ols_result.slope ** 2)

    fig = px.scatter(
        y=df_resid.cds_resid,
        x=df_resid.eq_resid,
        title="Equity Return Residual vs. CDS Return Residual",
        labels=dict(y="cds residual (n)", x="equity residual (n-1)"),
    )

    fig.add_scatter(
        x=[-0.3, 0.3],
        y=[ols_result.intercept + ols_result.slope * x for x in [-0.3, 0.3]],
        mode="lines",
    )

    fig.update_layout(showlegend=False)

    return fig


def make_rsq_comparison(df_resid_pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for w in df_resid_pred.columns.levels[0]:
        rsq = 1 - df_resid_pred[w].resid_sq.sum() / df_resid_pred[w].error_sq.sum()
        fig.add_trace(go.Bar(x=rsq.index, y=rsq.values, name=w))

    fig.update_layout(title="Comparision of R-squared: Window Type by Length")

    return fig


def make_rsq_ticker_comparison(df_resid_pred: pd.DataFrame) -> go.Figure:
    tickers = {
        "BA": (1, 1),
        "C": (1, 2),
        "DD": (2, 1),
        "F": (2, 2),
        "GE": (3, 1),
        "JPM": (3, 2),
        "LOW": (4, 1),
        "LUV": (4, 2),
        "MAR": (5, 1),
        "T": (5, 2),
        "WFC": (6, 1),
        "XOM": (6, 2),
    }

    fig = make_subplots(
        rows=6,
        cols=2,
        subplot_titles=list(tickers.keys()),
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    for t, (r, c) in tickers.items():
        for j, w in enumerate(df_resid_pred.columns.levels[1]):
            fig.add_trace(
                go.Bar(
                    x=df_resid_pred.index,
                    y=df_resid_pred[t][w],
                    name=w,
                    marker=dict(color=COLORS[j]),
                    showlegend=True if t == "BA" else False,
                    legendgroup=w,
                ),
                row=r,
                col=c,
            )

    fig.update_layout(title="Comparision of R-squared: Window Type by Length")

    fig.update_layout(width=1000, height=1600)
    return fig
