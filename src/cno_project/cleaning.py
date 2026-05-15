"""Cleaning functions for raw CNO project datasets."""

import pandas as pd

from cno_project.config import START_DATE


def clean_cno_export_prices(df):
    """Clean monthly coconut oil export prices."""
    output = df[["Date", "Price"]].copy()

    output["date"] = pd.to_datetime(output["Date"], errors="coerce")
    output["cno_price"] = pd.to_numeric(output["Price"], errors="coerce")

    output = output[["date", "cno_price"]].dropna()
    output["date"] = output["date"].dt.to_period("M").dt.to_timestamp()
    output = output[output["date"] >= START_DATE]

    return (
        output.groupby("date", as_index=False)["cno_price"]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )


def clean_coconut_production(df):
    """Clean quarterly coconut production and forward-fill to monthly."""
    output = df[["Year", "Quarter", "Volume"]].copy()

    output["Year"] = output["Year"].ffill()
    output["Volume"] = pd.to_numeric(output["Volume"], errors="coerce")

    quarter_months = {
        "Quarter1": 1,
        "Quarter2": 4,
        "Quarter3": 7,
        "Quarter4": 10,
    }

    output["month"] = output["Quarter"].map(quarter_months)

    output["date"] = pd.to_datetime(
        output["Year"].astype("Int64").astype(str)
        + "-"
        + output["month"].astype("Int64").astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )

    output = (
        output[["date", "Volume"]]
        .rename(columns={"Volume": "coconut_prod"})
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    monthly_dates = pd.date_range(
        output["date"].min(),
        output["date"].max(),
        freq="MS",
    )

    monthly = pd.DataFrame({"date": monthly_dates})
    monthly = monthly.merge(output, on="date", how="left")
    monthly["coconut_prod"] = monthly["coconut_prod"].ffill()
    monthly = monthly[monthly["date"] >= START_DATE]

    return monthly.reset_index(drop=True)


def clean_copra_farmgate(df):
    """Clean copra farmgate data from wide format into monthly format."""
    output = df.copy()

    output = output.melt(
        var_name="date_raw",
        value_name="copra_farmgate",
    )

    output["date_raw"] = (
        output["date_raw"]
        .astype(str)
        .str.replace(" Copra Corriente", "", regex=False)
        .str.strip()
    )

    output["date"] = pd.to_datetime(
        output["date_raw"],
        format="%Y %B",
        errors="coerce",
    )

    output["copra_farmgate"] = pd.to_numeric(
        output["copra_farmgate"],
        errors="coerce",
    )

    output = output[["date", "copra_farmgate"]].dropna()
    output = output[output["date"] >= START_DATE]

    return output.sort_values("date").reset_index(drop=True)


def clean_cpo_futures(df):
    """Clean crude palm oil futures data."""
    output = df[["Date", "Price"]].copy()

    output["date"] = pd.to_datetime(output["Date"], errors="coerce")
    output["cpo_price"] = (
        output["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    output["cpo_price"] = pd.to_numeric(output["cpo_price"], errors="coerce")

    output = output[["date", "cpo_price"]].dropna()
    output = output[output["date"] >= START_DATE]

    return output.sort_values("date").reset_index(drop=True)


def clean_oni_monthly(df):
    """Clean monthly ONI data."""
    output = df[["date", "oni"]].copy()

    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["oni"] = pd.to_numeric(output["oni"], errors="coerce")

    output = output.dropna()
    output = output[output["date"] >= START_DATE]

    return output.sort_values("date").reset_index(drop=True)


def clean_usd_php(df):
    """Clean USD/PHP exchange rate data."""
    output = df[["Date", "Price"]].copy()

    output["date"] = pd.to_datetime(output["Date"], errors="coerce")
    output["usd_php"] = pd.to_numeric(output["Price"], errors="coerce")

    output = output[["date", "usd_php"]].dropna()
    output = output[output["date"] >= START_DATE]

    return output.sort_values("date").reset_index(drop=True)


def clean_all(raw):
    """Clean all raw datasets."""
    return {
        "cno": clean_cno_export_prices(raw["cno"]),
        "production": clean_coconut_production(raw["production"]),
        "copra": clean_copra_farmgate(raw["copra"]),
        "cpo": clean_cpo_futures(raw["cpo"]),
        "oni": clean_oni_monthly(raw["oni"]),
        "usd_php": clean_usd_php(raw["usd_php"]),
    }