import argparse
import io
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(content.decode("utf-8")))


def _download_month_csv(year: int, region: str, month: str) -> Optional[pd.DataFrame]:
    headers = {"User-Agent": "Mozilla/5.0"}
    base_url = f"https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{year}_"
    url = f"{base_url}{region}_{month}.csv"
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        return _read_csv_bytes(resp.content)
    print(f"Skipping {year}-{month} (csv): HTTP {resp.status_code}")

    zip_urls = [
        f"https://nemweb.com.au/Reports/Archive/PriceAndDemand/PRICE_AND_DEMAND_{year}_{region}_{month}.zip",
        f"https://nemweb.com.au/REPORTS/ARCHIVE/PRICEANDDEMAND/PRICE_AND_DEMAND_{year}_{region}_{month}.zip",
    ]
    for zip_url in zip_urls:
        try:
            resp = requests.get(zip_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"Skipping {year}-{month} (zip): HTTP {resp.status_code}")
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    print(f"Skipping {year}-{month} (zip): no CSV in archive")
                    continue
                with zf.open(csv_names[0]) as fh:
                    return _read_csv_bytes(fh.read())
        except Exception as exc:
            print(f"Skipping {year}-{month} (zip): {exc}")
    return None


def download_aemo_data(year: int, region: str = "VIC1") -> pd.DataFrame:
    """Download AEMO NEM Price and Demand data for a given year and region.

    Returns hourly resampled data with columns: timestamp, load, region, region_type.
    """
    frames: List[pd.DataFrame] = []
    months = [f"{m:02d}" for m in range(1, 13)]

    print(f"Downloading AEMO {region} for {year}...")
    for m in months:
        try:
            df = _download_month_csv(year, region, m)
            if df is not None:
                frames.append(df)
        except Exception as exc:
            print(f"Skipping {year}-{m}: {exc}")

    if not frames:
        raise RuntimeError(
            f"No data downloaded for {year} {region}. "
            f"Check network/firewall and AEMO/NEMWeb access."
        )

    full_df = pd.concat(frames, ignore_index=True)
    if "SETTLEMENTDATE" not in full_df.columns or "TOTALDEMAND" not in full_df.columns:
        raise ValueError("Missing expected columns SETTLEMENTDATE/TOTALDEMAND")

    full_df["SETTLEMENTDATE"] = pd.to_datetime(full_df["SETTLEMENTDATE"], errors="coerce")
    full_df = full_df.dropna(subset=["SETTLEMENTDATE"]).sort_values("SETTLEMENTDATE")

    clean_df = full_df[["SETTLEMENTDATE", "TOTALDEMAND"]].rename(
        columns={"SETTLEMENTDATE": "timestamp", "TOTALDEMAND": "load"}
    )

    clean_df = (
        clean_df.set_index("timestamp")
        .resample("H")
        .mean()
        .reset_index()
    )

    clean_df["region"] = region
    clean_df["region_type"] = "real_grid"
    return clean_df


def parse_years(values: Iterable[str]) -> List[int]:
    years = []
    for v in values:
        try:
            years.append(int(v))
        except ValueError as exc:
            raise ValueError(f"Invalid year: {v}") from exc
    if not years:
        raise ValueError("No valid years provided")
    return years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", default=["2023", "2024"], help="Years to download")
    parser.add_argument("--region", default="VIC1", help="Region code, e.g. VIC1")
    parser.add_argument(
        "--out",
        default="data/external/aemo_vic/load.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    frames = [download_aemo_data(y, region=args.region) for y in years]
    df = pd.concat(frames, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"AEMO dataset ready: {df.shape}, saved to {out_path}")


if __name__ == "__main__":
    main()
