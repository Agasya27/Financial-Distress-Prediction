"""
Step 2 data pipeline for multimodal financial distress prediction.

Builds three modalities from the filtered ECL dataset:
1) Tabular financial features from SEC XBRL companyfacts API.
2) MD&A text files from 10-K filings using sec-edgar-downloader.
3) Interfirm graph from SEC SIC codes.

This script runs in test mode first (first 500 unique CIKs by default),
saves intermediate outputs every 100 CIKs, and logs failures.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from torch_geometric.data import Data
from tqdm import tqdm


SEC_BASE = "https://data.sec.gov"


RAW_XBRL_TAGS = {
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_liabilities": ["Liabilities"],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ],
    "net_income": ["NetIncomeLoss"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "ebit": ["OperatingIncomeLoss"],
    "depreciation": ["DepreciationDepletionAndAmortization"],
    "short_term_debt": ["DebtCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent"],
    "inventory": ["InventoryNet"],
    "accounts_payable": ["AccountsPayableCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "equity": ["StockholdersEquity"],
}

# 15 raw + 10 derived = 25 node features for graph branch.
FEATURE_COLUMNS_25 = [
    "assets",
    "current_assets",
    "current_liabilities",
    "total_liabilities",
    "revenue",
    "net_income",
    "cash",
    "ebit",
    "depreciation",
    "short_term_debt",
    "long_term_debt",
    "inventory",
    "accounts_payable",
    "retained_earnings",
    "equity",
    "current_ratio",
    "debt_ratio",
    "return_on_assets",
    "return_on_equity",
    "working_capital",
    "asset_turnover",
    "debt_to_equity",
    "cash_ratio",
    "operating_margin",
    "inventory_turnover",
]


@dataclass
class PipelinePaths:
    """All file-system paths used in Step 2."""

    ecl_csv: str
    raw_dir: str
    processed_dir: str
    mda_dir: str
    downloads_dir: str
    failed_log: str
    tabular_csv: str
    graph_pt: str
    tabular_partial_csv: str
    state_json: str
    sic_partial_json: str


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML config.

    Args:
        config_path: Path to config file.

    Returns:
        Parsed config dictionary.
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_paths(config: dict[str, Any]) -> PipelinePaths:
    """Resolve all paths used in Step 2.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        PipelinePaths container.
    """
    data_cfg = config["data"]
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    mda_dir = os.path.join(raw_dir, "mda_texts")
    downloads_dir = os.path.join(raw_dir, "sec_downloads")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(mda_dir, exist_ok=True)
    os.makedirs(downloads_dir, exist_ok=True)

    return PipelinePaths(
        ecl_csv=data_cfg["ecl_csv_path"],
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        mda_dir=mda_dir,
        downloads_dir=downloads_dir,
        failed_log=os.path.join(raw_dir, "failed_ciks.log"),
        tabular_csv=os.path.join(processed_dir, "tabular.csv"),
        graph_pt=os.path.join(processed_dir, "graph.pt"),
        tabular_partial_csv=os.path.join(processed_dir, "tabular_partial.csv"),
        state_json=os.path.join(processed_dir, "pipeline_state.json"),
        sic_partial_json=os.path.join(processed_dir, "sic_map_partial.json"),
    )


def log_failure(paths: PipelinePaths, cik: int, stage: str, message: str) -> None:
    """Append a failure entry to the log file.

    Args:
        paths: Pipeline paths.
        cik: Company CIK.
        stage: Stage name (xbrl/mda/sic/etc).
        message: Error or context message.
    """
    with open(paths.failed_log, "a", encoding="utf-8") as f:
        f.write(f"cik={cik}\tstage={stage}\tmessage={message}\n")


def normalize_ecl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the expected ECL columns.

    Args:
        df: Raw ECL dataframe.

    Returns:
        Normalized dataframe with standard column names.
    """
    rename_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    if "" in df.columns:
        df = df.drop(columns=[""])
    return df


def load_and_filter_ecl(ecl_csv_path: str) -> pd.DataFrame:
    """Load ECL CSV and apply required filtering.

    Args:
        ecl_csv_path: Path to `ECL (1).csv`.

    Returns:
        Filtered dataframe.
    """
    df = pd.read_csv(ecl_csv_path, low_memory=False)
    df = normalize_ecl_columns(df)

    required = ["cik", "datadate", "filing_date", "can_label", "qualified", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in ECL CSV: {missing}")

    df["cik"] = pd.to_numeric(df["cik"], errors="coerce")
    df = df.dropna(subset=["cik"]).copy()
    df["cik"] = df["cik"].astype(int)

    df["datadate"] = pd.to_datetime(df["datadate"], dayfirst=True, errors="coerce")
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["label"] = df["label"].astype(bool)

    df = df[df["can_label"] == True]  # noqa: E712
    df = df[df["qualified"] == "Yes"]
    df = df.dropna(subset=["datadate", "filing_date"]).reset_index(drop=True)
    return df


def get_sec_session(user_agent: str) -> requests.Session:
    """Create a session for SEC requests.

    Args:
        user_agent: SEC-compliant user agent string.

    Returns:
        Configured requests session.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return s


def fetch_companyfacts(session: requests.Session, cik: int) -> Optional[dict[str, Any]]:
    """Fetch SEC companyfacts JSON for a CIK.

    Args:
        session: Configured SEC requests session.
        cik: Integer CIK.

    Returns:
        Parsed JSON dict or None.
    """
    padded = str(cik).zfill(10)
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{padded}.json"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except requests.RequestException:
        return None


def fetch_submissions(session: requests.Session, cik: int) -> Optional[dict[str, Any]]:
    """Fetch SEC submissions JSON for a CIK.

    Args:
        session: Configured SEC requests session.
        cik: Integer CIK.

    Returns:
        Parsed JSON dict or None.
    """
    padded = str(cik).zfill(10)
    url = f"{SEC_BASE}/submissions/CIK{padded}.json"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except requests.RequestException:
        return None


def pick_tag_value_by_year(companyfacts: dict[str, Any], tag_candidates: list[str], year: int) -> float:
    """Extract annual value for a candidate tag by fiscal year.

    Args:
        companyfacts: SEC companyfacts payload.
        tag_candidates: Candidate us-gaap tags (without prefix).
        year: Year to match.

    Returns:
        Numeric value or NaN.
    """
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    for tag in tag_candidates:
        concept = us_gaap.get(tag, {})
        units = concept.get("units", {})
        for unit_list in units.values():
            for item in unit_list:
                if item.get("fy") == int(year) and item.get("form") in {"10-K", "10-K/A"}:
                    val = item.get("val")
                    if isinstance(val, (int, float)):
                        return float(val)
    return float("nan")


def compute_derived_ratios(row: pd.Series) -> dict[str, float]:
    """Compute derived financial ratios requested in Step 2.

    Args:
        row: Series containing raw financial values.

    Returns:
        Dict of 10 derived ratio features.
    """
    def div(a: float, b: float) -> float:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return float("nan")
        return float(a) / float(b)

    assets = row["assets"]
    current_assets = row["current_assets"]
    current_liabilities = row["current_liabilities"]
    total_liabilities = row["total_liabilities"]
    revenue = row["revenue"]
    net_income = row["net_income"]
    equity = row["equity"]
    cash = row["cash"]
    ebit = row["ebit"]
    inventory = row["inventory"]

    return {
        "current_ratio": div(current_assets, current_liabilities),
        "debt_ratio": div(total_liabilities, assets),
        "return_on_assets": div(net_income, assets),
        "return_on_equity": div(net_income, equity),
        "working_capital": (
            float(current_assets - current_liabilities)
            if not pd.isna(current_assets) and not pd.isna(current_liabilities)
            else float("nan")
        ),
        "asset_turnover": div(revenue, assets),
        "debt_to_equity": div(total_liabilities, equity),
        "cash_ratio": div(cash, current_liabilities),
        "operating_margin": div(ebit, revenue),
        "inventory_turnover": div(revenue, inventory) if (not pd.isna(inventory) and inventory > 0) else float("nan"),
    }


def extract_item7_text_from_html(html: str) -> str:
    """Extract Item 7 (MD&A) section text from filing HTML.

    Args:
        html: Full filing HTML content.

    Returns:
        Extracted MD&A text (empty string if not found).
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    start_match = re.search(r"(?i)\bitem\s*7\b", text)
    if not start_match:
        return ""
    start = start_match.start()

    end_match = re.search(r"(?i)\bitem\s*7a\b|\bitem\s*8\b", text[start:])
    if end_match:
        end = start + end_match.start()
    else:
        end = min(len(text), start + 150000)

    section = text[start:end].strip()
    return section


def build_xbrl_rows_for_cik(
    cik_rows: pd.DataFrame,
    companyfacts: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build tabular feature rows for all filings of one CIK.

    Args:
        cik_rows: Rows for a single CIK from filtered ECL dataframe.
        companyfacts: SEC companyfacts payload for this CIK.

    Returns:
        List of records with raw + derived financial features.
    """
    records: list[dict[str, Any]] = []
    for _, r in cik_rows.iterrows():
        year = int(pd.Timestamp(r["datadate"]).year)
        record: dict[str, Any] = {
            "cik": int(r["cik"]),
            "company": r.get("company", ""),
            "datadate": pd.Timestamp(r["datadate"]).strftime("%Y-%m-%d"),
            "filing_date": pd.Timestamp(r["filing_date"]).strftime("%Y-%m-%d"),
            "label": bool(r["label"]),
        }

        for feature_name, tags in RAW_XBRL_TAGS.items():
            if companyfacts is None:
                record[feature_name] = float("nan")
            else:
                record[feature_name] = pick_tag_value_by_year(companyfacts, tags, year)

        record.update(compute_derived_ratios(pd.Series(record)))
        records.append(record)
    return records


def save_intermediate(
    paths: PipelinePaths,
    tabular_records: list[dict[str, Any]],
    processed_ciks: list[int],
    sic_map: dict[int, str],
) -> None:
    """Persist intermediate progress every N CIKs.

    Args:
        paths: Pipeline path object.
        tabular_records: Accumulated tabular rows.
        processed_ciks: Successfully iterated CIK list.
        sic_map: Partial SIC mapping.
    """
    pd.DataFrame(tabular_records).to_csv(paths.tabular_partial_csv, index=False)
    with open(paths.state_json, "w", encoding="utf-8") as f:
        json.dump({"processed_ciks": processed_ciks, "count": len(processed_ciks)}, f, indent=2)
    with open(paths.sic_partial_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sic_map.items()}, f, indent=2)


def process_mda_for_cik(
    cik: int,
    cik_rows: pd.DataFrame,
    paths: PipelinePaths,
    dl: Downloader,
) -> None:
    """Download 10-Ks and save Item 7 text files for one CIK.

    Args:
        cik: Current CIK.
        cik_rows: Filtered rows for the CIK.
        paths: Pipeline paths.
        dl: sec-edgar-downloader client.
    """
    years = sorted({int(pd.Timestamp(d).year) for d in cik_rows["filing_date"]})
    cik_str = str(int(cik))
    cik_download_dir = os.path.join(paths.downloads_dir, f"CIK{str(int(cik)).zfill(10)}")
    os.makedirs(cik_download_dir, exist_ok=True)

    for year in years:
        try:
            dl.get("10-K", cik_str, limit=1, after=str(year - 1), before=str(year + 1))
        except Exception:
            continue

    html_files: list[str] = []
    for root, _dirs, files in os.walk(cik_download_dir):
        for fn in files:
            if fn.lower().endswith((".htm", ".html")):
                html_files.append(os.path.join(root, fn))

    html_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if not html_files:
        for _, row in cik_rows.iterrows():
            out_name = f"{int(cik)}_{pd.Timestamp(row['datadate']).strftime('%Y-%m-%d')}.txt"
            with open(os.path.join(paths.mda_dir, out_name), "w", encoding="utf-8") as f:
                f.write("")
        return

    for _, row in cik_rows.iterrows():
        datadate_str = pd.Timestamp(row["datadate"]).strftime("%Y-%m-%d")
        out_name = f"{int(cik)}_{datadate_str}.txt"
        out_path = os.path.join(paths.mda_dir, out_name)
        if os.path.exists(out_path):
            continue
        content = ""
        for html_path in html_files[:3]:
            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    section = extract_item7_text_from_html(f.read())
                if section:
                    content = section
                    break
            except OSError:
                continue
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)


def build_graph(tabular_df: pd.DataFrame, sic_map: dict[int, str], out_path: str) -> None:
    """Build SIC-based graph with 25-dim node features and save as PyG Data.

    Args:
        tabular_df: Final tabular dataframe (contains CIK + 25 features).
        sic_map: Mapping cik -> 2-digit SIC string.
        out_path: Output path for graph.pt.
    """
    # Aggregate node features by CIK then fill NaN->0 only for graph.
    node_df = tabular_df.groupby("cik")[FEATURE_COLUMNS_25].mean().reset_index()
    node_df[FEATURE_COLUMNS_25] = node_df[FEATURE_COLUMNS_25].fillna(0.0)

    cik_list = node_df["cik"].astype(int).tolist()
    cik_to_node = {cik: idx for idx, cik in enumerate(cik_list)}
    node_to_cik = {idx: cik for cik, idx in cik_to_node.items()}

    edges: list[list[int]] = [[], []]
    sic_groups: dict[str, list[int]] = {}
    for cik in cik_list:
        sic2 = sic_map.get(cik, "")
        if sic2:
            sic_groups.setdefault(sic2, []).append(cik)

    for _, ciks in sic_groups.items():
        for i in range(len(ciks)):
            for j in range(i + 1, len(ciks)):
                a = cik_to_node[ciks[i]]
                b = cik_to_node[ciks[j]]
                edges[0].extend([a, b])
                edges[1].extend([b, a])

    x = torch.tensor(node_df[FEATURE_COLUMNS_25].values, dtype=torch.float32)  # (num_nodes, 25)
    if len(edges[0]) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)  # (2, 0)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long)  # (2, num_edges)

    graph = Data(x=x, edge_index=edge_index)
    graph.node_to_cik = node_to_cik
    torch.save(graph, out_path)


def run_pipeline(config_path: str = "config.yaml") -> None:
    """Execute Step 2 pipeline.

    Args:
        config_path: YAML config path.
    """
    config = load_config(config_path)
    data_cfg = config["data"]

    paths = build_paths(config)
    ecl_df = load_and_filter_ecl(paths.ecl_csv)

    # Test-batch-first policy
    unique_ciks = ecl_df["cik"].drop_duplicates().tolist()
    if not data_cfg.get("run_full_ciks", False):
        unique_ciks = unique_ciks[: int(data_cfg.get("test_batch_cik_limit", 500))]
        print(f"[INFO] TEST BATCH MODE: processing first {len(unique_ciks)} unique CIKs.")
    else:
        print(f"[INFO] FULL MODE: processing all {len(unique_ciks)} unique CIKs.")

    save_every = int(data_cfg.get("intermediate_save_every_ciks", 100))
    sec_user_agent = data_cfg.get("sec_user_agent", "RCOEM Student agasyabutolia@gmail.com")
    session = get_sec_session(sec_user_agent)

    # sec-edgar-downloader client
    dl = Downloader("RCOEM Student", "agasyabutolia@gmail.com", download_folder=paths.downloads_dir)

    tabular_records: list[dict[str, Any]] = []
    sic_map: dict[int, str] = {}
    processed_ciks: list[int] = []

    cik_loop = tqdm(unique_ciks, desc="CIK pipeline", unit="cik")
    for idx, cik in enumerate(cik_loop, start=1):
        cik_rows = ecl_df[ecl_df["cik"] == cik].copy()

        # XBRL fetch
        companyfacts = fetch_companyfacts(session, cik)
        if companyfacts is None:
            log_failure(paths, cik, "xbrl", "companyfacts unavailable")
        tabular_records.extend(build_xbrl_rows_for_cik(cik_rows, companyfacts))
        time.sleep(0.11)

        # SIC fetch
        submissions = fetch_submissions(session, cik)
        if submissions is None:
            sic_map[int(cik)] = ""
            log_failure(paths, cik, "sic", "submissions unavailable")
        else:
            sic_raw = submissions.get("sic", "")
            sic_map[int(cik)] = str(sic_raw)[:2] if sic_raw else ""
        time.sleep(0.11)

        # MD&A download/extract
        try:
            process_mda_for_cik(cik, cik_rows, paths, dl)
        except Exception as exc:
            log_failure(paths, cik, "mda", str(exc))

        processed_ciks.append(int(cik))

        if idx % save_every == 0:
            save_intermediate(paths, tabular_records, processed_ciks, sic_map)
            cik_loop.set_postfix({"saved": idx})

    # Final save
    tabular_df = pd.DataFrame(tabular_records)
    # Keep exact output contract requested in fresh prompt.
    keep_cols = ["cik", "datadate"] + FEATURE_COLUMNS_25 + ["label"]
    for c in keep_cols:
        if c not in tabular_df.columns:
            tabular_df[c] = np.nan
    tabular_df = tabular_df[keep_cols]
    tabular_df.to_csv(paths.tabular_csv, index=False)

    build_graph(tabular_df, sic_map, paths.graph_pt)
    save_intermediate(paths, tabular_records, processed_ciks, sic_map)

    print(f"[DONE] tabular saved: {paths.tabular_csv}")
    print(f"[DONE] graph saved:   {paths.graph_pt}")
    print(f"[DONE] failures log: {paths.failed_log}")
    print(f"[DONE] processed CIKs: {len(processed_ciks)}")


if __name__ == "__main__":
    run_pipeline()
