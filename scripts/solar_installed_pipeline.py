from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    import geopandas as gpd
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires geopandas. Install with: pip install geopandas pyarrow openpyxl") from e


EXCLUDE_STATE_FIPS = {"60", "66", "69", "72", "78"}  # AS, GU, MP, PR, VI
STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH",
    "NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT",
    "VT","VA","WA","WV","WI","WY"
}


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_column(columns: list[str], keywords: list[str], required: bool = True) -> Optional[str]:
    norm = {normalize_name(c): c for c in columns}
    for kw in keywords:
        key = normalize_name(kw)
        if key in norm:
            return norm[key]
    for kw in keywords:
        key = normalize_name(kw)
        for n, orig in norm.items():
            if key in n:
                return orig
    if required:
        raise KeyError(f"Could not find column for {keywords}. Available columns: {columns}")
    return None


def detect_header_row_excel(xls: pd.ExcelFile, sheet_name: str, probes: list[str], max_rows: int = 10) -> int:
    preview = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=max_rows)
    norm_probes = [normalize_name(x) for x in probes]
    for i in range(len(preview)):
        row = [normalize_name(x) for x in preview.iloc[i].tolist() if pd.notna(x)]
        if any(any(p in cell for cell in row) for p in norm_probes):
            return i
    return 0


def read_excel_with_detected_header(path_or_buffer, sheet_name: str, probes: list[str]) -> pd.DataFrame:
    xls = pd.ExcelFile(path_or_buffer)
    header_row = detect_header_row_excel(xls, sheet_name, probes)
    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def download_to_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def fetch_eia_annual_zip(year: int, outdir: Path) -> Path:
    # EIA annual zip URL pattern is stable in practice but not formally guaranteed.
    url = f"https://www.eia.gov/electricity/data/eia860/xls/eia860{year}.zip"
    content = download_to_bytes(url)
    path = outdir / f"eia860_{year}.zip"
    path.write_bytes(content)
    return path


def fetch_eia_860m_xls(year: int, month: str, outdir: Path) -> Path:
    month = month.lower()
    url = f"https://www.eia.gov/electricity/data/eia860m/xls/{month}_generator{year}.xlsx"
    content = download_to_bytes(url)
    path = outdir / f"eia860m_{year}_{month}.xlsx"
    path.write_bytes(content)
    return path


def extract_zip(zip_path: Path, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(outdir)
    return list(outdir.rglob("*"))


def find_file(root: Path, contains: list[str], suffix: str) -> Path:
    candidates = []
    for p in root.rglob(f"*{suffix}"):
        name = p.name.lower()
        if all(token.lower() in name for token in contains):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No file found in {root} matching contains={contains}, suffix={suffix}")
    return sorted(candidates)[0]


def standardize_eia_operable(plant_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    plant_cols = plant_df.columns.tolist()
    gen_cols = gen_df.columns.tolist()

    plant_id_plant = find_column(plant_cols, ["plant code", "plant id", "plantcode"])
    state_col = find_column(plant_cols, ["state", "plant state", "state abbreviation"])
    county_col = find_column(plant_cols, ["county", "county name"], required=False)

    plant = plant_df[[plant_id_plant, state_col] + ([county_col] if county_col else [])].copy()
    plant.columns = ["plant_id_eia", "state", *(["county_name"] if county_col else [])]
    plant["plant_id_eia"] = pd.to_numeric(plant["plant_id_eia"], errors="coerce")
    plant["state"] = plant["state"].astype(str).str.upper()
    if county_col:
        plant["county_name"] = plant["county_name"].astype(str).str.strip()

    plant = plant[plant["state"].isin(STATE_ABBR)].dropna(subset=["plant_id_eia"]).drop_duplicates("plant_id_eia")

    plant_id_gen = find_column(gen_cols, ["plant code", "plant id", "plantcode"])
    status_col = find_column(gen_cols, ["status", "status code"], required=False)
    source_col = find_column(gen_cols, ["energy source code", "energy source 1", "energy source"])
    technology_col = find_column(gen_cols, ["technology"], required=False)
    prime_mover_col = find_column(gen_cols, ["prime mover"], required=False)
    cap_col = find_column(gen_cols, ["nameplate capacity", "nameplate capacity (mw)", "summer capacity (mw)", "winter capacity (mw)"])
    gen_id_col = find_column(gen_cols, ["generator id"], required=False)

    keep = [plant_id_gen, source_col, cap_col]
    for c in [status_col, technology_col, prime_mover_col, gen_id_col]:
        if c:
            keep.append(c)
    gen = gen_df[keep].copy()
    rename = {
        plant_id_gen: "plant_id_eia",
        source_col: "energy_source_code",
        cap_col: "capacity_mw",
    }
    if status_col:
        rename[status_col] = "status"
    if technology_col:
        rename[technology_col] = "technology"
    if prime_mover_col:
        rename[prime_mover_col] = "prime_mover"
    if gen_id_col:
        rename[gen_id_col] = "generator_id"
    gen = gen.rename(columns=rename)
    gen["plant_id_eia"] = pd.to_numeric(gen["plant_id_eia"], errors="coerce")
    gen["capacity_mw"] = pd.to_numeric(gen["capacity_mw"], errors="coerce")

    # Solar filter
    solar_mask = gen["energy_source_code"].astype(str).str.upper().str.contains("SUN", na=False)
    if "technology" in gen.columns:
        solar_mask |= gen["technology"].astype(str).str.contains("solar|pv|photovoltaic|thermal", case=False, na=False)
    if "prime_mover" in gen.columns:
        solar_mask |= gen["prime_mover"].astype(str).str.contains("PV|CP|ST", case=False, na=False)
    gen = gen[solar_mask].copy()

    merged = gen.merge(plant, on="plant_id_eia", how="left")
    merged = merged[merged["state"].isin(STATE_ABBR)].copy()
    merged = merged.dropna(subset=["capacity_mw"])
    return merged


def process_eia_annual_zip(zip_path: Path, year: int) -> pd.DataFrame:
    workdir = zip_path.parent / f"unzipped_{zip_path.stem}"
    if not workdir.exists():
        extract_zip(zip_path, workdir)

    plant_xlsx = find_file(workdir, ["plant", str(year)], ".xlsx")
    gen_xlsx = find_file(workdir, ["generator", str(year)], ".xlsx")

    plant_df = read_excel_with_detected_header(plant_xlsx, sheet_name=0, probes=["Plant Code", "State", "County"])
    xls = pd.ExcelFile(gen_xlsx)
    operable_sheet = next((s for s in xls.sheet_names if "operable" in s.lower()), xls.sheet_names[0])
    gen_df = read_excel_with_detected_header(gen_xlsx, sheet_name=operable_sheet, probes=["Plant Code", "Generator ID", "Energy Source Code"])
    annual = standardize_eia_operable(plant_df, gen_df)
    annual["year"] = year
    annual["source"] = f"EIA860_{year}"
    return annual


def process_eia_860m(monthly_xlsx: Path, plant_crosswalk_annual: pd.DataFrame, year: int) -> pd.DataFrame:
    xls = pd.ExcelFile(monthly_xlsx)
    operable_sheet = next((s for s in xls.sheet_names if "operable" in s.lower()), xls.sheet_names[0])
    gen_df = read_excel_with_detected_header(monthly_xlsx, sheet_name=operable_sheet, probes=["Plant Code", "Generator ID", "Energy Source Code"])

    # Reuse the annual plant crosswalk for county/state.
    annual_plant = plant_crosswalk_annual[["plant_id_eia", "state", "county_name"]].drop_duplicates("plant_id_eia")
    standardized = standardize_eia_operable(annual_plant.rename(columns={"plant_id_eia": "Plant Code", "state": "State", "county_name": "County"}), gen_df)
    standardized["year"] = year
    standardized["source"] = f"EIA860M_{year}"
    return standardized


def load_counties(county_path: Path) -> gpd.GeoDataFrame:
    counties = gpd.read_file(county_path)
    counties.columns = [c.lower() for c in counties.columns]
    geoid_col = find_column(counties.columns.tolist(), ["geoid", "county_fips", "fips"])
    name_col = find_column(counties.columns.tolist(), ["namelsad", "county_name", "name"], required=False)
    state_col = find_column(counties.columns.tolist(), ["statefp", "state_fips"], required=False)
    counties["county_fips"] = counties[geoid_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
    counties["county_name"] = counties[name_col].astype(str) if name_col else counties["county_fips"]
    if state_col:
        counties["state_fips"] = counties[state_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(2)
        counties = counties[~counties["state_fips"].isin(EXCLUDE_STATE_FIPS)].copy()
    counties = counties.to_crs(4326)
    return counties[["county_fips", "county_name", "geometry"]]


def attach_county_fips_by_name(df: pd.DataFrame, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    # Build a lightweight state-agnostic county name crosswalk. This is imperfect for duplicate county names,
    # so use only as a fallback if a lat/lon spatial join is not available.
    county_lookup = counties[["county_fips", "county_name"]].copy()
    county_lookup["county_key"] = county_lookup["county_name"].str.lower().str.replace(" county", "", regex=False).str.strip()

    out = df.copy()
    out["county_key"] = out["county_name"].astype(str).str.lower().str.replace(" county", "", regex=False).str.strip()
    out = out.merge(county_lookup.drop_duplicates("county_key"), on="county_key", how="left", suffixes=("", "_from_shape"))
    out["county_fips"] = out["county_fips"] if "county_fips" in out.columns else out["county_fips_from_shape"]
    out = out.drop(columns=[c for c in ["county_key", "county_fips_from_shape"] if c in out.columns])
    return out


def aggregate_capacity(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(by, dropna=False)["capacity_mw"]
        .sum()
        .reset_index(name="solar_mw")
        .sort_values(by)
    )
    return out


def choose_gem_column(columns: list[str], groups: list[list[str]], required: bool = True) -> Optional[str]:
    for group in groups:
        try:
            return find_column(columns, group, required=True)
        except Exception:
            continue
    if required:
        raise KeyError(f"Could not infer GEM column from choices={groups}. Columns={columns}")
    return None


def process_gem_csv(gem_csv: Path, counties: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
    gem = pd.read_csv(gem_csv, low_memory=False)
    gem.columns = [str(c).strip() for c in gem.columns]
    cols = gem.columns.tolist()

    country_col = choose_gem_column(cols, [["country", "country/area"]])
    status_col = choose_gem_column(cols, [["status"], ["phase status"]])
    cap_col = choose_gem_column(cols, [["capacity", "mwac"], ["capacity mwac"], ["capacity_mwac"], ["capacity", "mw"]])
    state_col = choose_gem_column(cols, [["state"], ["province/state"], ["subnational"], ["state/province"]], required=False)
    county_col = choose_gem_column(cols, [["county"], ["county/parish"]], required=False)
    year_col = choose_gem_column(cols, [["start year"], ["operating year"], ["commissioning year"], ["start date"]], required=False)
    lat_col = choose_gem_column(cols, [["latitude"], ["lat"]], required=False)
    lon_col = choose_gem_column(cols, [["longitude"], ["lon"]], required=False)
    project_col = choose_gem_column(cols, [["project name"], ["name"], ["project"]], required=False)

    out = gem.copy()
    out = out[out[country_col].astype(str).str.contains("United States|USA|US", case=False, na=False)].copy()
    out = out[out[status_col].astype(str).str.contains("operating", case=False, na=False)].copy()
    out["capacity_mw"] = pd.to_numeric(out[cap_col], errors="coerce")
    out = out.dropna(subset=["capacity_mw"])
    if state_col:
        out["state"] = out[state_col].astype(str).str.upper().str[:2]
    else:
        out["state"] = np.nan
    if county_col:
        out["county_name"] = out[county_col].astype(str)
    else:
        out["county_name"] = np.nan
    if year_col:
        out["operating_year"] = pd.to_numeric(out[year_col].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    else:
        out["operating_year"] = np.nan
    out["project_name"] = out[project_col].astype(str) if project_col else ""

    # Optional spatial county assignment if county_name is missing but coordinates exist.
    if counties is not None and out["county_name"].isna().all() and lat_col and lon_col:
        points = out.copy()
        points[lat_col] = pd.to_numeric(points[lat_col], errors="coerce")
        points[lon_col] = pd.to_numeric(points[lon_col], errors="coerce")
        points = points.dropna(subset=[lat_col, lon_col]).copy()
        gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points[lon_col], points[lat_col]), crs=4326)
        joined = gpd.sjoin(gdf, counties[["county_fips", "county_name", "geometry"]], how="left", predicate="within")
        out = pd.DataFrame(joined.drop(columns=[c for c in ["geometry", "index_right"] if c in joined.columns]))

    if counties is not None:
        out = attach_county_fips_by_name(out, counties)

    out["source"] = "GEM"
    return out


def state_year_from_eia_and_gem(eia_df: pd.DataFrame, gem_df: pd.DataFrame, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eia_all = []
    gem_all = []
    for y in years:
        eia_y = eia_df[eia_df["year"] == y].copy()
        eia_y["source"] = eia_y["source"].astype(str)
        eia_all.append(aggregate_capacity(eia_y, ["year", "state"]))

        gem_y = gem_df.copy()
        gem_y = gem_y[(gem_y["operating_year"].isna()) | (gem_y["operating_year"] <= y)].copy()
        gem_y["year"] = y
        gem_all.append(aggregate_capacity(gem_y, ["year", "state"]))

    eia_state = pd.concat(eia_all, ignore_index=True).rename(columns={"solar_mw": "solar_mw_eia"})
    gem_state = pd.concat(gem_all, ignore_index=True).rename(columns={"solar_mw": "solar_mw_gem"})
    compare = eia_state.merge(gem_state, on=["year", "state"], how="outer")
    compare["abs_diff_mw"] = (compare["solar_mw_eia"] - compare["solar_mw_gem"]).abs()
    compare["pct_diff_vs_eia"] = compare["abs_diff_mw"] / compare["solar_mw_eia"].replace(0, np.nan)
    return eia_state, gem_state, compare


def write_outputs(outdir: Path, eia_2021: pd.DataFrame, eia_2025: pd.DataFrame, gem: pd.DataFrame, counties: gpd.GeoDataFrame, years: list[int]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # County/state summaries from EIA
    eia_all = pd.concat([eia_2021, eia_2025], ignore_index=True)
    eia_all = attach_county_fips_by_name(eia_all, counties)
    eia_county = aggregate_capacity(eia_all.dropna(subset=["county_fips"]), ["year", "state", "county_fips", "county_name"])
    eia_state = aggregate_capacity(eia_all, ["year", "state"])

    gem_county_pieces = []
    gem_state_pieces = []
    for y in years:
        gy = gem[(gem["operating_year"].isna()) | (gem["operating_year"] <= y)].copy()
        gy["year"] = y
        if "county_fips" not in gy.columns:
            gy["county_fips"] = np.nan
        gem_county_pieces.append(aggregate_capacity(gy.dropna(subset=["county_fips"]), ["year", "state", "county_fips", "county_name"]))
        gem_state_pieces.append(aggregate_capacity(gy, ["year", "state"]))
    gem_county = pd.concat(gem_county_pieces, ignore_index=True)
    gem_state = pd.concat(gem_state_pieces, ignore_index=True)

    eia_state_cmp, gem_state_cmp, compare = state_year_from_eia_and_gem(eia_all, gem, years)

    eia_all.to_csv(outdir / "eia_plant_level_solar.csv", index=False)
    eia_county.to_csv(outdir / "eia_solar_by_county.csv", index=False)
    eia_state.to_csv(outdir / "eia_solar_by_state.csv", index=False)
    gem.to_csv(outdir / "gem_plant_level_solar.csv", index=False)
    gem_county.to_csv(outdir / "gem_solar_by_county.csv", index=False)
    gem_state.to_csv(outdir / "gem_solar_by_state.csv", index=False)
    compare.to_csv(outdir / "eia_vs_gem_state_compare.csv", index=False)

    with open(outdir / "notes.json", "w") as f:
        json.dump(
            {
                "years": years,
                "rule_for_gem": "Include Operating projects with operating_year <= target year.",
                "rule_for_eia_2025": "Prefer annual EIA-860 final 2024 for location crosswalk + EIA-860M December 2025 for preliminary 2025 inventory.",
                "recommended_choice": "If EIA and GEM state totals are close, use GEM for the 2025 plant-level supplement and keep EIA 2021 as baseline."
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Process EIA/GEM solar plant data to county/state tables for 2021 and 2025 (or 2024 fallback).")
    parser.add_argument("--county-boundaries", required=True, help="County boundaries file (geojson/gpkg/shp)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--eia-annual-2021-zip", help="Local path to eia860 2021 zip. If omitted, the script will try to download it.")
    parser.add_argument("--eia-annual-crosswalk-zip", help="Local path to annual EIA zip used for plant county/state crosswalk. Use 2024 if you are processing 2025 monthly EIA-860M.")
    parser.add_argument("--eia-860m-xlsx", help="Local path to 2025 EIA-860M xlsx. If omitted and --download-2025-month is set, the script will try to download it.")
    parser.add_argument("--download-2025-month", choices=[
        "january","february","march","april","may","june","july","august","september","october","november","december"
    ], help="If set, download that 2025 EIA-860M month from EIA.")
    parser.add_argument("--gem-csv", required=True, help="Local GEM solar download CSV")
    parser.add_argument("--year2", type=int, default=2025, help="Second comparison year: 2025 by default, or set to 2024 if you want a final annual fallback")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    counties = load_counties(Path(args.county_boundaries))

    # EIA annual 2021
    eia_2021_zip = Path(args.eia_annual_2021_zip) if args.eia_annual_2021_zip else fetch_eia_annual_zip(2021, outdir)
    eia_2021 = process_eia_annual_zip(eia_2021_zip, 2021)

    # EIA year2
    if args.year2 == 2024 and args.eia_annual_crosswalk_zip:
        eia_year2 = process_eia_annual_zip(Path(args.eia_annual_crosswalk_zip), 2024)
    else:
        crosswalk_zip = Path(args.eia_annual_crosswalk_zip) if args.eia_annual_crosswalk_zip else fetch_eia_annual_zip(2024, outdir)
        crosswalk_annual = process_eia_annual_zip(crosswalk_zip, 2024)
        if args.eia_860m_xlsx:
            monthly_xlsx = Path(args.eia_860m_xlsx)
        else:
            if not args.download_2025_month:
                raise SystemExit("Provide --eia-860m-xlsx or use --download-2025-month for 2025")
            monthly_xlsx = fetch_eia_860m_xls(2025, args.download_2025_month, outdir)
        eia_year2 = process_eia_860m(monthly_xlsx, crosswalk_annual, 2025)

    gem = process_gem_csv(Path(args.gem_csv), counties=counties)
    write_outputs(outdir, eia_2021, eia_year2, gem, counties, years=[2021, args.year2])
    print(f"Finished. Outputs written to {outdir}")


if __name__ == "__main__":
    main()
