from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely import wkt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires geopandas and shapely. Install with: pip install geopandas shapely pyarrow"
    ) from e

try:
    from sklearn.cluster import KMeans
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires scikit-learn. Install with: pip install scikit-learn") from e


EXCLUDE_STATE_FIPS = {"60", "66", "69", "72", "78"}  # AS, GU, MP, PR, VI

LAT_CANDIDATES = [
    "lat", "latitude", "site_lat", "y", "centroid_lat", "resource_lat", "cpa_lat", "lat_dd"
]
LON_CANDIDATES = [
    "lon", "lng", "longitude", "site_lon", "x", "centroid_lon", "resource_lon", "cpa_lon", "lon_dd"
]
CAPACITY_CANDIDATES = [
    "mw", "capacity_mw", "max_capacity_mw", "available_mw", "resource_mw", "cap_mw",
    "cluster_capacity_mw", "new_build_mw", "potential_mw", "solar_mw"
]
LCOE_CANDIDATES = [
    "lcoe", "lcoe_mwh", "lcoe_per_mwh", "total_lcoe", "all_in_lcoe", "approx_lcoe", "site_lcoe"
]
INTERCONNECT_CANDIDATES = [
    "interconnection_cost", "interconnect_cost", "spur_cost_per_mw", "tx_cost_per_mw",
    "interconnect_adder_per_mw"
]
CF_CANDIDATES = [
    "cf", "avg_cf", "mean_cf", "capacity_factor", "avg_capacity_factor"
]
SITE_ID_CANDIDATES = [
    "site_id", "cpa_id", "id", "resource_id", "gid", "cluster", "project_id"
]
STATE_CANDIDATES = ["state", "state_abbr", "state_code", "st"]
COUNTY_FIPS_CANDIDATES = ["county_fips", "countyfp", "geoid", "county_geoid", "fips"]
COUNTY_NAME_CANDIDATES = ["county_name", "name", "namelsad", "county"]


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_column(columns: Iterable[str], candidates: list[str], required: bool = True) -> Optional[str]:
    cols = list(columns)
    normalized = {normalize_name(c): c for c in cols}
    for cand in candidates:
        key = normalize_name(cand)
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = normalize_name(cand)
        for n, orig in normalized.items():
            if key in n:
                return orig
    if required:
        raise KeyError(f"Could not find a matching column for candidates={candidates}. Columns: {cols}")
    return None


def safe_weighted_mean(v: pd.Series, w: pd.Series) -> float:
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def read_table(path: Path) -> pd.DataFrame | gpd.GeoDataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return gpd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".geojson", ".json", ".gpkg", ".shp"}:
        return gpd.read_file(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_counties(county_path: Path) -> gpd.GeoDataFrame:
    counties = gpd.read_file(county_path)
    counties.columns = [c.lower() for c in counties.columns]

    geoid_col = find_column(counties.columns, COUNTY_FIPS_CANDIDATES)
    counties["county_fips"] = counties[geoid_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)

    name_col = find_column(counties.columns, COUNTY_NAME_CANDIDATES, required=False)
    counties["county_name"] = counties[name_col].astype(str) if name_col else counties["county_fips"]

    state_col = find_column(counties.columns, ["statefp", "state_fips", "statefp20"], required=False)
    if state_col:
        counties["state_fips"] = counties[state_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(2)
        counties = counties[~counties["state_fips"].isin(EXCLUDE_STATE_FIPS)].copy()

    keep_cols = ["county_fips", "county_name", "geometry"]
    if "state_fips" in counties.columns:
        keep_cols.append("state_fips")

    counties = counties[keep_cols].copy().to_crs(4326)
    return counties


def table_to_points(df: pd.DataFrame | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if isinstance(df, gpd.GeoDataFrame) and df.geometry.notna().any():
        if df.crs is None:
            df = df.set_crs(4326, allow_override=True)
        return df.to_crs(4326)

    cols = list(df.columns)
    lat_col = find_column(cols, LAT_CANDIDATES, required=False)
    lon_col = find_column(cols, LON_CANDIDATES, required=False)

    if lat_col and lon_col:
        d = df.copy()
        d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
        d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
        d = d.dropna(subset=[lat_col, lon_col]).copy()
        return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d[lon_col], d[lat_col]), crs=4326)

    geom_col = find_column(cols, ["geometry", "geom", "wkt"], required=False)
    if geom_col is not None:
        d = df.copy()
        d["geometry"] = d[geom_col].apply(
            lambda x: wkt.loads(x) if isinstance(x, str) and x.startswith(("POINT", "POLYGON", "MULTI")) else None
        )
        d = d.dropna(subset=["geometry"]).copy()
        return gpd.GeoDataFrame(d, geometry="geometry", crs=4326)

    raise ValueError("Could not infer site geometry. Provide a file with latitude/longitude or geometry.")


def standardize_sites(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = list(gdf.columns)
    out = gdf.copy()

    site_id = find_column(cols, SITE_ID_CANDIDATES, required=False)
    out["site_id"] = out[site_id].astype(str) if site_id else [f"site_{i}" for i in range(len(out))]

    lat_col = find_column(cols, LAT_CANDIDATES, required=False)
    lon_col = find_column(cols, LON_CANDIDATES, required=False)
    if lat_col is None or lon_col is None:
        out["lat"] = out.geometry.y
        out["lon"] = out.geometry.x
    else:
        out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
        out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")

    cap_col = find_column(cols, CAPACITY_CANDIDATES)
    out["available_mw"] = pd.to_numeric(out[cap_col], errors="coerce")

    lcoe_col = find_column(cols, LCOE_CANDIDATES, required=False)
    out["lcoe"] = pd.to_numeric(out[lcoe_col], errors="coerce") if lcoe_col else np.nan

    ic_col = find_column(cols, INTERCONNECT_CANDIDATES, required=False)
    out["interconnection_cost"] = pd.to_numeric(out[ic_col], errors="coerce") if ic_col else np.nan

    cf_col = find_column(cols, CF_CANDIDATES, required=False)
    out["capacity_factor"] = pd.to_numeric(out[cf_col], errors="coerce") if cf_col else np.nan

    st_col = find_column(cols, STATE_CANDIDATES, required=False)
    out["state"] = out[st_col].astype(str).str.upper() if st_col else None

    out = out.dropna(subset=["available_mw"]).copy()
    out = out[out["available_mw"] > 0].copy()
    out = out[(out.geometry.x.between(-180, 180)) & (out.geometry.y.between(-90, 90))].copy()
    return out


def join_sites_to_counties(
    sites: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
    county_modifier_csv: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    # use intersects instead of within so boundary-touching points/polygons still join
    joined = gpd.sjoin(sites, counties[["county_fips", "county_name", "geometry"]], how="left", predicate="intersects")
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])

    if county_modifier_csv is not None:
        mod = pd.read_csv(county_modifier_csv)
        mod.columns = [c.lower() for c in mod.columns]
        geoid_col = find_column(mod.columns, ["geoid", "county_fips", "fips"])
        mod["county_fips"] = mod[geoid_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
        keep = ["county_fips"] + [c for c in ["full_score", "cost_modifier_full"] if c in mod.columns]
        joined = joined.merge(mod[keep].drop_duplicates("county_fips"), on="county_fips", how="left")

    return joined


def make_county_summary(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    grouped = joined.dropna(subset=["county_fips"]).copy()

    if grouped.empty:
        return pd.DataFrame(
            columns=[
                "county_fips",
                "county_name",
                "site_count",
                "available_mw",
                "median_lcoe",
                "min_lcoe",
                "mean_interconnection_cost",
                "mean_cost_modifier",
                "weighted_capacity_factor",
            ]
        )

    summary = (
        grouped.groupby("county_fips", dropna=False)
        .agg(
            county_name=("county_name", lambda s: s.dropna().iloc[0] if s.dropna().any() else None),
            site_count=("site_id", "nunique"),
            available_mw=("available_mw", "sum"),
            median_lcoe=("lcoe", "median"),
            min_lcoe=("lcoe", "min"),
            mean_interconnection_cost=("interconnection_cost", "mean"),
            mean_cost_modifier=("cost_modifier_full", "mean") if "cost_modifier_full" in grouped.columns else ("available_mw", lambda s: np.nan),
        )
        .reset_index()
    )

    wcf_records = [
        {
            "county_fips": county_fips,
            "weighted_capacity_factor": safe_weighted_mean(g["capacity_factor"], g["available_mw"]),
        }
        for county_fips, g in grouped.groupby("county_fips", dropna=False)
    ]
    wcf = pd.DataFrame(wcf_records, columns=["county_fips", "weighted_capacity_factor"])

    summary = summary.merge(wcf, on="county_fips", how="left")
    summary["site_count"] = summary["site_count"].fillna(0).astype(int)
    return summary


def cluster_sites_by_county(joined: gpd.GeoDataFrame, target_cluster_mw: float = 500.0, max_clusters_per_county: int = 10) -> pd.DataFrame:
    records: list[dict] = []
    clustered = joined.dropna(subset=["county_fips", "lat", "lon", "available_mw"]).copy()

    if clustered.empty:
        return pd.DataFrame(
            columns=[
                "county_fips", "county_name", "cluster_id", "n_sites", "cluster_available_mw",
                "cluster_lcoe_weighted", "cluster_interconnection_cost_weighted",
                "cluster_capacity_factor_weighted", "centroid_lat", "centroid_lon",
            ]
        )

    for county_fips, g in clustered.groupby("county_fips"):
        g = g.copy().reset_index(drop=True)
        n_sites = len(g)
        total_mw = float(g["available_mw"].sum())

        if n_sites <= 10:
            k = 1
        else:
            k = max(1, math.ceil(total_mw / target_cluster_mw))
            k = min(k, max_clusters_per_county, n_sites)

        if k == 1:
            g["cluster_local"] = 0
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            g["cluster_local"] = km.fit_predict(g[["lon", "lat"]])

        for cluster_local, cg in g.groupby("cluster_local"):
            weights = cg["available_mw"].fillna(0)
            centroid_lon = np.average(cg["lon"], weights=weights) if weights.sum() > 0 else cg["lon"].mean()
            centroid_lat = np.average(cg["lat"], weights=weights) if weights.sum() > 0 else cg["lat"].mean()
            records.append({
                "county_fips": county_fips,
                "county_name": cg["county_name"].dropna().iloc[0] if cg["county_name"].notna().any() else county_fips,
                "cluster_id": f"{county_fips}_{int(cluster_local)+1}",
                "n_sites": int(len(cg)),
                "cluster_available_mw": float(cg["available_mw"].sum()),
                "cluster_lcoe_weighted": safe_weighted_mean(cg["lcoe"], cg["available_mw"]),
                "cluster_interconnection_cost_weighted": safe_weighted_mean(cg["interconnection_cost"], cg["available_mw"]),
                "cluster_capacity_factor_weighted": safe_weighted_mean(cg["capacity_factor"], cg["available_mw"]),
                "centroid_lat": float(centroid_lat),
                "centroid_lon": float(centroid_lon),
            })
    return pd.DataFrame(records)


def add_available_flag(counties: gpd.GeoDataFrame, summary: pd.DataFrame) -> gpd.GeoDataFrame:
    out = counties.merge(summary, on="county_fips", how="left", suffixes=("", "_summary"))
    if "county_name_summary" in out.columns:
        out["county_name"] = out["county_name"].fillna(out["county_name_summary"])
        out = out.drop(columns=["county_name_summary"])
    out["site_count"] = out["site_count"].fillna(0).astype(int)
    out["available_mw"] = out["available_mw"].fillna(0.0)
    out["county_has_candidate_sites"] = out["site_count"] > 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Join PowerGenome candidate solar sites to counties and create county-level outputs.")
    parser.add_argument("--sites", required=True, help="PowerGenome solar resource-group file (.parquet or .csv)")
    parser.add_argument("--counties", required=True, help="County boundaries (.geojson, .gpkg, .shp, etc.)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--county-modifier", help="Optional county_cost_modifier_full_centered.csv")
    parser.add_argument("--target-cluster-mw", type=float, default=500.0)
    parser.add_argument("--max-clusters-per-county", type=int, default=10)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    counties = load_counties(Path(args.counties))
    sites_raw = read_table(Path(args.sites))
    site_points = table_to_points(sites_raw)
    sites = standardize_sites(site_points)

    joined = join_sites_to_counties(
        sites,
        counties,
        county_modifier_csv=Path(args.county_modifier) if args.county_modifier else None,
    )

    matched_sites = int(joined["county_fips"].notna().sum()) if "county_fips" in joined.columns else 0
    print(f"Total candidate sites after cleaning: {len(sites)}")
    print(f"Candidate sites matched to counties: {matched_sites}")

    summary = make_county_summary(joined)
    county_layer = add_available_flag(counties, summary)
    clusters = cluster_sites_by_county(
        joined,
        target_cluster_mw=args.target_cluster_mw,
        max_clusters_per_county=args.max_clusters_per_county,
    )

    joined.drop(columns="geometry").to_csv(outdir / "solar_candidate_sites_with_county.csv", index=False)
    summary.to_csv(outdir / "county_candidate_summary.csv", index=False)
    clusters.to_csv(outdir / "county_solar_clusters.csv", index=False)
    county_layer.to_file(outdir / "county_candidate_layer.geojson", driver="GeoJSON")

    with open(outdir / "column_inference_report.json", "w") as f:
        json.dump(
            {
                "site_columns": list(site_points.columns),
                "matched_sites_to_counties": matched_sites,
                "normalized_output_columns": [
                    "site_id", "lat", "lon", "available_mw", "lcoe", "interconnection_cost",
                    "capacity_factor", "county_fips", "county_name"
                ],
                "notes": [
                    "If matched_sites_to_counties is 0, inspect your PowerGenome coordinates/geometry.",
                    "The hosted PowerGenome GUI does not expose a county-layer upload control; use your own GUI or a local fork.",
                ],
            },
            f,
            indent=2,
        )

    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
