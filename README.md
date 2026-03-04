# County Propensity Score Calculation

This repo contains a notebook for calculating **county-level solar siting propensity scores** from the Solar-NIMBY dataset using coefficients from the **Capacity Intensity** regression results.

## What the notebook does

The notebook builds two county-level scores:

- **Social-component score**  
  Uses only the significant **socio-political** terms from the expanded model:
  - Black population share
  - Asian population share
  - Median income
  - Graduate/professional education
  - GDP per capita

- **Full propensity score**  
  Uses both significant **techno-economic** and **socio-political** terms from the expanded model:
  - **Techno-economic:** GHI, Protected_Land, Slope, Population_Density  
  - **Socio-political:** Black, Asian, Income, Graduate education, GDP per capita

The notebook then:
1. loads county-level data from the **Solar-NIMBY** repo
2. merges county techno-economic and socio-political variables
3. computes raw weighted scores using the regression coefficients
4. normalizes the raw values to **0–1**
5. exports county-level CSV files for later backtesting / model input

## Input data

The notebook expects data from the `Solar-NIMBY/data/` directory:

- `suitability_scores/suitability_scores_county.csv`
- `county_clean/social_factors_merged.csv`
- `GDP_percapita.csv`

## Output files

The notebook writes:

- `county_propensity_scores_social_only.csv`
- `county_propensity_scores_full.csv`

Each output CSV contains:

- `county_name`
- `score`

## Important implementation notes

- `Protected_Land` is used as the land variable in the score calculation, consistent with Jenny’s clarification.
- `GDP_percapita.csv` is merged onto the county social table by **county + state name**.
- The **social-component score** is a **constructed subset of the expanded model**, not a separately estimated social-only regression.
- `Population_Density` is currently used as the available processed county variable corresponding to the population-related term in the model.

## Notebook

Main notebook:

- `county_propensity_score_calculation.ipynb`

## Suggested workflow

1. Make sure the Solar-NIMBY repo is available locally
2. Update the `basepath` in the notebook if needed
3. Run the notebook from top to bottom
4. Review the output CSV files and sanity-check the rankings

## Repo purpose

This notebook is intended to support:
- **backtesting / historical validation**
- **county-level propensity score construction**
- future integration into **forward-looking capacity expansion modeling**
