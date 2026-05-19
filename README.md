# The Proximity Paradox

**If everyone in a city were constrained to 15-minute local living, how much would experienced segregation change — and where?**

This project implements the full pipeline from raw GPS data to a baseline counterfactual simulation (S1) for a multi-country study on whether proximity-based urban planning (the 15-minute city) can reconcile environmental sustainability with social mixing.

The pipeline processes mobile phone GPS pings into individual activity patterns, links them to urban structure and socioeconomic context, and imposes a 15-minute walking constraint to quantify the segregation cost of proximity under current urban structure.

## Overview

- **Part A (Steps 1–6)**: Mobility processing — GPS pings → stays → home/work → POI visits
- **Part B (Steps 7–10)**: Counterfactual simulation — isochrones → trip rewiring → segregation measurement

The mobility pipeline follows conventions from [geo-social-mixing](https://github.com/MobiSegInsights/geo-social-mixing). The counterfactual simulation is new to this project.

## Project Structure

```
proximity-paradox/
├── .devcontainer/          # Docker container (CUDA + Java + R + Python 3.11)
├── config/
│   ├── default.yaml        # Global pipeline parameters
│   ├── countries/           # Country-specific overrides
│   ├── poi_categories.yaml  # Unified POI category schema (12 categories)
│   └── schema.py            # Config validation (pydantic)
├── src/                     # Pipeline modules (Steps 0–10 + utilities)
├── r_scripts/               # r5r isochrone computation
├── scripts/                 # CLI entry points
├── notebooks/               # Per-step diagnostics
├── tests/
└── requirements.txt
```

> **Status**: Under active development. See [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for implementation roadmap.

## Setup

Requires the dev container (recommended) or a conda environment with Python 3.11, Java 21, and R.

```bash
# Build and open in VS Code dev container
code --folder-uri vscode-remote://dev-container+$(printf '%s' "$PWD" | xxd -p)/workspace

# Or use conda directly
micromamba create -f .devcontainer/environment.yml -n geoenv
micromamba activate geoenv
```
