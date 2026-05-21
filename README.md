
# Deep Learning for Crop Classification Using Multi-Source Satellite Data

Poject for the Neural Networks course — Master 2 SII, USTHB.

## Overview

This project explores crop type classification from Sentinel-2 time-series imagery using deep learning, across two US regions: **Arkansas** and **California**.

## Project Structure

The project is organized into three parts:

**Part 1 — MCTNet Reproduction**\
Faithful reimplementation of the MCTNet architecture (Wang et al., 2024), a hybrid CNN-Transformer network designed for pixel-based crop mapping from Sentinel-2 time-series.

**Part 2 — Environmental Covariates Integration**\
Ablation study evaluating the impact of adding static environmental covariates (climate, soil, topography) to the MCTNet backbone via Late Fusion.

**Part 3 — Improved Model**\
Design and evaluation of an improved model building on the findings of Parts 1 and 2.

## Data

- **Sentinel-2** time-series (36 timesteps, 10 bands, 2021) — acquired via Google Earth Engine
- **CDL** (Cropland Data Layer 2021) — ground truth labels
- **ERA5** — climate covariates
- **OpenLandMap SoilGrids** — soil covariates
- **ETOPO1 + ALOS ERGo** — topographic covariates

## Regions & Classes

| Region | Classes |
| --- | --- |
| Arkansas | Corn, Cotton, Rice, Soybean, Others |
| California | Grapes, Rice, Alfalfa, Almonds, Pistachios, Others |

## Reference

- Wang et al. (2024). *A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery.* Computers and Electronics in Agriculture.
- Remote Sens. (2025). Deep Learning Applications for Crop Mapping Using Multi-Temporal Sentinel-2 Data and Red-Edge Vegetation Indices: Integrating Convolutional and Recurrent Neural Networks.
