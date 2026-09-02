# Oil Fabrication Tower

## Biological Hydrocarbon Synthesis Skyscraper

A single-file standalone interactive 3D model and first-order engineering study of a multi-story building (converted skyscraper) that uses photosynthesis, water, and CO2 to fabricate oil via bacterial bioconversion.

```
CO2 + H2O + sunlight → bacteria fix carbon → excrete hydrocarbons → oil
trickles down floor by floor to a spout → fresh oil flows out
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Dependencies](#dependencies)
- [CLI Commands](#cli-commands)
- [Interactive UI -- 5 Tabs](#interactive-ui----5-tabs)
- [Keyboard & Mouse Controls](#keyboard--mouse-controls)
- [Tower Size Presets](#tower-size-presets)
- [Three Bacterial Species](#three-bacterial-species)
- [Biological Model Subsystems](#biological-model-subsystems)
- [Synergy Multipliers](#synergy-multipliers)
- [Economic Model](#economic-model)
- [Environmental Stability & Risk](#environmental-stability--risk)
- [Oil Processing & Refining](#oil-processing--refining)
- [Nighttime Production (H2 Storage)](#nighttime-production-h2-storage)
- [Advanced Features](#advanced-features)
- [BacteriaModel.py Module Interface](#bacteriamodelpy-module-interface)
- [File Structure](#file-structure)
- [Scientific References](#scientific-references)

---

## Quick Start

```bash
# Install dependencies
pip install pygame numpy

# Launch the interactive 3D viewer
python OilFabricationTower.py

# Run the headless self-test (build + physics + biology + render)
python OilFabricationTower.py --selftest

# Generate a feasibility report
python OilFabricationTower.py --feasibility

# Generate a detailed production report
python OilFabricationTower.py --report

# Export the 3D tower model to OBJ + MTL
python OilFabricationTower.py --export-obj
```

---

## Dependencies

| Package  | Purpose                          |
|----------|----------------------------------|
| `pygame` | Window, input, 2D rendering      |
| `numpy`  | 3D math, matrix ops, painter     |

No other dependencies. The 3D renderer is a pure-Python software renderer (numpy painter + flat shading + backface culling). No OpenGL, no external 3D libraries.

---

## CLI Commands

| Command           | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `python OilFabricationTower.py`             | Launch interactive 3D viewer with 5-tab UI |
| `python OilFabricationTower.py --selftest`  | Headless build + physics + biology + render verification |
| `python OilFabricationTower.py --feasibility` | Full feasibility report with economics |
| `python OilFabricationTower.py --report`    | Detailed production report (all metrics) |
| `python OilFabricationTower.py --export-obj` | Export 3D tower geometry to OBJ + MTL files |
| `python BacteriaModel.py`                   | Bacteria model showcase report |
| `python BacteriaModel.py --species 0`       | Single species report (B. braunii) |
| `python BacteriaModel.py --species 2`       | Single species report (R. eutropha) |
| `python BacteriaModel.py --compare`         | Side-by-side species comparison |
| `python BacteriaModel.py --json`            | JSON export of all bacteria model data |

---

## Interactive UI -- 5 Tabs

The application opens a pygame window with a top tab bar. Switch tabs with `Tab` key, number keys `1-5`, or by clicking tab labels.

### Tab 1: TOWER
- Interactive 3D software-rendered view of the full tower
- Orbit camera (drag), pan (right-drag), zoom (wheel)
- Cross-section view (`X`), exploded view (`E`), labels (`L`)
- Info panel (right side) with multiple pages (`I` / `Shift+I` or `PgUp`/`PgDn`)
- Hotkey overlay panel (left side) listing all controls
- Left stats panel: real-time production metrics

### Tab 2: BACTERIA
- Animated species-specific cell diagrams with organelles
- Phospholipid bilayer membranes, DNA strands, metabolic pathways
- Hover organelles for tooltip popups
- Click organelles to select and see detail panel
- Click species panel to switch between species
- Exploded view mode (`E`) separates organelles for clarity
- Left/Right arrow keys to switch species

### Tab 3: PRODUCTION
- Full production report in scrollable UI format
- Species-by-species oil output breakdown
- Light capture, CO2 fixation, biomass, water recycling
- Solar capture details (tracking, seasonal derate, panel degradation)
- Biomass turnover and maintenance carbon

### Tab 4: ECONOMICS
- Construction cost (CapEx) breakdown
- Operating cost (OpEx): labor, maintenance, insurance, depreciation, tax
- Revenue: fuel sales + carbon credits
- Net profit, payback period, ROI
- Oil processing/refining output (gasoline, jet, diesel, chemicals)

### Tab 5: SYNERGY
- Step-by-step flow diagram showing all 3 species side-by-side
- 6 metabolic steps per species (18 total, 96px each) with animated flow arrows
- Cross-species synergy arrows routing horizontally between specific step rows (O2, H2, exudates exchange)
- KPI strip at top: oil, CO2, O2, H2, revenue, stability
- Efficiency rings and bars on each step box
- Cached glow effects, animated grid background, trailing particles
- Responsive layout with text truncation (adapts to window width)
- Synergy multiplier values from `BACTERIA_SYNERGY` dict with gradient mini bars
- Scientific proofs & references section with peer-reviewed citations
- Live tower model output with computed synergy factor
- Performance caches: `_glow_cache`, `_grid_cache`, `_panel_cache`

---

## Keyboard & Mouse Controls

### All Tabs
| Input           | Action                              |
|-----------------|-------------------------------------|
| `Tab`           | Cycle to next tab                   |
| `1`-`5`         | Jump to specific tab                |
| Click tab label | Switch to clicked tab               |
| `Up`/`Down`     | Scroll content                      |
| `Home`/`End`    | Scroll to top / bottom              |
| `P`             | Pause/resume simulation             |
| `O`             | Export OBJ + MTL                    |
| `H` or `F1`     | Toggle help overlay                 |
| `ESC`           | Quit (or close help)                |

### Tower Tab
| Input           | Action                              |
|-----------------|-------------------------------------|
| Mouse drag      | Orbit camera                        |
| RMB drag        | Pan view                            |
| Wheel           | Zoom (or scroll info panel)         |
| `X`             | Toggle cross-section                |
| `E`             | Toggle exploded view                |
| `L`             | Toggle labels                       |
| `I` / `Shift+I` | Next / prev info page               |
| `PgUp`/`PgDn`   | Next / prev info page               |
| `1`             | Reset camera view                   |
| `2`             | Front view                          |
| `3`             | Aerial view                         |
| `4`             | Exploded view preset                |

### Bacteria Tab
| Input           | Action                              |
|-----------------|-------------------------------------|
| Hover organelle | Tooltip popup                       |
| Click organelle | Select + detail panel               |
| Click species   | Switch species                      |
| `Left`/`Right`  | Switch species                      |
| `E`             | Toggle exploded view                |
| Click empty     | Deselect                            |

---

## Tower Size Presets

Five tower size presets are available, each scaling all dimensions and parameters:

| Preset    | Label                        | Height | Floors | Width  | Target (bbl/day) |
|-----------|------------------------------|--------|--------|--------|-------------------|
| `pilot`   | Pilot Plant (R&D)            | 30 m   | 6      | 10 m   | 10-50             |
| `small`   | Small Commercial             | 80 m   | 16     | 18 m   | 200-800           |
| `medium`  | Medium Industrial            | 140 m  | 28     | 25 m   | 1,000-3,000       |
| `large`   | Large Industrial (Default)   | 200 m  | 40     | 30 m   | 2,000-5,000       |
| `mega`    | Mega Scale (National)        | 320 m  | 64     | 50 m   | 8,000-20,000      |

Each preset adjusts tower dimensions, tray count, tray size, panel rows, vent count, and ground storage capacity.

---

## Three Bacterial Species

The tower houses a hybrid culture of three engineered bacterial species, each filling a distinct ecological niche:

### 1. Botryococcus braunii (Race B) -- Primary Oil Producer
- **Type**: Photoautotroph (green alga)
- **Oil**: Botryococcene (C34H58) -- triterpene hydrocarbon
- **Oil fraction**: 75% dry weight (engineered to 85%)
- **Doubling time**: 48 hours
- **CO2 tolerance**: 5%
- **Genome**: 14,000 genes, GC 54.5%
- **Harvest**: Extracellular excretion (continuous milking, 95% efficiency)
- **Key pathway**: MEP pathway → FPP → SSL-1/2/3 → botryococcene
- **Key advantage**: Oil excreted outside cell -- no cell disruption needed

### 2. Synechocystis sp. PCC 6803 -- Fast Alkane Producer
- **Type**: Photoautotroph (cyanobacterium)
- **Oil**: Alkanes C15-C17 (direct diesel-range blend)
- **Oil fraction**: 15% dry weight (engineered to 25%)
- **Doubling time**: 8 hours
- **CO2 tolerance**: 5%
- **Genome**: 3,672 genes, GC 47.8%
- **Harvest**: Partial excretion + in-line separation (82% efficiency)
- **Key pathway**: Fatty acid synthase → AAR → ADO → alkane
- **Key advantage**: Direct diesel-range fuel, no processing needed (95% direct blend)
- **Engineering**: Truncated antenna (+20% light distribution), CRISPR aar+ado overexpression

### 3. Ralstonia eutropha H16 (Cupriavidus necator) -- Nighttime Producer
- **Type**: Chemoautotroph (Knallgas bacterium)
- **Oil**: PHB (polyhydroxybutyrate, C4H6O2)n
- **Oil fraction**: 87% dry weight (engineered to 92%)
- **Doubling time**: 3.5 hours (fastest)
- **CO2 tolerance**: 10%
- **Genome**: 6,626 genes, GC 66.5%
- **Harvest**: Cell disruption + depolymerization (75% efficiency)
- **Key pathway**: CBB cycle → acetyl-CoA → PhaA/B/C → PHB
- **Key advantage**: Produces oil at night using stored H2 (no sunlight needed)
- **Energy**: [NiFe] hydrogenase oxidizes H2 → NADH + ATP

### Species Cross-Feeding Synergies
- **B. braunii → R. eutropha**: Daytime O2 feeds nighttime Knallgas (H2 oxidation)
- **Synechocystis → R. eutropha**: Photosynthetic exudates feed daytime mixotrophic growth
- **R. eutropha → B. braunii**: Dark fermentation H2 supports N-fixation pathway
- **R. eutropha → Synechocystis**: H2 for mixotrophic CO2 fixation (enhanced carbon flux)

---

## Biological Model Subsystems

The simulation includes 20+ individual biological model functions, each modeling one aspect of bacterial physiology:

### Core Metabolism
- **`photosynthesis_model`**: Photon absorption, PSII/PSI electron transport, NADPH production, O2 release, photoinhibition dynamics
- **`co2_fixation_model`**: RuBisCO kinetics (kcat, Km for CO2/O2), carboxysome/pyrenoid CCM, carbonic anhydrase, bicarbonate equilibrium
- **`oil_biosynthesis_model`**: MEP pathway flux (DXS overexpression), botryococcene synthase (SSL-1/2/3), alkane pathway (AAR+ADO), PHB polymerization (PhaA/B/C)
- **`growth_model`**: Biomass yield on CO2, maintenance energy, dark respiration, N-limitation effects

### Cellular Structures
- **`biofilm_model`**: Attachment/detachment rates, biofilm density, diffusion limitations, surface area enhancement (3.5x)
- **`slayer_model`**: S-layer protein lattice integrity, mechanical stress, pH protection
- **`flagellar_model`**: Motility, chemotaxis, energy cost
- **`phasin_model`**: Phasin coat proteins on PHB granules, granule size regulation

### Stress & Adaptation
- **`stress_response_model`**: ROS production, antioxidant capacity, heat/cold shock, UV damage
- **`quorum_sensing_model`**: Autoinducer production, quorum threshold, coordinated oil excretion (+12%)

### Genetic Engineering
- **`genetic_engineering_model`**: CRISPR-Cas12a expression fold (2.5x), plasmid copy number, metabolic burden, gene silencing rate, promoter leakiness

### Nutrient & Storage
- **`nutrient_uptake_model`**: N/P/Fe uptake rates, trace metal boost, B12 supplementation, adaptive N-pulsing
- **`polyphosphate_model`**: PolyP accumulation/depletion, energy reserve
- **`cyanophycin_model`**: Multi-L-arginyl-poly-L-aspartate (Synechocystis N reserve)
- **`glycogen_model`**: Carbon reserve polymer dynamics

### Energy & Nighttime
- **`chemoautotrophy_model`**: [NiFe] hydrogenase, H2 → NADH+ATP, CBB cycle with H2 energy
- **`dark_fermentation_model`**: Acetate-butanol pathway, H2 yield from biomass turnover, waste carbon recovery
- **`o2_balance_model`**: O2 production (photosynthesis) vs consumption (Knallgas), O2 toxicity, venting, scavenging

### Turnover & Stability
- **`biomass_turnover_model`**: Cell death/renewal rates, carbon recycling from dead biomass
- **`oil_processing_model`**: Botryococcene hydrocracking, alkane direct blend, PHB depolymerization
- **`stability_model`**: Contamination, temperature excursion, nutrient imbalance, pH drift, light failure -- with cascade failures, sensor redundancy, predictive maintenance

### Economic
- **`economic_model`**: Construction cost, operating costs (labor, maintenance, insurance), depreciation, tax, revenue (fuel + carbon credits), ROI, payback period

---

## Synergy Multipliers

The `BACTERIA_SYNERGY` dictionary defines cross-process multipliers that stack non-linearly:

| Synergy Key                    | Name                  | Value  | Description                              |
|--------------------------------|-----------------------|--------|------------------------------------------|
| `photo_growth_synergy`         | Photo → Growth        | 1.30x  | Photosynthesis feeds biomass growth      |
| `quorum_oil_synergy`           | Quorum → Oil          | 1.50x  | Quorum sensing coordinates oil excretion |
| `stress_turnover_synergy`      | Stress → Turnover     | 1.40x  | Stress accelerates biomass turnover      |
| `nutrient_growth_synergy`      | Nutrient → Growth     | 1.60x  | Nutrient availability boosts growth      |
| `genetic_oil_synergy`          | Genetic → Oil         | 1.80x  | CRISPR upregulation of oil pathway genes |
| `biofilm_stability_synergy`    | Biofilm → Stability   | 1.20x  | Biofilm protects culture stability       |
| `chemo_darkferm_synergy`       | Chemo → Dark Ferm     | 1.30x  | Chemoautotrophy + dark ferm complement   |
| `adaptation_stress_synergy`    | Adaptation → Stress   | 1.40x  | Adaptation reduces stress impact         |
| `cross_feeding_synergy`        | Cross-feeding         | 1.15x  | Species exchange O2, H2, exudates        |
| `global_synergy_multiplier`    | Global stacking       | 1.10x  | Nonlinear stacking of all synergies      |

The `compute_synergy_factor(state)` function computes the live synergy multiplier from a `BacteriaState` by raising each synergy to a factor based on the state's current conditions (photo efficiency, growth rate, quorum status, nutrient levels, genetic boost, adaptation level).

---

## Economic Model

### Revenue
- **Fuel revenue**: Saleable fuel (gasoline + jet + diesel + chemicals) priced at $3.00/gal
- **Carbon credits**: $50/ton CO2 fixed
- **Total revenue**: Fuel + carbon credits (annual)

### Costs
- **CapEx**: Construction ($5,000/m² bio area) + permitting ($500K one-time)
- **OpEx**: Labor ($500K/yr) + maintenance (2% of construction) + insurance (1% of construction)
- **Depreciation**: 25-year straight-line
- **Tax**: 21% corporate rate

### Key Metrics (Large Tower, Default)
- **Oil output**: ~2,966 bbl/day (~1,082,503 bbl/yr)
- **Total boost**: 325x over baseline
- **Stability index**: 99.8%
- **Payback period**: Varies by size preset

---

## Environmental Stability & Risk

The tower models 5 independent failure modes with cascade dynamics:

| Risk Type         | Base Rate (/hr) | Severity | Recovery (hr) | Mitigation              |
|-------------------|-----------------|----------|---------------|-------------------------|
| Contamination     | 0.10%           | 30% loss | 12 hr         | UV sterilization (-50%) |
| Temp excursion    | 0.20%           | 20% loss | 4 hr          | Redundant HVAC (-60%)   |
| Nutrient imbalance| 0.15%           | 15% loss | 6 hr          | Auto-dosing (-70%)      |
| pH drift          | 0.10%           | 10% loss | 3 hr          | pH auto-buffer (-80%)   |
| Light failure     | 0.05%           | 25% loss | 2 hr          | LED redundancy (-50%)   |

Additional features:
- **Cascade failure modeling**: 5% chance one failure triggers another
- **Sensor redundancy**: 3x sensors per parameter (2-of-3 voting)
- **Predictive maintenance**: AI-driven, 30% further risk reduction
- **Gradual recovery**: Exponential recovery dynamics (not instant)
- **Drain-to-ground storage**: Oil diverts to underground tank when stability < 80%
- **Backup power**: 99.9% uptime with battery + generator

---

## Oil Processing & Refining

The tower includes on-site processing of raw bio-oil to saleable fuels:

| Product    | Source Species    | Process                  | Efficiency |
|------------|-------------------|--------------------------|------------|
| Gasoline   | B. braunii        | Botryococcene hydrocrack | 67% of C34 |
| Jet fuel   | B. braunii        | Botryococcene hydrocrack | 15% of C34 |
| Diesel     | Synechocystis     | Direct alkane blend      | 95% direct |
| Chemicals  | R. eutropha       | PHB depolymerization     | 80% to propylene + butyric acid |

Processing energy: 5% of produced oil energy used for refining.

---

## Nighttime Production (H2 Storage)

The tower achieves 24-hour production through H2 storage:

1. **Daytime**: Excess solar energy (35%) stored as H2 via SOEC electrolysis (85% efficiency)
2. **H2 compression**: 350 bar Type IV composite tanks, 85% storage efficiency
3. **Nighttime**: R. eutropha oxidizes stored H2 via [NiFe] hydrogenase → NADH + ATP
4. **CBB cycle**: Uses H2 energy to fix CO2 → pyruvate → acetyl-CoA → PHB
5. **O2 recycling**: 60% of daytime O2 available for nighttime Knallgas reaction

---

## Advanced Features

### Monte Carlo Uncertainty Quantification
- `run_bacteria_monte_carlo(species_idx, n_runs, seed, duration)` perturbs 9 biological parameters
- Reports p5, p50, p95 percentiles for peak oil production
- Priors: photosynthesis max eff, RuBisCO kcat, MEP pathway flux, PHB synthesis rate, N uptake rate, heat shock threshold, optimal temp, biomass yield, gene silencing rate

### Scenario Comparison
- `compare_bacteria_scenarios(species_idx, duration)` runs 9 protocols:
  - Baseline, high light, N-limitation, CRISPR boost, low CO2, high temp, contamination, no engineering, optimized
- Reports total oil (kg) for each scenario

### Benefit/Risk Assessment
- `assess_bacteria_benefit_risk(species_idx, duration)` computes:
  - Benefit score (oil output vs theoretical max)
  - Cumulative risk % (contamination + stability + stress + genetic drift + nutrient depletion)
  - Recommendation: OPTIMIZE / PROCEED WITH MONITORING / REDUCE INTENSITY

### Temporal Simulation
- `simulate_bacteria(species_idx, duration_hr, dt_hr)` runs hour-by-hour simulation
- Tracks 20+ time series: oil, CO2, O2, biomass, stress, stability, growth, biofilm, quorum, adaptation, contamination, genetic stability, temperature, polyphosphate, glycogen, S-layer integrity
- Records peak oil, min stability, contamination events, stability failures

### 3D Software Renderer
- Pure Python/numpy -- no OpenGL
- Flat shading with directional light
- Backface culling
- Z-sorting painter's algorithm
- 291 parts, 35,136 vertices, 27,619 faces (large tower)
- OBJ + MTL export

---

## BacteriaModel.py Module Interface

`BacteriaModel.py` re-exports all bacteria model functions from `OilFabricationTower.py` as a standalone library:

```python
from BacteriaModel import compute_bacteria_model, BACTERIA_TEMPLATES

result = compute_bacteria_model(0)  # B. braunii
print(result['photosynthesis']['co2_fixed_kg_hr'])
```

### Exports

**Data structures**: `BIO`, `DIMS`, `BACTERIA_TEMPLATES`, `GENETIC_CODE`, `TOWER_SIZES`, `BACTERIA_SYNERGY`, `MONTE_CARLO_PRIORS`

**Model functions**: `photosynthesis_model`, `co2_fixation_model`, `oil_biosynthesis_model`, `growth_model`, `biofilm_model`, `stress_response_model`, `quorum_sensing_model`, `genetic_engineering_model`, `nutrient_uptake_model`, `chemoautotrophy_model`, `dark_fermentation_model`, `o2_balance_model`, `biomass_turnover_model`, `oil_processing_model`, `stability_model`, `economic_model`, `polyphosphate_model`, `cyanophycin_model`, `glycogen_model`, `slayer_model`, `flagellar_model`, `phasin_model`, `compute_bacteria_model`

**Simulation**: `BacteriaState`, `BacteriaSimulationResult`, `initialize_bacteria_state`, `simulate_bacteria`, `run_bacteria_monte_carlo`, `compare_bacteria_scenarios`, `assess_bacteria_benefit_risk`, `compute_synergy_factor`

**Utilities**: `OilProductionModel`, `TowerState`, `protein_properties`, `apply_tower_size`, `run_bacteria_report`, `run_selftest`

---

## File Structure

```
OilSynthesis/
├── OilFabricationTower.py    # Main application (8,250+ lines)
├── BacteriaModel.py          # Module interface / standalone library
├── OilFabricationTower.obj   # Exported 3D model (OBJ format)
├── OilFabricationTower.mtl   # Exported materials (MTL format)
├── ChiefJuiceV1.17.py        # Reference codebase (pattern source)
├── ReferenceCode/            # Additional reference implementations
├── informational.md          # Project notes
├── README.md                 # This file
├── ABOUT.md                  # Project background & philosophy
└── OVERVIEW.md               # Technical architecture overview
```

---

## Scientific References

| Claim                                          | Citation                                              |
|------------------------------------------------|-------------------------------------------------------|
| B. braunii: 75% DW botryococcene               | Metzger & Largeau 2005, Appl Microbiol Biotechnol     |
| SSL-1/2/3 genes identified                     | Niehaus et al. 2011, PNAS 108:12669                   |
| Extracellular oil excretion                    | Allard & Casadevall 1990, Energy Convers Mgmt         |
| Synechocystis: AAR+ADO alkane pathway          | Schirmer et al. 2010, Science 329:559                 |
| CRISPR-Cas12a genome editing                   | Umena et al. 2012, Nature 483:513                     |
| Truncated antenna +20% light                   | Kirst et al. 2014, PNAS 111:16606                     |
| R. eutropha: 87% DW PHB accumulation           | Steinbuchel & Schlegel 1991, Mol Microbiol            |
| [NiFe] hydrogenase Knallgas                    | Burgdorf et al. 2005, J Bacteriol 187:6589            |
| pHG1 megaplasmid (autotrophy)                  | Pohlmann et al. 2006, Nat Biotechnol 24:1257          |
| RuBisCO engineering (reduced photorespiration) | Whitney et al. 2011, PNAS 108:17773                   |
| Quorum sensing oil coordination                | Schuster et al. 2013, Front Microbiol 4:62            |

---

## License

This project is provided as-is for research and educational purposes.
