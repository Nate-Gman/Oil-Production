# Technical Architecture Overview

## Oil Fabrication Tower -- System Design & Code Structure

This document provides a detailed technical walkthrough of the codebase architecture, data flow, rendering pipeline, and model integration.

---

## Table of Contents

- [Architecture at a Glance](#architecture-at-a-glance)
- [File Layout](#file-layout)
- [Code Organization (OilFabricationTower.py)](#code-organization-oilfabricationtowerpy)
- [Data Flow](#data-flow)
- [3D Rendering Pipeline](#3d-rendering-pipeline)
- [Biological Model Architecture](#biological-model-architecture)
- [Temporal Simulation Engine](#temporal-simulation-engine)
- [Monte Carlo & Scenario Systems](#monte-carlo--scenario-systems)
- [Synergy Computation](#synergy-computation)
- [UI System Architecture](#ui-system-architecture)
- [Synergy Tab Rendering Pipeline](#synergy-tab-rendering-pipeline)
- [Bacteria Tab Rendering Pipeline](#bacteria-tab-rendering-pipeline)
- [Economic Model Architecture](#economic-model-architecture)
- [Stability & Risk Model](#stability--risk-model)
- [Configuration System](#configuration-system)
- [Color & Visual Constants](#color--visual-constants)
- [Helper Function Reference](#helper-function-reference)
- [Key Data Structures](#key-data-structures)
- [Self-Test Pipeline](#self-test-pipeline)
- [OBJ Export Pipeline](#obj-export-pipeline)
- [Performance Characteristics](#performance-characteristics)

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────┐
│                    OilFabricationTower.py                │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Constants   │  │  BIO dict    │  │  BACTERIA_     │  │
│  │  (DIMS)      │  │  (80+ params)│  │  TEMPLATES     │  │
│  │  TOWER_SIZES │  │              │  │  (3 species)   │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘  │
│         │                 │                  │            │
│         ▼                 ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Biological Model Functions              │ │
│  │  (20+ functions: photosynthesis, CO2, oil, growth,  │ │
│  │   biofilm, stress, quorum, genetic, nutrient, etc.)  │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                 │
│           ┌─────────────┼──────────────┐                  │
│           ▼             ▼              ▼                  │
│  ┌──────────────┐ ┌───────────┐ ┌──────────────┐         │
│  │ compute_     │ │ simulate_ │ │ run_bacteria_│         │
│  │ bacteria_    │ │ bacteria  │ │ monte_carlo  │         │
│  │ model        │ │ (temporal)│ │              │         │
│  └──────┬───────┘ └─────┬─────┘ └──────┬───────┘         │
│         │               │              │                  │
│         ▼               ▼              ▼                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           OilProductionModel                         │ │
│  │  (integrates all models → tower-level production)    │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                 │
│                         ▼                                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              TowerState                              │ │
│  │  (holds model + camera + animation state)            │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                 │
│           ┌─────────────┼──────────────┐                  │
│           ▼             ▼              ▼                  │
│  ┌──────────────┐ ┌───────────┐ ┌──────────────┐         │
│  │  3D Renderer │ │ 5-Tab UI  │ │ CLI Reports  │         │
│  │  (numpy)     │ │ (pygame)  │ │ (--selftest, │         │
│  │              │ │           │ │  --report)   │         │
│  └──────────────┘ └───────────┘ └──────────────┘         │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ BacteriaModel.py│  (re-exports for library use)
└─────────────────┘
```

---

## File Layout

```
OilSynthesis/
├── OilFabricationTower.py     # Main: model + renderer + UI (8,200+ lines)
├── BacteriaModel.py           # Module interface (re-exports, 722 lines)
├── OilFabricationTower.obj    # Exported 3D geometry (1.7 MB)
├── OilFabricationTower.mtl    # Exported materials (12 KB)
├── ChiefJuiceV1.17.py         # Reference codebase (pattern source)
├── ReferenceCode/             # Additional reference implementations
├── informational.md           # Project notes
├── README.md                  # User documentation
├── ABOUT.md                   # Project background & philosophy
└── OVERVIEW.md                # This file -- technical architecture
```

---

## Code Organization (OilFabricationTower.py)

The file is organized in sequential sections, each separated by banner comments:

| Section                | Lines (approx) | Description                                      |
|------------------------|-----------------|--------------------------------------------------|
| Header & Imports       | 1-50            | Module docstring, imports, pygame init           |
| Tower Dimensions       | 50-410          | `DIMS` dict, `TOWER_SIZES` presets, `apply_tower_size()` |
| Colors                 | 423-437         | Color constants (BG, C_ACCENT, C_TEXT, etc.)     |
| Math Helpers           | 439-454         | Rotation matrices, clamp, mix, detail segmentation|
| Primitive Builders     | 456-580         | `_solid_cylinder`, `_hollow_cylinder`, `_box`, `_panel_mesh`, `_pipe`, `_tray`, `_sphere` |
| Tower Assembly         | 580-1580        | `build_tower()` -- assembles 291 parts from primitives |
| HUD Helpers            | 1580-1612       | `_label()`, `_panel()`, `_bar()`, `_mini_bar()`, `_section_hdr()` |
| BIO Constants          | 1620-1700       | 80+ biological constants (RuBisCO kcat, MEP flux, etc.) |
| Bacteria Templates     | 1700-1850       | `BACTERIA_TEMPLATES` dict (3 species with genome, GC, genes) |
| Genetic Code           | 1850-1950       | `GENETIC_CODE` dict (codon table, amino acid properties) |
| Biological Models      | 1950-3150       | 20+ model functions (photosynthesis, CO2, oil, growth, etc.) |
| Bacteria Synergy       | 3150-3170       | `BACTERIA_SYNERGY` dict (10 multipliers)          |
| Monte Carlo Priors     | 3170-3190       | `MONTE_CARLO_PRIORS` list (9 parameters to perturb)|
| Temporal Simulation    | 3190-3400       | `BacteriaState`, `BacteriaSimulationResult`, `initialize_bacteria_state()`, `simulate_bacteria()` |
| Synergy Computation    | 3400-3430       | `compute_synergy_factor()`                        |
| Monte Carlo & Scenarios| 3430-3600       | `run_bacteria_monte_carlo()`, `compare_bacteria_scenarios()`, `assess_bacteria_benefit_risk()` |
| Oil Production Model   | 3600-1580       | `OilProductionModel` class (integrates all models) |
| Tower State            | 1580-1620       | `TowerState` class (model + camera + animation)   |
| 3D Renderer            | 1620-5700       | `TowerRenderer` class (projection, shading, painter) |
| CLI Reports            | 5700-5830       | `run_selftest()`, `run_feasibility()`, `run_report()` |
| OBJ Export             | 5830-5870       | `export_obj()`                                    |
| Top Bar & Tabs         | 5850-5910       | `TAB_MODES`, `TAB_LABELS`, `draw_topbar()`, `handle_tab_click()` |
| UI Helpers             | 5900-5920       | `_mini_bar()`, `_section_hdr()`, `_clamp_x()`     |
| Bacteria Cell Diagram  | 5920-6800       | Cell drawing functions (bilayer, DNA, organelles)  |
| Bacteria Tab           | 6800-7370       | `draw_bacteria_tab()` -- interactive cell viewer   |
| Synergy Tab            | 7370-7840       | `draw_synergy_tab()` -- flow diagram with KPIs, responsive layout, cached glow/grid |
| HUD Draw               | 7840-8100       | `draw_hud()` -- tower tab info panels              |
| Main Loop              | 8100-8250       | Event handling, tab routing, render dispatch       |

---

## Data Flow

### Production Model Flow
```
DIMS (dimensions)
    │
    ├──→ Solar capture (facade area, sun tracking, seasonal derate)
    │         │
    │         ▼
    │    Light power (MW)
    │         │
    │         ├──→ LED supplement (excess solar → electrolysis → H2 → LED)
    │         │
    │         ▼
    │    Photon flux per tray
    │         │
    BACTERIA_TEMPLATES (3 species)
    │         │
    │         ▼
    │    compute_bacteria_model(species_idx)
    │         │
    │         ├──→ photosynthesis_model → CO2 fixed, O2 produced
    │         ├──→ co2_fixation_model → carbon flux
    │         ├──→ oil_biosynthesis_model → oil kg/hr per species
    │         ├──→ growth_model → biomass, doubling
    │         ├──→ genetic_engineering_model → CRISPR boost (2.5x)
    │         ├──→ nutrient_uptake_model → N/P/Fe uptake
    │         ├──→ biofilm_model → surface area enhancement
    │         ├──→ quorum_sensing_model → oil boost (+12%)
    │         ├──→ stress_response_model → stress factor
    │         ├──→ chemoautotrophy_model → nighttime production
    │         └──→ dark_fermentation_model → H2 yield
    │         │
    │         ▼
    │    Per-species oil production (kg/hr)
    │         │
    │         ▼
    │    Sum → Total oil (kg/hr → bbl/day → bbl/yr)
    │         │
    │         ├──→ oil_processing_model → gasoline, jet, diesel, chemicals
    │         ├──→ economic_model → revenue, costs, profit, ROI
    │         ├──→ stability_model → stability index, risk rates
    │         └──→ biomass_turnover_model → renewal, recycling
    │
    └──→ summary() → dict of 200+ metrics
```

### UI Data Flow
```
TowerState
    │
    ├── model: OilProductionModel
    │       │
    │       └── summary() → dict
    │               │
    │               ▼
    │    draw_topbar()     ← current_tab
    │    draw_hud()        ← info_page, scroll_offset
    │    draw_bacteria_tab() ← selected_species, scroll_offset
    │    draw_production_tab() ← scroll_offset
    │    draw_economics_tab() ← scroll_offset
    │    draw_synergy_tab() ← scroll_offset
    │
    ├── renderer: TowerRenderer
    │       │
    │       └── render(screen, rect, interactive=True)
    │
    └── camera: yaw, pitch, zoom, pan_x, pan_y
```

---

## 3D Rendering Pipeline

The renderer is a **pure-Python software renderer** using numpy for matrix operations. No OpenGL.

### Pipeline Steps

1. **Geometry Construction** (`build_tower()`)
   - 291 parts, each with a list of meshes
   - Each mesh has vertices (Nx3 numpy array) and faces (Mx3 indices)
   - Parts: base, ground storage, floors, trays, facade panels, pipes, spout, vents, LED panels, solar panels

2. **Camera Transformation** (`TowerRenderer.render()`)
   - Model rotation: `rot_y(yaw) @ rot_x(pitch)`
   - Translation: pan offsets
   - Perspective projection: `x' = x * f / z`, `y' = y * f / z`
   - Screen mapping: projected coords → pixel coords

3. **Backface Culling**
   - Compute face normal via cross product of edge vectors
   - Dot product with view direction → skip if negative (facing away)

4. **Flat Shading**
   - Light direction: `C_LIGHT_DIR = [0.5, 0.7, 0.6]`
   - Face normal · light direction → brightness factor
   - Color = base_color × brightness (clamped)

5. **Z-Sorting (Painter's Algorithm)**
   - Compute centroid Z for each face
   - Sort faces back-to-front
   - Draw in order (far faces first, near faces last)

6. **Screen Rendering**
   - `pygame.draw.polygon()` for each face
   - Outline mode for cross-section views
   - Label rendering for annotated views

### Performance
- 35,136 vertices, 27,619 faces (large tower)
- ~30-60 FPS on modern hardware
- `_detail_seg()` adjusts geometric detail based on `VISUAL_DETAIL` setting

---

## Biological Model Architecture

Each biological model function follows a consistent pattern:

```python
def some_model(state: BacteriaState, dt: float, params: dict) -> dict:
    """Model one aspect of bacterial physiology."""
    # Extract relevant constants from BIO dict
    kcat = BIO["some_kcat"]
    km = BIO["some_km"]
    
    # Compute rates from state
    rate = kcat * state.some_param / (km + state.some_param)
    
    # Apply engineering boost
    if state.genetic_boost > 0:
        rate *= BIO["crispr_expression_fold"]
    
    # Apply stress penalty
    rate *= (1.0 - state.stress_level * 0.3)
    
    return {
        "rate": rate,
        "product_kg": rate * dt * state.biomass_kg,
        "details": {...},
    }
```

### `compute_bacteria_model(species_idx)` -- Master Function
Integrates all individual model functions for one species and returns a comprehensive result dict:

```python
result = compute_bacteria_model(0)  # B. braunii
# result contains:
#   result['photosynthesis'] = {co2_fixed_kg_hr, o2_produced_kg_hr, ...}
#   result['co2_fixation'] = {carbon_flux_kg_hr, rubisco_efficiency, ...}
#   result['oil_biosynthesis'] = {oil_kg_hr, oil_fraction, pathway, ...}
#   result['growth'] = {growth_rate_per_hr, doubling_time_hr, ...}
#   result['genetic_engineering'] = {boost_factor, expression_fold, ...}
#   result['nutrient_uptake'] = {n_uptake, p_uptake, fe_uptake, ...}
#   result['biofilm'] = {attachment_rate, surface_area_m2, ...}
#   result['quorum_sensing'] = {autoinducer_level, quorum_active, ...}
#   result['stress_response'] = {stress_level, ros_production, ...}
#   result['chemoautotrophy'] = {h2_oxidized, nighttime_oil_kg_hr, ...}
#   result['dark_fermentation'] = {h2_produced_kg_hr, ...}
#   result['o2_balance'] = {net_o2_kg_hr, ...}
#   result['summary'] = {total_oil_kg_hr, total_co2_kg_hr, ...}
```

---

## Temporal Simulation Engine

### `BacteriaState` -- Mutable State Object
Tracks all time-varying parameters:

```python
@dataclass
class BacteriaState:
    species_idx: int
    hour: float
    biomass_kg: float
    growth_rate_per_hr: float
    photo_efficiency: float
    oil_kg_per_hr: float
    co2_fixed_kg_per_hr: float
    o2_produced_kg_per_hr: float
    stress_level: float          # 0-1
    stability: float             # 0-1
    adaptation_level: float      # 0-1
    quorum_active: bool
    contamination_level: float   # 0-1
    nutrient_combined_boost: float
    net_boost: float             # total genetic boost
    # ... plus storage granule levels, biofilm parameters, etc.
```

### `simulate_bacteria(species_idx, duration_hr, dt_hr)`
Runs hour-by-hour simulation:

```
for each timestep dt:
    1. Update photosynthesis (photo_efficiency, photoinhibition, repair)
    2. Update CO2 fixation (RuBisCO kinetics, CCM)
    3. Update oil biosynthesis (pathway flux, N-limitation)
    4. Update growth (biomass yield, maintenance, dark respiration)
    5. Update biofilm (attachment, detachment, diffusion)
    6. Update stress (ROS, heat shock, UV, antioxidants)
    7. Update quorum sensing (autoinducer accumulation, threshold)
    8. Update genetic stability (silencing, drift)
    9. Update nutrient status (N/P/Fe uptake, depletion)
    10. Update adaptation (gradual acclimation to conditions)
    11. Check contamination (probabilistic, with recovery)
    12. Update stability (cascade failure check)
    13. Record time series data point
```

### `BacteriaSimulationResult`
Contains:
- `time_series`: dict of lists (20+ tracked variables)
- `peak_oil_kg_hr`: maximum oil production during simulation
- `avg_oil_kg_hr`: average oil production
- `cumulative_oil_kg`: total oil produced
- `min_stability`: lowest stability reached
- `contamination_events`: list of (hour, severity) tuples
- `stability_failures`: list of (hour, type) tuples
- `final_state`: BacteriaState at end of simulation

---

## Monte Carlo & Scenario Systems

### `run_bacteria_monte_carlo(species_idx, n_runs, seed, duration)`

1. For each of `n_runs` iterations:
   - Perturb 9 parameters from `MONTE_CARLO_PRIORS` using normal distribution
   - Parameters: photosynthesis_max_eff, rubisco_kcat, mep_pathway_flux, phb_synthesis_rate, nitrogen_uptake_rate, heat_shock_threshold, optimal_temp, biomass_yield_on_co2, gene_silencing_rate
   - Run `simulate_bacteria()` with perturbed parameters
   - Record peak oil
2. Sort results, report p5, p50, p95 percentiles

### `compare_bacteria_scenarios(species_idx, duration)`

Runs 9 predefined protocols:
1. **Baseline**: Default parameters
2. **High light**: 2x photon flux
3. **N-limitation**: 0.3x nitrogen (triggers oil accumulation)
4. **CRISPR boost**: 3.0x expression fold (vs 2.5x default)
5. **Low CO2**: 0.5x CO2 concentration
6. **High temp**: +5°C above optimal
7. **Contamination**: 5x contamination risk
8. **No engineering**: 1.0x expression (no CRISPR)
9. **Optimized**: All parameters at best-case values

Reports total oil (kg) for each scenario.

### `assess_bacteria_benefit_risk(species_idx, duration)`

Computes:
- **Benefit score**: `peak_oil / theoretical_max_oil * 100`
- **Risk %**: `contamination_risk + stability_risk + stress_risk + genetic_drift_risk + nutrient_depletion_risk`
- **Recommendation**:
  - risk < 5% → "OPTIMIZE"
  - risk < 15% → "PROCEED WITH MONITORING"
  - risk >= 15% → "REDUCE INTENSITY"

---

## Synergy Computation

### `compute_synergy_factor(state: BacteriaState) -> float`

```python
def compute_synergy_factor(state):
    s = BACTERIA_SYNERGY
    # Each synergy is raised to a power based on how active that synergy is
    
    # Photo → Growth: active when both photo and growth are high
    photo_factor = min(1.0, state.photo_efficiency / 0.3)
    growth_factor = min(1.0, state.growth_rate_per_hr / 0.05)
    synergy1 = s["photo_growth_synergy"] ** (photo_factor * growth_factor)
    
    # Quorum → Oil: active when quorum is active
    quorum_factor = 1.0 if state.quorum_active else 0.5
    synergy2 = s["quorum_oil_synergy"] ** (quorum_factor * 0.5)
    
    # Nutrient → Growth: active when nutrients are available
    nutrient_factor = min(1.0, state.nutrient_combined_boost / 2.0)
    synergy3 = s["nutrient_growth_synergy"] ** (nutrient_factor * 0.5)
    
    # Genetic → Oil: active when genetic boost is high
    genetic_factor = min(1.0, state.net_boost / 3.0)
    synergy4 = s["genetic_oil_synergy"] ** (genetic_factor * 0.5)
    
    # Adaptation → Stress: active when adaptation is high
    adapt_factor = state.adaptation_level
    synergy5 = s["adaptation_stress_synergy"] ** (adapt_factor * 0.3)
    
    # Stack all synergies with global multiplier
    total = synergy1 * synergy2 * synergy3 * synergy4 * synergy5 * s["global_synergy_multiplier"]
    return total
```

The result is a float (typically ~1.51x) representing the combined cross-process synergy multiplier.

---

## UI System Architecture

### Tab System
```python
TAB_MODES = ["tower", "bacteria", "production", "economics", "synergy"]
TAB_LABELS = ["TOWER", "BACTERIA", "PRODUCTION", "ECONOMICS", "SYNERGY"]
```

### Main Loop Architecture
```python
while running:
    # 1. Handle events
    for event in pygame.event.get():
        # Key events → tab switching, scroll, camera, special keys
        # Mouse events → tab clicks, orbit drag, pan, zoom, bacteria clicks
    
    # 2. Update state
    if not paused:
        state.update(dt)  # TowerState.update()
    
    # 3. Render
    screen.fill(BG)
    draw_topbar(screen, font, big_font, W, current_tab)
    content_rect = pygame.Rect(0, TOP_BAR_H, W, H - TOP_BAR_H - 20)
    
    if current_tab == 0:
        renderer.render(screen, content_rect, state, interactive=True)
        if show_info:
            draw_hud(screen, font, big_font, content_rect, state, renderer, info_page, scroll_offset)
    elif current_tab == 1:
        draw_bacteria_tab(screen, font, big_font, content_rect, state, scroll_offset, selected_species)
    elif current_tab == 2:
        draw_production_tab(screen, font, big_font, content_rect, state, scroll_offset)
    elif current_tab == 3:
        draw_economics_tab(screen, font, big_font, content_rect, state, scroll_offset)
    elif current_tab == 4:
        draw_synergy_tab(screen, font, big_font, content_rect, state, scroll_offset)
    
    # 4. Help overlay
    if show_help:
        draw_help_overlay(screen, font, big_font)
    
    # 5. Bottom status bar
    draw_status_bar(screen, font, W, H, current_tab, clock)
    
    pygame.display.flip()
```

### Font System
Two font sizes used throughout:
- `font`: 18px (body text, labels, values)
- `big_font`: 24px (headers, titles, KPI values)

### Panel System
`_panel(surf, x, y, w, h, alpha=200)` draws a semi-transparent rounded rect with border. Uses a cache (`_panel_cache`) keyed by `(w, h, alpha)` to avoid recreating surfaces.

---

## Synergy Tab Rendering Pipeline

The `draw_synergy_tab()` function is the most visually complex tab. Rendering order:

1. **Animated grid background** -- Pulsing alpha grid lines on a cached SRCALPHA surface (only `set_alpha()` varies per frame)
2. **KPI strip** (non-scrolling, always visible) -- 6 cards with top accent bars, value rendered with `big_font` and unit positioned via actual rendered width
3. **Title panel** -- With cached glow effect and left accent stripe
4. **Species headers** (3 columns, 88px height) -- With cached glow, accent bars, ringed dots, 5 text lines (name, role, oil type, oil fraction, live output)
5. **Step boxes** (18 total: 3 species × 6 steps, 96px height each) -- Each with:
   - Cached glow border (pulsing with `sin(t)`)
   - Left accent stripe
   - Ringed number circle
   - Multi-line description (starts at sy+32, 3 lines at 14px spacing)
   - Efficiency ring (circular arc indicator at sy+72)
   - Efficiency bar (horizontal at sy+86)
6. **Vertical flow arrows** (15 total) -- With 3-particle trailing dots (comet effect)
7. **Cross-species synergy arrows** (4 total) -- Route **horizontally** between specific step rows (right edge of source → left edge of target). Each with:
   - Glow underlay (thicker, darker)
   - Main line (24-segment bezier curve with slight vertical offset)
   - Filled triangular arrowhead
   - 3-particle trailing animation
   - Label pill at curve apex (semi-transparent background with colored border)
8. **Legend panel** -- With mini curved arrow icons, items wrap to new line if panel width exceeded; panel height adapts
9. **Synergy multipliers panel** -- Responsive column positions (fractions of `cw`), accent header, gradient mini bars, text truncation with ellipsis, combined total
10. **Scientific proofs panel** -- Responsive claim/reference columns (52% split), blue accent header, text truncation
11. **Tower model output panel** -- Responsive 2-column layout (42% label / 58% value), gold accent header, text truncation

### Helper Functions (synergy tab specific)
- `_glow_rect(surf, x, y, w, h, color, intensity)` -- 3-layer glowing border, **cached** via `_glow_cache` dict keyed by `(w, h, color, round(intensity, 2))`; single surface with 3 border layers drawn at once
- `_accent_header(surf, font, big_font, x, y, w, text, color)` -- Section header with colored left bar and separator line
- `_efficiency_ring(surf, cx, cy, r, frac, color)` -- Circular progress arc drawn as polyline

### Performance Caches (synergy tab)
- `_glow_cache` -- Caches glow border surfaces by `(w, h, color, intensity)` to avoid recreating 3-layer surfaces per call
- `_grid_cache` -- Caches grid background surface by `(cw, rect.h, grid_spacing)`; only `set_alpha()` is called per frame for the pulsing effect
- `_panel_cache` -- (pre-existing) Caches semi-transparent panel surfaces by `(w, h, alpha)`

### Responsive Layout System
All bottom panels (synergy multipliers, proofs, model output) use fractional positions based on `cw` (content width) instead of hardcoded pixel offsets:
- Synergy panel: name at `cx+16`, value at `cw*0.28`, description at `cw*0.42`, bar at `cw*0.82`
- Proofs: claim column at 52% of `cw-32`, reference column fills remainder
- Model output: label at 42% of half-column width, value fills remainder
- Text truncation with ellipsis (`...`) applied when rendered text exceeds available column width

---

## Bacteria Tab Rendering Pipeline

The `draw_bacteria_tab()` function renders animated cell diagrams:

1. **Species selector** (top) -- 3 clickable panels showing species name, type, oil type
2. **Cell diagram** (center) -- Species-specific rendering:
   - **B. braunii**: Chloroplasts with thylakoids, pyrenoid, oil droplets, nucleus, cell wall
   - **Synechocystis**: Carboxysomes, thylakoid membranes, phycobilisomes, nucleoid, PHB granules
   - **R. eutropha**: Hydrogenase complexes, PHB granules (large), nucleoid, S-layer, flagellum
3. **Organelle rendering**: Each organelle is drawn with:
   - `_draw_phospholipid_bilayer()` -- Double ring of dots for membranes
   - `_draw_dna_strands()` -- Animated wavy lines for DNA/chromatin
   - Animated pulsing/rotation based on `synergy_anim_t` or separate animation timer
4. **Hover tooltips** -- `_bacteria_hover_regions` list rebuilt each frame
5. **Detail panel** (right side) -- Shows selected organelle details
6. **Exploded view** -- `_bacteria_explode_amt` animates organelles separating

---

## Economic Model Architecture

The `OilProductionModel` class computes economics in its `__init__`:

```
Section 10d: ECONOMIC MODEL
    ├── Construction cost = $5,000/m² × TOTAL_BIO_AREA + $500K permitting
    ├── Operating costs:
    │   ├── Labor: $500K/yr (fixed)
    │   ├── Maintenance: 2% of construction
    │   └── Insurance: 1% of construction
    ├── Depreciation: construction / 25 years (straight-line)
    ├── Tax: 21% of (revenue - OpEx - depreciation)
    ├── Revenue:
    │   ├── Fuel: saleable_fuel_gal/day × $3.00/gal × 365
    │   └── Carbon: CO2 tons/yr × $50/ton
    ├── Net profit: revenue - OpEx - depreciation - tax
    ├── Payback: total_capex / net_profit
    └── ROI: net_profit / total_capex × 100
```

All economic parameters are in `DIMS` and can be overridden per tower size preset.

---

## Stability & Risk Model

### `stability_model()` -- 5 Independent Failure Modes

Each failure mode has:
- **Base rate** (probability per hour)
- **Severity** (production loss fraction if triggered)
- **Recovery time** (hours to recover)
- **Mitigation** (engineering reduction factor)

```
Risk = base_rate × (1 - mitigation_factor) × (1 + cascade_bonus)

If any risk triggers:
    1. Apply severity × production_loss
    2. Start recovery timer
    3. 5% chance to cascade to another failure mode
    4. During recovery: gradual exponential recovery (not instant)
```

### Stability Index
```
stability_index = 1.0 - average(risk_rates) × severity_weight
```

### Ground Drain System
When `stability_index < 0.80`:
- Oil production diverts to underground storage tank
- Drain efficiency: 90% of lost oil captured
- Storage capacity: 2,000,000 bbl (large tower)
- Depth: 70m, Radius: 25m

---

## Configuration System

### `DIMS` Dictionary
Contains all tower dimension and configuration parameters:
- Tower geometry: `tower_width_m`, `tower_depth_m`, `tower_height_m`, `n_floors`
- Tray parameters: `tray_count_per_floor`, `tray_size_m`, `tray_depth_m`
- Panel parameters: `panel_rows_per_floor`, `facade_transparency`
- Biological: `co2_concentration_pct`, `nutrient_boost`, `temp_c`
- Engineering: `electrolysis_efficiency`, `led_supplement_fraction`
- Economic: `oil_price_per_barrel`, `fuel_price_per_gallon`, `carbon_credit_per_ton`
- Stability: `contamination_risk_base`, `temp_risk_base`, etc.

### `TOWER_SIZES` Dictionary
Five presets (`pilot`, `small`, `medium`, `large`, `mega`), each overriding `DIMS` values.

### `apply_tower_size(size_name)`
```python
def apply_tower_size(size_name):
    global DIMS, FLOOR_AREA, TOTAL_BIO_AREA, FACADE_AREA, CURRENT_SIZE
    preset = TOWER_SIZES[size_name]
    CURRENT_SIZE = size_name
    for key, val in preset.items():
        if key not in ("label", "target_bbl_day", "description"):
            DIMS[key] = val
    FLOOR_AREA = DIMS["tower_width_m"] * DIMS["tower_depth_m"]
    TOTAL_BIO_AREA = FLOOR_AREA * DIMS["floor_area_utilization"] * DIMS["n_floors"]
    FACADE_AREA = 4 * DIMS["tower_width_m"] * DIMS["tower_height_m"]
```

---

## Color & Visual Constants

```python
BG          = (10, 14, 22)     # Dark navy background
C_STRUCTURE = (72, 78, 86)     # Tower structure
C_FACADE    = (40, 80, 120)    # Transparent facade panels
C_TRAY      = (56, 62, 70)     # Bacteria trays
C_LIQUID    = (30, 90, 60)     # Liquid medium
C_OIL       = (60, 40, 20)     # Oil color
C_ACCENT    = (80, 200, 120)   # Green accent (UI)
C_SUN       = (255, 220, 100)  # Sun/light color
C_TEXT      = (224, 230, 238)  # Body text
C_DIM       = (150, 160, 176)  # Dimmed text
C_WARN      = (240, 182, 60)   # Warning color
```

Species colors (used in bacteria and synergy tabs):
- B. braunii: `(100, 200, 120)` -- Green
- Synechocystis: `(100, 160, 240)` -- Blue
- R. eutropha: `(240, 160, 60)` -- Orange

---

## Helper Function Reference

### UI Helpers
| Function | Description |
|----------|-------------|
| `_panel(surf, x, y, w, h, alpha=200)` | Semi-transparent panel with border (cached via `_panel_cache`) |
| `_label(surf, font, text, pos, accent=False)` | Text with background label |
| `_bar(surf, font, x, y, w, frac, color, label, val)` | Progress bar with label |
| `_mini_bar(surf, x, y, w, frac, color)` | Small progress bar |
| `_section_hdr(surf, font, x, y, text, color)` | Section header with underline |
| `_clamp_x(v)` | Clamp x to non-negative |

### Synergy Tab Helpers
| Function | Description |
|----------|-------------|
| `_glow_rect(surf, x, y, w, h, color, intensity)` | 3-layer glowing border (cached via `_glow_cache`) |
| `_accent_header(surf, font, big_font, x, y, w, text, color)` | Section header with colored left bar + separator line |
| `_efficiency_ring(surf, cx, cy, r, frac, color)` | Circular progress arc as polyline |

### Math Helpers
| Function | Description |
|----------|-------------|
| `rot_x(a)`, `rot_y(a)`, `rot_z(a)` | 3D rotation matrices (numpy) |
| `clamp(x, lo, hi)` | Clamp value to range |
| `_mix(c1, c2, t)` | Linear interpolation between two colors |
| `_detail_seg(seg)` | Detail level for geometric segmentation |

---

## Key Data Structures

### `OilProductionModel.summary()` → dict
Returns 200+ metrics including:
- `oil_kg_per_hr`, `oil_barrels_per_day`, `oil_barrels_per_year`
- `co2_uptake_kg_per_hr`, `o2_kg_per_hr`
- `braunii_oil_kg_per_hr`, `synecho_oil_kg_per_hr`, `eutropha_day_oil_kg_per_hr`, `eutropha_night_oil_kg_per_hr`
- `braunii_oil_frac`, `synecho_oil_frac`, `eutropha_oil_frac`
- `h2_storage_kg`, `h2_energy_MWh`
- `biomass_kg`, `biofilm_area_m2`
- `total_boost`, `env_mult`, `genetic_mult`, `light_dist_mult`
- `stability_index`, `stability_loss_fraction`
- `contam_risk`, `temp_risk`, `nutrient_risk`, `ph_risk`, `light_risk`
- `gasoline_kg_per_hr`, `jet_fuel_kg_per_hr`, `diesel_kg_per_hr`, `chemicals_kg_per_hr`
- `revenue_fuel_per_year`, `revenue_carbon_per_year`, `total_revenue_per_year`
- `construction_cost`, `operating_cost_per_year`, `annual_profit`, `payback_years`, `roi_pct`
- `total_solar_power_MW`, `led_power_MW`, `electrolysis_type`
- And many more...

### `BACTERIA_TEMPLATES` → dict
```python
{
    0: {
        "name": "Botryococcus braunii (Race B)",
        "type": "green_alga",
        "oil_type": "botryococcene",
        "oil_fraction_dw": 0.75,
        "doubling_time_hr": 48.0,
        "co2_tolerance_pct": 5.0,
        "genome_size_genes": 14000,
        "gc_content": 0.545,
        "cell_diameter_um": 13.0,
        "photosynthetic_type": "oxygenic",
        "pigments": ["chlorophyll_a", "chlorophyll_b", "carotenoids"],
        "key_pathway": "MEP -> botryococcene (SSL-1/2/3)",
        "harvest_method": "extracellular_excretion",
        "harvest_efficiency": 0.95,
        # ... more fields
    },
    1: { ... },  # Synechocystis
    2: { ... },  # R. eutropha
}
```

### `BIO` → dict
80+ biological constants organized by category:
- Photosynthesis: `photosynthesis_baseline_eff`, `photosynthesis_max_eff`, `photoinhibition_threshold_umol`, `psii_repair_rate`, `quantum_yield_baseline`, `quantum_yield_engineered`
- CO2 fixation: `rubisco_kcat`, `rubisco_km_co2`, `rubisco_km_o2`, `rubisco_specificity_factor`, `carbonic_anhydrase_kcat`
- Oil biosynthesis: `mep_pathway_flux_baseline`, `mep_pathway_flux_engineered`, `phb_synthesis_rate`, `phb_max_fraction`, `botryococcene_export_rate`
- Growth: `biomass_yield_on_co2`, `maintenance_energy_frac`, `dark_respiration_frac`, `n_limitation_oil_boost`
- Biofilm: `biofilm_attachment_rate`, `biofilm_detachment_rate`, `biofilm_density_g_m2`, `biofilm_diffusion_limit`
- Stress: `ros_production_rate`, `antioxidant_capacity_baseline`, `heat_shock_threshold_c`, `uv_damage_rate`
- Quorum: `autoinducer_production_rate`, `quorum_threshold_cells_m2`, `quorum_oil_boost`
- Genetic: `crispr_expression_fold`, `plasmid_copy_number`, `metabolic_burden_frac`, `gene_silencing_rate`
- Nutrients: `nitrogen_uptake_rate`, `phosphorus_uptake_rate`, `iron_uptake_rate`, `trace_metal_boost`
- Temperature: `optimal_temp_c`, `temp_tolerance_range_c`, `q10_growth`, `thermal_death_threshold_c`
- O2: `o2_toxicity_threshold_mg_l`, `o2_production_per_co2`, `o2_consumption_knallgas`
- Storage: `polyphosphate_accumulation_rate`, `cyanophycin_synthesis_rate`, `glycogen_synthesis_rate`

---

## Self-Test Pipeline

`run_selftest()` executes 8 test stages:

| Stage | Test | What It Verifies |
|-------|------|------------------|
| 1 | Building tower geometry | 291 parts, 35K vertices, 28K faces |
| 2 | Production model | All 3 species, mass balance, CO2/O2/H2 |
| 3 | Mass balance | Conservation of mass (CO2 + H2O = Oil + O2 + Biomass) |
| 4 | Bacteria templates | Genome sizes, GC content, gene counts match literature |
| 5 | Render test | Software renderer produces valid output |
| 6 | Bacteria model functions | Per-species oil production, growth rates, genetic boost |
| 7 | Temporal simulation | 168-hour (1 week) simulation with stability tracking |
| 8 | Advanced features | Monte Carlo, scenarios, benefit/risk, synergy factor |

All stages must pass for `SELF TEST PASSED`.

---

## OBJ Export Pipeline

`export_obj(path="OilFabricationTower")`:

1. Build full tower geometry (`build_tower(lite=False)`)
2. For each part, for each mesh:
   - Write vertices as `v x y z`
   - Track material (color → MTL entry)
   - Write faces as `f v1 v2 v3` (with vertex offset)
3. Write `.mtl` file with material definitions (Kd color, Ke emission)
4. Output files: `OilFabricationTower.obj` (1.7 MB) + `OilFabricationTower.mtl` (12 KB)

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| File size | ~495 KB (OilFabricationTower.py) |
| Lines of code | ~8,250 |
| Tower parts | 291 |
| Vertices | 35,136 |
| Faces | 27,619 |
| BIO constants | 80+ |
| Bacteria templates | 3 species |
| Model functions | 20+ |
| Summary dict keys | 200+ |
| UI tabs | 5 |
| Render FPS | 30-60 (typical) |
| Self-test time | ~5 seconds |
| Dependencies | pygame, numpy |
