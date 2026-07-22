# Plant Simulation Reality Audit

## Scope

This audit evaluates how closely the simulation models real plant behavior. It deliberately excludes browser, rendering, UI, and runtime concerns.

## Summary

The project is a strong qualitative houseplant simulation: approximately **3/5 for biological structure**, but it is not yet quantitatively realistic. It models many correct causal directions—water, light, temperature, carbon, roots, senescence, and species habits—and it has unusually thorough internal tests. The principal gap is that normalized game values are tuned for visible play rather than calibrated to biological units or observed plant trajectories.

## Existing strengths

- The model separates photosynthetic income from continuous respiration and requires banked carbon plus water for elongation. See `engine/src/sim/plant.rs`.
- It represents drought, root damage, pot binding, nutrient limitation, leaf aging, self-shading, and species-specific growth habits as stateful mechanisms rather than visual effects.
- Canopy self-shading produces diminishing returns instead of unlimited leaf accumulation.
- The native simulation suite has 286 passing tests, including long-running playthrough and regression tests. This is strong evidence of internal consistency.

## Highest-priority improvements

### 1. Establish a biological time scale

**Finding:** One full day is 400 simulation seconds, or five real seconds, and physiological rates are tuned to this compressed clock (`engine/src/sim/config.rs`, `TimeConfig`). Leaf lifespan is similarly expressed in game seconds.

**Impact:** The game can demonstrate plant processes quickly, but its durations cannot be interpreted as real days, weeks, or seasons.

**Improve:** Define a canonical model clock in hours or days. Accelerate presentation independently of that clock. Keep rates in explicit units such as leaf area per day, carbon per day, or water per day.

**Confidence:** High.

### 2. Replace visual light intensity with plant-usable radiation

**Finding:** Photosynthesis multiplies leaf area by a `0..1` sun intensity and an efficiency constant. The model has no PPFD/PAR, daily light integral (DLI), orientation, window transmission, or photosynthetic saturation. An ambient light floor remains available even when the plant is beyond the window-light zone.

**Impact:** Low-light tolerance, growth ceilings, and differences among window positions cannot be calibrated against real indoor conditions.

**Improve:** Calculate PPFD at the plant, integrate it to DLI, and use a saturating photosynthesis response. A simple rectangular-hyperbola response would be a substantial upgrade without requiring a full leaf-energy model.

**Confidence:** High.

### 3. Rework VPD and stomatal coupling

**Status:** Implemented. `Humidity::vpd_factor` now uses a Tetens saturation-vapor-pressure calculation and derives VPD from temperature and relative humidity (`engine/src/sim/humidity.rs`). Dry air therefore produces a VPD at ordinary room temperatures instead of only above an arbitrary temperature threshold.

**Status:** Implemented. VPD and effective root-water availability now combine into a shared stomatal-conductance factor. That factor gates both transpiration and CO2 assimilation, preventing the two processes from diverging under atmospheric or root stress.

**Remaining limitation:** The VPD closure curve is a shared heuristic, not yet a calibrated species-specific stomatal model.

**Confidence:** High.

### 4. Model the root zone rather than a single soil-water scalar

**Finding:** Soil moisture and nutrient availability are independent scalar reservoirs with linear thresholds (`engine/src/sim/soil.rs`). The model does not represent pot volume, substrate type, water potential, drainage, air-filled porosity, root biomass, or salt concentration.

**Impact:** Waterlogging, underwatering, fertilizer burn, and repotting are directionally plausible but cannot reproduce realistic timing or tradeoffs across potting media.

**Improve:** Add a compact root-zone subsystem: water content to water potential, air-filled porosity to oxygen stress, and electrical conductivity to salinity stress. It can stay aggregate rather than spatial.

**Confidence:** High.

### 5. Separate species physiology from shared defaults

**Finding:** Peace lily and pothos are largely overrides of a Dracaena configuration (`engine/src/sim/config.rs`, `PlantConfig::peace_lily` and `PlantConfig::pothos`). They retain much of the same response curves, pest response, senescence, mechanics, and leaf-movement behavior.

**Impact:** The species differ visibly and architecturally, but not enough physiologically. In particular, applying pronounced daily folding and heliotropism to every selected species is not biologically credible.

**Improve:** Split traits into physiology, shoot architecture, phenology, mechanics, and movement capabilities. Make leaf movement opt-in by species.

**Confidence:** High.

### 6. Treat branching and flowering as developmental state transitions

**Finding:** Branches appear after height and carbon thresholds; bloom cycles are explicitly cosmetic and do not affect carbon allocation (`engine/src/sim/plant.rs`).

**Impact:** Dracaena behavior is plausible in silhouette, but branch timing, flowering, pruning response, and resource competition are simplified.

**Improve:** Represent shoot meristems as active, dormant lateral bud, terminal inflorescence, and damaged/pruned. Make flowering and branch release compete for carbon and depend on species-specific conditions.

**Confidence:** Medium-high.

### 7. Model pests as populations with introduction events

**Finding:** Pest pressure grows deterministically whenever humidity is below a threshold (`engine/src/sim/pests.rs`).

**Impact:** Dry conditions can create an infestation from nothing. The model also omits disease risks that can rise under persistently humid conditions.

**Improve:** Separate pest presence from environmental suitability. Model acquisition, reproduction, treatment, and optional dispersal; treat humidity as a modifier rather than the source of pests.

**Confidence:** High.

## Architecture recommendations

The separation between `sim`, rendering, and configuration is sound. The next improvement is to decompose the large `Plant::step` flow into domain systems:

```text
Environment -> root-zone hydraulics -> stomata/gas exchange
            -> carbon allocation -> development/meristems
            -> structural mechanics -> mortality/biotics
```

This would reduce coupling in `Plant::step` and permit more meaningful species differentiation. Prefer distinct state and trait objects over a growing monolithic `PlantConfig`.

Recommended order of work:

1. Add explicit biological units and a canonical simulation clock.
2. Implement PPFD/DLI and real VPD calculations.
3. Extract a root-zone hydraulics subsystem.
4. Split species traits from mutable plant state.
5. Add calibration tests based on measured or published trajectories.

## Validation strategy

Current tests strongly prove internal consistency: monotonic responses, bounded leaf count, intended lifecycle behavior, and gameplay pacing. Add validation tests that compare model outputs with observed ranges instead:

- Height, leaf-count, and biomass ranges after a specified number of model days.
- Growth responses across known DLI, temperature, and humidity scenarios.
- Dry-down and recovery curves for a defined pot/substrate volume.
- Sensitivity analysis showing which parameters dominate outcomes.
- Species-specific scenario tests, especially for low-light growth, branching, and leaf movement.

Do not assign precise biological constants without a cited data source. Use explicit calibration targets and uncertainty ranges when empirical data becomes available.
