// Classifies each HUD gauge by how it's currently affecting the plant, not
// just its raw number — kept separate from EngineCanvas.tsx so it's unit-
// testable without a wasm module or a canvas, same reasoning as
// formatStats.ts. Thresholds mirror the engine's own config defaults
// (engine/src/sim/config.rs) rather than being arbitrary UI choices, so a
// "caution"/"bad" color lines up with an actual game mechanic kicking in.

export type HealthTier = "good" | "caution" | "bad";

// --- Water (Stats::water_level, soil moisture 0..1) ----------------------
// Mirrors SoilConfig::moisture_gate_threshold (0.35 — growth is drought-
// gated below this) and SoilConfig::waterlogged_threshold (0.97 — root rot
// risk begins above this). The "caution" bands are a UI-only margin before
// each hard threshold, so a player sees it coming rather than only after
// growth is already gated or roots are already damaged.
const WATER_DRY_BAD = 0.2;
const WATER_DRY_CAUTION = 0.35;
const WATER_WET_CAUTION = 0.9;
const WATER_WET_BAD = 0.97;

export function waterTier(level: number): HealthTier {
  if (level < WATER_DRY_BAD || level > WATER_WET_BAD) return "bad";
  if (level < WATER_DRY_CAUTION || level > WATER_WET_CAUTION) return "caution";
  return "good";
}

// --- Nutrient (Stats::nutrient_level, 0..1 typical, can exceed 1.0) ------
// Mirrors SoilConfig::nutrient_gate_threshold (0.1 — starved/growth-gated
// below this) and SoilConfig::overfeed_threshold (1.4 — fertilizer-burn
// root damage above this).
const NUTRIENT_STARVED_BAD = 0.05;
const NUTRIENT_STARVED_CAUTION = 0.1;
const NUTRIENT_OVERFED_CAUTION = 1.2;
const NUTRIENT_OVERFED_BAD = 1.4;

export function nutrientTier(level: number): HealthTier {
  if (level < NUTRIENT_STARVED_BAD || level > NUTRIENT_OVERFED_BAD) return "bad";
  if (level < NUTRIENT_STARVED_CAUTION || level > NUTRIENT_OVERFED_CAUTION) return "caution";
  return "good";
}

// --- Humidity (Stats::humidity_level, 0..1) ------------------------------
// Mirrors PestConfig::safe_humidity (0.5 — pests start thriving below
// this).
const HUMIDITY_BAD = 0.35;
const HUMIDITY_CAUTION = 0.5;

export function humidityTier(level: number): HealthTier {
  if (level < HUMIDITY_BAD) return "bad";
  if (level < HUMIDITY_CAUTION) return "caution";
  return "good";
}

// --- Root health (Stats::root_health, 0..1, higher is better) ------------
const ROOT_HEALTH_BAD = 0.4;
const ROOT_HEALTH_CAUTION = 0.7;

export function rootHealthTier(level: number): HealthTier {
  if (level < ROOT_HEALTH_BAD) return "bad";
  if (level < ROOT_HEALTH_CAUTION) return "caution";
  return "good";
}

// --- Pest infestation (Stats::pest_infestation, 0..1, lower is better) ---
const PEST_BAD = 0.4;
const PEST_CAUTION = 0.15;

export function pestTier(level: number): HealthTier {
  if (level > PEST_BAD) return "bad";
  if (level > PEST_CAUTION) return "caution";
  return "good";
}

// --- Temperature (Stats::temperature_c) ----------------------------------
// Mirrors PlantConfig::optimal_temperature_c (24), ::temperature_tolerance_c
// (10, the bell-curve falloff width used for the *caution* band here) and
// ::cold_stress_threshold_c (12 — a hard cold-stress cutoff the engine
// itself gates on). The engine has no equivalent named hot-side cutoff, so
// this mirrors the cold cutoff's own distance from optimal onto the hot
// side as a reasonable symmetric stand-in.
const TEMP_OPTIMAL_C = 24;
const TEMP_CAUTION_BAND_C = 5; // half of temperature_tolerance_c
const TEMP_COLD_BAD_C = 12;
const TEMP_HOT_BAD_C = TEMP_OPTIMAL_C + (TEMP_OPTIMAL_C - TEMP_COLD_BAD_C);

export function temperatureTier(temperatureC: number): HealthTier {
  if (temperatureC < TEMP_COLD_BAD_C || temperatureC > TEMP_HOT_BAD_C) return "bad";
  if (Math.abs(temperatureC - TEMP_OPTIMAL_C) > TEMP_CAUTION_BAND_C) return "caution";
  return "good";
}
