//! Soil water and nutrient reservoir. A real pot has finite water-holding
//! capacity (field capacity) — excess drains away rather than pooling
//! indefinitely — and loses moisture to evaporation (faster in light/heat)
//! and to whatever the plant actually draws out through its roots. It also
//! holds a second, independent reservoir — dissolved nutrients — with its
//! own two-sided failure mode (starvation *and* overdose, see
//! `nutrient_factor`/`overfeed_stress`), and can be kept unrealistically
//! oversaturated long enough to suffocate roots (see `waterlog_stress`).
//! Tunable numbers live in `config::SoilConfig`, passed in rather than
//! hardcoded here.

use super::config::SoilConfig;

#[derive(Debug, Clone, Copy)]
pub struct Soil {
    /// Fraction of field capacity, 0.0 (bone dry) ..= 1.0 (saturated).
    pub moisture: f64,
    /// Standing dissolved-nutrient level — unlike `moisture` this has no
    /// hard physical ceiling from field capacity (real fertilizer salts
    /// keep accumulating the more you add), only the soft
    /// `SoilConfig::max_nutrient` clamp `fertilize` applies.
    pub nutrient: f64,
}

/// A generic, ungated baseline (full moisture, ample nutrient) — exists so
/// tests that only care about one field can write `Soil { moisture: X,
/// ..Default::default() }` instead of naming every field. Not used by
/// `Soil::new` itself, which always derives both fields from a real
/// `SoilConfig` instead.
impl Default for Soil {
    fn default() -> Self {
        Soil { moisture: 1.0, nutrient: 1.0 }
    }
}

impl Soil {
    pub fn new(config: &SoilConfig) -> Self {
        Soil {
            moisture: config.initial_moisture,
            nutrient: config.initial_nutrient,
        }
    }

    /// Adds water (e.g. from a watering-can action), clamped at field
    /// capacity — a draining pot can't hold more than that no matter how
    /// much is poured in.
    pub fn water(&mut self, amount: f64) {
        self.moisture = (self.moisture + amount.max(0.0)).min(1.0);
    }

    /// Adds fertilizer, clamped at `SoilConfig::max_nutrient` — see that
    /// field's doc comment on why this is a soft bound, not a physical one
    /// the way `water`'s field-capacity clamp is.
    pub fn fertilize(&mut self, amount: f64, config: &SoilConfig) {
        self.nutrient = (self.nutrient + amount.max(0.0)).min(config.max_nutrient);
    }

    /// Advances soil moisture and nutrient by `dt` seconds: evaporation
    /// (scaled by `light_intensity`, since heat/light drives surface
    /// evaporation) plus whatever the plant drew out via `uptake_rate`
    /// (fraction of field capacity per second, computed from the plant's
    /// own transpiration) depletes moisture; that same uptake (nutrients
    /// move with transpiration's mass flow) plus a slow background leach
    /// rate depletes nutrient.
    pub fn update(&mut self, dt: f64, light_intensity: f64, uptake_rate: f64, config: &SoilConfig) {
        // Some evaporation continues at night (residual warmth), but most
        // of it tracks daytime light/heat.
        let evaporation = config.evaporation_rate_per_sec * (0.25 + 0.75 * light_intensity) * dt;
        self.moisture = (self.moisture - evaporation - uptake_rate * dt).max(0.0);

        let nutrient_drawn = uptake_rate * config.nutrient_uptake_coeff * dt;
        let nutrient_leached = config.nutrient_leach_rate_per_sec * dt;
        self.nutrient = (self.nutrient - nutrient_drawn - nutrient_leached).max(0.0);
    }

    /// 1.0 above the gating threshold, ramping linearly to 0.0 at bone dry
    /// — see `SoilConfig::moisture_gate_threshold`.
    pub fn water_factor(&self, config: &SoilConfig) -> f64 {
        (self.moisture / config.moisture_gate_threshold).clamp(0.0, 1.0)
    }

    /// 1.0 above the gating threshold, ramping linearly to 0.0 at zero
    /// nutrient — same shape as `water_factor`, see
    /// `SoilConfig::nutrient_gate_threshold`.
    pub fn nutrient_factor(&self, config: &SoilConfig) -> f64 {
        (self.nutrient / config.nutrient_gate_threshold).clamp(0.0, 1.0)
    }

    /// 0.0 below `waterlogged_threshold`, ramping to 1.0 at full saturation
    /// — how oxygen-starved the root zone is right now. Doesn't by itself
    /// track *duration*; see `Plant::waterlogged_duration`, which is what
    /// actually gates whether root damage occurs (a brief touch of
    /// saturation right after watering shouldn't matter, only sustained
    /// flooding).
    pub fn waterlog_stress(&self, config: &SoilConfig) -> f64 {
        let range = (1.0 - config.waterlogged_threshold).max(1e-9);
        ((self.moisture - config.waterlogged_threshold) / range).clamp(0.0, 1.0)
    }

    /// 0.0 below `overfeed_threshold`, ramping to 1.0 at `max_nutrient` —
    /// osmotic "fertilizer burn" stress, the overdose-direction twin of
    /// `waterlog_stress`.
    pub fn overfeed_stress(&self, config: &SoilConfig) -> f64 {
        let range = (config.max_nutrient - config.overfeed_threshold).max(1e-9);
        ((self.nutrient - config.overfeed_threshold) / range).clamp(0.0, 1.0)
    }

    /// Returns how much of `budget` it actually spent.
    pub fn apply_auto_water(&mut self, enabled: bool, budget: f64, price_per_unit: f64, config: &SoilConfig) -> f64 {
        if !enabled || self.moisture >= config.auto_water_floor {
            return 0.0;
        }
        let full_delta = config.auto_water_floor - self.moisture;
        if price_per_unit <= 0.0 {
            self.moisture = config.auto_water_floor;
            return 0.0;
        }
        let full_cost = full_delta * price_per_unit;
        if full_cost <= budget {
            self.moisture = config.auto_water_floor;
            full_cost
        } else {
            let affordable_delta = (budget / price_per_unit).min(full_delta).max(0.0);
            self.moisture += affordable_delta;
            affordable_delta * price_per_unit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SoilConfig {
        SoilConfig::default()
    }

    #[test]
    fn evaporation_dries_out_soil_over_time() {
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let config = config();
        for _ in 0..1000 {
            soil.update(1.0, 1.0, 0.0, &config);
        }
        assert!(soil.moisture < 1.0);
    }

    #[test]
    fn watering_cannot_exceed_field_capacity() {
        let mut soil = Soil { moisture: 0.9, ..Default::default() };
        soil.water(5.0);
        assert_eq!(soil.moisture, 1.0);
    }

    #[test]
    fn watering_never_goes_negative_even_with_bad_input() {
        let mut soil = Soil { moisture: 0.1, ..Default::default() };
        soil.water(-5.0); // shouldn't happen, but shouldn't drain the pot either
        assert!(soil.moisture >= 0.1);
    }

    #[test]
    fn water_factor_saturates_above_threshold_and_hits_zero_when_dry() {
        let config = config();
        assert_eq!(Soil { moisture: 1.0, ..Default::default() }.water_factor(&config), 1.0);
        assert_eq!(
            Soil { moisture: config.moisture_gate_threshold, ..Default::default() }
            .water_factor(&config),
            1.0
        );
        assert_eq!(Soil { moisture: 0.0, ..Default::default() }.water_factor(&config), 0.0);
        let half = Soil { moisture: config.moisture_gate_threshold / 2.0, ..Default::default() }
        .water_factor(&config);
        assert!((half - 0.5).abs() < 1e-9);
    }

    #[test]
    fn fertilize_raises_nutrient_clamped_at_the_soft_max() {
        let config = config();
        let mut soil = Soil { nutrient: 1.5, ..Default::default() };
        soil.fertilize(10.0, &config);
        assert_eq!(soil.nutrient, config.max_nutrient, "expected the soft overdose ceiling to clamp a huge dose");
    }

    #[test]
    fn nutrient_factor_saturates_above_threshold_and_hits_zero_when_depleted() {
        let config = config();
        assert_eq!(Soil { nutrient: 1.0, ..Default::default() }.nutrient_factor(&config), 1.0);
        assert_eq!(Soil { nutrient: 0.0, ..Default::default() }.nutrient_factor(&config), 0.0);
        let half = Soil { nutrient: config.nutrient_gate_threshold / 2.0, ..Default::default() }
            .nutrient_factor(&config);
        assert!((half - 0.5).abs() < 1e-9);
    }

    #[test]
    fn nutrient_depletes_via_uptake_and_slow_leaching() {
        let config = config();
        let mut soil = Soil { nutrient: 1.0, ..Default::default() };
        for _ in 0..1000 {
            soil.update(1.0, 1.0, 0.001, &config);
        }
        assert!(soil.nutrient < 1.0, "expected nutrient to deplete over time from uptake and leaching");
    }

    #[test]
    fn waterlog_stress_is_zero_below_threshold_and_ramps_to_one_at_saturation() {
        let config = config();
        assert_eq!(Soil { moisture: config.waterlogged_threshold, ..Default::default() }.waterlog_stress(&config), 0.0);
        assert_eq!(Soil { moisture: 1.0, ..Default::default() }.waterlog_stress(&config), 1.0);
        assert_eq!(Soil { moisture: 0.5, ..Default::default() }.waterlog_stress(&config), 0.0);
    }

    #[test]
    fn overfeed_stress_is_zero_below_threshold_and_ramps_to_one_at_the_soft_max() {
        let config = config();
        assert_eq!(Soil { nutrient: config.overfeed_threshold, ..Default::default() }.overfeed_stress(&config), 0.0);
        assert_eq!(Soil { nutrient: config.max_nutrient, ..Default::default() }.overfeed_stress(&config), 1.0);
        assert_eq!(Soil { nutrient: 0.5, ..Default::default() }.overfeed_stress(&config), 0.0);
    }

    #[test]
    fn plant_uptake_dries_soil_faster_than_evaporation_alone() {
        let config = config();
        let mut with_uptake = Soil { moisture: 1.0, ..Default::default() };
        let mut without_uptake = Soil { moisture: 1.0, ..Default::default() };
        for _ in 0..100 {
            with_uptake.update(1.0, 1.0, 0.001, &config);
            without_uptake.update(1.0, 1.0, 0.0, &config);
        }
        assert!(with_uptake.moisture < without_uptake.moisture);
    }

    #[test]
    fn auto_water_tops_up_below_the_floor_when_enabled() {
        let config = config();
        let mut soil = Soil { moisture: 0.1, ..Default::default() };
        soil.apply_auto_water(true, 0.0, 0.0, &config);
        assert_eq!(soil.moisture, config.auto_water_floor);
    }

    #[test]
    fn auto_water_does_nothing_when_disabled() {
        let config = config();
        let mut soil = Soil { moisture: 0.1, ..Default::default() };
        soil.apply_auto_water(false, 0.0, 0.0, &config);
        assert_eq!(soil.moisture, 0.1, "disabled auto-water shouldn't touch moisture at all");
    }

    #[test]
    fn auto_water_never_lowers_moisture_thats_already_above_the_floor() {
        let config = config();
        let mut soil = Soil { moisture: 0.9, ..Default::default() };
        soil.apply_auto_water(true, 0.0, 0.0, &config);
        assert_eq!(soil.moisture, 0.9, "auto-water is a floor, not a target to snap to");
    }

    #[test]
    fn auto_water_keeps_moisture_from_ever_dropping_below_the_floor_over_time() {
        let config = config();
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        for _ in 0..100_000 {
            // Heavy draw (bright light, thirsty plant) — without auto-water
            // this would crash straight to bone dry.
            soil.update(1.0, 1.0, 0.01, &config);
            soil.apply_auto_water(true, 0.0, 0.0, &config);
        }
        assert_eq!(soil.moisture, config.auto_water_floor);
    }

    #[test]
    fn auto_water_spends_exactly_the_delta_times_price_when_affordable() {
        let config = config();
        let mut soil = Soil { moisture: 0.1, ..Default::default() };
        let delta = config.auto_water_floor - 0.1;
        let spent = soil.apply_auto_water(true, 100.0, 2.0, &config);
        assert!((spent - delta * 2.0).abs() < 1e-9);
        assert_eq!(soil.moisture, config.auto_water_floor);
    }

    #[test]
    fn auto_water_does_nothing_when_the_budget_is_zero() {
        let config = config();
        let mut soil = Soil { moisture: 0.1, ..Default::default() };
        let spent = soil.apply_auto_water(true, 0.0, 2.0, &config);
        assert_eq!(spent, 0.0);
        assert_eq!(soil.moisture, 0.1);
    }

    #[test]
    fn auto_water_partially_tops_up_when_the_budget_only_covers_part_of_the_gap() {
        let config = config();
        let mut soil = Soil { moisture: 0.1, ..Default::default() };
        let full_delta = config.auto_water_floor - 0.1;
        let budget = full_delta * 2.0 * 0.5;
        let spent = soil.apply_auto_water(true, budget, 2.0, &config);
        assert!((spent - budget).abs() < 1e-9, "expected to spend the whole (insufficient) budget, got {spent}");
        assert!(soil.moisture > 0.1 && soil.moisture < config.auto_water_floor);
    }
}
