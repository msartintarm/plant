//! Ambient air humidity — a separate reservoir from `soil::Soil`'s moisture,
//! representing dry indoor air rather than the pot's own water content.
//! Modeled as small, explicit, caller-owned state the same way `Soil` is
//! (`Humidity::update` is called by whoever drives the loop, same as
//! `Soil::update`), rather than folded into `Plant`/`Plant::step` itself —
//! this keeps `Plant::step`'s own signature stable (it takes the *current*
//! humidity level as a plain `f64`, the same way it already takes `&SunState`/
//! `&ClimateState` as pre-computed snapshots) while still letting a player
//! action (`mist`) have a real, persisting effect across ticks.

use super::config::HumidityConfig;

#[derive(Debug, Clone, Copy)]
pub struct Humidity {
    /// 0.0 (bone dry indoor air) ..= 1.0 (saturated/humid).
    pub level: f64,
}

impl Humidity {
    pub fn new(config: &HumidityConfig) -> Self {
        Humidity { level: config.initial_humidity }
    }

    /// Misting: raises humidity immediately, clamped at full saturation.
    pub fn mist(&mut self, amount: f64) {
        self.level = (self.level + amount.max(0.0)).min(1.0);
    }

    /// Eases humidity back down toward `dry_air_floor` — faster in heat
    /// (warm air both holds and loses moisture capacity faster than cool
    /// air), same idiom as every other eased target in this codebase
    /// (`Leaf::droop`, `Plant::bloom_intensity`, etc).
    pub fn update(&mut self, dt: f64, temperature_c: f64, config: &HumidityConfig) {
        // Never below some floor multiplier — a cold room's air still dries
        // out eventually, just slower, not instantly.
        let heat_factor = (temperature_c / config.vpd_reference_temperature_c).max(0.3);
        let rate = (config.decay_rate_per_sec * heat_factor * dt).min(1.0);
        self.level += (config.dry_air_floor - self.level) * rate;
        self.level = self.level.clamp(0.0, 1.0);
    }

    /// Saturation vapor pressure (kPa) from air temperature using the FAO-56
    /// Tetens approximation. This is the pressure of water vapor in fully
    /// saturated air at `temperature_c`; actual indoor RH determines how
    /// much of that capacity is already occupied.
    pub fn saturation_vapor_pressure_kpa(temperature_c: f64) -> f64 {
        0.6108 * (17.27 * temperature_c / (temperature_c + 237.3)).exp()
    }

    /// Vapor-pressure deficit (kPa): the gap between saturated air and the
    /// air's actual water-vapor pressure. Unlike the previous heuristic, it
    /// is nonzero whenever RH is below 100%, including at normal indoor
    /// temperatures.
    pub fn vapor_pressure_deficit_kpa(&self, temperature_c: f64) -> f64 {
        let relative_humidity = self.level.clamp(0.0, 1.0);
        Self::saturation_vapor_pressure_kpa(temperature_c) * (1.0 - relative_humidity)
    }

    /// VPD-driven multiplier on transpiration. The multiplier is neutral at
    /// saturation, then rises with the physically calculated VPD relative to
    /// the configured reference. Stomatal closure remains a future model
    /// extension; this function only expresses the atmospheric pull on a
    /// transpiring, well-watered leaf.
    pub fn vpd_factor(&self, temperature_c: f64, config: &HumidityConfig) -> f64 {
        let reference = config.vpd_reference_kpa.max(1e-9);
        1.0 + config.vpd_strength * self.vapor_pressure_deficit_kpa(temperature_c) / reference
    }

    /// Fraction of maximum stomatal conductance available to the leaf. Dry
    /// roots restrict opening directly; high VPD closes stomata progressively
    /// to conserve water. The same output must gate both CO2 assimilation
    /// and transpiration so the two processes cannot diverge.
    pub fn stomatal_conductance_factor(
        &self,
        root_water_factor: f64,
        temperature_c: f64,
        config: &HumidityConfig,
    ) -> f64 {
        let root_opening = root_water_factor.clamp(0.0, 1.0);
        let closure_scale = config.stomatal_vpd_closure_kpa.max(1e-9);
        let vpd = self.vapor_pressure_deficit_kpa(temperature_c);
        // A squared response keeps modest indoor VPD near the open state
        // while still producing a strong closure response once VPD reaches
        // the configured half-closure scale.
        let vpd_ratio = vpd / closure_scale;
        let vpd_opening = 1.0 / (1.0 + vpd_ratio * vpd_ratio);
        let residual = config.stomatal_min_conductance.clamp(0.0, 1.0);
        root_opening * (residual + (1.0 - residual) * vpd_opening)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> HumidityConfig {
        HumidityConfig::default()
    }

    #[test]
    fn misting_raises_humidity_clamped_at_saturation() {
        let mut humidity = Humidity { level: 0.4 };
        humidity.mist(0.3);
        assert!((humidity.level - 0.7).abs() < 1e-9);
        humidity.mist(10.0);
        assert_eq!(humidity.level, 1.0);
    }

    #[test]
    fn humidity_dries_back_out_toward_the_dry_air_floor_over_time() {
        let config = config();
        let mut humidity = Humidity { level: 1.0 };
        for _ in 0..100_000 {
            humidity.update(1.0, config.vpd_reference_temperature_c, &config);
        }
        assert!((humidity.level - config.dry_air_floor).abs() < 1e-6);
    }

    #[test]
    fn humidity_dries_out_faster_in_heat() {
        let config = config();
        let mut cool = Humidity { level: 1.0 };
        let mut hot = Humidity { level: 1.0 };
        for _ in 0..500 {
            cool.update(1.0, config.vpd_reference_temperature_c, &config);
            hot.update(1.0, config.vpd_reference_temperature_c * 1.5, &config);
        }
        assert!(hot.level < cool.level, "expected hotter air to dry out faster: hot {} vs cool {}", hot.level, cool.level);
    }

    #[test]
    fn saturation_vapor_pressure_matches_a_known_room_temperature_value() {
        let saturation = Humidity::saturation_vapor_pressure_kpa(20.0);
        assert!((saturation - 2.338).abs() < 0.01, "expected about 2.338 kPa at 20 C, got {saturation}");
    }

    #[test]
    fn vpd_is_nonzero_in_dry_air_at_ordinary_room_temperature() {
        let dry = Humidity { level: 0.0 };
        let vpd = dry.vapor_pressure_deficit_kpa(20.0);
        assert!((vpd - 2.338).abs() < 0.01, "expected dry-air VPD near saturation pressure, got {vpd}");
        assert!(dry.vpd_factor(20.0, &config()) > 1.0);
    }

    #[test]
    fn vpd_factor_is_neutral_at_full_saturation_even_when_hot() {
        let config = config();
        let humid = Humidity { level: 1.0 };
        let factor = humid.vpd_factor(config.vpd_reference_temperature_c + 10.0, &config);
        assert_eq!(factor, 1.0, "fully saturated air has no room left to pull extra water from leaves");
    }

    #[test]
    fn vpd_factor_rises_with_heat_in_dry_air() {
        let config = config();
        let dry = Humidity { level: 0.0 };
        let mild = dry.vpd_factor(20.0, &config);
        let hot = dry.vpd_factor(35.0, &config);
        assert!(hot > mild, "expected more heat to pull transpiration up further: {hot} vs {mild}");
        assert!(mild > 1.0);
    }

    #[test]
    fn vpd_factor_is_worse_in_drier_air_at_the_same_hot_temperature() {
        let config = config();
        let dry = Humidity { level: 0.1 };
        let humid = Humidity { level: 0.9 };
        let hot_temp = config.vpd_reference_temperature_c + 10.0;
        assert!(
            dry.vpd_factor(hot_temp, &config) > humid.vpd_factor(hot_temp, &config),
            "expected dry air to amplify transpiration more than humid air at the same heat"
        );
    }

    #[test]
    fn stomata_close_under_high_vpd_but_close_fully_when_roots_are_dry() {
        let config = config();
        let humid = Humidity { level: 0.95 };
        let dry = Humidity { level: 0.1 };
        let humid_opening = humid.stomatal_conductance_factor(1.0, 25.0, &config);
        let dry_opening = dry.stomatal_conductance_factor(1.0, 25.0, &config);
        assert!(dry_opening < humid_opening, "high VPD should close stomata: dry {dry_opening}, humid {humid_opening}");
        assert_eq!(dry.stomatal_conductance_factor(0.0, 25.0, &config), 0.0);
    }
}
