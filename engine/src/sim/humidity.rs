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

    /// Vapor-pressure-deficit-style multiplier on transpiration: hot *and*
    /// dry air pulls dramatically more water out of leaves than either
    /// factor alone — exactly 1.0 (no adjustment) at or below `vpd_
    /// reference_temperature_c`, or whenever humidity is already at 1.0
    /// (fully saturated air has no additional pull left to add), growing
    /// with both how much hotter than the reference it is and how dry the
    /// air currently is.
    pub fn vpd_factor(&self, temperature_c: f64, config: &HumidityConfig) -> f64 {
        let heat_above_ref =
            ((temperature_c - config.vpd_reference_temperature_c) / config.vpd_reference_temperature_c).max(0.0);
        let dryness = 1.0 - self.level.clamp(0.0, 1.0);
        1.0 + config.vpd_strength * heat_above_ref * dryness
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
    fn vpd_factor_is_neutral_at_or_below_the_reference_temperature_regardless_of_humidity() {
        let config = config();
        let dry = Humidity { level: 0.0 };
        let humid = Humidity { level: 1.0 };
        assert_eq!(dry.vpd_factor(config.vpd_reference_temperature_c, &config), 1.0);
        assert_eq!(humid.vpd_factor(config.vpd_reference_temperature_c, &config), 1.0);
        assert_eq!(dry.vpd_factor(config.vpd_reference_temperature_c - 5.0, &config), 1.0);
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
        let mild = dry.vpd_factor(config.vpd_reference_temperature_c + 5.0, &config);
        let hot = dry.vpd_factor(config.vpd_reference_temperature_c + 15.0, &config);
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
}
