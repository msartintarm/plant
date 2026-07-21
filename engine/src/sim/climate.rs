//! Ambient temperature as a pure function of where we are in the day/night
//! cycle (`sun::DayProgress`), plus the general temperature-response curves
//! (`temperature_factor`, `q10_factor`) real plant physiology follows.
//! Tunable numbers live in `config::ClimateConfig`
//! (temperature itself)/`config::PlantConfig` (how a *plant* responds to
//! it), passed in rather than hardcoded here — same pattern as `sun.rs`.

use std::f64::consts::PI;

use super::config::ClimateConfig;
use super::sun::DayProgress;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClimateState {
    pub temperature_c: f64,
}

/// A room's air temperature tracks the day/night cycle, but real diurnal
/// temperature lags peak solar intensity by a couple of hours (thermal
/// mass — walls/air don't heat up instantly) — modeled here as a fixed
/// phase offset on the same cosine curve, rather than as separately-tracked
/// state that eases toward a target over time. Kept a pure function of
/// `day_progress` (no hidden state to thread through `Plant::step` beyond
/// the value itself) for the same reason `sun::sun_state` is: cheap to call
/// from anywhere, and trivially regression-testable without simulating a
/// history of previous ticks.
pub fn climate_state(day_progress: DayProgress, config: &ClimateConfig) -> ClimateState {
    let day_progress = day_progress.rem_euclid(1.0);
    // Solar noon is day_progress == 0.5 (see `SunConfig`'s doc comment);
    // peak temperature lags that by `temperature_peak_offset`.
    let phase = 2.0 * PI * (day_progress - 0.5 - config.temperature_peak_offset);
    let temperature_c = config.base_temperature_c + config.day_night_swing_c * phase.cos();
    ClimateState { temperature_c }
}

/// How favorable `temperature_c` is for temperature-sensitive processes
/// (photosynthesis, elongation) relative to `optimal_c` — a bell curve
/// (1.0 exactly at the optimum, falling off symmetrically in both
/// directions) rather than a hard cutoff, matching how enzyme-driven
/// reaction rates actually respond to temperature in real plants: gradual
/// falloff on both the cold and hot side of an optimum, not a cliff.
/// `tolerance_c` is the curve's width — at `optimal_c ± tolerance_c`, the
/// factor is `1/e ≈ 0.37`.
pub fn temperature_factor(temperature_c: f64, optimal_c: f64, tolerance_c: f64) -> f64 {
    let z = (temperature_c - optimal_c) / tolerance_c;
    (-z * z).exp()
}

/// The Q10 relationship: respiration (unlike photosynthesis) doesn't have
/// an optimum that falls off on the hot side within any range a houseplant
/// actually experiences — it keeps climbing with temperature, roughly
/// multiplying by `q10` (canonically ~2 for plant respiration) for every
/// 10°C rise above `reference_c` (the temperature `PlantConfig::
/// respiration_rate` was itself tuned at). Falls below 1.0 in the other
/// direction (a cold plant's metabolism genuinely slows down), which is
/// also real and intentional.
pub fn q10_factor(temperature_c: f64, reference_c: f64, q10: f64) -> f64 {
    q10.powf((temperature_c - reference_c) / 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ClimateConfig {
        ClimateConfig::default()
    }

    #[test]
    fn temperature_swings_around_the_base_within_the_configured_range() {
        let config = config();
        let samples: Vec<f64> = (0..100)
            .map(|i| climate_state(i as f64 / 100.0, &config).temperature_c)
            .collect();
        let max = samples.iter().cloned().fold(f64::MIN, f64::max);
        let min = samples.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            (max - config.base_temperature_c - config.day_night_swing_c).abs() < 1e-6,
            "expected the peak to reach base + swing, got max {max}"
        );
        assert!(
            (min - config.base_temperature_c + config.day_night_swing_c).abs() < 1e-6,
            "expected the trough to reach base - swing, got min {min}"
        );
    }

    #[test]
    fn peak_temperature_lags_solar_noon() {
        let config = config();
        // Solar noon is day_progress == 0.5 — peak temperature should land
        // *after* that, not exactly on it.
        let at_noon = climate_state(0.5, &config).temperature_c;
        let a_bit_after_noon = climate_state(0.5 + config.temperature_peak_offset, &config).temperature_c;
        assert!(
            a_bit_after_noon > at_noon,
            "expected temperature to still be climbing just after solar noon: {at_noon} -> {a_bit_after_noon}"
        );
    }

    #[test]
    fn day_progress_wraps() {
        let config = config();
        let a = climate_state(0.9, &config).temperature_c;
        let b = climate_state(1.9, &config).temperature_c;
        assert!((a - b).abs() < 1e-9);
    }

    #[test]
    fn temperature_factor_peaks_at_1_exactly_at_the_optimum() {
        assert_eq!(temperature_factor(24.0, 24.0, 10.0), 1.0);
    }

    #[test]
    fn temperature_factor_falls_off_symmetrically_in_both_directions() {
        let cold = temperature_factor(14.0, 24.0, 10.0);
        let hot = temperature_factor(34.0, 24.0, 10.0);
        assert!((cold - hot).abs() < 1e-9, "expected a symmetric bell curve: cold {cold} vs hot {hot}");
        assert!(cold < 1.0 && cold > 0.0);
    }

    #[test]
    fn temperature_factor_is_worse_further_from_the_optimum() {
        let near = temperature_factor(20.0, 24.0, 10.0);
        let far = temperature_factor(4.0, 24.0, 10.0);
        assert!(far < near, "expected a temperature further from the optimum to fare worse: {far} vs {near}");
    }

    #[test]
    fn q10_factor_is_1_at_the_reference_temperature() {
        assert_eq!(q10_factor(20.0, 20.0, 2.0), 1.0);
    }

    #[test]
    fn q10_factor_roughly_doubles_per_10_degrees_above_reference() {
        let factor = q10_factor(30.0, 20.0, 2.0);
        assert!((factor - 2.0).abs() < 1e-9, "expected exactly 2x at +10C with q10=2, got {factor}");
    }

    #[test]
    fn q10_factor_drops_below_1_when_colder_than_reference() {
        let factor = q10_factor(10.0, 20.0, 2.0);
        assert!(factor < 1.0, "expected slowed metabolism below the reference temperature, got {factor}");
    }
}
