//! Pest pressure, modeled on spider mites — the classic dry-indoor-air
//! houseplant pest, a real threat orthogonal to the water/light/nutrient
//! economy this simulation otherwise models: a plant with every other dial
//! set correctly can still be lost to pests if humidity is neglected. Pure
//! functions of explicit inputs, same discipline as `plant::self_shading_
//! factors`/`plant::stem_droop_target` — the actual `pest_infestation`
//! value they operate on lives on `Plant` itself (it's plant-tissue state,
//! like `Plant::senescence`), not tracked here.

use super::config::PestConfig;

/// How fast infestation grows per second at the current air humidity — zero
/// at/above `PestConfig::safe_humidity` (real spider mites specifically
/// struggle in humid air), ramping linearly to `growth_rate` at bone-dry
/// air.
pub fn pest_growth_rate(humidity: f64, config: &PestConfig) -> f64 {
    let dryness = (config.safe_humidity - humidity).clamp(0.0, config.safe_humidity);
    let dryness_fraction = dryness / config.safe_humidity.max(1e-9);
    config.growth_rate * dryness_fraction
}

/// Photosynthesis multiplier from current infestation severity — sap-
/// sucking pests are a direct carbon-income tax, not just an added stress
/// signal (see `plant::leaf_stress_signal`, which infestation also feeds
/// into for senescence).
pub fn photosynthesis_penalty(infestation: f64, config: &PestConfig) -> f64 {
    (1.0 - config.photosynthesis_penalty * infestation.clamp(0.0, 1.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PestConfig {
        PestConfig::default()
    }

    #[test]
    fn no_pest_growth_at_or_above_safe_humidity() {
        let config = config();
        assert_eq!(pest_growth_rate(config.safe_humidity, &config), 0.0);
        assert_eq!(pest_growth_rate(1.0, &config), 0.0);
    }

    #[test]
    fn pest_growth_is_fastest_in_bone_dry_air() {
        let config = config();
        assert_eq!(pest_growth_rate(0.0, &config), config.growth_rate);
    }

    #[test]
    fn pest_growth_rate_increases_monotonically_as_humidity_drops() {
        let config = config();
        let humid = pest_growth_rate(0.4, &config);
        let dry = pest_growth_rate(0.1, &config);
        assert!(dry > humid, "expected drier air to grow pests faster: {dry} vs {humid}");
    }

    #[test]
    fn photosynthesis_penalty_is_neutral_with_no_infestation() {
        assert_eq!(photosynthesis_penalty(0.0, &config()), 1.0);
    }

    #[test]
    fn photosynthesis_penalty_worsens_with_infestation_but_never_goes_negative() {
        let config = config();
        let mild = photosynthesis_penalty(0.3, &config);
        let severe = photosynthesis_penalty(1.0, &config);
        assert!(severe < mild && mild < 1.0);
        assert!(severe >= 0.0);
    }
}
