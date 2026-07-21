//! A slow year-length cycle layered on top of the fast day/night cycle —
//! real houseplants slow their growth in winter specifically because of
//! shorter days (photoperiodism), independent of temperature, which is
//! already modeled separately (`climate.rs`). Modeled the same "pure
//! function of elapsed time" way `climate::climate_state` itself is (see
//! that module's doc comment for the reasoning): no persistent state to
//! thread through `Plant::step` beyond the single `Plant::total_time`
//! accumulator already needed to evaluate it.

use std::f64::consts::PI;

use super::config::SeasonConfig;

/// Which quarter of the year cycle `total_time` currently falls in — a
/// discrete label for display (HUD/wall), distinct from the continuous
/// `day_length_factor` a plant's own growth math actually uses. Ordered
/// Summer → Autumn → Winter → Spring → (back to Summer), matching how
/// `season_state` places `total_time == 0.0` at midsummer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Season {
    Summer,
    Autumn,
    Winter,
    Spring,
}

impl Season {
    /// The label a HUD/wall display shows — see `render::Simulation::stats`.
    pub fn name(self) -> &'static str {
        match self {
            Season::Spring => "Spring",
            Season::Summer => "Summer",
            Season::Autumn => "Autumn",
            Season::Winter => "Winter",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeasonState {
    /// 1.0 at midsummer (long days, no dormancy suppression) down to
    /// `SeasonConfig::winter_floor` at midwinter — multiplies directly into
    /// elongation (see `PlantConfig::dormancy_elongation_sensitivity`).
    pub day_length_factor: f64,
    /// 0.0..1.0 fraction of the way through the current year-length cycle —
    /// `day_length_factor` alone can't distinguish "heading into winter"
    /// from "heading out of it" (a cosine takes the same value on both
    /// sides of each extreme), so this is what `season` and any other
    /// display that needs to know *which half* of the cycle it is reads
    /// instead.
    pub phase: f64,
    /// Discrete quarter-of-the-year label derived from `phase` — see
    /// `Season`.
    pub season: Season,
}

/// `total_time == 0.0` always lands at midsummer (`day_length_factor ==
/// 1.0`) — a fresh plant/session always starts in the growing season,
/// matching how someone would typically start a houseplant rather than
/// beginning mid-winter.
pub fn season_state(total_time: f64, config: &SeasonConfig) -> SeasonState {
    let period = config.season_length_sim_seconds.max(1e-9);
    let phase = (total_time / period).rem_euclid(1.0);
    let angle = 2.0 * PI * phase;
    let midpoint = (1.0 + config.winter_floor) / 2.0;
    let amplitude = (1.0 - config.winter_floor) / 2.0;
    let season = if phase < 0.25 {
        Season::Summer
    } else if phase < 0.5 {
        Season::Autumn
    } else if phase < 0.75 {
        Season::Winter
    } else {
        Season::Spring
    };
    SeasonState {
        day_length_factor: midpoint + amplitude * angle.cos(),
        phase,
        season,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SeasonConfig {
        SeasonConfig::default()
    }

    #[test]
    fn a_fresh_session_starts_at_midsummer_with_no_dormancy_suppression() {
        let config = config();
        assert_eq!(season_state(0.0, &config).day_length_factor, 1.0);
    }

    #[test]
    fn midwinter_drops_to_the_configured_floor() {
        let config = config();
        let midwinter = config.season_length_sim_seconds / 2.0;
        let factor = season_state(midwinter, &config).day_length_factor;
        assert!((factor - config.winter_floor).abs() < 1e-6);
    }

    #[test]
    fn the_cycle_returns_to_midsummer_after_a_full_period() {
        let config = config();
        let start = season_state(0.0, &config).day_length_factor;
        let after_a_year = season_state(config.season_length_sim_seconds, &config).day_length_factor;
        assert!((start - after_a_year).abs() < 1e-6);
    }

    #[test]
    fn day_length_factor_never_drops_below_the_floor_or_exceeds_full_summer() {
        let config = config();
        for i in 0..100 {
            let t = i as f64 / 100.0 * config.season_length_sim_seconds;
            let factor = season_state(t, &config).day_length_factor;
            assert!(factor >= config.winter_floor - 1e-9 && factor <= 1.0 + 1e-9, "factor {factor} out of range at t={t}");
        }
    }

    #[test]
    fn a_fresh_session_starts_in_summer_at_phase_zero() {
        let config = config();
        let state = season_state(0.0, &config);
        assert_eq!(state.season, Season::Summer);
        assert_eq!(state.phase, 0.0);
    }

    #[test]
    fn the_year_visits_all_four_seasons_in_order() {
        let config = config();
        let period = config.season_length_sim_seconds;
        assert_eq!(season_state(period * 0.1, &config).season, Season::Summer);
        assert_eq!(season_state(period * 0.3, &config).season, Season::Autumn);
        assert_eq!(season_state(period * 0.6, &config).season, Season::Winter);
        assert_eq!(season_state(period * 0.9, &config).season, Season::Spring);
    }

    #[test]
    fn phase_wraps_back_to_zero_after_a_full_year_instead_of_climbing_forever() {
        let config = config();
        let period = config.season_length_sim_seconds;
        let phase_at_one_and_a_quarter_years = season_state(period * 1.25, &config).phase;
        assert!((phase_at_one_and_a_quarter_years - 0.25).abs() < 1e-9);
    }

    #[test]
    fn phase_alone_distinguishes_heading_into_winter_from_heading_out_of_it() {
        // The whole reason `phase` exists alongside `day_length_factor`:
        // a symmetric cosine takes the same value on both sides of each
        // extreme, so `day_length_factor` alone can't tell autumn (heading
        // into winter) apart from spring (heading out of it) even though
        // they're clearly different seasons.
        let config = config();
        let period = config.season_length_sim_seconds;
        let autumn = season_state(period * 0.3, &config);
        let spring = season_state(period * 0.7, &config);
        assert!((autumn.day_length_factor - spring.day_length_factor).abs() < 1e-6);
        assert_ne!(autumn.season, spring.season);
        assert!(autumn.phase < spring.phase);
    }
}
