//! Where the pot sits relative to the window — a player-chosen, static
//! placement decision, distinct from `plant::height_light_factor`'s
//! *vertical* falloff (how tall the plant itself has grown relative to the
//! window). Modeled as a pure adjustment applied to an already-computed
//! `SunState`/`ClimateState` (the same pattern `climate::climate_state`
//! itself already establishes: cheap, stateless, no persistent state to
//! thread through `Plant::step`), so the caller derives the *effective*
//! sun/climate for wherever the pot currently sits, then hands those
//! adjusted values into `Plant::step` exactly as it already does today.

use super::climate::ClimateState;
use super::config::RoomConfig;
use super::sun::SunState;

/// 0.0 (right at the windowsill — brightest, but drafty) ..= 1.0 (as far
/// back into the room as this game models — dim, but climate-stable).
pub type PotPosition = f64;

/// Applies the pot's room position to an already-computed sun/climate
/// snapshot — see the module doc comment. Returns adjusted copies rather
/// than mutating in place, matching `sun::sun_state`/`climate::
/// climate_state`'s own "pure function returning a fresh value" style.
pub fn apply_pot_position(
    sun: SunState,
    climate: ClimateState,
    position: PotPosition,
    config: &RoomConfig,
) -> (SunState, ClimateState) {
    let position = position.clamp(0.0, 1.0);
    let light_factor = 1.0 + (config.window_light_floor - 1.0) * position;
    let mut adjusted_sun = sun;
    adjusted_sun.intensity = (sun.intensity * light_factor).clamp(0.0, 1.0);

    let mut adjusted_climate = climate;
    adjusted_climate.temperature_c -= config.window_draft_cold_c * (1.0 - position);

    (adjusted_sun, adjusted_climate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RoomConfig {
        RoomConfig::default()
    }

    fn full_sun() -> SunState {
        SunState { elevation: 1.0, azimuth: 0.5, intensity: 1.0, color: [1.0, 1.0, 1.0] }
    }

    fn climate(temperature_c: f64) -> ClimateState {
        ClimateState { temperature_c }
    }

    #[test]
    fn right_at_the_window_keeps_full_light_but_applies_the_full_draft_penalty() {
        let config = config();
        let (sun, climate) = apply_pot_position(full_sun(), climate(21.0), 0.0, &config);
        assert_eq!(sun.intensity, 1.0, "expected no light falloff right at the window");
        assert_eq!(climate.temperature_c, 21.0 - config.window_draft_cold_c);
    }

    #[test]
    fn far_from_the_window_dims_light_but_removes_the_draft() {
        let config = config();
        let (sun, climate) = apply_pot_position(full_sun(), climate(21.0), 1.0, &config);
        assert_eq!(sun.intensity, config.window_light_floor);
        assert_eq!(climate.temperature_c, 21.0, "expected no draft penalty far from the window");
    }

    #[test]
    fn light_and_draft_both_scale_monotonically_with_position() {
        let config = config();
        let (near_sun, near_climate) = apply_pot_position(full_sun(), climate(21.0), 0.25, &config);
        let (far_sun, far_climate) = apply_pot_position(full_sun(), climate(21.0), 0.75, &config);
        assert!(near_sun.intensity > far_sun.intensity, "expected closer-to-window to stay brighter");
        assert!(near_climate.temperature_c < far_climate.temperature_c, "expected closer-to-window to stay colder (draftier)");
    }

    #[test]
    fn position_is_clamped_to_the_valid_range() {
        let config = config();
        let (over, _) = apply_pot_position(full_sun(), climate(21.0), 5.0, &config);
        let (far, _) = apply_pot_position(full_sun(), climate(21.0), 1.0, &config);
        assert_eq!(over.intensity, far.intensity, "expected out-of-range input to clamp rather than extrapolate");
    }
}
