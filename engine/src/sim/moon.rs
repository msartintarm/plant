//! Real lunar phase, grounded in today's actual date rather than an
//! arbitrary in-game cycle — see `MoonConfig::initial_phase` for how "today"
//! feeds in. Phase progresses afterward using the real synodic month length
//! applied to the game's own (already-compressed) day unit, since location
//! doesn't change the phase itself (only moonrise/set timing, which this
//! stylized side-profile scene doesn't model at all).

use std::f64::consts::PI;

use super::config::{MoonConfig, SunConfig};
use super::sun::SunState;

/// Julian Day Number for a Gregorian calendar date (proleptic Gregorian,
/// valid for any reasonable date) — the standard integer-day algorithm.
fn julian_day_number(year: i32, month: u32, day: u32) -> f64 {
    let a = (14 - month as i32) / 12;
    let y = year + 4800 - a;
    let m = month as i32 + 12 * a - 3;
    let jdn = day as i32 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    jdn as f64
}

/// 0.0 (new moon) ..< 1.0 (back to new moon), 0.5 = full moon, for a given
/// date — a known reference new moon (2000-01-06, ~18:14 UTC, JDN 2451550.26)
/// and the mean synodic month length (29.530588853 days).
pub fn phase_for_date(year: i32, month: u32, day: u32) -> f64 {
    const SYNODIC_MONTH_DAYS: f64 = 29.530588853;
    // `julian_day_number` returns the JDN for *noon* UTC (the standard
    // convention); the reference new moon itself fell at ~18:14 UTC that
    // same calendar day, 0.26 of a day later.
    const REFERENCE_NEW_MOON_JDN: f64 = 2451550.26;
    let jdn = julian_day_number(year, month, day);
    ((jdn - REFERENCE_NEW_MOON_JDN) / SYNODIC_MONTH_DAYS).rem_euclid(1.0)
}

/// How far `phase` sits from 0.0 on a wrapping 0..1 cycle — 0.99 and 0.01
/// are both "close to 0," a plain subtraction doesn't know that.
#[cfg(test)]
fn circular_distance_from_zero(phase: f64) -> f64 {
    phase.min(1.0 - phase)
}

/// Current phase (0.0..1.0, wrapping), given how many sim-seconds have
/// elapsed and `MoonConfig::initial_phase`/`cycle_length_sim_seconds`.
pub fn current_phase(total_time: f64, config: &MoonConfig) -> f64 {
    (config.initial_phase + total_time / config.cycle_length_sim_seconds).rem_euclid(1.0)
}

/// What the phase actually looks like — illuminated fraction (0 = new,
/// 1 = full) and which side is lit (waxing = right, waning = left, in this
/// scene's convention), for `scene`'s two-disc crescent rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonAppearance {
    pub illuminated_fraction: f64,
    pub waxing: bool,
}

pub fn appearance(phase: f64) -> MoonAppearance {
    let phase = phase.rem_euclid(1.0);
    MoonAppearance {
        illuminated_fraction: (1.0 - (2.0 * PI * phase).cos()) / 2.0,
        waxing: phase < 0.5,
    }
}

/// The moon's own position across the sky — (azimuth, elevation), same
/// 0.0..1.0 conventions `scene::sky_object_transform` already expects from
/// `SunState`. Deliberately *not* just reusing `sun_state`'s own azimuth/
/// elevation for the moon: those are only ever meant to be looked at while
/// the sun is up (see `SunState::azimuth`'s doc comment — it holds at
/// whatever it was at sunset/sunrise once the sun sets, since nothing used
/// to render it at night). That was invisible until the moon started being
/// drawn *at* night using that same held value — the moon would sit frozen
/// at the sunset-side edge all evening, then instantly snap to the sunrise-
/// side edge the moment `day_progress` wrapped past midnight. This instead
/// sweeps smoothly across its own rise-to-set span, with no discontinuity
/// at the `day_progress` wrap.
///
/// Rise/set timing shifts with `phase` (the sun-earth-moon elongation, same
/// 0..1 convention `appearance` uses) rather than always sitting exactly
/// opposite the sun: new moon (phase 0) rises/transits/sets alongside the
/// sun; full moon (0.5) transits at midnight (the old, phase-blind
/// assumption); quarters transit at sunset/sunrise.
pub fn arc_position(day_progress: f64, phase: f64, sun_config: &SunConfig) -> (f64, f64) {
    let day_progress = day_progress.rem_euclid(1.0);
    let moonrise_progress = (sun_config.sunrise + phase).rem_euclid(1.0);
    let since_moonrise = (day_progress - moonrise_progress).rem_euclid(1.0);
    let azimuth = (since_moonrise / 0.5).clamp(0.0, 1.0);
    let elevation = (2.0 * PI * since_moonrise).sin();
    (azimuth, elevation)
}

/// Bright-limb tilt (radians) for `scene::moon_shadow_transform` — a
/// stylized approximation (not real parallactic-angle ephemeris math,
/// since `arc_position`'s azimuth/elevation aren't true equatorial
/// coordinates either). Bright limb faces the sun's own current position;
/// the tilt fades to 0 (pure vertical split) toward the poles and to full
/// strength toward the equator, matching how a real crescent reads more
/// vertical at high latitude and more "smile"-shaped near the equator.
pub fn crescent_tilt_angle(
    moon: (f64, f64),
    sun: (f64, f64),
    latitude_degrees: f64,
) -> f64 {
    let (moon_azimuth, moon_elevation) = moon;
    let (sun_azimuth, sun_elevation) = sun;
    let toward_sun = (sun_elevation - moon_elevation).atan2(sun_azimuth - moon_azimuth);
    let latitude_factor = 1.0 - (latitude_degrees.to_radians().abs()).sin();
    toward_sun * latitude_factor
}

/// Adds the moon's own light to the sun's — a full moon genuinely
/// brightens a dark room a little, on top of (never replacing) whatever the
/// sun itself is contributing; see `MoonConfig::max_light_contribution`.
/// Only `intensity` changes — `elevation`/`azimuth` stay the sun's own (see
/// `arc_position` for the moon's actual on-screen position, a separate
/// concern from how much light it's contributing to growth) and `color`
/// stays untouched too, since this is meant as a subtle boost, not a visual
/// re-tint. `moon_elevation` gates the contribution to 0 when the moon
/// isn't actually above the horizon (see `arc_position`).
pub fn apply_moonlight(sun: SunState, appearance: MoonAppearance, moon_elevation: f64, config: &MoonConfig) -> SunState {
    let moonlight = if moon_elevation > 0.0 {
        appearance.illuminated_fraction * config.max_light_contribution
    } else {
        0.0
    };
    SunState { intensity: (sun.intensity + moonlight).min(1.0), ..sun }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_day_number_matches_the_known_reference_new_moon_epoch() {
        // 2000-01-06 (noon UTC) is JDN 2451550 exactly — verified
        // independently against a Python reference implementation.
        assert_eq!(julian_day_number(2000, 1, 6), 2451550.0);
    }

    #[test]
    fn phase_for_date_is_near_zero_at_the_reference_new_moon() {
        let phase = phase_for_date(2000, 1, 6);
        assert!(circular_distance_from_zero(phase) < 0.02, "expected phase near 0 (new moon), got {phase}");
    }

    #[test]
    fn phase_for_date_returns_to_the_same_point_after_a_full_synodic_month() {
        let start = phase_for_date(2000, 1, 6);
        // 2000-02-05 is one synodic month (29.53 days) later.
        let one_month_later = phase_for_date(2000, 2, 5);
        let delta = (start - one_month_later).rem_euclid(1.0);
        let circular_delta = delta.min(1.0 - delta);
        assert!(circular_delta < 0.05, "expected roughly the same phase one synodic month later, drifted by {circular_delta}");
    }

    #[test]
    fn current_phase_advances_over_a_full_cycle_and_wraps() {
        let config = MoonConfig { initial_phase: 0.0, cycle_length_sim_seconds: 1000.0, ..MoonConfig::default() };
        assert_eq!(current_phase(0.0, &config), 0.0);
        assert!((current_phase(500.0, &config) - 0.5).abs() < 1e-9);
        assert!((current_phase(1000.0, &config) - 0.0).abs() < 1e-9);
    }

    /// Regression test for a real "moon cycle feels frozen" bug report: it
    /// turned out to be `render::mod` driving `current_phase` off `Plant::
    /// total_time` (which resets to 0 on every restart/species-switch/
    /// cutting) instead of a persistent session clock — every restart
    /// snapped the moon back near its starting phase, which looked like it
    /// had barely moved across a session with a few restarts in it, even
    /// though a lone phase computation was always mathematically correct.
    /// That wiring lives in wasm-only code this crate can't unit-test
    /// directly, so this instead pins down the piece that actually *is*
    /// testable here: given the real default `MoonConfig`/`TimeConfig`
    /// rates, does one real-world minute of play move the phase by a
    /// clearly noticeable amount at the default (1x) speed? If this ever
    /// regresses to something glacial, this fails independently of
    /// whichever clock ends up feeding it.
    #[test]
    fn default_config_advances_the_moon_phase_at_a_perceptible_real_time_rate() {
        use super::super::config::TimeConfig;
        let moon_config = MoonConfig::default();
        let time_config = TimeConfig::default();
        let real_seconds = 60.0;
        let sim_seconds = real_seconds * time_config.sim_seconds_per_real_second;
        let start = MoonConfig { initial_phase: 0.0, ..moon_config };
        let phase_before = current_phase(0.0, &start);
        let phase_after = current_phase(sim_seconds, &start);
        let advanced = (phase_after - phase_before).rem_euclid(1.0);
        assert!(
            advanced > 0.1,
            "expected at least a tenth of a full cycle in one real minute at 1x speed, got {advanced}"
        );
    }

    /// A full cycle should complete within a few real minutes at 1x speed
    /// — not, say, take longer than a real year because of a units mixup
    /// (sim-seconds vs. real-seconds) somewhere in how `MoonConfig::
    /// cycle_length_sim_seconds` was derived.
    #[test]
    fn default_config_completes_a_full_moon_cycle_within_a_few_real_minutes_at_1x_speed() {
        use super::super::config::TimeConfig;
        let moon_config = MoonConfig::default();
        let time_config = TimeConfig::default();
        let real_seconds_per_cycle = moon_config.cycle_length_sim_seconds / time_config.sim_seconds_per_real_second;
        assert!(
            real_seconds_per_cycle < 600.0,
            "expected a full moon cycle well under 10 real minutes at 1x speed, got {real_seconds_per_cycle}s"
        );
    }

    #[test]
    fn appearance_is_new_at_phase_zero_and_full_at_phase_half() {
        assert!((appearance(0.0).illuminated_fraction - 0.0).abs() < 1e-9);
        assert!((appearance(0.5).illuminated_fraction - 1.0).abs() < 1e-9);
    }

    #[test]
    fn appearance_is_half_lit_at_the_quarters() {
        assert!((appearance(0.25).illuminated_fraction - 0.5).abs() < 1e-9);
        assert!((appearance(0.75).illuminated_fraction - 0.5).abs() < 1e-9);
    }

    #[test]
    fn waxing_is_the_first_half_of_the_cycle_waning_the_second() {
        assert!(appearance(0.1).waxing);
        assert!(appearance(0.49).waxing);
        assert!(!appearance(0.51).waxing);
        assert!(!appearance(0.9).waxing);
    }

    // Full-moon (`phase` 0.5) is the one case where the old, phase-blind
    // `arc_position` happened to be exactly correct — it's genuinely in
    // opposition, so it does rise at sunset and set at sunrise every time.
    // These pin that special case down explicitly now that it's no longer
    // the *only* case the function models.

    #[test]
    fn full_moon_rises_at_sunset_and_sets_at_sunrise() {
        let config = SunConfig::default();
        let (azimuth_at_sunset, elevation_at_sunset) = arc_position(config.sunset, 0.5, &config);
        assert!((azimuth_at_sunset - 0.0).abs() < 1e-9);
        assert!(elevation_at_sunset.abs() < 1e-9, "moonrise should be right at the horizon");
        let (azimuth_at_sunrise, elevation_at_sunrise) = arc_position(config.sunrise, 0.5, &config);
        assert!((azimuth_at_sunrise - 1.0).abs() < 1e-9);
        assert!(elevation_at_sunrise.abs() < 1e-9, "moonset should be right at the horizon");
    }

    #[test]
    fn full_moon_peaks_at_midnight() {
        let config = SunConfig::default();
        let night_duration = (1.0 - config.sunset) + config.sunrise;
        let midnight_ish = (config.sunset + night_duration / 2.0).rem_euclid(1.0);
        let (azimuth, elevation) = arc_position(midnight_ish, 0.5, &config);
        assert!((azimuth - 0.5).abs() < 1e-6);
        assert!((elevation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn full_moon_sits_below_the_horizon_during_the_day() {
        // opposition to the sun: at its nadir when the sun's at its peak
        let config = SunConfig::default();
        let (_, elevation) = arc_position(0.5, 0.5, &config);
        assert!((elevation - -1.0).abs() < 1e-6, "expected the full moon at its nadir at solar noon, got {elevation}");
    }

    #[test]
    fn new_moon_is_up_during_the_day_alongside_the_sun_not_at_night() {
        // conjunction with the sun: reverse of the full-moon case above
        let config = SunConfig::default();
        let (_, elevation_at_noon) = arc_position(0.5, 0.0, &config);
        assert!(elevation_at_noon > 0.9, "expected the new moon near its own peak at solar noon, got {elevation_at_noon}");
        let midnight = 0.0;
        let (_, elevation_at_midnight) = arc_position(midnight, 0.0, &config);
        assert!(elevation_at_midnight < -0.9, "expected the new moon well below the horizon at midnight, got {elevation_at_midnight}");
    }

    #[test]
    fn first_quarter_moon_transits_at_sunset_last_quarter_transits_at_sunrise() {
        let config = SunConfig::default();
        let (_, elevation_first_quarter_at_sunset) = arc_position(config.sunset, 0.25, &config);
        assert!(
            (elevation_first_quarter_at_sunset - 1.0).abs() < 1e-6,
            "expected first quarter moon to peak right at sunset, got {elevation_first_quarter_at_sunset}"
        );
        let (_, elevation_last_quarter_at_sunrise) = arc_position(config.sunrise, 0.75, &config);
        assert!(
            (elevation_last_quarter_at_sunrise - 1.0).abs() < 1e-6,
            "expected last quarter moon to peak right at sunrise, got {elevation_last_quarter_at_sunrise}"
        );
    }

    #[test]
    fn arc_position_sweeps_continuously_across_the_day_progress_wrap() {
        let config = SunConfig::default();
        for phase in [0.0, 0.25, 0.5, 0.75] {
            let just_before_wrap = arc_position(0.999, phase, &config);
            let just_after_wrap = arc_position(0.001, phase, &config);
            assert!(
                (just_after_wrap.0 - just_before_wrap.0).abs() < 0.01
                    || (just_after_wrap.0 - just_before_wrap.0).abs() > 0.99,
                "azimuth should sweep smoothly across midnight (or wrap cleanly 1.0->0.0), not jump: {just_before_wrap:?} -> {just_after_wrap:?} at phase {phase}"
            );
        }
    }

    #[test]
    fn apply_moonlight_adds_more_light_at_full_moon_than_new_moon_while_both_are_up() {
        let config = MoonConfig { max_light_contribution: 0.05, ..MoonConfig::default() };
        let dark_sun = SunState { elevation: -0.5, azimuth: 1.0, intensity: 0.0, color: [1.0, 1.0, 1.0] };
        let moon_up = 0.5;
        let new_moon = apply_moonlight(dark_sun, appearance(0.0), moon_up, &config);
        let full_moon = apply_moonlight(dark_sun, appearance(0.5), moon_up, &config);
        assert_eq!(new_moon.intensity, 0.0, "a new moon adds no light at all");
        assert!((full_moon.intensity - 0.05).abs() < 1e-9, "a full moon adds the configured maximum");
    }

    #[test]
    fn apply_moonlight_contributes_nothing_while_the_moon_is_below_the_horizon() {
        let config = MoonConfig { max_light_contribution: 0.05, ..MoonConfig::default() };
        let dark_sun = SunState { elevation: -0.5, azimuth: 1.0, intensity: 0.0, color: [1.0, 1.0, 1.0] };
        let moon_down = -0.1;
        let lit = apply_moonlight(dark_sun, appearance(0.5), moon_down, &config);
        assert_eq!(lit.intensity, 0.0, "a below-horizon moon should contribute no light even at full phase");
    }

    #[test]
    fn apply_moonlight_never_pushes_intensity_past_full_daylight() {
        let config = MoonConfig { max_light_contribution: 0.05, ..MoonConfig::default() };
        let bright_sun = SunState { elevation: 1.0, azimuth: 0.5, intensity: 1.0, color: [1.0, 1.0, 1.0] };
        let lit = apply_moonlight(bright_sun, appearance(0.5), 0.5, &config);
        assert_eq!(lit.intensity, 1.0, "moonlight is irrelevant once the sun already provides full intensity");
    }

    #[test]
    fn apply_moonlight_leaves_position_and_color_untouched() {
        let config = MoonConfig::default();
        let sun = SunState { elevation: -0.3, azimuth: 0.4, intensity: 0.1, color: [0.9, 0.8, 0.7] };
        let lit = apply_moonlight(sun, appearance(0.5), 0.5, &config);
        assert_eq!(lit.elevation, sun.elevation);
        assert_eq!(lit.azimuth, sun.azimuth);
        assert_eq!(lit.color, sun.color);
    }

    #[test]
    fn crescent_tilt_is_zero_at_the_poles_and_nonzero_near_the_equator() {
        let moon = (0.5, 0.5);
        let sun = (0.9, -0.5);
        assert_eq!(crescent_tilt_angle(moon, sun, 90.0), 0.0);
        assert!(crescent_tilt_angle(moon, sun, 0.1).abs() > 0.1);
    }

    #[test]
    fn crescent_tilt_points_toward_the_suns_position() {
        let moon = (0.5, 0.5);
        let toward_sun_below = crescent_tilt_angle(moon, (0.5, -0.5), 0.0);
        let toward_sun_above = crescent_tilt_angle(moon, (0.5, 1.5), 0.0);
        assert!(toward_sun_below < 0.0);
        assert!(toward_sun_above > 0.0);
    }
}
