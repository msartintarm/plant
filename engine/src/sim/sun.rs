//! Sun position/intensity/color as a pure function of where we are in the
//! (accelerated, in-game) day/night cycle. No wall-clock time involved —
//! see the `day_progress` parameter. Tunable numbers live in
//! `config::SunConfig`, passed in rather than hardcoded here.

use std::f64::consts::PI;

use super::config::SunConfig;

/// Fraction of a full day/night cycle elapsed, wrapping at 1.0. 0.0 is
/// midnight.
pub type DayProgress = f64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunState {
    /// -1.0 (midnight) ..= 1.0 (solar noon); above 0 means the sun is up.
    pub elevation: f64,
    /// 0.0 (sunrise horizon) ..= 1.0 (sunset horizon). Holds at its
    /// sunrise/sunset clamp outside daytime, since azimuth only drives
    /// anything visible while the sun is up.
    pub azimuth: f64,
    /// 0.0 at night, up to 1.0 at solar noon — what the plant's growth model
    /// integrates over time.
    pub intensity: f64,
    /// 0.0-1.0-per-channel tint: warm near the horizon, neutral white at
    /// solar noon.
    pub color: [f32; 3],
}

fn elevation_unit(day_progress: DayProgress, config: &SunConfig) -> f64 {
    (2.0 * PI * (day_progress - config.sunrise)).sin()
}

fn azimuth_unit(day_progress: DayProgress, config: &SunConfig) -> f64 {
    ((day_progress - config.sunrise) / (config.sunset - config.sunrise)).clamp(0.0, 1.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn sun_state(day_progress: DayProgress, config: &SunConfig) -> SunState {
    let day_progress = day_progress.rem_euclid(1.0);
    let elevation = elevation_unit(day_progress, config);
    let daylight = elevation.max(0.0);
    // A soft twilight tail below the horizon — see `SunConfig::
    // twilight_depth`'s doc comment — rather than snapping straight to
    // night the instant the sun dips below it.
    let twilight = if elevation < 0.0 {
        (1.0 + elevation / config.twilight_depth).clamp(0.0, 1.0) * config.twilight_intensity
    } else {
        0.0
    };
    let intensity = (daylight + twilight).clamp(0.0, 1.0);
    // Climbs fast off the horizon and tapers near noon, so dawn/dusk read as
    // a brief warm window rather than a slow fade — sqrt of a 0..1 value
    // rises faster than linear near 0.
    let warmth = (elevation.max(0.0) as f32).sqrt();
    let color = [
        lerp(config.dawn_color[0], config.noon_color[0], warmth),
        lerp(config.dawn_color[1], config.noon_color[1], warmth),
        lerp(config.dawn_color[2], config.noon_color[2], warmth),
    ];
    SunState {
        elevation,
        azimuth: azimuth_unit(day_progress, config),
        intensity,
        color,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(day_progress: DayProgress) -> SunState {
        sun_state(day_progress, &SunConfig::default())
    }

    #[test]
    fn midnight_is_dark() {
        let s = state(0.0);
        assert_eq!(s.intensity, 0.0);
        assert!(s.elevation < 0.0);
    }

    #[test]
    fn noon_is_brightest() {
        let noon = state(0.5);
        let mid_morning = state(0.35);
        let dusk = state(0.7);
        assert!(noon.intensity > mid_morning.intensity);
        assert!(noon.intensity > dusk.intensity);
        assert!((noon.intensity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_intensity_deep_in_the_night() {
        for i in 0..100 {
            // 0.80..1.20, which wraps past midnight back to well before
            // sunrise — comfortably clear of the twilight tail on both
            // ends (see `twilight_tail_glows_briefly_after_sunset_then_
            // fades_to_true_night`), unlike the moment right at sunset/
            // sunrise itself.
            let t = 0.80 + (i as f64 / 100.0) * 0.40;
            assert_eq!(state(t).intensity, 0.0, "expected true night at day_progress {t}");
        }
    }

    #[test]
    fn twilight_tail_glows_briefly_after_sunset_then_fades_to_true_night() {
        let config = SunConfig::default();
        let just_after = state(config.sunset + 0.005);
        let later = state(config.sunset + 0.015);
        let deep_night = state(config.sunset + config.twilight_depth + 0.05);
        assert!(just_after.intensity > 0.0, "expected a residual glow just after sunset, not an instant cutoff");
        assert!(
            later.intensity < just_after.intensity,
            "expected twilight to keep fading as the sun sinks further below the horizon"
        );
        assert_eq!(deep_night.intensity, 0.0, "expected twilight to fully fade out past its configured depth");
    }

    #[test]
    fn twilight_never_outshines_actual_daylight() {
        let config = SunConfig::default();
        let noon = state(0.5);
        let twilight_peak = state(config.sunset);
        assert!(twilight_peak.intensity < noon.intensity);
    }

    #[test]
    fn azimuth_sweeps_monotonically_across_the_day() {
        let config = SunConfig::default();
        let samples: Vec<f64> = (0..=10)
            .map(|i| state(config.sunrise + i as f64 * 0.05).azimuth)
            .collect();
        for pair in samples.windows(2) {
            assert!(pair[1] >= pair[0], "azimuth moved backwards: {samples:?}");
        }
        assert_eq!(*samples.first().unwrap(), 0.0);
        assert_eq!(*samples.last().unwrap(), 1.0);
    }

    #[test]
    fn day_progress_wraps() {
        // Not bit-exact — rem_euclid on 1.9 vs. 0.9 lands a float ULP or two
        // apart, which sin() amplifies slightly — so compare with a tight
        // tolerance rather than equality.
        let a = state(0.9);
        let b = state(1.9);
        assert!((a.elevation - b.elevation).abs() < 1e-9);
        assert!((a.azimuth - b.azimuth).abs() < 1e-9);
        assert!((a.intensity - b.intensity).abs() < 1e-9);
    }

    #[test]
    fn color_warms_toward_white_at_noon() {
        let config = SunConfig::default();
        let dawn = state(config.sunrise + 0.001);
        let noon = state(0.5);
        // Noon should be closer to neutral white (near-equal channels) than
        // dawn's warm orange.
        let dawn_spread = dawn.color[0] - dawn.color[2];
        let noon_spread = noon.color[0] - noon.color[2];
        assert!(noon_spread < dawn_spread);
    }
}
