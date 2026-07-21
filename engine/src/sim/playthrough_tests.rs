//! Integration-level tests that mirror the *actual* real-time pacing the
//! live render loop uses (`GrowthConfig::time`), instead of the
//! fixed-dt/fixed-sun loops in `plant.rs`'s unit tests. These answer "what
//! does a player actually see within N minutes of real play" directly and
//! fast, via `cargo test` — not by loading the game in a browser and
//! waiting on a wall clock, which is slow, hard to reproduce exactly, and
//! gives no way to inspect *why* something did or didn't happen.

use super::climate;
use super::config::{GrowthConfig, PlantConfig};
use super::plant::{Plant, Stage};
use super::soil::Soil;
use super::sun;

/// Ambient humidity `play()` runs at — deliberately pinned at fully humid
/// rather than tracking a real, decaying `Humidity` (see `sim::humidity`):
/// every test using this shared harness predates the humidity/pest/VPD
/// mechanics and isn't about them, and `humidity == 1.0` is exactly the
/// value that makes both effects mathematically inert (see `Humidity::
/// vpd_factor`'s doc comment and `pests::pest_growth_rate`, both zero at
/// full saturation) — so pinning it here keeps every existing assertion in
/// this file exercising only the mechanism it was actually written to check.
/// See `room_tests` below for dedicated coverage of pot-position/humidity/
/// pest/dormancy/pruning/repotting scenarios instead.
const INERT_HUMIDITY: f64 = 1.0;

/// Steps `plant`/`soil` forward by `real_seconds` of wall-clock time using
/// the same real-time-to-sim-time conversion `render/mod.rs`'s frame loop
/// does, at a fixed 1-real-second sub-step (coarser than an actual frame,
/// but the model has no per-substep-granularity-dependent behavior other
/// than exactly when a discrete threshold — a leaf/branch spawning, a stage
/// transition — gets crossed, which isn't what these tests are checking).
/// `water_every` (real seconds), if `Some`, tops up the soil by
/// `water_amount` on that cadence — `None` models "nobody manually waters
/// it." `auto_water` mirrors the render loop calling
/// `Soil::apply_auto_water` once per tick — see `Simulation::set_auto_water`.
fn play(
    plant: &mut Plant,
    soil: &mut Soil,
    config: &GrowthConfig,
    real_seconds: f64,
    water_every: Option<f64>,
    water_amount: f64,
    auto_water: bool,
) {
    let real_dt = 1.0;
    let mut elapsed_real = 0.0;
    let mut day_progress = 0.0;
    let mut since_last_watering = 0.0;
    while elapsed_real < real_seconds {
        let sim_dt = real_dt * config.time.sim_seconds_per_real_second;
        day_progress =
            (day_progress + sim_dt / config.time.day_length_sim_seconds).rem_euclid(1.0);
        let sun_state = sun::sun_state(day_progress, &config.sun);
        let climate_state = climate::climate_state(day_progress, &config.climate);
        plant.step(sim_dt, &sun_state, &climate_state, soil, INERT_HUMIDITY, config);
        soil.apply_auto_water(auto_water, &config.soil);

        elapsed_real += real_dt;
        since_last_watering += real_dt;
        if let Some(interval) = water_every {
            if since_last_watering >= interval {
                soil.water(water_amount);
                since_last_watering = 0.0;
            }
        }
    }
}

#[test]
fn germinates_and_reaches_vegetative_within_the_first_real_minute() {
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    play(&mut plant, &mut soil, &config, 60.0, None, 0.0, false);
    assert_eq!(plant.stage, Stage::Vegetative);
    assert!(!plant.leaves.is_empty());
}

#[test]
fn without_ever_watering_growth_eventually_plateaus() {
    // A real playthrough where the player never touches the Water button
    // and never enables auto-water — growth should stall once soil runs
    // dry, not keep climbing forever on no water at all.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    play(&mut plant, &mut soil, &config, 600.0, None, 0.0, false);
    let height_at_10_min = plant.height;

    play(&mut plant, &mut soil, &config, 600.0, None, 0.0, false);
    let height_at_20_min = plant.height;

    assert_eq!(soil.moisture, 0.0, "soil should be bone dry with nobody watering it");
    let growth_in_second_window = height_at_20_min - height_at_10_min;
    assert!(
        growth_in_second_window < height_at_10_min * 0.1,
        "expected growth to have largely stalled once soil ran dry: {height_at_10_min} -> {height_at_20_min}"
    );
}

#[test]
fn auto_water_sustains_growth_and_branching_without_any_manual_watering() {
    // The auto-water toggle exists precisely so a player doesn't have to
    // babysit the Water button for a fast-growing plant's ever-increasing
    // transpiration — this is the end-to-end proof it actually substitutes
    // for manual watering, not just that `Soil::apply_auto_water` itself
    // holds a floor (see soil.rs's own unit tests for that). Kept to 45
    // real seconds, not longer: past that (at this demo's fast pacing) the
    // plant's own height starts running past `PlantConfig::
    // window_light_zone_height`, and a plant genuinely starved of light
    // eventually stops affording new leaves at all — a *separate*,
    // deliberate mechanism (see `height_light_factor`), not something this
    // test is about.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    play(&mut plant, &mut soil, &config, 45.0, None, 0.0, true);

    assert_eq!(plant.stage, Stage::Vegetative);
    assert!(
        soil.moisture > 0.0,
        "auto-water should have kept the pot from ever crashing to bone dry"
    );
    assert!(
        !plant.branches.is_empty(),
        "expected at least one branch within 5 real minutes of auto-watered growth"
    );
    assert!(
        plant.branches.iter().any(|b| !b.leaves.is_empty()),
        "expected at least one branch to have grown its own leaves by then too"
    );
}

#[test]
fn with_sustaining_manual_watering_branches_appear_within_a_reasonable_session() {
    // Demonstrates the branching mechanism works end-to-end given adequate
    // *manual* water too (not just auto-water) — a full top-up every 20
    // real seconds is generous but no longer the exploit-grade "every 15
    // seconds forever" the old, miscalibrated `transpiration_coeff` used to
    // require just to keep a leafy plant alive at all (see that field's own
    // doc comment on the recalibration). Kept short (45s, not 5 minutes) —
    // see `auto_water_sustains_growth_and_branching_without_any_manual_
    // watering`'s comment on why: past that the plant's height runs into
    // `height_light_factor`'s falloff, a separate, deliberate mechanism.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    play(&mut plant, &mut soil, &config, 45.0, Some(20.0), 1.0, false);

    assert_eq!(plant.stage, Stage::Vegetative);
    assert!(
        plant.height >= config.plant.min_height_for_branching,
        "expected to clear the branching height threshold within 45 real seconds: height {}",
        plant.height
    );
    assert!(
        !plant.branches.is_empty(),
        "expected at least one branch within 45 real seconds of sustained watering"
    );
    assert!(
        plant.branches.iter().any(|b| !b.leaves.is_empty()),
        "expected at least one branch to have grown its own leaves by then too"
    );
}

#[test]
fn every_branch_grows_its_own_leaves_roughly_fairly_over_a_long_multi_stem_session() {
    // Regression test for a real bug found while auditing multi-stem
    // behavior specifically: processing "the main stem, then each branch
    // in creation order, each fully catching up its own leaf backlog
    // before moving to the next" let whichever grower ran first
    // permanently monopolize carbon — over a long enough session, the
    // *oldest* branch(es) would end up covered in leaves while the most
    // recently formed branch(es) stayed completely bare no matter how
    // tall they grew, since carbon always ran out before their turn came
    // around. Fixed with a persistent round-robin (see
    // `Plant::spawn_due_leaves_fairly` — "persistent" specifically because
    // a rotation that reset every tick never advanced past its first slot,
    // since most ticks only afford a handful of leaf-spawns in total).
    // Kept to 90 real seconds — comfortably long enough for every branch to
    // form and the fairness mechanism to even out, but short enough to stay
    // within the plant's own light zone (`height_light_factor`); past that,
    // a genuinely light-starved plant eventually stops affording leaves at
    // all regardless of how fair the allocation is, which would defeat the
    // point of this specific test.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    play(&mut plant, &mut soil, &config, 90.0, None, 0.0, true);

    assert_eq!(
        plant.branches.len(),
        config.plant.max_branches,
        "expected the crown to fill out to its cap over a 90-second session"
    );
    for (i, branch) in plant.branches.iter().enumerate() {
        assert!(
            !branch.leaves.is_empty(),
            "branch {i} (attached at height {:.2}, grew to its own height {:.2}) never grew a single leaf",
            branch.attach_height,
            branch.height
        );
    }

    let leaf_counts: Vec<usize> = plant.branches.iter().map(|b| b.leaves.len()).collect();
    let max_count = *leaf_counts.iter().max().unwrap();
    let min_count = *leaf_counts.iter().min().unwrap();
    assert!(
        (max_count - min_count) as f64 <= max_count as f64 * 0.25,
        "expected roughly balanced leaf counts across every branch, got {leaf_counts:?}"
    );
}

#[test]
fn leaf_count_stays_bounded_instead_of_growing_without_limit_over_a_long_session() {
    // A real plant doesn't keep accumulating leaves indefinitely — self-
    // shading among its own leaves (see `PlantConfig::leaf_self_shading_
    // coeff`) makes each additional leaf's marginal photosynthetic income
    // shrink while its maintenance respiration cost doesn't, and a leaf
    // heavily overtopped by newer growth above it senesces immediately
    // rather than waiting out a full lifespan (see `age_and_senesce_
    // leaves`) — so total standing leaf count should settle into a small,
    // bounded range instead of climbing without limit as a fast-growing,
    // auto-watered plant keeps producing new nodes over an extended
    // session.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    let total_leaves = |plant: &Plant| -> usize {
        plant.leaves.len() + plant.branches.iter().map(|b| b.leaves.len()).sum::<usize>()
    };

    play(&mut plant, &mut soil, &config, 120.0, None, 0.0, true);
    let leaves_at_2_min = total_leaves(&plant);
    assert!(
        leaves_at_2_min < 60,
        "expected leaf count to stay well short of an unrealistic pile-up: {leaves_at_2_min}"
    );

    play(&mut plant, &mut soil, &config, 120.0, None, 0.0, true);
    let leaves_at_4_min = total_leaves(&plant);
    assert!(
        leaves_at_4_min < 60,
        "expected leaf count to stay bounded rather than keep climbing over a longer session: {leaves_at_4_min}"
    );
}

#[test]
fn peace_lily_never_branches_over_a_long_auto_watered_session() {
    let mut config = GrowthConfig::default();
    config.plant = PlantConfig::peace_lily();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    play(&mut plant, &mut soil, &config, 10.0 * 60.0, None, 0.0, true);

    assert_eq!(plant.stage, Stage::Vegetative);
    assert!(
        plant.branches.is_empty(),
        "expected a basal-rosette habit to never grow lateral branches, got {}",
        plant.branches.len()
    );
}

#[test]
fn peace_lily_stays_much_shorter_than_dracaena_but_still_grows_plenty_of_leaves() {
    // The whole point of scaling `base_elongation_rate` and
    // `plastochron_height_interval` down by the same factor (see
    // `PlantConfig::peace_lily`'s doc comment): height should stay much
    // lower than the caning habit over the same session, without leaf
    // *count* falling behind by anywhere near as much.
    let mut dracaena_config = GrowthConfig::default();
    dracaena_config.plant = PlantConfig::dracaena();
    let mut dracaena = Plant::new();
    let mut dracaena_soil = Soil::new(&dracaena_config.soil);
    play(&mut dracaena, &mut dracaena_soil, &dracaena_config, 5.0 * 60.0, None, 0.0, true);

    let mut peace_lily_config = GrowthConfig::default();
    peace_lily_config.plant = PlantConfig::peace_lily();
    let mut peace_lily = Plant::new();
    let mut peace_lily_soil = Soil::new(&peace_lily_config.soil);
    play(&mut peace_lily, &mut peace_lily_soil, &peace_lily_config, 5.0 * 60.0, None, 0.0, true);

    assert!(
        peace_lily.height < dracaena.height * 0.5,
        "expected the rosette habit to stay much shorter: peace lily {} vs dracaena {}",
        peace_lily.height,
        dracaena.height
    );
    assert!(
        !peace_lily.leaves.is_empty(),
        "expected the rosette habit to still grow leaves despite its low elongation rate"
    );
    // "Roughly comparable" rather than equal — both habits' income/cost
    // dynamics differ enough (a squat plant's leaf area, and so its
    // photosynthesis, grows differently over time than a tall branching
    // one's) that exact parity isn't the point; the point is that it isn't
    // *starved* of leaves the way a naively-scaled-down elongation rate
    // (without also scaling the plastochron interval) would leave it.
    assert!(
        peace_lily.leaves.len() as f64 > dracaena.leaves.len() as f64 * 0.25,
        "expected leaf count to stay in the same ballpark despite the height difference: peace lily {} vs dracaena {}",
        peace_lily.leaves.len(),
        dracaena.leaves.len()
    );
}

#[test]
fn a_player_who_waters_every_single_real_second_damages_root_health_and_can_kill_the_plant() {
    // The overwatering failure mode end-to-end: a player mashing the Water
    // button far more often than the plant could ever draw the pot back
    // down keeps soil continuously saturated, which — sustained long
    // enough past `SoilConfig::waterlog_grace_period` — actually damages
    // `Plant::root_health`, unlike a normal watering cadence (see the
    // sibling `with_sustaining_manual_watering_branches_appear_within_a_
    // reasonable_session` test just above, which waters every 20 real
    // seconds and never triggers this). At this demo's own aggressively
    // fast validation pacing (see `sim::config::TimeConfig`'s doc comment),
    // sustained flooding is severe enough to kill the plant within a couple
    // of real minutes — a strong, legible version of the real lesson
    // ("more water isn't always better") this mechanic exists to teach, not
    // a bug; a slower, gameplay-tuned pacing would stretch this out
    // considerably.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    play(&mut plant, &mut soil, &config, 120.0, Some(1.0), 1.0, false);

    assert_eq!(plant.stage, Stage::Dead, "expected sustained flooding to eventually kill the plant");
    assert_eq!(plant.root_health, 0.0, "expected it to have died specifically from total root loss");
}

#[test]
fn pruning_a_freshly_grown_stem_immediately_produces_branches_well_before_the_normal_branching_height() {
    // The pruning payoff end-to-end: a stem below `min_height_for_branching`
    // (so it would *never* branch on its own yet) still gets an immediate
    // crown release the instant it's pruned, once past the much lower
    // `prune_min_height` bar — a direct, player-triggered shaping action
    // rather than waiting on the plant's own automatic timeline.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    play(&mut plant, &mut soil, &config, 3.0, Some(2.0), 1.0, false);

    assert_eq!(plant.stage, Stage::Vegetative);
    assert!(
        plant.height >= config.plant.prune_min_height,
        "expected this short session to at least clear the prunable height, got {}",
        plant.height
    );
    assert!(plant.branches.is_empty(), "shouldn't have branched on its own yet in a session this short");

    assert!(plant.prune_main_stem(&config.plant));
    assert!(!plant.branches.is_empty(), "expected pruning to release branches immediately");
}

#[test]
fn default_no_input_play_keeps_growing_leaves_instead_of_stalling_at_one() {
    // Unlike `play()` (pinned humidity, see its own doc comment), this
    // tracks a real decaying `Humidity` the way the live render loop does —
    // catches the class of bug where ambient humidity alone drifts into
    // pest territory and permanently taxes photosynthesis.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    let mut humidity = super::humidity::Humidity::new(&config.humidity);
    let mut day_progress = 0.0;
    for _ in 0..90 {
        let sim_dt = config.time.sim_seconds_per_real_second;
        day_progress = (day_progress + sim_dt / config.time.day_length_sim_seconds).rem_euclid(1.0);
        let sun_state = sun::sun_state(day_progress, &config.sun);
        let climate_state = climate::climate_state(day_progress, &config.climate);
        humidity.update(sim_dt, climate_state.temperature_c, &config.humidity);
        plant.step(sim_dt, &sun_state, &climate_state, &mut soil, humidity.level, &config);
        soil.apply_auto_water(true, &config.soil);
    }
    assert!(
        plant.leaves.len() > 1,
        "expected leaf growth to continue past the first leaf over a 90-second default session"
    );
}

#[test]
fn true_zero_input_session_crashes_bone_dry_within_a_real_minute_and_starves() {
    // The actual UI default is auto-water OFF and no manual watering — a
    // player who loads the page and does nothing. At this demo's 80x-sim-
    // speed pacing, soil crashes to bone dry in well under a real minute
    // regardless of the plant, so this genuinely requires watering or
    // auto-water almost immediately, not a leisurely "check back later."
    // Leaf count does climb past 1 early (drought hadn't bitten yet), then
    // collapses as the plant sheds leaves under sustained drought, ending
    // in starvation. Pinned here as the documented current behavior.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    let mut humidity = super::humidity::Humidity::new(&config.humidity);
    let mut day_progress = 0.0;
    let mut leaves_at_30s = 0;
    for i in 0..150 {
        let sim_dt = config.time.sim_seconds_per_real_second;
        day_progress = (day_progress + sim_dt / config.time.day_length_sim_seconds).rem_euclid(1.0);
        let sun_state = sun::sun_state(day_progress, &config.sun);
        let climate_state = climate::climate_state(day_progress, &config.climate);
        humidity.update(sim_dt, climate_state.temperature_c, &config.humidity);
        plant.step(sim_dt, &sun_state, &climate_state, &mut soil, humidity.level, &config);
        if i == 30 {
            leaves_at_30s = plant.leaves.len();
        }
    }
    assert!(leaves_at_30s > 1, "expected leaf growth past the first leaf before drought bites, got {leaves_at_30s}");
    assert_eq!(soil.moisture, 0.0, "expected soil to have crashed bone dry with zero watering");
    assert_eq!(plant.stage, Stage::Dead, "expected sustained drought to eventually starve the plant");
    assert_eq!(plant.death_cause, Some(super::plant::DeathCause::Starvation));
}

#[test]
fn a_plant_that_permanently_outgrows_its_light_source_eventually_dies_rather_than_freezing_forever() {
    // Regression test: a leaf-aging bug (fixed in age_and_senesce_leaves)
    // let leaves live forever under zero external stress, so a plant that
    // raced past its window and crashed to net-negative carbon income got
    // stuck in undead limbo — frozen solid, neither recovering nor dying.
    // Real plants given permanently inadequate light do eventually decline
    // and die; this confirms the sim now does too.
    let config = GrowthConfig::default();
    let mut plant = Plant::new();
    let mut soil = Soil::new(&config.soil);
    let mut humidity = super::humidity::Humidity::new(&config.humidity);
    let mut day_progress = 0.0;
    for _ in 0..600 {
        let sim_dt = config.time.sim_seconds_per_real_second;
        day_progress = (day_progress + sim_dt / config.time.day_length_sim_seconds).rem_euclid(1.0);
        let sun_state = sun::sun_state(day_progress, &config.sun);
        let climate_state = climate::climate_state(day_progress, &config.climate);
        humidity.update(sim_dt, climate_state.temperature_c, &config.humidity);
        plant.step(sim_dt, &sun_state, &climate_state, &mut soil, humidity.level, &config);
        soil.apply_auto_water(true, &config.soil);
    }
    assert_eq!(plant.stage, Stage::Dead);
    assert_eq!(plant.death_cause, Some(super::plant::DeathCause::Starvation));
}
