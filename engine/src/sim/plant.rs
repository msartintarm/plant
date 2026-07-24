//! Plant growth model — a sprout growing into a small houseplant, driven by
//! light (via `sim::sun`) and soil water (via `sim::soil`), grounded in a
//! handful of real plant-physiology heuristics rather than an arbitrary
//! curve:
//!
//! - **Germination** is gated by soil moisture, not light — real seeds
//!   imbibe water to swell and split their coat. The hypocotyl then
//!   elongates on *stored seed reserves*, independent of photosynthesis,
//!   until it breaches the soil (`Stage::Sprout`).
//! - **Photosynthesis is gated by light *and* water together** (Liebig's
//!   law of the minimum): under drought, stomata close to conserve water,
//!   blocking CO2 uptake regardless of available light.
//! - **Elongation is turgor-driven** (the Lockhart equation): cell
//!   expansion needs water pressure directly, not just stored sugar.
//! - **Secondary thickening follows the "pipe model"** (Shinozaki et al.
//!   1964): stem cross-sectional area tracks the leaf area it supplies with
//!   water, so it thickens with *cumulative water throughput*, not simply
//!   with age.
//! - **Etiolation / shade avoidance**: low light biases carbon allocation
//!   toward elongation over thickening — a real seedling races for light at
//!   the expense of a sturdy stem.
//! - **Wilting**: turgor loss under drought droops leaves *and*, once
//!   support tissue itself loses rigidity, physically sags the stem/branches
//!   under their own weight (`Plant::stem_droop`/`Branch::droop`) — both
//!   reversible on rewatering, but the stem's own droop lags further behind
//!   (a whole stem losing turgor is slower and more attenuated by thickness
//!   than a single leaf's motor cells) and is dampened by stem radius: a
//!   thin young stem flops dramatically, an established thick one barely
//!   bends, for the same water stress.
//! - **Leaf initiation follows a plastochron** (a fixed amount of stem
//!   elongation between successive new leaves — see
//!   `PlantConfig::plastochron_height_interval`), not a pure carbon-cost
//!   race against other sinks. New nodes appear as the shoot actually
//!   grows, so a bare stretch of stem with no leaves on it never happens
//!   just because carbon happened to be busy funding something else that
//!   tick.
//!
//! Plants also *move* under their own control, via two physiologically
//! distinct mechanisms this model keeps separate:
//!
//! - **Phototropism** (`Plant::lean_angle`) is slow, cumulative, and
//!   effectively irreversible on short timescales: differential growth
//!   redistributes auxin to a stem's shaded side, elongating it more and
//!   bending the whole stem toward light (the Cholodny-Went model). It only
//!   ever accumulates toward the light, it doesn't relax back when light
//!   drops, because it's built from actual new tissue, not a reversible
//!   motion.
//! - **Heliotropism and nyctinasty** (`Leaf::helio_angle`, `Leaf::fold`) are
//!   fast and fully reversible within a single day: a turgor-pressure motor
//!   organ at the leaf base (a pulvinus, in plants that have one) physically
//!   reorients the leaf toward the light during the day and folds it down
//!   at night ("sleep movement") — no new growth involved, just water
//!   moving in and out of motor cells, which is why it can reverse on the
//!   same timescale as the day/night cycle itself.
//!
//! **Crown branching** (`Plant::branches`) is modeled on *Dracaena*'s
//! well-known growth habit — sold commercially as "branched"/multi-head
//! specifically because of this. A stem's growing tip suppresses the buds
//! below it (apical dominance, via auxin); once that tip's activity pauses
//! (naturally, e.g. after producing a terminal flower, or from injury —
//! here, simply once it's old/tall enough), several of the *topmost*
//! lateral buds release together and grow out as co-dominant branches,
//! rather than one bud quietly taking over. Each branch is then its own
//! smaller growing point — own leaves, own thickness, own phototropic
//! lean — but all branches and the main stem draw from one shared
//! `carbon_pool`: real phloem transport isn't compartmentalized per-branch,
//! sugar goes wherever the plant is currently investing it.
//!
//! Every rate/threshold here is a parameter on `config::PlantConfig` /
//! `config::SoilConfig`, passed in by the caller — this module has no
//! module-level constants of its own to keep in sync with tests or tuning.

use super::climate::{self, ClimateState};
use super::config::{GrowthConfig, GrowthHabit, HumidityConfig, PestConfig, PlantConfig, SeasonConfig, SoilConfig};
use super::humidity::Humidity;
use super::pests;
use super::season;
use super::soil::Soil;
use super::sun::SunState;

/// Hard cap on how many historical segments a single stem or branch
/// records (see `Plant::stem_segment_history`/`Branch::segment_history`) —
/// bounds memory for an arbitrarily long-running session. Past this, the
/// still-growing tip segment simply keeps extending indefinitely (absorbing
/// all further height) instead of subdividing further — a reasonable
/// degradation, since it just means very old, already-established growth
/// renders as one long straight run instead of finer-grained history, and
/// real lower stem sections are usually the straightest, most lignified
/// part anyway. `render::scene`'s instance pool for stem segments is sized
/// off this same constant, so nothing sim ever records goes undrawn.
pub const MAX_STEM_SEGMENTS: usize = 40;

/// Hard cap on `Plant::aerial_roots`, same reasoning as `MAX_STEM_SEGMENTS`
/// — generous relative to any realistic `trellis_height /
/// aerial_root_height_interval`, so it's a memory bound in practice, not a
/// visible truncation.
pub const MAX_AERIAL_ROOTS: usize = 40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Ungerminated — nothing above soil yet.
    Seed,
    /// Hypocotyl pushing up on stored seed reserves; cotyledons unfurl at
    /// the end of this stage.
    Sprout,
    /// True leaves forming, driven by photosynthesis.
    Vegetative,
    /// Root establishment.
    Rooting,
    /// Terminal — either `Plant::root_health` reached zero (total root
    /// loss, from sustained waterlogging/fertilizer burn — see
    /// `SoilConfig::waterlogged_threshold`/`overfeed_threshold`) or
    /// `carbon_pool` stayed at zero for longer than `PlantConfig::
    /// starvation_death_threshold` (prolonged carbon starvation). `step`
    /// becomes a no-op once here — previously there was no failure state
    /// at all, so a neglected plant just idled forever as a bare, leafless
    /// cane rather than actually dying.
    Dead,
}

/// Which specific failure actually killed the plant — see `Plant::
/// death_cause`. Surfaced to the player (via `render::Simulation::stats`)
/// specifically because the two read identically at a glance (a dead plant
/// is a dead plant) but call for opposite corrective action: water less vs.
/// water/feed/light it more, so *just* showing "Dead" leaves a player no way
/// to learn from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeathCause {
    /// `root_health` reached zero — sustained waterlogging or fertilizer-
    /// burn overdose (see `SoilConfig::waterlogged_threshold`/
    /// `overfeed_threshold`), *not* underwatering.
    RootRot,
    /// `carbon_pool` stayed at zero, with no true leaves left anywhere to
    /// ever earn more, for longer than `PlantConfig::starvation_death_
    /// threshold` — the ordinary "neglected it too long" death: drought,
    /// cold, or pests stripped every leaf and it never recovered enough
    /// carbon to grow a new one.
    Starvation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy)]
pub struct Leaf {
    /// Absolute stem height at the moment this leaf attached — fixed for
    /// life. Real stems only extend at the apical meristem (the tip); an
    /// older node doesn't creep upward as the plant keeps growing above it.
    pub attach_height: f64,
    pub side: Side,
    /// 0.0 (just budded) ..= 1.0 (fully expanded) — leaves unfurl gradually
    /// rather than appearing at full size.
    pub maturity: f64,
    /// 0.0 (turgid) ..= 1.0 (fully wilted) — eases toward a target set by
    /// current water stress, not snapped, so wilting/recovery both read as
    /// gradual. A stress response, distinct from the two below.
    pub droop: f64,
    /// Heliotropic reorientation, roughly -1.0..1.0 — eases toward a target
    /// derived from the sun's current position in the window. Reversible
    /// within the day; see the module docs.
    pub helio_angle: f64,
    /// Nyctinastic fold, 0.0 (open, daytime posture) ..= 1.0 (folded,
    /// nighttime posture) — eases toward a target driven by light
    /// intensity. Reversible within the day; see the module docs.
    pub fold: f64,
    /// Sim-seconds since this leaf spawned — only used to gate when
    /// senescence starts (see `PlantConfig::leaf_mature_lifespan`), not
    /// read anywhere else.
    pub age: f64,
    /// 0.0 (healthy) ..= 1.0 (dead) — eases toward 1.0 once `age` passes
    /// `leaf_mature_lifespan`, faster under drought/cold stress. Distinct
    /// from `droop` (reversible turgor loss): senescence is one-directional
    /// aging, and past `PlantConfig::leaf_abscission_senescence_threshold`
    /// the leaf is removed from the plant entirely. Shown to the player as
    /// the leaf's color shifting green → yellow → brown (see
    /// `render::scene::leaf_transform_in_frame`), the most "in context" way
    /// to surface a plant's health without a HUD gauge for every leaf.
    pub senescence: f64,
}

/// A small anchoring root emerging from the main stem while it's trained
/// against a support (`PlantConfig::trellis_height`) — real *aerial roots*
/// (adventitious roots along the stem, distinct from below-ground roots),
/// which species like Pothos/Monstera put out specifically in response to
/// continuous contact with a humid, textured surface like a moss pole, not
/// as a general capability every climbing habit has. This is the real
/// mechanism *Epipremnum aureum* actually climbs by — it presses flat
/// against its support and roots into it, it doesn't twine or send out
/// coiling tendrils the way a pea or morning glory would (see `spawn_due_
/// aerial_roots`). Purely cosmetic (doesn't feed back into growth), same
/// spirit as `Plant`'s terminal flower.
#[derive(Debug, Clone, Copy)]
pub struct AerialRoot {
    /// Absolute main-stem height at the moment this root emerged — fixed
    /// for life, same reasoning as `Leaf::attach_height`.
    pub attach_height: f64,
}

/// A lateral branch off the main stem — a smaller growing point with its
/// own leaves, thickness, and lean, but no carbon of its own (see the
/// module docs: everything draws from `Plant::carbon_pool`). Modeled on
/// *Dracaena*'s crown-branching habit.
#[derive(Debug, Clone)]
pub struct Branch {
    /// Height on the *main stem* where this branch emerges — fixed for
    /// life, same reasoning as `Leaf::attach_height`.
    pub attach_height: f64,
    pub side: Side,
    /// This branch's own length, independent of the main stem's height.
    pub height: f64,
    pub stem_radius: f64,
    /// Slow phototropic lean, same mechanism as `Plant::lean_angle` but
    /// local to this branch. Only the *still-growing* tip actually renders
    /// at this live value — see `segment_history` and
    /// `render::scene::StemCurve`.
    pub lean_angle: f64,
    /// Physical gravity droop under water stress, same mechanism as
    /// `Plant::stem_droop` but local to this branch. Like `lean_angle`,
    /// only the still-growing tip segment renders at this live value.
    pub droop: f64,
    pub leaves: Vec<Leaf>,
    /// This branch's own `height` the last time it grew a new leaf — see
    /// `Plant::height_at_last_leaf` and `PlantConfig::plastochron_height_interval`.
    height_at_last_leaf: f64,
    next_leaf_side: Side,
    /// `lean_angle` frozen at the moment each of this branch's own
    /// completed segments stopped being the growing tip — see
    /// `Plant::stem_segment_history`'s doc comment for the full reasoning
    /// (identical mechanism, just local to this branch instead of the main
    /// stem).
    pub segment_history: Vec<f64>,
    /// This branch's own `height` the last time a segment was recorded —
    /// see `Plant::height_at_last_stem_segment`.
    height_at_last_stem_segment: f64,
}

impl Branch {
    pub fn new(attach_height: f64, side: Side) -> Self {
        Branch {
            attach_height,
            side,
            height: 0.0,
            stem_radius: 0.0,
            lean_angle: 0.0,
            droop: 0.0,
            leaves: Vec::new(),
            height_at_last_leaf: 0.0,
            next_leaf_side: Side::Left,
            segment_history: Vec::new(),
            height_at_last_stem_segment: 0.0,
        }
    }

    fn leaf_area(&self, config: &PlantConfig) -> f64 {
        self.leaves.iter().map(|l| l.maturity * config.leaf_area_per_leaf).sum()
    }
}

/// Attempts to spawn *one* leaf if this grower (the main stem, or one
/// branch) has advanced at least one more `plastochron_height_interval`
/// past `height_at_last_leaf` *and* `carbon_pool` can afford
/// `new_leaf_carbon_cost`. Deliberately just one attempt per call, not a
/// catch-up loop draining a single grower's entire backlog in one go — see
/// `Plant::spawn_due_leaves_fairly`, which calls this in a round-robin
/// across every grower specifically so a long-standing backlog on one of
/// them (typically the oldest branch, or the main stem) can't spend 100% of
/// a tick's carbon on itself before its siblings get a turn. Each leaf
/// attaches at the *interval position* it's due at (`height_at_last_leaf`
/// after incrementing), not at the grower's current total height, so a
/// multi-round catch-up still distributes its leaves up the stem rather
/// than stacking them all at one point. Returns whether a leaf spawned.
fn spawn_one_due_leaf(
    height: f64,
    height_at_last_leaf: &mut f64,
    carbon_pool: &mut f64,
    plant: &PlantConfig,
    leaves: &mut Vec<Leaf>,
    next_side: &mut Side,
) -> bool {
    if height - *height_at_last_leaf >= plant.plastochron_height_interval
        && *carbon_pool > plant.new_leaf_carbon_cost
    {
        *carbon_pool -= plant.new_leaf_carbon_cost;
        *height_at_last_leaf += plant.plastochron_height_interval;
        leaves.push(Leaf {
            attach_height: *height_at_last_leaf,
            side: *next_side,
            maturity: 0.0,
            droop: 0.0,
            helio_angle: 0.0,
            fold: 1.0,
            age: 0.0,
            senescence: 0.0,
        });
        *next_side = match *next_side {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        };
        true
    } else {
        false
    }
}

/// A stem/branch's target physical gravity droop this tick: scales with
/// water stress (`1.0 - water_factor`, same drought signal `Leaf::droop`
/// targets) but attenuated by `stem_radius` — flexural stiffness grows
/// sharply with thickness in real stems, so a thin young one droops close
/// to `stem_droop_max_angle` at full stress while an established thick one
/// barely moves for the same stress. Shared by the main stem and every
/// branch (each with its own `stem_radius`), same reasoning as
/// `spawn_one_due_leaf`.
fn stem_droop_target(water_factor: f64, stem_radius: f64, plant: &PlantConfig) -> f64 {
    let flexibility = plant.stem_droop_reference_radius / (stem_radius + plant.stem_droop_reference_radius);
    (1.0 - water_factor) * flexibility * plant.stem_droop_max_angle
}

/// This tick's target for `Plant::bloom_intensity` — 0.0 (not mature
/// enough to flower at all yet) or, once mature, 1.0 while within
/// `bloom_duration` of the current cycle and 0.0 while resting for
/// `bloom_rest_duration` after it — see `PlantConfig`'s flowering doc
/// comments. A pure function of already-known values (no `Plant` access,
/// no `%` computed twice) — independently testable with plain numbers, no
/// growth simulation required.
fn bloom_intensity_target(mature_enough: bool, bloom_cycle_position: f64, plant: &PlantConfig) -> f64 {
    if !mature_enough {
        return 0.0;
    }
    let cycle_length = plant.bloom_duration + plant.bloom_rest_duration;
    if cycle_length <= 0.0 {
        return 0.0;
    }
    if bloom_cycle_position % cycle_length < plant.bloom_duration {
        1.0
    } else {
        0.0
    }
}

/// How much window light actually reaches a grower currently at `height` —
/// distance from the light source, not just time of day. 1.0 within the
/// window's own height range, easing down to `ambient_light_floor` over
/// `window_light_falloff_range` further up. Growth that pushes height well
/// past this naturally starves for carbon rather than continuing forever,
/// which is why real houseplants don't grow indefinitely past their own
/// window.
pub fn height_light_factor(height: f64, plant: &PlantConfig) -> f64 {
    if height <= plant.window_light_zone_height {
        1.0
    } else {
        let excess = height - plant.window_light_zone_height;
        let t = (excess / plant.window_light_falloff_range).clamp(0.0, 1.0);
        1.0 + (plant.ambient_light_floor - 1.0) * t
    }
}

/// Whether a grower currently at `height` is bending toward light on its
/// own (a freestanding stem always is) or being held straight by a climbing
/// habit's support instead — see `PlantConfig::trellis_height`'s doc
/// comment. A single pure predicate shared by the main stem
/// (`step_vegetative`) and every branch (`step_branch`, using its own
/// `attach_height + height`) rather than duplicated inline at each call
/// site, so the "is this grower still on its support" rule has exactly one
/// definition to test and change.
fn leans_freely(height: f64, trellis_height: Option<f64>) -> bool {
    trellis_height.is_none_or(|trellis_height| height > trellis_height)
}

/// How stressed the whole plant is right now, for leaf senescence purposes
/// — whichever of drought or cold is currently worse (not summed: a plant
/// shedding leaves over one bad condition isn't shedding twice as fast just
/// because a second condition also happens to be present). 0.0 is
/// unstressed, 1.0 is maximally stressed.
fn leaf_stress_signal(water_factor: f64, temperature_c: f64, pest_infestation: f64, plant: &PlantConfig) -> f64 {
    let drought_stress = 1.0 - water_factor;
    let cold_stress = ((plant.cold_stress_threshold_c - temperature_c) / plant.temperature_tolerance_c)
        .clamp(0.0, 1.0);
    let pest_stress = pest_infestation.clamp(0.0, 1.0);
    drought_stress.max(cold_stress).max(pest_stress)
}

/// Each leaf's own light factor from self-shading by this *same grower's*
/// newer, higher leaves — see `PlantConfig::leaf_self_shading_coeff`'s doc
/// comment for the real-world grounding. `leaves` must be in ascending
/// `attach_height` order, which both `Plant::leaves` and `Branch::leaves`
/// always are (new leaves are only ever pushed at the grower's current,
/// ever-increasing height — see `spawn_one_due_leaf`), so leaf area at a
/// *later* index is always physically above and shading the leaf at an
/// earlier one. Returned in the same order/length as `leaves`, so callers
/// can zip them back together. `pub` so `render` can also use it to darken
/// occluded leaves visually, not just discount their photosynthesis.
pub fn self_shading_factors(leaves: &[Leaf], plant: &PlantConfig) -> Vec<f64> {
    let mut area_above = 0.0;
    let mut factors = vec![0.0; leaves.len()];
    for i in (0..leaves.len()).rev() {
        factors[i] = (-plant.leaf_self_shading_coeff * area_above).exp();
        area_above += leaves[i].maturity * plant.leaf_area_per_leaf;
    }
    factors
}

/// Ages every leaf in `leaves` by `dt` and sheds (removes) any whose
/// senescence has crossed `leaf_abscission_senescence_threshold`. Two
/// separate senescence pressures, both feeding the same `senescence` value:
///
/// - **Age/environmental stress** (drought, cold): only starts once a leaf
///   is past `leaf_mature_lifespan` — a real leaf doesn't age out early just
///   because water or temperature is briefly bad.
/// - **Self-shading** (`shade_factors`, from `self_shading_factors`): starts
///   *immediately*, with no age gate at all. This is deliberate and mirrors
///   real shade-induced leaf drop, which isn't a slower version of ordinary
///   old-age senescence — a leaf that gets rapidly overtopped by vigorous
///   new growth above it can yellow and drop within weeks, long before
///   anything like its full potential lifespan, precisely because it's
///   already a net carbon liability (full respiration upkeep, negligible
///   photosynthetic income) from the moment it's buried, not just once it's
///   additionally old. Gating this the same way as age/stress would let a
///   fast-growing plant's initial burst of new leaves pile up largely
///   unchecked for its entire `leaf_mature_lifespan` before any of them
///   could be shed, which is exactly the unrealistic runaway-leaf-count
///   behavior this mechanism exists to prevent.
///
/// `shade_factors` must be the same length/order as `leaves` — see
/// `self_shading_factors`. Shared by the main stem and every branch, same
/// reasoning as `spawn_one_due_leaf`/`stem_droop_target`.
fn age_and_senesce_leaves(
    leaves: &mut Vec<Leaf>,
    stress_signal: f64,
    shade_factors: &[f64],
    dt: f64,
    plant: &PlantConfig,
) {
    for (leaf, &shade_factor) in leaves.iter_mut().zip(shade_factors) {
        leaf.age += dt;
        // Bug fix: this used to be `if past_lifespan { stress_signal } else
        // { 0.0 }`, so zero external stress meant zero age-driven decay —
        // a leaf could live forever past its own lifespan in perfect
        // conditions, contradicting "baseline rate... absent any stress"
        // above. `past_lifespan` alone should gate baseline decay; stress
        // only multiplies the rate on top.
        let past_lifespan = leaf.age >= plant.leaf_mature_lifespan;
        let shade_stress = 1.0 - shade_factor;
        if past_lifespan || shade_stress > 0.0 {
            let stress = if past_lifespan { stress_signal.max(shade_stress) } else { shade_stress };
            let rate = plant.leaf_senescence_rate * (1.0 + stress * (plant.leaf_stress_senescence_multiplier - 1.0));
            leaf.senescence += (1.0 - leaf.senescence) * (rate * dt).min(1.0);
        }
    }
    leaves.retain(|leaf| leaf.senescence < plant.leaf_abscission_senescence_threshold);
}

/// Freezes `lean_angle` into `segment_history` for every
/// `stem_segment_height_interval` of `height` grown since the last one was
/// recorded — a `while` loop (not a single `if`), same reasoning as
/// `spawn_one_due_leaf`: a coarse timestep can cross several interval
/// boundaries in a single tick, and each one should still get its own
/// (slightly different) historical angle rather than only the last.
/// Stops recording new segments past `MAX_STEM_SEGMENTS` — see that
/// constant's doc comment on why that's a reasonable degradation rather
/// than an error. Shared by the main stem and every branch, same reasoning
/// as `spawn_one_due_leaf`/`stem_droop_target`.
fn record_stem_segments(
    height: f64,
    height_at_last_stem_segment: &mut f64,
    lean_angle: f64,
    segment_history: &mut Vec<f64>,
    plant: &PlantConfig,
) {
    while height - *height_at_last_stem_segment >= plant.stem_segment_height_interval
        && segment_history.len() < MAX_STEM_SEGMENTS
    {
        segment_history.push(lean_angle);
        *height_at_last_stem_segment += plant.stem_segment_height_interval;
    }
}

/// Spawns a new `AerialRoot` for every `aerial_root_height_interval` of
/// height grown since the last one, while `climbing_now` (see `leans_
/// freely` — this is its exact negation, passed in rather than
/// recomputed, since the caller already has it). A `while` loop (not a
/// single `if`), same reasoning as `record_stem_segments`: a coarse
/// timestep can cross several interval boundaries in one tick. Stops past
/// `MAX_AERIAL_ROOTS`, same reasoning as that constant's doc comment. A
/// pure function taking every input explicitly (no hidden state, nothing
/// read off `self`) — independently testable with a bare `Vec` and a
/// couple of `f64`s, no `Plant`/`Soil`/`SunState` setup required.
fn spawn_due_aerial_roots(
    height: f64,
    height_at_last_aerial_root: &mut f64,
    aerial_roots: &mut Vec<AerialRoot>,
    climbing_now: bool,
    plant: &PlantConfig,
) {
    if !climbing_now {
        return;
    }
    while height - *height_at_last_aerial_root >= plant.aerial_root_height_interval
        && aerial_roots.len() < MAX_AERIAL_ROOTS
    {
        *height_at_last_aerial_root += plant.aerial_root_height_interval;
        aerial_roots.push(AerialRoot { attach_height: *height_at_last_aerial_root });
    }
}

/// A snapshot of one `Plant::step` call's inputs and the decisions it
/// produced from them — lets tests (and, later, a debug HUD) assert on
/// *why* the plant did or didn't grow this tick, not just infer it from
/// running thousands of ticks and eyeballing the aggregate result. Cheap
/// single-tick tests over this are the preferred way to pin down a specific
/// mechanism; the long-running aggregate tests below are for genuinely
/// cumulative/emergent behavior (leaf count over time, lean saturating)
/// that no single tick's decision can show on its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    Seed {
        water_factor: f64,
        threshold: f64,
        germinated: bool,
    },
    Sprout {
        water_factor: f64,
        height_gained: f64,
        transitioned_to_vegetative: bool,
    },
    Vegetative {
        // Inputs.
        sun_intensity: f64,
        sun_azimuth: f64,
        water_factor: f64,
        /// `water_factor` further discounted by `root_health` and pot-bound
        /// stress — what growth/uptake/wilting actually use, see
        /// `step_vegetative`. Equal to `water_factor` whenever roots are
        /// fully healthy and not yet pot-bound.
        effective_water_factor: f64,
        /// Shared stomatal opening factor, constrained by both effective
        /// root water and atmospheric VPD. Gates photosynthesis and
        /// transpiration together.
        stomatal_conductance: f64,
        root_health: f64,
        pest_infestation: f64,
        /// `season::season_state` this tick — 1.0 at midsummer, dropping
        /// toward `SeasonConfig::winter_floor` in winter; multiplies
        /// elongation only (see `PlantConfig::dormancy_elongation_
        /// sensitivity`).
        dormancy_factor: f64,
        leaf_area: f64,
        /// `light_weighted_leaf_area` this tick — what transpiration
        /// (`uptake_rate`) uses; see `height_light_factor`.
        lit_leaf_area: f64,
        /// `photosynthesis_leaf_area` this tick — what photosynthesis
        /// actually uses; additionally discounted by self-shading among
        /// this plant's own leaves on top of `lit_leaf_area`'s room-
        /// position falloff, see `self_shading_factors`.
        photosynthesis_area: f64,
        carbon_pool_before: f64,
        temperature_c: f64,
        /// `climate::temperature_factor` this tick — gates photosynthesis
        /// and elongation.
        temp_factor: f64,
        /// `climate::q10_factor` this tick — multiplies respiration.
        respiration_q10_factor: f64,
        // Carbon economy.
        photosynthesis: f64,
        respiration: f64,
        carbon_pool_after: f64,
        // Growth allocation.
        elongation: f64,
        /// True if the carbon pool, not water/light, was the binding
        /// constraint on elongation this tick (see `step_vegetative`).
        elongation_carbon_limited: bool,
        thickening_delta: f64,
        leaf_spawned: bool,
        branch_spawned: bool,
        // Movement targets shared by every leaf this tick (each leaf eases
        // toward these at its own current rate — see `Leaf`).
        lean_delta: f64,
        helio_target: f64,
        fold_target: f64,
        droop_target: f64,
        /// This tick's target for `Plant::stem_droop` — see the module docs
        /// on why the *stem's* own droop is a separate, slower-easing,
        /// thickness-attenuated mechanism from `droop_target` above (which
        /// only drives leaf blades).
        stem_droop_target: f64,
    },
}

#[derive(Debug, Clone)]
pub struct Plant {
    pub stage: Stage,
    pub height: f64,
    pub stem_radius: f64,
    pub carbon_pool: f64,
    pub cumulative_water_uptake: f64,
    pub leaves: Vec<Leaf>,
    /// Slow, cumulative phototropic stem lean (radians) — see module docs.
    /// Only the main stem's still-growing tip segment actually renders at
    /// this live value; every completed segment stays frozen at whatever
    /// `stem_segment_history` recorded for it — see that field and
    /// `render::scene::StemCurve`.
    pub lean_angle: f64,
    /// Physical gravity droop under water stress (radians) — see module
    /// docs and `PlantConfig::stem_droop_max_angle`. Distinct from
    /// `Leaf::droop`, which only tips individual leaf blades. Like
    /// `lean_angle`, only the still-growing tip segment renders at this
    /// live value.
    pub stem_droop: f64,
    /// Crown branches off the main stem — see `Branch` and the module docs.
    pub branches: Vec<Branch>,
    /// What the most recent `step` call actually did and why — see
    /// `Decision`. `None` only before the very first `step`.
    pub last_decision: Option<Decision>,
    /// This stem's own `height` the last time it grew a new leaf — see
    /// `PlantConfig::plastochron_height_interval`.
    height_at_last_leaf: f64,
    next_side: Side,
    next_branch_side: Side,
    /// `lean_angle` frozen at the moment each completed segment of the main
    /// stem stopped being the growing tip — real stem tissue keeps whatever
    /// curvature it had when it stiffened rather than retroactively
    /// straightening (or bending further) just because the growing tip
    /// keeps leaning more as the plant ages. Rendering walks this history
    /// (see `render::scene::StemCurve`) instead of treating the whole stem
    /// as one rigid rotation, so a long-lived, still-leaning plant reads as
    /// a gentle sweep — straighter low down (formed early, before much lean
    /// had accumulated), more bent up high (recent growth, under whatever
    /// lean is current *now*) — not a single straight line pivoting from
    /// the pot. Capped at `MAX_STEM_SEGMENTS`.
    pub stem_segment_history: Vec<f64>,
    /// This stem's own `height` the last time a segment was recorded — see
    /// `PlantConfig::stem_segment_height_interval`.
    height_at_last_stem_segment: f64,
    /// Aerial roots along the *main stem only* (see `AerialRoot`) — a
    /// deliberate scope simplification: a Pothos's lateral shoots
    /// (`branches`) typically just hang free off an already-anchored vine
    /// rather than independently re-rooting into the same support, so this
    /// isn't mirrored on `Branch` the way `stem_segment_history` is.
    /// Always empty for a freestanding habit (`PlantConfig::trellis_height:
    /// None`). Capped at `MAX_AERIAL_ROOTS`.
    pub aerial_roots: Vec<AerialRoot>,
    /// This stem's own `height` the last time an aerial root was spawned —
    /// see `PlantConfig::aerial_root_height_interval`.
    height_at_last_aerial_root: f64,
    /// Sim-seconds since `height` first reached `flowering_height_
    /// threshold` — stays 0 until then, only advances once mature (see
    /// `bloom_intensity_target`). Used, modulo `bloom_duration +
    /// bloom_rest_duration`, to determine which phase of the bloom cycle
    /// the plant is currently in.
    bloom_cycle_position: f64,
    rooting_elapsed: f64,
    /// 0.0 (fully closed/no visible bloom) ..= 1.0 (fully open) — eases
    /// toward its current cycle phase's target at `bloom_response_rate`,
    /// same idiom as `Leaf::droop`/`helio_angle`/`fold`. Purely cosmetic;
    /// see `render::scene::flower_transform`.
    pub bloom_intensity: f64,
    /// Which grower (main stem = slot 0, else `branches[slot - 1]`) gets
    /// first claim on carbon in `spawn_due_leaves_fairly`'s next round —
    /// persisted across ticks (not reset per call) so the rotation actually
    /// advances: most ticks only afford a handful of leaf-spawns total, far
    /// fewer than a full lap around every grower, so a rotation that reset
    /// every call would only ever get through its first one or two slots
    /// and never reach the rest.
    leaf_priority_rotation: usize,
    /// 1.0 (fully healthy) ..= 0.0 (totally rotted, kills the plant) —
    /// damaged by sustained waterlogging/fertilizer-burn stress, recovered
    /// slowly absent it (or restored on `repot`). Multiplies directly into
    /// `effective_water_factor`, so damaged roots can't take up water even
    /// when the soil itself is wet — the real, counterintuitive symptom
    /// that makes overwatering something to diagnose, not just "more of a
    /// good thing." See module docs and `Soil::waterlog_stress`.
    pub root_health: f64,
    /// Sim-seconds soil has stayed continuously at/above `SoilConfig::
    /// waterlogged_threshold` (or nutrient at/above `overfeed_threshold`) —
    /// resets to zero the instant it drops back below. `root_health` only
    /// starts decaying once this exceeds `SoilConfig::waterlog_grace_
    /// period`, so a brief touch of saturation right after watering never
    /// matters, only a pot kept artificially flooded does.
    waterlogged_duration: f64,
    /// 0.0 (pest-free) ..= 1.0 (severe infestation) — modeled on spider
    /// mites, favored by dry air (see `pests::pest_growth_rate`). Directly
    /// taxes photosynthesis and feeds into `leaf_stress_signal`; knocked
    /// down by `treat_pests`.
    pub pest_infestation: f64,
    /// Multiplies `PlantConfig::initial_pot_capacity` to get this plant's
    /// *current* pot-bound ceiling — starts at 1.0 (the pot it germinated
    /// in), multiplied by `PlantConfig::repot_capacity_multiplier` each time
    /// `repot` is called. See `pot_bound_factor`.
    pot_capacity_multiplier: f64,
    /// 0.0 (no setback) ..= 1.0 (maximally shocked) — a shared "just had a
    /// physical setback" signal pushed up by both `prune` (cut tissue) and
    /// `repot` (disturbed roots), easing back down to zero over time (see
    /// `PlantConfig::shock_recovery_rate`). Multiplies down elongation only
    /// — a shocked plant still ticks over, just slower, the same real
    /// tradeoff both actions exist for.
    pub growth_shock: f64,
    /// Sim-seconds `carbon_pool` has stayed pinned at exactly zero —
    /// resets the instant it recovers above zero. Crossing `PlantConfig::
    /// starvation_death_threshold` kills the plant (see `Stage::Dead`); a
    /// grace period, not an instant kill, since one bad night of zero net
    /// income is normal and recoverable.
    starvation_timer: f64,
    /// Total sim-seconds this plant has been stepped, accumulated
    /// regardless of stage — the only state `season::season_state` needs
    /// to evaluate the slow year-length dormancy cycle, the same "pure
    /// function of elapsed time" pattern `climate::climate_state` already
    /// uses for the day/night cycle.
    total_time: f64,
    /// Which failure actually killed the plant — `None` while alive, set
    /// exactly once at the tick `stage` becomes `Stage::Dead` and never
    /// cleared afterward. See `DeathCause`.
    pub death_cause: Option<DeathCause>,
    /// Cumulative leaves ever spawned over this plant's life — main stem
    /// and every branch, including ones later shed by senescence or cut off
    /// by pruning. Distinct from the *current* leaf count (`leaves.len()`
    /// plus each branch's own): see `max_leaves_at_once` for the concurrent
    /// high-water mark. Monotonically increasing — a scoring metric, not a
    /// live gauge.
    pub leaves_produced_total: u32,
    /// Highest concurrent leaf count (main stem + every branch) this plant
    /// has ever held at once — a running high-water mark, unlike the
    /// current count itself, which drops as leaves senesce or get pruned.
    /// Monotonically increasing.
    pub max_leaves_at_once: u32,
    /// Tallest `height` this plant has ever reached — a running high-water
    /// mark that survives `height` itself dropping back down (taking a
    /// cutting shortens the parent; pruning shortens whatever it cuts).
    /// Monotonically increasing.
    pub max_height_reached: f64,
    /// Sim-seconds this plant has spent anywhere other than `Stage::Dead`.
    /// Unlike `total_time` (which keeps accumulating after death — see its
    /// own doc comment), this freezes the instant `stage` becomes
    /// `Stage::Dead`, so it reads as "how long it lived," not "how long
    /// it's existed."
    pub alive_duration: f64,
    /// Chosen once, at germination — eases elongation toward zero as
    /// height/branch-length approach `PlantConfig::realistic_max_height`
    /// (see `realistic_scale_taper`) instead of today's default unbounded
    /// growth, which otherwise never truly stops (`height_light_factor`
    /// bottoms out at a nonzero floor, not zero).
    pub realistic_scale: bool,
}

impl Default for Plant {
    fn default() -> Self {
        Plant {
            stage: Stage::Seed,
            height: 0.0,
            stem_radius: 0.0,
            carbon_pool: 0.0,
            cumulative_water_uptake: 0.0,
            leaves: Vec::new(),
            lean_angle: 0.0,
            stem_droop: 0.0,
            branches: Vec::new(),
            last_decision: None,
            height_at_last_leaf: 0.0,
            next_side: Side::Left,
            next_branch_side: Side::Left,
            leaf_priority_rotation: 0,
            stem_segment_history: Vec::new(),
            height_at_last_stem_segment: 0.0,
            aerial_roots: Vec::new(),
            height_at_last_aerial_root: 0.0,
            bloom_cycle_position: 0.0,
            bloom_intensity: 0.0,
            rooting_elapsed: 0.0,
            root_health: 1.0,
            waterlogged_duration: 0.0,
            pest_infestation: 0.0,
            pot_capacity_multiplier: 1.0,
            growth_shock: 0.0,
            starvation_timer: 0.0,
            total_time: 0.0,
            death_cause: None,
            leaves_produced_total: 0,
            max_leaves_at_once: 0,
            max_height_reached: 0.0,
            alive_duration: 0.0,
            realistic_scale: false,
        }
    }
}

/// Eases elongation toward zero as `height` nears `max_height` — 1.0 well
/// below it, 0.0 at/past it. See `Plant::realistic_scale`.
fn realistic_scale_taper(height: f64, max_height: f64) -> f64 {
    if max_height <= 0.0 {
        return 0.0;
    }
    (1.0 - height / max_height).clamp(0.0, 1.0)
}

impl Plant {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the realistic-scale choice at creation (see `realistic_scale`'s
    /// own doc comment) — a builder method, not a struct literal, since
    /// most of this struct's fields are private to this module.
    pub fn with_realistic_scale(mut self, enabled: bool) -> Self {
        self.realistic_scale = enabled;
        self
    }

    /// Total sim-seconds this plant has been stepped — the only input
    /// `season::season_state` needs; exposed read-only so a caller (e.g.
    /// `render::Simulation::stats`) can evaluate the dormancy cycle for
    /// display without this module needing to know anything about seasons
    /// itself.
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    pub fn true_leaf_area(&self, config: &PlantConfig) -> f64 {
        self.leaves.iter().map(|l| l.maturity * config.leaf_area_per_leaf).sum()
    }

    pub fn cotyledon_fade_fraction(&self, config: &PlantConfig) -> f64 {
        if self.stage == Stage::Seed {
            return 0.0;
        }
        (1.0 - self.leaves_produced_total as f64 / config.cotyledon_fade_over_leaves).clamp(0.0, 1.0)
    }

    fn total_leaf_area(&self, config: &PlantConfig) -> f64 {
        let cotyledon_area = self.cotyledon_fade_fraction(config) * config.cotyledon_leaf_area;
        let branch_area: f64 = self.branches.iter().map(|b| b.leaf_area(config)).sum();
        cotyledon_area + self.true_leaf_area(config) + branch_area
    }

    fn light_weighted_leaf_area(&self, config: &PlantConfig) -> f64 {
        let cotyledon_area = self.cotyledon_fade_fraction(config) * config.cotyledon_leaf_area;
        let main_stem_factor = height_light_factor(self.height, config);
        let main_stem_area = (cotyledon_area + self.true_leaf_area(config)) * main_stem_factor;
        let branch_area: f64 = self
            .branches
            .iter()
            .map(|b| b.leaf_area(config) * height_light_factor(b.attach_height + b.height, config))
            .sum();
        main_stem_area + branch_area
    }

    // Monsi-Saeki canopy photosynthesis
    fn photosynthesis_leaf_area(&self, config: &PlantConfig) -> f64 {
        let cotyledon_area = self.cotyledon_fade_fraction(config) * config.cotyledon_leaf_area;
        let main_stem_factor = height_light_factor(self.height, config);
        let self_shaded_true_leaf_area: f64 = self
            .leaves
            .iter()
            .zip(self_shading_factors(&self.leaves, config))
            .map(|(leaf, shade_factor)| leaf.maturity * config.leaf_area_per_leaf * shade_factor)
            .sum();
        let main_stem_area = (cotyledon_area + self_shaded_true_leaf_area) * main_stem_factor;
        let branch_area: f64 = self
            .branches
            .iter()
            .map(|b| {
                let self_shaded: f64 = b
                    .leaves
                    .iter()
                    .zip(self_shading_factors(&b.leaves, config))
                    .map(|(leaf, shade_factor)| leaf.maturity * config.leaf_area_per_leaf * shade_factor)
                    .sum();
                self_shaded * height_light_factor(b.attach_height + b.height, config)
            })
            .sum();
        main_stem_area + branch_area
    }

    fn spawn_leaf(&mut self) {
        self.leaves.push(Leaf {
            attach_height: self.height,
            side: self.next_side,
            maturity: 0.0,
            droop: 0.0,
            helio_angle: 0.0,
            // Starts folded, like an unopened bud, and unfurls via the
            // same nyctinasty dynamics that reopen it every subsequent
            // morning.
            fold: 1.0,
            age: 0.0,
            senescence: 0.0,
        });
        self.next_side = match self.next_side {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        };
        self.leaves_produced_total += 1;
    }

    fn spawn_branch(&mut self) {
        self.branches.push(Branch::new(self.height, self.next_branch_side));
        self.next_branch_side = match self.next_branch_side {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        };
    }

    /// Fairly interleaves leaf initiation across the main stem and every
    /// branch — all draw from the shared `carbon_pool` (see the module
    /// docs), and simply processing "the main stem, then each branch in
    /// creation order, each catching up its *entire* backlog before moving
    /// on" let whichever grower ran first spend 100% of a tick's available
    /// carbon on itself, permanently starving every later sibling once any
    /// grower had a standing backlog — which a long enough real session
    /// always eventually produces (a fast-elongating grower outpaces how
    /// much carbon a single tick can fund). One "round" gives every
    /// eligible grower at most one leaf (`spawn_one_due_leaf`); rounds
    /// repeat until nobody both has backlog and can afford one. Returns
    /// whether any leaf spawned anywhere this tick.
    fn spawn_due_leaves_fairly(&mut self, plant: &PlantConfig) -> bool {
        // Slot 0 is the main stem, slot i+1 is `self.branches[i]` — treating
        // both uniformly by index (rather than "main stem, then iterate
        // branches") is what lets the starting slot rotate below.
        let total_growers = 1 + self.branches.len();
        // A newly-created branch means there are now more slots than the
        // rotation counter accounted for — wrapping it into range here
        // (rather than only ever taking a modulo at use sites) keeps it a
        // meaningful "whose turn" position instead of an ever-growing
        // counter.
        self.leaf_priority_rotation %= total_growers;
        let mut spawned_anywhere = false;
        loop {
            let mut spawned_this_round = false;
            for offset in 0..total_growers {
                let slot = (self.leaf_priority_rotation + offset) % total_growers;
                let spawned = if slot == 0 {
                    spawn_one_due_leaf(
                        self.height,
                        &mut self.height_at_last_leaf,
                        &mut self.carbon_pool,
                        plant,
                        &mut self.leaves,
                        &mut self.next_side,
                    )
                } else {
                    let branch = &mut self.branches[slot - 1];
                    spawn_one_due_leaf(
                        branch.height,
                        &mut branch.height_at_last_leaf,
                        &mut self.carbon_pool,
                        plant,
                        &mut branch.leaves,
                        &mut branch.next_leaf_side,
                    )
                };
                if spawned {
                    self.leaves_produced_total += 1;
                }
                spawned_this_round = spawned_this_round || spawned;
            }
            // Advances every round regardless of whether it spawned
            // anything — a carbon-starved tick where *no* grower could
            // afford a leaf still needs the rotation to progress, or
            // whichever slot is first the moment carbon does recover would
            // permanently keep first claim.
            self.leaf_priority_rotation = (self.leaf_priority_rotation + 1) % total_growers;
            if !spawned_this_round {
                break;
            }
            spawned_anywhere = true;
        }
        spawned_anywhere
    }

    /// Advances the simulation by `dt` seconds given the current sun state
    /// and soil (which this also draws water from / evaporates further).
    /// `humidity_level` is the current ambient air humidity (0.0..1.0) —
    /// passed in as an already-computed snapshot, same reasoning as `sun`/
    /// `climate`: `sim::humidity::Humidity` is small, explicit, caller-owned
    /// state (a `mist` action needs to persist across ticks) rather than
    /// threaded through here as another `&mut`, which would mean every
    /// existing call site of `step` gains a second piece of mutable state to
    /// manage alongside `soil` for no benefit — the sim only ever needs to
    /// *read* the current level, never mutate it itself.
    pub fn step(
        &mut self,
        dt: f64,
        sun: &SunState,
        climate: &ClimateState,
        soil: &mut Soil,
        humidity_level: f64,
        config: &GrowthConfig,
    ) {
        if dt <= 0.0 {
            return;
        }
        // Day/night and season advance on the simulation clock; biological
        // processes use the independently configurable physiology clock.
        let physiology_dt = config.time.physiology_dt(dt);
        self.total_time += dt;
        if self.stage != Stage::Dead {
            self.alive_duration += dt;
        }
        match self.stage {
            Stage::Seed => self.step_seed(physiology_dt, climate, soil, &config.plant, &config.soil),
            Stage::Sprout => self.step_sprout(physiology_dt, climate, soil, &config.plant, &config.soil),
            Stage::Vegetative | Stage::Rooting => self.step_vegetative(
                physiology_dt,
                sun,
                climate,
                soil,
                humidity_level,
                &config.plant,
                &config.soil,
                &config.humidity,
                &config.pest,
                &config.season,
            ),
            // Terminal — see `Stage::Dead`'s doc comment.
            Stage::Dead => {}
        }
        if self.height > self.max_height_reached {
            self.max_height_reached = self.height;
        }
        let leaf_count = self.leaves.len() + self.branches.iter().map(|b| b.leaves.len()).sum::<usize>();
        if leaf_count as u32 > self.max_leaves_at_once {
            self.max_leaves_at_once = leaf_count as u32;
        }
    }

    fn step_seed(
        &mut self,
        dt: f64,
        climate: &ClimateState,
        soil: &mut Soil,
        plant: &PlantConfig,
        soil_cfg: &SoilConfig,
    ) {
        // Bone-dry soil still evaporates/settles even with nothing planted
        // above it yet, and no light-dependent uptake exists pre-sprout.
        soil.update(dt, 0.0, 0.0, soil_cfg);
        let water_factor = soil.water_factor(soil_cfg);
        // Real seeds need both moisture *and* warmth to imbibe and
        // germinate — see `PlantConfig::germination_min_temperature_c`.
        let germinated = water_factor >= plant.germination_water_factor
            && climate.temperature_c >= plant.germination_min_temperature_c;
        if germinated {
            self.stage = Stage::Sprout;
        }
        self.last_decision = Some(Decision::Seed {
            water_factor,
            threshold: plant.germination_water_factor,
            germinated,
        });
    }

    fn step_sprout(
        &mut self,
        dt: f64,
        climate: &ClimateState,
        soil: &mut Soil,
        plant: &PlantConfig,
        soil_cfg: &SoilConfig,
    ) {
        // Elongates on stored seed reserves — gated by turgor (water_factor)
        // like all elongation, but not by light or carbon: there's no
        // photosynthetic surface yet to produce either. Still
        // temperature-sensitive, like any turgor-driven cell expansion.
        soil.update(dt, 0.0, 0.0, soil_cfg);
        let water_factor = soil.water_factor(soil_cfg);
        let temp_factor = climate::temperature_factor(
            climate.temperature_c,
            plant.optimal_temperature_c,
            plant.temperature_tolerance_c,
        );
        let height_gained = plant.sprout_growth_rate * water_factor * temp_factor * dt;
        self.height += height_gained;
        let transitioned_to_vegetative = self.height >= plant.sprout_height_threshold;
        if transitioned_to_vegetative {
            self.stage = Stage::Vegetative;
            self.spawn_leaf();
        }
        self.last_decision = Some(Decision::Sprout {
            water_factor,
            height_gained,
            transitioned_to_vegetative,
        });
    }

    fn step_vegetative(
        &mut self,
        dt: f64,
        sun: &SunState,
        climate: &ClimateState,
        soil: &mut Soil,
        humidity_level: f64,
        plant: &PlantConfig,
        soil_cfg: &SoilConfig,
        humidity_cfg: &HumidityConfig,
        pest_cfg: &PestConfig,
        season_cfg: &SeasonConfig,
    ) {
        let rooting = self.stage == Stage::Rooting;
        let water_factor = soil.water_factor(soil_cfg);

        // Root health: overwatering / fertilizer burn (see module docs and
        // `Soil::waterlog_stress`/`overfeed_stress`) — whichever stress is
        // currently worse (same "worst, not summed" pattern as
        // `leaf_stress_signal`), gated by a grace period so a single
        // watering dose that's then allowed to drain normally never
        // matters, only a pot kept artificially flooded/oversalted does.
        let waterlog_stress = soil.waterlog_stress(soil_cfg);
        let overfeed_stress = soil.overfeed_stress(soil_cfg);
        let root_stress = waterlog_stress.max(overfeed_stress);
        if root_stress > 0.0 {
            self.waterlogged_duration += dt;
        } else {
            self.waterlogged_duration = 0.0;
        }
        if self.waterlogged_duration > soil_cfg.waterlog_grace_period {
            self.root_health = (self.root_health - plant.root_rot_rate * root_stress * dt).max(0.0);
        } else {
            self.root_health = (self.root_health + plant.root_recovery_rate * dt).min(1.0);
        }

        // Pot-bound stress: a real container caps how large a root system
        // can get before it needs a bigger pot — see `pot_bound_factor`.
        let pot_bound_factor = self.pot_bound_factor(plant);
        // What growth/uptake/wilting actually use: raw soil water further
        // discounted by how well the roots can actually take it up right
        // now — a rotted or pot-bound root system can't draw water even
        // when the soil itself is wet, the real symptom that makes
        // overwatering a mistake to diagnose rather than just "not enough
        // of a good thing."
        let effective_water_factor = water_factor * self.root_health * pot_bound_factor;

        let nutrient_factor = soil.nutrient_factor(soil_cfg);

        // Growth shock: a shared setback signal from pruning/repotting (see
        // module docs) — decays back toward zero over time.
        let shock_factor = 1.0 - plant.shock_growth_penalty * self.growth_shock;
        self.growth_shock = (self.growth_shock - plant.shock_recovery_rate * dt).max(0.0);

        // Dormancy: winter's shorter days suppress elongation independent
        // of temperature (a real photoperiod response) — see
        // `season::season_state`.
        let season_state = season::season_state(self.total_time, season_cfg);
        let dormancy_factor =
            (1.0 - plant.dormancy_elongation_sensitivity * (1.0 - season_state.day_length_factor)).max(0.0);

        // Pests: modeled on spider mites, favored by dry air (see
        // `pests::pest_growth_rate`) — a threat orthogonal to the water/
        // light/nutrient economy.
        let pest_growth = pests::pest_growth_rate(humidity_level, pest_cfg);
        self.pest_infestation = (self.pest_infestation + pest_growth * dt).min(1.0);
        let pest_factor = pests::photosynthesis_penalty(self.pest_infestation, pest_cfg);

        let leaf_area = self.total_leaf_area(plant);
        let lit_leaf_area = self.light_weighted_leaf_area(plant);
        let photosynthesis_area = self.photosynthesis_leaf_area(plant);
        let carbon_pool_before = self.carbon_pool;

        // Temperature response: photosynthesis and elongation both run on
        // enzyme-driven reactions with an optimum (`temperature_factor` — a
        // bell curve, not "hotter is always better"); respiration instead
        // keeps climbing with heat (the Q10 relationship — see
        // `climate::q10_factor`), which is also real: a plant's own
        // maintenance cost goes up in the heat even as its net income falls.
        let temp_factor = climate::temperature_factor(
            climate.temperature_c,
            plant.optimal_temperature_c,
            plant.temperature_tolerance_c,
        );
        let respiration_q10_factor = climate::q10_factor(
            climate.temperature_c,
            plant.respiration_reference_temperature_c,
            plant.respiration_q10,
        );
        // How hard this plant is stressed right now, for leaf senescence
        // (see `age_and_senesce_leaves`) — whichever of drought (via the
        // *effective* water factor, so root damage counts too), cold, or
        // pests is currently worse, not added together (a plant shedding
        // leaves over one bad condition isn't shedding them twice as fast
        // just because a second condition is also present).
        let stress_signal =
            leaf_stress_signal(effective_water_factor, climate.temperature_c, self.pest_infestation, plant);

        let humidity = Humidity { level: humidity_level };
        let stomatal_conductance = humidity.stomatal_conductance_factor(
            effective_water_factor,
            climate.temperature_c,
            humidity_cfg,
        );

        // Photosynthesis: gated by light AND stomatal conductance AND temperature AND
        // nutrient availability together (Liebig's law of the minimum,
        // extended to a second resource), further taxed by pest damage —
        // abundant light with closed (drought/root-damage-stressed)
        // stomata, a cold snap, nutrient starvation, or a bad infestation
        // each independently yield near-zero carbon income regardless of
        // how favorable everything else is.
        let photosynthesis = photosynthesis_area
            * sun.intensity
            * plant.light_use_efficiency
            * stomatal_conductance
            * temp_factor
            * nutrient_factor
            * pest_factor;
        // Maintenance respiration runs continuously, day or night, scaled
        // by living/metabolically-active tissue — leaf area, not raw stem
        // height. (Height used to contribute directly here, which created a
        // runaway feedback once branching could pause main-stem leaf
        // production: a tall, sparse-leaved stem would rack up respiration
        // cost from height alone with no matching income, making it *less*
        // able to ever afford a branch the taller it got. A bare cane's
        // upkeep cost is low; a leafy crown's is what actually matters.)
        let respiration = plant.respiration_rate * (1.0 + leaf_area) * respiration_q10_factor;
        self.carbon_pool = (self.carbon_pool + (photosynthesis - respiration) * dt)
            .clamp(0.0, plant.max_carbon_pool);

        let leafless =
            self.leaves.is_empty() && self.branches.iter().all(|b| b.leaves.is_empty());
        if leafless && self.carbon_pool <= 0.0 {
            self.starvation_timer += dt;
        } else {
            self.starvation_timer = 0.0;
        }

        // Transpiration ~ stomatal opening, which is shared with
        // photosynthesis above so atmospheric/root stress cannot increase
        // water loss while leaving CO2 uptake unconstrained. This depletes soil moisture and
        // (via cumulative_water_uptake below) drives stem thickening. Also
        // scaled by vapor-pressure deficit (hot *and* dry air pulls
        // dramatically more water out of leaves than either factor alone —
        // see `Humidity::vpd_factor`), the temperature-scaling this used to
        // deliberately omit before humidity existed as a mechanic.
        let vpd_factor = humidity.vpd_factor(climate.temperature_c, humidity_cfg);
        let uptake_rate =
            lit_leaf_area * sun.intensity * stomatal_conductance * plant.transpiration_coeff * vpd_factor;
        self.cumulative_water_uptake += uptake_rate * dt;
        soil.update(dt, sun.intensity, uptake_rate, soil_cfg);

        // Elongation: needs both banked carbon *and* turgor pressure
        // (water) — carbon alone can't push a wilted cell wall outward.
        // Shade avoidance: starved of light, the plant biases what carbon
        // it does spend toward racing upward rather than bulking up.
        // Dormancy and growth-shock both bias elongation down further, on
        // top of water/temperature — see their own computation above.
        // Carbon-*limited* rather than gated by a hard reserve threshold:
        // scales smoothly down as the pool runs low instead of an all-or-
        // nothing cliff, so a slow-photosynthesizing plant still creeps
        // upward rather than never growing at all.
        let etiolation_bias =
            1.0 + plant.shade_avoidance_strength * (1.0 - sun.intensity.min(1.0));
        let desired_elongation = if rooting {
            0.0
        } else {
            plant.base_elongation_rate
                * effective_water_factor
                * etiolation_bias
                * temp_factor
                * dormancy_factor
                * shock_factor
                * dt
        };
        let desired_cost = desired_elongation * plant.elongation_carbon_cost;
        let mut elongation = 0.0;
        let mut elongation_carbon_limited = false;
        if desired_cost > 0.0 {
            let affordable_fraction = (self.carbon_pool / desired_cost).min(1.0);
            elongation_carbon_limited = affordable_fraction < 1.0;
            elongation = desired_elongation * affordable_fraction;
            if self.realistic_scale {
                elongation *= realistic_scale_taper(self.height, plant.realistic_max_height);
            }
            self.height += elongation;
            self.carbon_pool -= desired_cost * affordable_fraction;
        }

        // Secondary thickening (pipe model): target radius set by the leaf
        // area actually being supplied; actual radius eases toward that
        // target at a rate proportional to water throughput, not just time
        // — a stem that's moved more water has built more xylem.
        let target_radius = plant.pipe_model_coeff * leaf_area.sqrt();
        let radius_before = self.stem_radius;
        if !rooting && target_radius > self.stem_radius {
            self.stem_radius += (target_radius - self.stem_radius)
                * (plant.thickening_rate_coeff * uptake_rate * dt).min(1.0);
        }
        let thickening_delta = self.stem_radius - radius_before;

        // Crown branching: its own, rarer, more expensive event than
        // routine leaf initiation (see below) — checked here, ahead of any
        // leaf spawning, so an eligible stem always gets first claim on
        // *this* tick's carbon for it rather than potentially losing out to
        // the fair leaf round-robin below.
        let branch_eligible =
            self.height >= plant.min_height_for_branching && self.branches.len() < plant.max_branches;
        let branch_spawned = !rooting && branch_eligible && self.carbon_pool > plant.new_branch_carbon_cost;
        if branch_spawned {
            self.carbon_pool -= plant.new_branch_carbon_cost;
            self.spawn_branch();
        }

        // Each branch is its own smaller growing point, sharing the same
        // carbon pool and environment as the main stem — see
        // `Plant::step_branch` (elongation/thickening/lean/droop only; leaf
        // initiation for every grower happens together, below).
        for i in 0..self.branches.len() {
            self.step_branch(
                i,
                dt,
                sun,
                effective_water_factor,
                temp_factor,
                stress_signal,
                dormancy_factor,
                shock_factor,
                plant,
            );
        }

        // Leaf initiation is height-gated (a plastochron — see module docs
        // and `PlantConfig::plastochron_height_interval`), fairly
        // interleaved across the main stem and every branch — see
        // `spawn_due_leaves_fairly` for why "the main stem, then each
        // branch in turn, each fully catching up its own backlog" would
        // otherwise let whichever grower ran first starve every later one.
        let leaf_spawned = !rooting && self.spawn_due_leaves_fairly(plant);

        // Phototropism: one-directional and cumulative (see module docs) —
        // it only grows toward the light, it never relaxes back at night.
        // A climbing habit (`PlantConfig::trellis_height`) suppresses this
        // entirely while still within reach of its support: a stem
        // twining/clinging to a moss pole is mechanically held straight,
        // not bending toward light, until it grows past the top of it.
        let lean_before = self.lean_angle;
        let leaning_freely = leans_freely(self.height, plant.trellis_height);
        if leaning_freely && !rooting {
            self.lean_angle =
                (self.lean_angle + plant.lean_rate * sun.intensity * dt).min(plant.max_lean_angle);
        }
        let lean_delta = self.lean_angle - lean_before;
        record_stem_segments(
            self.height,
            &mut self.height_at_last_stem_segment,
            self.lean_angle,
            &mut self.stem_segment_history,
            plant,
        );
        // `!leaning_freely` is exactly "still within reach of the support"
        // (see `leans_freely`) — real aerial roots only emerge while the
        // stem is actually pressed against something to root into.
        spawn_due_aerial_roots(
            self.height,
            &mut self.height_at_last_aerial_root,
            &mut self.aerial_roots,
            !leaning_freely,
            plant,
        );

        // Heliotropism/nyctinasty targets are shared across all leaves this
        // step (same sun, same moment) — computed once outside the loop.
        let helio_target = plant.helio_strength * (sun.azimuth * 2.0 - 1.0);
        let fold_target = 1.0 - sun.intensity.clamp(0.0, 1.0);
        // Wilting is driven by the *effective* water factor, not raw soil
        // moisture — see module docs: a rotted-root or badly pot-bound
        // plant can visibly wilt even though the soil itself reads wet.
        let droop_target = 1.0 - effective_water_factor;
        for leaf in &mut self.leaves {
            leaf.maturity = (leaf.maturity + plant.leaf_maturation_rate * dt).min(1.0);
            leaf.droop += (droop_target - leaf.droop) * (plant.droop_response_rate * dt).min(1.0);
            leaf.helio_angle +=
                (helio_target - leaf.helio_angle) * (plant.helio_response_rate * dt).min(1.0);
            leaf.fold += (fold_target - leaf.fold) * (plant.fold_response_rate * dt).min(1.0);
        }
        let shade_factors = self_shading_factors(&self.leaves, plant);
        age_and_senesce_leaves(&mut self.leaves, stress_signal, &shade_factors, dt, plant);

        // Wilting doesn't stop at the leaves: badly drought-stressed soft
        // tissue loses enough hydrostatic rigidity for the stem itself to
        // sag under its own weight — see module docs and
        // `stem_droop_target`. Eases more slowly than a leaf's own droop
        // (`stem_droop_response_rate` < `droop_response_rate`) and is
        // attenuated by how thick the stem already is.
        let stem_droop_target = stem_droop_target(effective_water_factor, self.stem_radius, plant);
        self.stem_droop +=
            (stem_droop_target - self.stem_droop) * (plant.stem_droop_response_rate * dt).min(1.0);

        // Blooming: purely cosmetic (see `bloom_intensity_target`'s doc
        // comment) — cycles open/rest rather than staying permanently in
        // bloom once mature, matching how real flowering plants flush and
        // rest rather than flower continuously forever.
        let mature_enough_to_bloom = !rooting && self.height >= plant.flowering_height_threshold;
        if mature_enough_to_bloom {
            self.bloom_cycle_position += dt;
        }
        let bloom_target = bloom_intensity_target(mature_enough_to_bloom, self.bloom_cycle_position, plant);
        self.bloom_intensity +=
            (bloom_target - self.bloom_intensity) * (plant.bloom_response_rate * dt).min(1.0);

        // Death: either total root loss or prolonged carbon starvation —
        // see `Stage::Dead`. Checked last, after every other mechanism this
        // tick has already run, so `last_decision` below still reflects
        // exactly what happened on the tick that killed it.
        // Root rot is checked first: a plant that's simultaneously leafless
        // *and* root-rotted (rare, but possible) died of the more specific,
        // more actionable-to-diagnose cause — "you overwatered it" beats a
        // generic "it starved."
        let death_cause = if self.root_health <= 0.0 {
            Some(DeathCause::RootRot)
        } else if self.starvation_timer >= plant.starvation_death_threshold {
            Some(DeathCause::Starvation)
        } else {
            None
        };
        if let Some(cause) = death_cause {
            self.stage = Stage::Dead;
            self.death_cause = Some(cause);
            // A dead plant needs to read as unmistakably dead regardless of
            // which failure actually killed it — a sudden total root loss,
            // for instance, could otherwise leave `step` freezing the plant
            // mid-frame still covered in green, undrooped leaves, which
            // would look like a rendering bug rather than the intended
            // terminal state. Force every leaf to its fully senesced/
            // wilted terminal appearance (reusing the exact same tint/
            // shrivel/droop machinery a naturally-aged leaf already uses,
            // rather than a separate "dead" rendering path) and collapse
            // the stem, instead of leaving it wherever it happened to be.
            for leaf in &mut self.leaves {
                leaf.senescence = 1.0;
                leaf.droop = 1.0;
            }
            for branch in &mut self.branches {
                for leaf in &mut branch.leaves {
                    leaf.senescence = 1.0;
                    leaf.droop = 1.0;
                }
                branch.droop = plant.stem_droop_max_angle;
            }
            self.stem_droop = plant.stem_droop_max_angle;
            self.bloom_intensity = 0.0;
        }

        if rooting && death_cause.is_none() {
            self.rooting_elapsed += dt;
            let duration = if self.realistic_scale {
                plant.rooting_duration_realistic
            } else {
                plant.rooting_duration
            };
            if self.rooting_elapsed >= duration {
                self.stage = Stage::Vegetative;
            }
        }

        self.last_decision = Some(Decision::Vegetative {
            sun_intensity: sun.intensity,
            sun_azimuth: sun.azimuth,
            water_factor,
            effective_water_factor,
            stomatal_conductance,
            root_health: self.root_health,
            pest_infestation: self.pest_infestation,
            dormancy_factor,
            leaf_area,
            lit_leaf_area,
            photosynthesis_area,
            carbon_pool_before,
            temperature_c: climate.temperature_c,
            temp_factor,
            respiration_q10_factor,
            photosynthesis,
            respiration,
            carbon_pool_after: self.carbon_pool,
            elongation,
            elongation_carbon_limited,
            thickening_delta,
            leaf_spawned,
            branch_spawned,
            lean_delta,
            helio_target,
            fold_target,
            droop_target,
            stem_droop_target,
        });
    }

    /// One branch's own growth this tick — a smaller-scale mirror of
    /// `step_vegetative` (elongation, thickening, leaf initiation, lean,
    /// per-leaf movement), but spending from the *shared* `self.carbon_pool`
    /// rather than one of its own (see module docs), and sized by its own
    /// leaf area, not the whole plant's.
    fn step_branch(
        &mut self,
        index: usize,
        dt: f64,
        sun: &SunState,
        water_factor: f64,
        temp_factor: f64,
        stress_signal: f64,
        dormancy_factor: f64,
        shock_factor: f64,
        plant: &PlantConfig,
    ) {
        let leaf_area = self.branches[index].leaf_area(plant);

        let etiolation_bias = 1.0 + plant.shade_avoidance_strength * (1.0 - sun.intensity.min(1.0));
        let desired_elongation = plant.base_elongation_rate
            * plant.branch_elongation_rate_factor
            * water_factor
            * etiolation_bias
            * temp_factor
            * dormancy_factor
            * shock_factor
            * dt;
        let desired_cost = desired_elongation * plant.elongation_carbon_cost;
        if desired_cost > 0.0 {
            let affordable_fraction = (self.carbon_pool / desired_cost).min(1.0);
            let mut branch_elongation = desired_elongation * affordable_fraction;
            if self.realistic_scale {
                branch_elongation *= realistic_scale_taper(self.branches[index].height, plant.realistic_max_height);
            }
            self.branches[index].height += branch_elongation;
            self.carbon_pool -= desired_cost * affordable_fraction;
        }

        // This branch's own transpiration is already folded into the main
        // stem's `light_weighted_leaf_area`-derived uptake/soil update —
        // this `uptake_rate` is only used locally, to drive how fast *this*
        // branch's own radius grows toward its pipe-model target. Weighted
        // by this branch's *own* height (not the main stem's), so a low
        // branch keeps thickening normally even once the main stem's tip
        // has grown past the window's light.
        let branch_light_factor =
            height_light_factor(self.branches[index].attach_height + self.branches[index].height, plant);
        let uptake_rate = leaf_area * sun.intensity * water_factor * branch_light_factor * plant.transpiration_coeff;
        let target_radius = plant.pipe_model_coeff * leaf_area.sqrt();
        if target_radius > self.branches[index].stem_radius {
            self.branches[index].stem_radius += (target_radius - self.branches[index].stem_radius)
                * (plant.thickening_rate_coeff * uptake_rate * dt).min(1.0);
        }

        // Same climbing-habit suppression as the main stem (see
        // `step_vegetative`), gated on this branch's own absolute height
        // along the support (same `attach_height + height` basis
        // `branch_light_factor` above already uses).
        let branch_leaning_freely = leans_freely(
            self.branches[index].attach_height + self.branches[index].height,
            plant.trellis_height,
        );
        if branch_leaning_freely {
            self.branches[index].lean_angle = (self.branches[index].lean_angle
                + plant.lean_rate * sun.intensity * dt)
                .min(plant.max_lean_angle);
        }
        let branch = &mut self.branches[index];
        record_stem_segments(
            branch.height,
            &mut branch.height_at_last_stem_segment,
            branch.lean_angle,
            &mut branch.segment_history,
            plant,
        );

        let helio_target = plant.helio_strength * (sun.azimuth * 2.0 - 1.0);
        let fold_target = 1.0 - sun.intensity.clamp(0.0, 1.0);
        let droop_target = 1.0 - water_factor;
        for leaf in &mut self.branches[index].leaves {
            leaf.maturity = (leaf.maturity + plant.leaf_maturation_rate * dt).min(1.0);
            leaf.droop += (droop_target - leaf.droop) * (plant.droop_response_rate * dt).min(1.0);
            leaf.helio_angle +=
                (helio_target - leaf.helio_angle) * (plant.helio_response_rate * dt).min(1.0);
            leaf.fold += (fold_target - leaf.fold) * (plant.fold_response_rate * dt).min(1.0);
        }
        let branch_shade_factors = self_shading_factors(&self.branches[index].leaves, plant);
        age_and_senesce_leaves(
            &mut self.branches[index].leaves,
            stress_signal,
            &branch_shade_factors,
            dt,
            plant,
        );

        // Same mechanism as the main stem's own droop (see
        // `stem_droop_target`), local to this branch's own radius.
        let branch_stem_droop_target =
            stem_droop_target(water_factor, self.branches[index].stem_radius, plant);
        self.branches[index].droop += (branch_stem_droop_target - self.branches[index].droop)
            * (plant.stem_droop_response_rate * dt).min(1.0);
    }

    /// 1.0 while `height` is still within this plant's current pot capacity
    /// (`PlantConfig::initial_pot_capacity * pot_capacity_multiplier`),
    /// ramping down to `pot_bound_floor` over the next `pot_bound_stress_
    /// range` of height past that — a real container caps how large a root
    /// system can get before it needs a bigger pot, a gradual squeeze, not
    /// a hard wall the instant the ceiling is reached. See `repot`.
    fn pot_bound_factor(&self, plant: &PlantConfig) -> f64 {
        let capacity = plant.initial_pot_capacity * self.pot_capacity_multiplier;
        if self.height <= capacity {
            1.0
        } else {
            let excess = self.height - capacity;
            (1.0 - excess / plant.pot_bound_stress_range.max(1e-9)).clamp(plant.pot_bound_floor, 1.0)
        }
    }

    /// Shared mechanics for cutting the main stem back to `new_height`,
    /// regardless of how that height was chosen — see `prune_main_stem`
    /// (a fixed fraction of current height) and `cut_main_stem_at` (an
    /// exact height, the click-to-prune tool's own mechanic). Removing the
    /// growing tip removes the bud suppressing the lateral buds below it,
    /// releasing several at once (real apical-dominance release) rather
    /// than waiting for the plant to reach branching height/carbon on its
    /// own — the same mechanism this module's crown branching already
    /// models automatically, just player-triggered.
    fn cut_main_stem_to(&mut self, new_height: f64, plant: &PlantConfig) {
        self.leaves.retain(|leaf| leaf.attach_height <= new_height);
        self.aerial_roots.retain(|root| root.attach_height <= new_height);
        // A branch attached above the cut has nothing left connecting it to
        // the plant — real pruning takes whatever was growing off the
        // removed section with it, not just the main stem's own leaves.
        self.branches.retain(|branch| branch.attach_height <= new_height);
        self.height = new_height;
        self.height_at_last_leaf = self.height_at_last_leaf.min(new_height);
        self.height_at_last_aerial_root = self.height_at_last_aerial_root.min(new_height);
        let kept_segments =
            (new_height / plant.stem_segment_height_interval).floor().max(0.0) as usize;
        self.stem_segment_history.truncate(kept_segments);
        self.height_at_last_stem_segment = kept_segments as f64 * plant.stem_segment_height_interval;

        // Real apical-dominance release frees several co-dominant buds
        // together, not just one — capped by however much room remains
        // under `max_branches`.
        let room_for_branches = plant.max_branches.saturating_sub(self.branches.len());
        let release_count = plant.prune_branch_release_count.min(room_for_branches);
        for _ in 0..release_count {
            self.spawn_branch();
        }

        self.growth_shock = (self.growth_shock + plant.prune_shock_amount).min(1.0);
    }

    /// Cuts the main stem back by a fixed fraction (`PlantConfig::
    /// prune_height_fraction`) — the "Prune stem" button's own mechanic.
    /// Returns whether pruning actually happened (it's a no-op below
    /// `PlantConfig::prune_min_height`, or once the plant is `Stage::Dead`).
    pub fn prune_main_stem(&mut self, plant: &PlantConfig) -> bool {
        if self.stage != Stage::Vegetative || self.height < plant.prune_min_height {
            return false;
        }
        let new_height = self.height * (1.0 - plant.prune_height_fraction);
        self.cut_main_stem_to(new_height, plant);
        true
    }

    /// Cuts the main stem at an exact height instead of a fixed fraction —
    /// the click-to-prune tool's own mechanic (see `render::mod`'s stem
    /// pick pass), letting a player choose precisely where to cut rather
    /// than always losing the same proportion. Shares every other
    /// consequence with `prune_main_stem` (shedding leaves/aerial roots
    /// above the cut, branch release, growth shock) — a cut is a cut
    /// regardless of how its height was chosen. Returns whether it
    /// actually happened: a no-op below `prune_min_height`, at or above
    /// the plant's current height (clicking the very tip isn't a cut), or
    /// once dead.
    pub fn cut_main_stem_at(&mut self, height: f64, plant: &PlantConfig) -> bool {
        if self.stage != Stage::Vegetative || self.height < plant.prune_min_height {
            return false;
        }
        let new_height = height.clamp(0.0, self.height);
        if new_height >= self.height {
            return false;
        }
        self.cut_main_stem_to(new_height, plant);
        true
    }

    /// Shared mechanics for cutting one branch back to `new_height` — see
    /// `prune_branch` (a fixed fraction) and `cut_branch_at` (an exact
    /// height). No further branch release, unlike the main stem: a
    /// branch's own laterals aren't modeled.
    fn cut_branch_to(&mut self, index: usize, new_height: f64, plant: &PlantConfig) {
        let branch = &mut self.branches[index];
        branch.leaves.retain(|leaf| leaf.attach_height <= new_height);
        branch.height = new_height;
        branch.height_at_last_leaf = branch.height_at_last_leaf.min(new_height);
        let kept_segments =
            (new_height / plant.stem_segment_height_interval).floor().max(0.0) as usize;
        branch.segment_history.truncate(kept_segments);
        branch.height_at_last_stem_segment = kept_segments as f64 * plant.stem_segment_height_interval;

        self.growth_shock = (self.growth_shock + plant.prune_shock_amount).min(1.0);
    }

    /// Cuts a single branch back by a fixed fraction — the "Prune branch"
    /// button's own mechanic. Returns whether pruning happened.
    pub fn prune_branch(&mut self, index: usize, plant: &PlantConfig) -> bool {
        if self.stage != Stage::Vegetative {
            return false;
        }
        let Some(branch) = self.branches.get(index) else {
            return false;
        };
        if branch.height < plant.prune_min_height {
            return false;
        }
        let new_height = branch.height * (1.0 - plant.prune_height_fraction);
        self.cut_branch_to(index, new_height, plant);
        true
    }

    /// Cuts one branch at an exact height instead of a fixed fraction —
    /// the click-to-prune tool's own mechanic, same reasoning as
    /// `cut_main_stem_at`. Returns whether it actually happened.
    pub fn cut_branch_at(&mut self, index: usize, height: f64, plant: &PlantConfig) -> bool {
        if self.stage != Stage::Vegetative {
            return false;
        }
        let Some(branch) = self.branches.get(index) else {
            return false;
        };
        let new_height = height.clamp(0.0, branch.height);
        if new_height >= branch.height {
            return false;
        }
        self.cut_branch_to(index, new_height, plant);
        true
    }

    /// Removes exactly one leaf — `slot` is a position in the same flat
    /// ordering the renderer fills its leaf pool with (main stem's own
    /// leaves first, then each branch's own leaves in branch order — see
    /// `render::mod`'s `leaf_slot` and `scene::leaf_depth`, which this
    /// mirrors so a player clicking on whatever they see on screen removes
    /// that exact leaf). Deliberately lighter than `prune_main_stem`/
    /// `prune_branch`: losing one leaf's photosynthetic area doesn't reset
    /// any height/segment history or apply `growth_shock` the way cutting
    /// back a whole stem does — it's closer to routine deadheading than a
    /// real setback. Returns whether a leaf actually existed at that slot.
    pub fn prune_leaf(&mut self, slot: usize) -> bool {
        if self.stage != Stage::Vegetative {
            return false;
        }
        if slot < self.leaves.len() {
            self.leaves.remove(slot);
            return true;
        }
        let mut remaining = slot - self.leaves.len();
        for branch in &mut self.branches {
            if remaining < branch.leaves.len() {
                branch.leaves.remove(remaining);
                return true;
            }
            remaining -= branch.leaves.len();
        }
        false
    }

    /// Moves the plant into a bigger pot: raises the pot-bound ceiling
    /// (`pot_capacity_multiplier`), restores at least `repot_root_health_
    /// restore` of root health (a real repot lets a grower trim off rotted
    /// roots and refresh the soil, so it's a genuine partial fix for root
    /// rot too, not just a size-cap reset), and clears any in-progress
    /// waterlog timer. Costs a temporary setback (`growth_shock`), the same
    /// real tradeoff pruning has — repotting too early is pure downside,
    /// repotting too late leaves the plant stunted, so there's a genuine
    /// judgment call about when it's worth it. Returns whether it happened
    /// (a no-op once `Stage::Dead`).
    pub fn repot(&mut self, plant: &PlantConfig) -> bool {
        if self.stage != Stage::Vegetative {
            return false;
        }
        self.pot_capacity_multiplier *= plant.repot_capacity_multiplier;
        self.growth_shock = (self.growth_shock + plant.repot_shock_amount).min(1.0);
        self.root_health = self.root_health.max(plant.repot_root_health_restore);
        self.waterlogged_duration = 0.0;
        true
    }

    /// Cane cuttings root from bare internodes; vine cuttings need a live
    /// node/leaf; a basal rosette (Spathiphyllum) has no stem to cut at all
    /// and propagates only by division.
    pub fn is_propagatable(&self, plant: &PlantConfig) -> bool {
        if self.stage != Stage::Vegetative || self.height < plant.cutting_min_height {
            return false;
        }
        if self.root_health < plant.cutting_min_root_health {
            return false;
        }
        match plant.growth_habit {
            GrowthHabit::BasalRosette => false,
            GrowthHabit::Vine => !self.leaves.is_empty(),
            GrowthHabit::UprightCane => true,
        }
    }

    pub fn take_cutting(&mut self, plant: &PlantConfig) -> bool {
        if !self.is_propagatable(plant) {
            return false;
        }
        let new_height = self.height * (1.0 - plant.cutting_cost_height_fraction);
        self.cut_main_stem_to(new_height, plant);
        true
    }

    /// `vigor` (0.0..=1.0, the parent's `root_health` at the moment the
    /// cutting was taken) scales starting reserves — a cutting off a
    /// struggling parent starts weaker.
    pub fn from_cutting(plant: &PlantConfig, vigor: f64) -> Plant {
        let vigor = vigor.clamp(0.0, 1.0);
        let mut fresh = Plant {
            stage: Stage::Rooting,
            height: plant.cutting_start_height,
            carbon_pool: plant.cutting_start_carbon * vigor,
            max_height_reached: plant.cutting_start_height,
            ..Plant::default()
        };
        let starter_leaves = ((plant.cutting_start_leaves as f64) * vigor).round().max(1.0) as usize;
        for _ in 0..starter_leaves {
            fresh.spawn_leaf();
        }
        fresh.max_leaves_at_once = fresh.leaves.len() as u32;
        fresh
    }

    pub fn rooting_progress(&self, plant: &PlantConfig) -> f64 {
        if self.stage != Stage::Rooting {
            return 1.0;
        }
        let duration = if self.realistic_scale {
            plant.rooting_duration_realistic
        } else {
            plant.rooting_duration
        };
        (self.rooting_elapsed / duration.max(1e-9)).clamp(0.0, 1.0)
    }

    pub fn is_dividable(&self, plant: &PlantConfig) -> bool {
        plant.growth_habit == GrowthHabit::BasalRosette
            && self.stage == Stage::Vegetative
            && self.leaves.len() >= plant.division_min_leaves
            && self.root_health >= plant.cutting_min_root_health
    }

    pub fn divide(&mut self, plant: &PlantConfig) -> Option<Plant> {
        if !self.is_dividable(plant) {
            return None;
        }
        let split = self.leaves.len() / 2;
        let offshoot_leaves = self.leaves.split_off(split);
        let offshoot_leaf_count = offshoot_leaves.len() as u32;
        let offshoot_carbon = self.carbon_pool * 0.5;
        self.carbon_pool -= offshoot_carbon;
        self.height *= 1.0 - plant.cutting_cost_height_fraction;
        self.growth_shock = (self.growth_shock + plant.prune_shock_amount).min(1.0);
        let mut offshoot = Plant {
            stage: Stage::Vegetative,
            height: self.height,
            carbon_pool: offshoot_carbon,
            root_health: self.root_health,
            growth_shock: plant.prune_shock_amount,
            leaves: offshoot_leaves,
            ..Plant::default()
        };
        offshoot.max_height_reached = offshoot.height;
        offshoot.max_leaves_at_once = offshoot_leaf_count;
        offshoot.leaves_produced_total = offshoot_leaf_count;
        Some(offshoot)
    }

    /// Treats a pest infestation (e.g. wiping leaves down / neem oil),
    /// knocking it down immediately by `PestConfig::treatment_reduction` —
    /// the active counterpart to letting humidity alone suppress future
    /// growth (see `pests::pest_growth_rate`).
    pub fn treat_pests(&mut self, pest_cfg: &PestConfig) {
        self.pest_infestation = (self.pest_infestation - pest_cfg.treatment_reduction).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::sun::sun_state;

    fn config() -> GrowthConfig {
        GrowthConfig::default()
    }

    fn noon(config: &GrowthConfig) -> SunState {
        sun_state(0.5, &config.sun)
    }

    fn midnight(config: &GrowthConfig) -> SunState {
        sun_state(0.0, &config.sun)
    }

    /// A climate at exactly `PlantConfig::dracaena()`'s own optimum — so
    /// `temperature_factor`/`q10_factor` both evaluate to their neutral
    /// (1.0) value and every test written before temperature existed keeps
    /// exercising whatever *other* mechanism it was actually about, without
    /// also incidentally becoming a temperature-gating test.
    fn neutral_climate() -> ClimateState {
        ClimateState {
            temperature_c: PlantConfig::dracaena().optimal_temperature_c,
        }
    }

    fn mature_leaf(side: Side) -> Leaf {
        Leaf {
            attach_height: 0.0,
            side,
            maturity: 1.0,
            droop: 0.0,
            helio_angle: 0.0,
            fold: 0.0,
            age: 0.0,
            senescence: 0.0,
        }
    }

    #[test]
    fn realistic_scale_taper_is_full_strength_well_below_the_cap_and_zero_at_or_past_it() {
        assert_eq!(realistic_scale_taper(0.0, 5.0), 1.0);
        assert!(realistic_scale_taper(2.5, 5.0) < 1.0);
        assert!(realistic_scale_taper(2.5, 5.0) > 0.0);
        assert_eq!(realistic_scale_taper(5.0, 5.0), 0.0);
        assert_eq!(realistic_scale_taper(6.0, 5.0), 0.0, "shouldn't go negative past the cap");
    }

    #[test]
    fn realistic_scale_plateaus_near_the_configured_cap_over_a_long_session() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            realistic_scale: true,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..200_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.moisture = 1.0;
        }
        let cap = config.plant.realistic_max_height;
        assert!(
            plant.height <= cap + 1e-6,
            "expected height to stay at/under the realistic cap {cap}, got {}",
            plant.height
        );
        assert!(
            plant.height > cap * 0.8,
            "expected height to have actually approached the cap {cap}, not stalled early, got {}",
            plant.height
        );
    }

    #[test]
    fn unbounded_scale_keeps_growing_well_past_where_realistic_scale_would_have_capped() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            realistic_scale: false,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..200_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.moisture = 1.0;
        }
        assert!(
            plant.height > config.plant.realistic_max_height,
            "expected today's default unbounded growth to exceed the realistic cap {} over a long session, got {}",
            config.plant.realistic_max_height,
            plant.height
        );
    }

    #[test]
    fn seed_does_not_germinate_in_dry_soil() {
        let config = config();
        let mut plant = Plant::new();
        let mut soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..10_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert_eq!(plant.stage, Stage::Seed);
    }

    #[test]
    fn seed_does_not_germinate_in_cold_soil_even_with_plenty_of_water() {
        let config = config();
        let mut plant = Plant::new();
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        let cold = ClimateState {
            temperature_c: config.plant.germination_min_temperature_c - 1.0,
        };
        for _ in 0..10_000 {
            plant.step(1.0, &sun, &cold, &mut soil, 1.0, &config);
        }
        assert_eq!(
            plant.stage,
            Stage::Seed,
            "expected cold soil to block germination even with ample water"
        );
    }

    #[test]
    fn seed_germinates_in_moist_soil_and_grows_a_first_leaf() {
        let config = config();
        let mut plant = Plant::new();
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        // Germination + reaching the sprout height threshold both happen
        // within single-digit ticks at full water/light — 500 is generous
        // headroom for that, while staying well short of
        // `leaf_mature_lifespan` (this test is about the *first* leaf
        // appearing, not surviving long enough to be shed of old age, which
        // is covered separately).
        for _ in 0..500 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert_eq!(plant.stage, Stage::Vegetative);
        assert!(!plant.leaves.is_empty());
        assert!(plant.height > 0.0);
    }

    #[test]
    fn no_meaningful_growth_at_night_despite_full_water() {
        let config = config();
        let mut day_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut night_plant = day_plant.clone();
        let mut day_soil = Soil { moisture: 1.0, ..Default::default() };
        let mut night_soil = Soil { moisture: 1.0, ..Default::default() };
        let (day_sun, night_sun) = (noon(&config), midnight(&config));

        for _ in 0..500 {
            day_plant.step(1.0, &day_sun, &neutral_climate(), &mut day_soil, 1.0, &config);
            night_plant.step(1.0, &night_sun, &neutral_climate(), &mut night_soil, 1.0, &config);
        }

        assert!(day_plant.height > night_plant.height);
        // Not `carbon_pool` directly — carbon banks up and then gets spent
        // in a lump the instant a new leaf is affordable, so it's a
        // sawtooth over time, not monotonically increasing; leaf count is
        // the robust signal that carbon was actually being earned.
        assert!(day_plant.leaves.len() > night_plant.leaves.len());
    }

    #[test]
    fn growth_stalls_in_dry_soil_even_with_full_light() {
        let config = config();
        let mut wet_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut dry_plant = wet_plant.clone();
        let mut wet_soil = Soil { moisture: 1.0, ..Default::default() };
        let mut dry_soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..2000 {
            wet_plant.step(1.0, &sun, &neutral_climate(), &mut wet_soil, 1.0, &config);
            dry_plant.step(1.0, &sun, &neutral_climate(), &mut dry_soil, 1.0, &config);
        }

        assert!(wet_plant.height > dry_plant.height);
        assert!(wet_plant.stem_radius > dry_plant.stem_radius);
    }

    #[test]
    fn stem_thickens_faster_with_more_supported_leaf_area() {
        let config = config();
        let base = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut few_leaves = base.clone();
        let mut many_leaves = base.clone();
        for i in 0..6 {
            let side = if i % 2 == 0 { Side::Left } else { Side::Right };
            many_leaves.leaves.push(mature_leaf(side));
        }
        let mut soil_a = Soil { moisture: 1.0, ..Default::default() };
        let mut soil_b = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..3000 {
            few_leaves.step(1.0, &sun, &neutral_climate(), &mut soil_a, 1.0, &config);
            many_leaves.step(1.0, &sun, &neutral_climate(), &mut soil_b, 1.0, &config);
        }

        assert!(
            many_leaves.stem_radius > few_leaves.stem_radius,
            "a stem supplying more leaf area should thicken more (pipe model): {} vs {}",
            many_leaves.stem_radius,
            few_leaves.stem_radius
        );
        assert!(many_leaves.cumulative_water_uptake > few_leaves.cumulative_water_uptake);
    }

    #[test]
    fn low_light_produces_etiolated_growth_relative_to_high_light() {
        let config = config();
        let start = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut bright_plant = start.clone();
        let mut dim_plant = start.clone();
        let mut bright_soil = Soil { moisture: 1.0, ..Default::default() };
        let mut dim_soil = Soil { moisture: 1.0, ..Default::default() };

        let bright_sun = SunState {
            elevation: 1.0,
            azimuth: 0.5,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
        };
        // Meaningfully dimmer than "bright" but still above this plant's
        // photosynthetic compensation point (where photosynthesis just
        // balances respiration) — below that point light is the whole
        // story (no growth at all, tested separately) and there's no
        // shape/allocation trade-off left to observe.
        let dim_sun = SunState {
            elevation: 0.4,
            azimuth: 0.5,
            intensity: 0.4,
            color: [1.0, 0.85, 0.65],
        };

        for _ in 0..3000 {
            bright_plant.step(1.0, &bright_sun, &neutral_climate(), &mut bright_soil, 1.0, &config);
            dim_plant.step(1.0, &dim_sun, &neutral_climate(), &mut dim_soil, 1.0, &config);
        }

        // Etiolation: the dim-light plant should be skinnier relative to
        // its own height than the bright-light plant, even if its absolute
        // height is smaller.
        let bright_ratio = bright_plant.stem_radius / bright_plant.height.max(1e-9);
        let dim_ratio = dim_plant.stem_radius / dim_plant.height.max(1e-9);
        assert!(
            dim_ratio < bright_ratio,
            "low light should bias toward thin/tall growth: dim ratio {dim_ratio} vs bright ratio {bright_ratio}"
        );
    }

    #[test]
    fn leaves_droop_under_drought_and_recover_after_watering() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..200 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        let wilted_droop = plant.leaves[0].droop;
        assert!(wilted_droop > 0.3, "expected noticeable wilting, got {wilted_droop}");

        soil.water(1.0);
        for _ in 0..200 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(plant.leaves[0].droop < wilted_droop);
    }

    #[test]
    fn a_young_unshaded_leaf_does_not_senesce_from_age_before_its_mature_lifespan() {
        // Exercises `age_and_senesce_leaves` directly (not a full `plant.step`
        // loop): a real plant this well-lit and well-watered would keep
        // growing *new* leaves above this one over a duration this long,
        // which would legitimately shade it and could trigger *early*,
        // shade-driven senescence (see `a_heavily_self_shaded_mature_leaf_
        // senesces_faster_than_an_unshaded_one`) — a real, deliberate,
        // age-independent mechanism, not a bug. A fixed `shade_factors: [1.0]`
        // isolates this test to just the age-gate half of the mechanism.
        let config = config();
        let mut leaves = vec![mature_leaf(Side::Left)];
        age_and_senesce_leaves(
            &mut leaves,
            0.0,
            &[1.0],
            config.plant.leaf_mature_lifespan - 100.0,
            &config.plant,
        );
        assert_eq!(
            leaves[0].senescence, 0.0,
            "expected no senescence before the mature lifespan elapses"
        );
    }

    #[test]
    fn an_unshaded_leaf_past_its_lifespan_still_senesces_under_zero_external_stress() {
        // Regression: age-driven senescence used to require nonzero
        // drought/cold/pest stress even past the lifespan, so a leaf in
        // perfect conditions lived forever.
        let config = config();
        let old_leaf = Leaf { age: config.plant.leaf_mature_lifespan + 1.0, ..mature_leaf(Side::Left) };
        let mut leaves = vec![old_leaf];
        age_and_senesce_leaves(&mut leaves, 0.0, &[1.0], 1.0, &config.plant);
        assert!(!leaves.is_empty() && leaves[0].senescence > 0.0, "expected baseline aging even with zero stress");
    }

    #[test]
    fn an_unreplaced_leaf_senesces_and_is_shed_once_far_enough_past_its_mature_lifespan() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        // Dry from the start (no water, ever) — prevents any *new* leaf
        // from ever growing in (elongation needs water), isolating this
        // test to "does the one original leaf age out and get shed," not
        // muddied by newer leaves also being present by the end. Drought
        // stress also accelerates the senescence this is testing, so it
        // doesn't need to wait through the full unstressed timeline either.
        let mut soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..8500 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(
            plant.leaves.is_empty(),
            "expected the original leaf to have senesced and been shed by now, got senescence {:?}",
            plant.leaves.iter().map(|l| l.senescence).collect::<Vec<_>>()
        );
    }

    #[test]
    fn drought_stress_accelerates_leaf_senescence_relative_to_well_watered() {
        let config = config();
        let aged_leaf = Leaf {
            age: config.plant.leaf_mature_lifespan + 1.0,
            ..mature_leaf(Side::Left)
        };
        let mut wet_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![aged_leaf],
            ..Plant::new()
        };
        let mut dry_plant = wet_plant.clone();
        let mut wet_soil = Soil { moisture: 1.0, ..Default::default() };
        let mut dry_soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..200 {
            wet_plant.step(1.0, &sun, &neutral_climate(), &mut wet_soil, 1.0, &config);
            dry_plant.step(1.0, &sun, &neutral_climate(), &mut dry_soil, 1.0, &config);
        }
        assert!(
            dry_plant.leaves[0].senescence > wet_plant.leaves[0].senescence,
            "expected drought to age a leaf faster: dry {} vs wet {}",
            dry_plant.leaves[0].senescence,
            wet_plant.leaves[0].senescence
        );
    }

    #[test]
    fn self_shading_factors_favor_the_newest_leaf_and_penalize_older_overtopped_ones() {
        let config = config();
        // Ascending attach_height, oldest (lowest) first — matches how
        // `Plant::leaves`/`Branch::leaves` are always populated in practice
        // (see `self_shading_factors`'s doc comment).
        let leaves = vec![
            Leaf { attach_height: 0.0, ..mature_leaf(Side::Left) },
            Leaf { attach_height: 1.0, ..mature_leaf(Side::Right) },
            Leaf { attach_height: 2.0, ..mature_leaf(Side::Left) },
        ];
        let factors = self_shading_factors(&leaves, &config.plant);
        assert_eq!(factors[2], 1.0, "the newest/topmost leaf has nothing of this plant's own shading it");
        assert!(
            factors[1] < factors[2],
            "the middle leaf should be shaded by the one leaf above it: {factors:?}"
        );
        assert!(
            factors[0] < factors[1],
            "the oldest/lowest leaf should be shaded by both leaves above it: {factors:?}"
        );
    }

    #[test]
    fn additional_leaf_area_yields_diminishing_photosynthesis_once_self_shaded() {
        let config = config();
        let leaf_at = |h: f64| Leaf { attach_height: h, ..mature_leaf(Side::Left) };

        let sparse = Plant {
            stage: Stage::Vegetative,
            leaves: vec![leaf_at(0.0), leaf_at(1.0)],
            ..Plant::new()
        };
        let dense = Plant {
            stage: Stage::Vegetative,
            leaves: vec![
                leaf_at(0.0),
                leaf_at(1.0),
                leaf_at(2.0),
                leaf_at(3.0),
                leaf_at(4.0),
                leaf_at(5.0),
            ],
            ..Plant::new()
        };

        let sparse_area = sparse.photosynthesis_leaf_area(&config.plant);
        let dense_area = dense.photosynthesis_leaf_area(&config.plant);
        let unshaded_dense_area = dense.true_leaf_area(&config.plant);

        assert!(
            dense_area > sparse_area,
            "more leaves should still capture at least somewhat more light: dense {dense_area} vs sparse {sparse_area}"
        );
        assert!(
            dense_area < unshaded_dense_area,
            "self-shading should discount the dense canopy below its raw (unshaded) area: {dense_area} vs {unshaded_dense_area}"
        );
        // The 4 extra leaves (2 -> 6) should contribute much less than
        // proportionally, precisely the diminishing-returns effect that
        // caps how much a plant benefits from piling on ever more leaves.
        let marginal_per_extra_leaf = (dense_area - sparse_area) / 4.0;
        let first_leaves_per_leaf = sparse_area / 2.0;
        assert!(
            marginal_per_extra_leaf < first_leaves_per_leaf,
            "later leaves should contribute less each than the first ones: marginal {marginal_per_extra_leaf} vs first {first_leaves_per_leaf}"
        );
    }

    #[test]
    fn a_heavily_self_shaded_mature_leaf_senesces_faster_than_an_unshaded_one() {
        let config = config();
        // Both leaves are equally old and equally well-watered/comfortable
        // (so drought/cold stress is identical, isolating this test to the
        // self-shading term) — the only difference is how much of this
        // same plant's own leaf area sits above each one.
        let aged = |h: f64| Leaf {
            attach_height: h,
            age: config.plant.leaf_mature_lifespan + 1.0,
            ..mature_leaf(Side::Left)
        };
        let mut buried_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![
                aged(0.0),
                aged(1.0),
                aged(2.0),
                aged(3.0),
                aged(4.0),
                aged(5.0),
            ],
            ..Plant::new()
        };
        let mut lone_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![aged(0.0)],
            ..Plant::new()
        };
        let mut soil_a = Soil { moisture: 1.0, ..Default::default() };
        let mut soil_b = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..50 {
            buried_plant.step(1.0, &sun, &neutral_climate(), &mut soil_a, 1.0, &config);
            lone_plant.step(1.0, &sun, &neutral_climate(), &mut soil_b, 1.0, &config);
        }
        assert!(
            buried_plant.leaves[0].senescence > lone_plant.leaves[0].senescence,
            "expected the heavily overtopped leaf to senesce faster: buried {} vs lone {}",
            buried_plant.leaves[0].senescence,
            lone_plant.leaves[0].senescence
        );
    }

    #[test]
    fn cold_stress_accelerates_leaf_senescence_relative_to_a_comfortable_temperature() {
        let config = config();
        let aged_leaf = Leaf {
            age: config.plant.leaf_mature_lifespan + 1.0,
            ..mature_leaf(Side::Left)
        };
        let mut warm_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![aged_leaf],
            ..Plant::new()
        };
        let mut cold_plant = warm_plant.clone();
        let mut warm_soil = Soil { moisture: 1.0, ..Default::default() };
        let mut cold_soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        let cold_climate = ClimateState {
            temperature_c: config.plant.cold_stress_threshold_c - config.plant.temperature_tolerance_c,
        };
        for _ in 0..200 {
            warm_plant.step(1.0, &sun, &neutral_climate(), &mut warm_soil, 1.0, &config);
            cold_plant.step(1.0, &sun, &cold_climate, &mut cold_soil, 1.0, &config);
        }
        assert!(
            cold_plant.leaves[0].senescence > warm_plant.leaves[0].senescence,
            "expected cold stress to age a leaf faster: cold {} vs warm {}",
            cold_plant.leaves[0].senescence,
            warm_plant.leaves[0].senescence
        );
    }

    #[test]
    fn height_light_factor_is_full_within_the_window_zone() {
        let config = config();
        assert_eq!(height_light_factor(0.0, &config.plant), 1.0);
        assert_eq!(height_light_factor(config.plant.window_light_zone_height, &config.plant), 1.0);
    }

    #[test]
    fn height_light_factor_reaches_the_ambient_floor_well_past_the_falloff_range() {
        let config = config();
        let far_above =
            config.plant.window_light_zone_height + config.plant.window_light_falloff_range * 10.0;
        assert!(
            (height_light_factor(far_above, &config.plant) - config.plant.ambient_light_floor).abs() < 1e-9
        );
    }

    #[test]
    fn height_light_factor_is_worse_further_past_the_zone() {
        let config = config();
        let a_bit_past = config.plant.window_light_zone_height + config.plant.window_light_falloff_range * 0.25;
        let further_past = config.plant.window_light_zone_height + config.plant.window_light_falloff_range * 0.75;
        let factor_a = height_light_factor(a_bit_past, &config.plant);
        let factor_b = height_light_factor(further_past, &config.plant);
        assert!(factor_b < factor_a, "expected further past the zone to fare worse: {factor_b} vs {factor_a}");
        assert!(factor_a < 1.0 && factor_b > config.plant.ambient_light_floor);
    }

    #[test]
    fn a_plant_that_has_outgrown_the_light_zone_photosynthesizes_less_than_an_identical_one_within_it() {
        let config = config();
        let base = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut within_zone = base.clone();
        within_zone.height = config.plant.window_light_zone_height * 0.5;
        let mut past_zone = base.clone();
        past_zone.height = config.plant.window_light_zone_height + config.plant.window_light_falloff_range * 0.5;

        let mut soil_a = Soil { moisture: 1.0, ..Default::default() };
        let mut soil_b = Soil { moisture: 1.0, ..Default::default() };
        within_zone.step(1.0, &noon(&config), &neutral_climate(), &mut soil_a, 1.0, &config);
        past_zone.step(1.0, &noon(&config), &neutral_climate(), &mut soil_b, 1.0, &config);

        let photosynthesis = |p: &Plant| match p.last_decision {
            Some(Decision::Vegetative { photosynthesis, .. }) => photosynthesis,
            other => panic!("expected a Vegetative decision, got {other:?}"),
        };
        assert!(
            photosynthesis(&past_zone) < photosynthesis(&within_zone),
            "expected less photosynthesis once past the window's light zone: {} vs {}",
            photosynthesis(&past_zone),
            photosynthesis(&within_zone)
        );
    }

    #[test]
    fn a_low_branch_stays_fully_lit_even_once_the_main_stem_has_outgrown_the_window() {
        let config = config();
        // The main stem's own height is *way* past the light zone, but this
        // branch attached low and hasn't grown much — it should still be
        // fully lit on its own terms.
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.window_light_zone_height * 5.0,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut low_branch = Branch::new(0.1, Side::Left);
        low_branch.leaves.push(mature_leaf(Side::Left));
        plant.branches.push(low_branch);

        assert_eq!(
            height_light_factor(plant.branches[0].attach_height + plant.branches[0].height, &config.plant),
            1.0,
            "expected the low branch to be unaffected by the main stem's own height"
        );
    }

    #[test]
    fn stem_droops_under_drought_and_recovers_after_watering() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            stem_radius: 0.01, // thin, young stem — should droop noticeably
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..200 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        let wilted_droop = plant.stem_droop;
        assert!(wilted_droop > 0.0, "expected the stem itself to sag under drought, got {wilted_droop}");

        soil.water(1.0);
        for _ in 0..200 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(
            plant.stem_droop < wilted_droop,
            "expected stem droop to recover after rewatering: {wilted_droop} -> {}",
            plant.stem_droop
        );
    }

    #[test]
    fn thinner_stems_droop_more_than_thicker_ones_under_the_same_drought() {
        let config = config();
        let mut thin_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            stem_radius: 0.005,
            ..Plant::new()
        };
        let mut thick_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            stem_radius: 0.2,
            ..Plant::new()
        };
        let mut thin_soil = Soil { moisture: 0.0, ..Default::default() };
        let mut thick_soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..200 {
            thin_plant.step(1.0, &sun, &neutral_climate(), &mut thin_soil, 1.0, &config);
            thick_plant.step(1.0, &sun, &neutral_climate(), &mut thick_soil, 1.0, &config);
        }

        assert!(
            thin_plant.stem_droop > thick_plant.stem_droop,
            "expected the thinner stem to droop more under the same water stress: thin {} vs thick {}",
            thin_plant.stem_droop,
            thick_plant.stem_droop
        );
    }

    #[test]
    fn a_branch_droops_under_drought_independently_of_the_main_stem() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.min_height_for_branching + 0.1,
            height_at_last_leaf: config.plant.min_height_for_branching,
            leaves: vec![mature_leaf(Side::Left)],
            carbon_pool: config.plant.new_branch_carbon_cost * 2.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        // First get a branch established under healthy water...
        for _ in 0..500 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.01);
        }
        assert!(!plant.branches.is_empty(), "expected a branch to have formed by now");

        // ...then let it dry out and confirm the branch's own droop responds.
        soil.moisture = 0.0;
        for _ in 0..300 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(
            plant.branches[0].droop > 0.0,
            "expected the branch's own stem to sag under drought too, got {}",
            plant.branches[0].droop
        );
    }

    #[test]
    fn new_leaves_spawn_over_time_with_good_conditions() {
        let config = config();
        let mut plant = Plant::new();
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        // 3000, not 50_000: past a few thousand ticks of uninterrupted full
        // noon growth the stem runs into `height_light_factor`'s falloff
        // (real height, not a demo-time artifact — see
        // `PlantConfig::window_light_zone_height`), which is a separate,
        // deliberate mechanism this test isn't about.
        for _ in 0..3000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001); // keep it from drying out over a long run
        }
        assert!(
            plant.leaves.len() >= 3,
            "expected multiple leaves after a long healthy run, got {}",
            plant.leaves.len()
        );
    }

    #[test]
    fn stem_leans_toward_light_over_time_and_saturates() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        assert_eq!(plant.lean_angle, 0.0);

        for _ in 0..1000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        let lean_after_1000 = plant.lean_angle;
        assert!(lean_after_1000 > 0.0, "should lean toward the light over time");

        // Saturates rather than bending indefinitely — real stems don't.
        for _ in 0..100_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert!((plant.lean_angle - config.plant.max_lean_angle).abs() < 1e-6);
    }

    #[test]
    fn a_climbing_habit_stays_perfectly_straight_while_within_reach_of_its_trellis() {
        let mut config = config();
        config.plant.trellis_height = Some(3.0);
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..500 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }

        assert!(
            plant.height < 3.0,
            "test setup should still be within trellis reach: height {}",
            plant.height
        );
        assert_eq!(
            plant.lean_angle, 0.0,
            "a trained climber should stay perfectly straight while still on its support"
        );
    }

    #[test]
    fn a_climbing_habit_resumes_ordinary_phototropism_once_it_outgrows_its_trellis() {
        let mut config = config();
        config.plant.trellis_height = Some(3.0);
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..60_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }

        assert!(
            plant.height > 3.0,
            "test setup should have outgrown the trellis: height {}",
            plant.height
        );
        assert!(
            plant.lean_angle > 0.0,
            "a climber that's outgrown its support should flop over and lean toward light like any freestanding stem"
        );
    }

    #[test]
    fn leans_freely_is_the_single_source_of_truth_for_the_climbing_suppression_rule() {
        // Exercises the extracted predicate directly, with no Plant/Soil/
        // Sun involved at all — the whole point of pulling it out of
        // `step_vegetative`/`step_branch` into its own pure function.
        assert!(leans_freely(0.0, None), "a freestanding habit always leans freely, even at height 0");
        assert!(leans_freely(100.0, None), "...and at any height");
        assert!(!leans_freely(1.0, Some(3.0)), "still within reach of the support");
        assert!(!leans_freely(3.0, Some(3.0)), "exactly at the support's own height is still \"on\" it");
        assert!(leans_freely(3.1, Some(3.0)), "past the support's height, it's on its own");
    }

    #[test]
    fn spawn_due_aerial_roots_only_spawns_while_climbing_and_respects_the_height_interval() {
        // Exercises the pure function directly — a bare Vec and a couple of
        // f64s, no Plant/Soil/SunState involved at all.
        let config = config();
        let mut roots: Vec<AerialRoot> = Vec::new();
        let mut height_at_last = 0.0;

        // Not climbing right now (e.g. past the trellis, or freestanding) —
        // should spawn nothing no matter how much height is behind it.
        spawn_due_aerial_roots(10.0, &mut height_at_last, &mut roots, false, &config.plant);
        assert!(roots.is_empty(), "should never spawn aerial roots when not currently climbing");

        // Climbing, and several intervals' worth of height has accumulated
        // at once (a coarse timestep) — should catch up in one call, same
        // reasoning as `record_stem_segments`'s own multi-crossing test.
        spawn_due_aerial_roots(1.0, &mut height_at_last, &mut roots, true, &config.plant);
        let expected = (1.0 / config.plant.aerial_root_height_interval).floor() as usize;
        assert_eq!(roots.len(), expected, "expected one root per interval crossed");
        for pair in roots.windows(2) {
            assert!(
                pair[1].attach_height > pair[0].attach_height,
                "attach heights should be strictly increasing: {roots:?}"
            );
        }
    }

    #[test]
    fn spawn_due_aerial_roots_stops_at_the_configured_maximum() {
        let config = config();
        let mut roots: Vec<AerialRoot> = Vec::new();
        let mut height_at_last = 0.0;
        spawn_due_aerial_roots(1_000_000.0, &mut height_at_last, &mut roots, true, &config.plant);
        assert_eq!(roots.len(), MAX_AERIAL_ROOTS);
    }

    #[test]
    fn a_climbing_plant_grows_aerial_roots_while_on_its_trellis_but_not_once_past_it() {
        let mut config = config();
        config.plant.trellis_height = Some(3.0);
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..500 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert!(
            plant.height < 3.0,
            "test setup should still be within trellis reach: height {}",
            plant.height
        );
        assert!(
            !plant.aerial_roots.is_empty(),
            "expected a climbing plant to have grown at least one aerial root by now"
        );

        // Keep stepping until it's *actually* outgrown the trellis (not
        // just a fixed tick count, which could still land mid-climb) —
        // only once height is unambiguously past 3.0 is "no more new
        // roots from here on" a meaningful thing to assert.
        for _ in 0..60_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
            if plant.height > 3.0 {
                break;
            }
        }
        assert!(
            plant.height > 3.0,
            "test setup should have outgrown the trellis: height {}",
            plant.height
        );
        let roots_just_past_trellis = plant.aerial_roots.len();

        for _ in 0..60_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert_eq!(
            plant.aerial_roots.len(),
            roots_just_past_trellis,
            "no new aerial roots should grow once past the trellis's own height"
        );
    }

    #[test]
    fn a_freestanding_habit_never_grows_any_aerial_roots() {
        let config = config();
        assert_eq!(config.plant.trellis_height, None);
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..5000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert!(plant.aerial_roots.is_empty());
    }

    // --- Blooming ----------------------------------------------------------

    #[test]
    fn bloom_intensity_target_is_zero_before_the_plant_is_mature_enough_to_flower() {
        let config = config();
        assert_eq!(bloom_intensity_target(false, 0.0, &config.plant), 0.0);
        // Even a `bloom_cycle_position` that would otherwise land in the
        // "open" phase shouldn't matter if not mature yet — immaturity
        // overrides everything else.
        assert_eq!(bloom_intensity_target(false, 1.0, &config.plant), 0.0);
    }

    #[test]
    fn bloom_intensity_target_cycles_open_then_resting_once_mature() {
        let mut config = config();
        config.plant.bloom_duration = 100.0;
        config.plant.bloom_rest_duration = 50.0;

        assert_eq!(bloom_intensity_target(true, 0.0, &config.plant), 1.0, "start of a cycle: open");
        assert_eq!(bloom_intensity_target(true, 99.0, &config.plant), 1.0, "still within bloom_duration");
        assert_eq!(bloom_intensity_target(true, 100.0, &config.plant), 0.0, "just past bloom_duration: resting");
        assert_eq!(bloom_intensity_target(true, 149.0, &config.plant), 0.0, "still resting");
        // A second cycle: 150 sim-seconds is exactly one full cycle length
        // (100 + 50) later than 0, so it should read as freshly open again.
        assert_eq!(bloom_intensity_target(true, 150.0, &config.plant), 1.0, "wrapped into a second cycle: open again");
    }

    #[test]
    fn a_plant_actually_blooms_and_then_rests_as_it_crosses_the_height_threshold_and_cycles() {
        let mut config = config();
        // A short, fast cycle keeps this test cheap while still exercising
        // the real `step_vegetative` wiring end to end (not just the pure
        // function in isolation).
        config.plant.bloom_duration = 200.0;
        config.plant.bloom_rest_duration = 100.0;
        config.plant.bloom_response_rate = 0.05;
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            height: config.plant.flowering_height_threshold,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);

        // 150 sim-seconds in: still well within the first cycle's
        // bloom_duration (200), and at bloom_response_rate 0.05 (an
        // exponential ease with a ~20-tick time constant) comfortably
        // enough time to have opened most of the way.
        for _ in 0..150 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert!(
            plant.bloom_intensity > 0.9,
            "expected the bloom to have eased most of the way open by now: {}",
            plant.bloom_intensity
        );

        // 120 more sim-seconds (total position 270): the target flipped to
        // resting at position 200, so this is 70 ticks of decay at the
        // same rate (0.95^70 ≈ 0.03 of whatever it started at) — plenty to
        // read as closed, while 270 is still within the rest phase
        // (200..300), not yet wrapped into a second cycle's bloom again.
        for _ in 0..120 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert!(
            plant.bloom_intensity < 0.1,
            "expected the bloom to have closed back down during the rest phase: {}",
            plant.bloom_intensity
        );
    }

    #[test]
    fn a_plant_below_the_flowering_height_threshold_never_blooms_at_all() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        assert!(plant.height < config.plant.flowering_height_threshold);
        // Bone dry throughout — elongation needs turgor, so height (and
        // thus maturity-to-bloom) never advances at all, isolating this to
        // "does an immature plant ever bloom," not muddied by it actually
        // growing tall enough to cross the threshold mid-test.
        let mut soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..5000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(plant.height < config.plant.flowering_height_threshold, "test setup should never have grown");
        assert_eq!(plant.bloom_intensity, 0.0);
    }

    #[test]
    fn stem_segments_get_recorded_as_height_crosses_each_interval_and_stay_non_decreasing() {
        // Regression test for a real "shouldn't stems curve?" question:
        // real stem tissue keeps whatever lean it had when it stiffened, it
        // doesn't retroactively straighten (or over-bend) just because the
        // growing tip keeps leaning more later. Recording (rather than
        // rendering one rigid rotation) is what makes that possible — see
        // `record_stem_segments` and `Plant::stem_segment_history`.
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..8000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }

        assert!(
            plant.height >= config.plant.stem_segment_height_interval * 2.0,
            "test setup should have grown past at least two segment boundaries: height {}",
            plant.height
        );
        assert!(
            plant.stem_segment_history.len() >= 2,
            "expected at least two recorded segments by now, got {}",
            plant.stem_segment_history.len()
        );
        // Lean only ever accumulates (see module docs), so a later-recorded
        // segment can never show *less* lean than an earlier one.
        for pair in plant.stem_segment_history.windows(2) {
            assert!(pair[1] >= pair[0], "segment history should be non-decreasing: {:?}", plant.stem_segment_history);
        }
        // Every recorded (frozen) segment reflects the lean at some point
        // *in the past* — never more than the plant's current total lean.
        for &recorded in &plant.stem_segment_history {
            assert!(recorded <= plant.lean_angle + 1e-9);
        }
    }

    #[test]
    fn stem_segment_history_never_exceeds_the_configured_maximum() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..200_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert!(
            plant.stem_segment_history.len() <= MAX_STEM_SEGMENTS,
            "got {} segments",
            plant.stem_segment_history.len()
        );
    }

    #[test]
    fn a_branch_records_its_own_segment_history_independent_of_the_main_stem() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.min_height_for_branching + 0.1,
            height_at_last_leaf: config.plant.min_height_for_branching,
            leaves: vec![mature_leaf(Side::Left)],
            carbon_pool: config.plant.new_branch_carbon_cost * 2.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        // Branches grow slower than the main stem
        // (`branch_elongation_rate_factor` < 1), so need more time to cross
        // the same number of segment boundaries.
        for _ in 0..20_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }

        assert!(!plant.branches.is_empty(), "expected at least one branch by now");
        let branch = &plant.branches[0];
        assert!(
            branch.height >= config.plant.stem_segment_height_interval * 2.0,
            "test setup should have grown the branch past at least two segment boundaries: height {}",
            branch.height
        );
        assert!(
            branch.segment_history.len() >= 2,
            "expected the branch to have recorded its own segment history, got {}",
            branch.segment_history.len()
        );
    }

    #[test]
    fn stem_does_not_lean_at_night() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = midnight(&config);
        for _ in 0..1000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert_eq!(plant.lean_angle, 0.0);
    }

    #[test]
    fn leaves_fold_at_night_and_reopen_by_day() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        plant.leaves[0].fold = 0.0; // start fully open, as if mid-afternoon
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let (night_sun, day_sun) = (midnight(&config), noon(&config));

        for _ in 0..300 {
            plant.step(1.0, &night_sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        let folded = plant.leaves[0].fold;
        assert!(folded > 0.5, "expected the leaf to fold down at night, got {folded}");

        for _ in 0..300 {
            plant.step(1.0, &day_sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(
            plant.leaves[0].fold < folded,
            "expected the leaf to reopen once light returns"
        );
    }

    #[test]
    fn leaves_track_the_suns_position_across_the_window() {
        let config = config();
        let mut morning_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut evening_plant = morning_plant.clone();
        let mut soil_a = Soil { moisture: 1.0, ..Default::default() };
        let mut soil_b = Soil { moisture: 1.0, ..Default::default() };

        // azimuth near 0 = sun low in the window on the sunrise side,
        // azimuth near 1 = sunset side — same intensity, opposite position.
        let morning_sun = SunState {
            elevation: 0.6,
            azimuth: 0.05,
            intensity: 0.6,
            color: [1.0, 0.8, 0.6],
        };
        let evening_sun = SunState {
            elevation: 0.6,
            azimuth: 0.95,
            intensity: 0.6,
            color: [1.0, 0.8, 0.6],
        };

        for _ in 0..500 {
            morning_plant.step(1.0, &morning_sun, &neutral_climate(), &mut soil_a, 1.0, &config);
            evening_plant.step(1.0, &evening_sun, &neutral_climate(), &mut soil_b, 1.0, &config);
        }

        assert!(
            morning_plant.leaves[0].helio_angle < 0.0,
            "sun on the sunrise side should bias the leaf that direction, got {}",
            morning_plant.leaves[0].helio_angle
        );
        assert!(
            evening_plant.leaves[0].helio_angle > 0.0,
            "sun on the sunset side should bias the leaf the other direction, got {}",
            evening_plant.leaves[0].helio_angle
        );
    }

    // --- Single-tick decision tests -----------------------------------
    //
    // These pin down *why* one specific step behaved a certain way, using
    // `Plant::last_decision` — much cheaper than the aggregate tests above
    // (one step instead of hundreds/thousands) and more precise about which
    // mechanism is being checked, since they can assert on the intermediate
    // quantities directly instead of inferring them from height/leaf count
    // after a long run.

    #[test]
    fn single_tick_seed_decision_reports_its_inputs() {
        let config = config();
        let mut plant = Plant::new();
        let mut dry_soil = Soil { moisture: 0.0, ..Default::default() };
        plant.step(1.0, &noon(&config), &neutral_climate(), &mut dry_soil, 1.0, &config);
        match plant.last_decision {
            Some(Decision::Seed {
                water_factor,
                threshold,
                germinated,
            }) => {
                assert_eq!(water_factor, 0.0);
                assert_eq!(threshold, config.plant.germination_water_factor);
                assert!(!germinated);
            }
            other => panic!("expected a Seed decision, got {other:?}"),
        }

        // Germination happens *within* this next step_seed call (stage
        // flips internally once water_factor clears the threshold) — the
        // decision recorded still describes that same Seed-stage tick, with
        // `germinated: true`; the dispatch to `step_sprout` (and so a
        // `Decision::Sprout`) only happens on the *following* call.
        let mut wet_soil = Soil { moisture: 1.0, ..Default::default() };
        plant.step(1.0, &noon(&config), &neutral_climate(), &mut wet_soil, 1.0, &config);
        match plant.last_decision {
            Some(Decision::Seed { germinated: true, .. }) => {}
            other => panic!("expected germination this tick, got {other:?}"),
        }
        assert_eq!(plant.stage, Stage::Sprout);

        plant.step(1.0, &noon(&config), &neutral_climate(), &mut wet_soil, 1.0, &config);
        match plant.last_decision {
            Some(Decision::Sprout { .. }) => {}
            other => panic!("expected a Sprout decision on the following tick, got {other:?}"),
        }
    }

    #[test]
    fn single_tick_vegetative_decision_has_zero_photosynthesis_at_night() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        plant.step(1.0, &midnight(&config), &neutral_climate(), &mut soil, 1.0, &config);

        match plant.last_decision {
            Some(Decision::Vegetative {
                sun_intensity,
                photosynthesis,
                elongation,
                ..
            }) => {
                assert_eq!(sun_intensity, 0.0);
                assert_eq!(photosynthesis, 0.0);
                assert_eq!(elongation, 0.0, "no carbon banked yet, so nothing to spend on elongation");
            }
            other => panic!("expected a Vegetative decision, got {other:?}"),
        }
    }

    #[test]
    fn single_tick_vegetative_decision_reports_temperature_and_its_factors() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let climate = ClimateState {
            temperature_c: config.plant.optimal_temperature_c + config.plant.temperature_tolerance_c,
        };
        plant.step(1.0, &noon(&config), &climate, &mut soil, 1.0, &config);

        match plant.last_decision {
            Some(Decision::Vegetative {
                temperature_c,
                temp_factor,
                respiration_q10_factor,
                ..
            }) => {
                assert_eq!(temperature_c, climate.temperature_c);
                // One tolerance-width away from the optimum: exp(-1).
                assert!((temp_factor - (-1.0_f64).exp()).abs() < 1e-9, "got {temp_factor}");
                let expected_q10 = climate::q10_factor(
                    climate.temperature_c,
                    config.plant.respiration_reference_temperature_c,
                    config.plant.respiration_q10,
                );
                assert!((respiration_q10_factor - expected_q10).abs() < 1e-9);
            }
            other => panic!("expected a Vegetative decision, got {other:?}"),
        }
    }

    #[test]
    fn cold_temperature_slows_photosynthesis_and_elongation_relative_to_the_optimum() {
        let config = config();
        let start = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut optimal_plant = start.clone();
        let mut cold_plant = start.clone();
        let mut optimal_soil = Soil { moisture: 1.0, ..Default::default() };
        let mut cold_soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        let optimal_climate = ClimateState {
            temperature_c: config.plant.optimal_temperature_c,
        };
        let cold_climate = ClimateState {
            temperature_c: config.plant.optimal_temperature_c - config.plant.temperature_tolerance_c * 2.0,
        };

        for _ in 0..2000 {
            optimal_plant.step(1.0, &sun, &optimal_climate, &mut optimal_soil, 1.0, &config);
            cold_plant.step(1.0, &sun, &cold_climate, &mut cold_soil, 1.0, &config);
        }

        assert!(
            cold_plant.height < optimal_plant.height,
            "expected cold to slow growth: cold height {} vs optimal height {}",
            cold_plant.height,
            optimal_plant.height
        );
    }

    #[test]
    fn heat_increases_respiration_via_q10_relative_to_the_reference_temperature() {
        let config = config();
        // No light, so photosynthesis is zero either way — isolates
        // respiration's own temperature response (Q10) from photosynthesis'
        // separate bell-curve one.
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            carbon_pool: config.plant.max_carbon_pool,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let hot_climate = ClimateState {
            temperature_c: config.plant.respiration_reference_temperature_c + 10.0,
        };
        plant.step(1.0, &midnight(&config), &hot_climate, &mut soil, 1.0, &config);

        match plant.last_decision {
            Some(Decision::Vegetative { respiration, .. }) => {
                let expected_baseline = config.plant.respiration_rate * (1.0 + plant.true_leaf_area(&config.plant));
                assert!(
                    respiration > expected_baseline,
                    "expected respiration at reference+10C to exceed the Q10=1 baseline: {respiration} vs {expected_baseline}"
                );
            }
            other => panic!("expected a Vegetative decision, got {other:?}"),
        }
    }

    #[test]
    fn single_tick_vegetative_decision_flags_carbon_limited_elongation() {
        let config = config();
        // Light below the photosynthetic compensation point (see
        // `low_light_produces_etiolated_growth_relative_to_high_light`'s
        // comment on the same threshold): respiration keeps running
        // regardless, so a fresh plant with no banked reserve runs a carbon
        // deficit this tick and its elongation should be entirely
        // carbon-limited.
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            carbon_pool: 0.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let dim_sun = SunState {
            elevation: 0.15,
            azimuth: 0.5,
            intensity: 0.15,
            color: [1.0, 0.7, 0.4],
        };
        plant.step(1.0, &dim_sun, &neutral_climate(), &mut soil, 1.0, &config);

        match plant.last_decision {
            Some(Decision::Vegetative {
                elongation_carbon_limited,
                elongation,
                water_factor,
                photosynthesis,
                respiration,
                ..
            }) => {
                assert!(elongation_carbon_limited);
                assert_eq!(elongation, 0.0);
                assert_eq!(water_factor, 1.0, "water was not the constraint here");
                assert!(
                    respiration > photosynthesis,
                    "this scenario is only meaningful if respiration actually exceeds income: {respiration} vs {photosynthesis}"
                );
            }
            other => panic!("expected a Vegetative decision, got {other:?}"),
        }
    }

    #[test]
    fn single_tick_vegetative_decision_reports_night_fold_target_and_day_helio_target() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };

        plant.step(1.0, &midnight(&config), &neutral_climate(), &mut soil, 1.0, &config);
        match plant.last_decision {
            Some(Decision::Vegetative { fold_target, .. }) => {
                assert!((fold_target - 1.0).abs() < 1e-9, "should target fully folded at night");
            }
            other => panic!("expected a Vegetative decision, got {other:?}"),
        }

        let sunset_side_sun = SunState {
            elevation: 0.5,
            azimuth: 1.0,
            intensity: 0.5,
            color: [1.0, 0.9, 0.8],
        };
        plant.step(1.0, &sunset_side_sun, &neutral_climate(), &mut soil, 1.0, &config);
        match plant.last_decision {
            Some(Decision::Vegetative { helio_target, .. }) => {
                assert!(helio_target > 0.0, "sunset-side sun should target a positive helio angle");
            }
            other => panic!("expected a Vegetative decision, got {other:?}"),
        }
    }

    // --- Regression test ------------------------------------------------

    #[test]
    fn regression_fixed_scenario_growth_snapshot() {
        let config = config();
        let mut plant = Plant::new();
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);

        for _ in 0..5000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }

        assert_eq!(plant.stage, Stage::Vegetative);
        assert_eq!(plant.leaves.len(), 10, "main-stem leaf count regression");
        assert_eq!(plant.branches.len(), config.plant.max_branches, "branch count regression");
        assert!(
            (plant.height - 5.007).abs() < 0.05,
            "height regression: got {}",
            plant.height
        );
        assert!(
            (plant.stem_radius - 0.0457).abs() < 0.002,
            "stem_radius regression: got {}",
            plant.stem_radius
        );
        assert!(
            (plant.lean_angle - config.plant.max_lean_angle).abs() < 1e-6,
            "lean should have saturated by now: got {}",
            plant.lean_angle
        );
        // A 0.001/tick top-up isn't enough to keep a plant this leafy fully
        // saturated (transpiration scales with total leaf area, which is
        // substantial by now across the main stem and 4 branches) — some
        // persistent stress, and so some nonzero stem droop, is expected
        // and realistic, not a bug.
        assert!(plant.stem_droop > 0.0, "expected some water stress by now: got {}", plant.stem_droop);
        for branch in &plant.branches {
            assert!(branch.height > 0.0);
            assert!(!branch.leaves.is_empty());
        }
    }

    // --- Crown branching -------------------------------------------------

    #[test]
    fn no_branches_before_the_height_threshold() {
        let config = config();
        // Well above the carbon cost of a branch, but shorter than the
        // height threshold — should still refuse to branch.
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.min_height_for_branching * 0.5,
            leaves: vec![mature_leaf(Side::Left)],
            carbon_pool: config.plant.new_branch_carbon_cost * 2.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        plant.step(1.0, &noon(&config), &neutral_climate(), &mut soil, 1.0, &config);
        assert!(plant.branches.is_empty());
        match plant.last_decision {
            Some(Decision::Vegetative { branch_spawned, .. }) => assert!(!branch_spawned),
            other => panic!("expected a Vegetative decision, got {other:?}"),
        }
    }

    #[test]
    fn single_tick_decision_reports_a_branch_spawning() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.min_height_for_branching + 0.1,
            // Already had its routine plastochron leaves up to the
            // branching threshold — without this, a freshly-constructed
            // test plant that jumps straight to this height (rather than
            // growing there gradually) would trigger a multi-leaf
            // catch-up burst (see `Plant::spawn_due_leaves_fairly`) that
            // eats the very carbon this test wants to reserve for the
            // branch.
            height_at_last_leaf: config.plant.min_height_for_branching,
            leaves: vec![mature_leaf(Side::Left)],
            carbon_pool: config.plant.new_branch_carbon_cost * 2.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        plant.step(1.0, &noon(&config), &neutral_climate(), &mut soil, 1.0, &config);

        assert_eq!(plant.branches.len(), 1);
        assert_eq!(plant.branches[0].attach_height, plant.height, "a branch attaches at the current (near-tip) height, not partway down the stem");
        match plant.last_decision {
            Some(Decision::Vegetative { branch_spawned, .. }) => assert!(branch_spawned),
            other => panic!("expected a Vegetative decision, got {other:?}"),
        }
    }

    #[test]
    fn branch_grows_its_own_height_radius_and_leaves_over_time() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.min_height_for_branching + 0.1,
            leaves: vec![mature_leaf(Side::Left)],
            carbon_pool: config.plant.new_branch_carbon_cost * 2.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..5000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }

        assert!(!plant.branches.is_empty(), "expected at least one branch after a long healthy run");
        let branch = &plant.branches[0];
        assert!(branch.height > 0.0, "branch should have elongated");
        assert!(branch.stem_radius > 0.0, "branch should have thickened");
        assert!(!branch.leaves.is_empty(), "branch should have grown its own leaves");
    }

    #[test]
    fn branch_count_never_exceeds_the_configured_maximum() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.min_height_for_branching + 0.1,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..50_000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert!(plant.branches.len() <= config.plant.max_branches);
    }

    #[test]
    fn leaves_keep_appearing_on_the_main_stem_well_past_the_branching_height_threshold() {
        // Regression test for a real reported bug: leaf initiation used to
        // be mutually exclusive with funding a branch (see
        // `plastochron_height_interval`'s doc comment) — once the stem
        // crossed `min_height_for_branching` it would typically never grow
        // another main-stem leaf again, no matter how much further it grew,
        // because elongation (cheap, unconditional) kept outcompeting the
        // branch's much higher carbon threshold every tick. The visible
        // symptom was "leaves only ever grow at the base."
        let config = config();
        let mut plant = Plant::new();
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..5000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }

        assert!(
            plant.height > config.plant.min_height_for_branching * 2.0,
            "expected the stem to have grown well past the branching threshold: {}",
            plant.height
        );
        let leaves_past_threshold = plant
            .leaves
            .iter()
            .filter(|l| l.attach_height > config.plant.min_height_for_branching)
            .count();
        assert!(
            leaves_past_threshold >= 3,
            "expected several main-stem leaves attached above the branching height, got {leaves_past_threshold} (all leaf heights: {:?})",
            plant.leaves.iter().map(|l| l.attach_height).collect::<Vec<_>>()
        );
    }

    #[test]
    fn leaf_attach_heights_spread_along_the_main_stem_instead_of_clustering_at_the_base() {
        let config = config();
        let mut plant = Plant::new();
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..2000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }

        let heights: Vec<f64> = plant.leaves.iter().map(|l| l.attach_height).collect();
        let max_height = heights.iter().cloned().fold(0.0, f64::max);
        let min_height = heights.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            max_height - min_height > config.plant.min_height_for_branching,
            "expected leaf attach heights to spread out over the stem's growth, got a span of {} (heights: {heights:?})",
            max_height - min_height
        );
    }

    // --- Root health: overwatering / fertilizer burn ----------------------

    #[test]
    fn sustained_waterlogging_damages_root_health_even_though_soil_reads_fully_watered() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        // Re-flood every tick, faster than the plant could ever draw it
        // down — models a player spamming the Water button instead of
        // letting the pot drain between waterings.
        for _ in 0..2000 {
            soil.water(1.0);
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(
            plant.root_health < 1.0,
            "expected sustained flooding to damage root health, got {}",
            plant.root_health
        );
    }

    #[test]
    fn a_root_damaged_plant_wilts_even_with_soil_reading_fully_watered() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            root_health: 0.2,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..300 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(
            plant.leaves[0].droop > 0.3,
            "expected wilting despite fully-watered soil once roots are damaged, got {}",
            plant.leaves[0].droop
        );
    }

    #[test]
    fn root_health_recovers_over_time_once_soil_is_no_longer_waterlogged() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            root_health: 0.5,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 0.5, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..5000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(
            plant.root_health > 0.5,
            "expected root health to recover once no longer waterlogged, got {}",
            plant.root_health
        );
    }

    #[test]
    fn overfeeding_damages_root_health_the_same_way_overwatering_does() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 0.5, nutrient: config.soil.max_nutrient, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..2000 {
            // Keep pinned at the overfeed ceiling, like repeatedly
            // over-fertilizing instead of letting it draw down.
            soil.nutrient = config.soil.max_nutrient;
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert!(
            plant.root_health < 1.0,
            "expected sustained over-fertilizing to damage roots, got {}",
            plant.root_health
        );
    }

    #[test]
    fn depleted_nutrient_reduces_photosynthesis_relative_to_well_fed_soil() {
        let config = config();
        let mut fed = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut starved = fed.clone();
        let mut fed_soil = Soil { moisture: 1.0, nutrient: 1.0 };
        let mut starved_soil = Soil { moisture: 1.0, nutrient: 0.0 };
        let sun = noon(&config);
        fed.step(1.0, &sun, &neutral_climate(), &mut fed_soil, 1.0, &config);
        starved.step(1.0, &sun, &neutral_climate(), &mut starved_soil, 1.0, &config);
        let photosynthesis = |p: &Plant| match p.last_decision {
            Some(Decision::Vegetative { photosynthesis, .. }) => photosynthesis,
            _ => panic!("expected a Vegetative decision"),
        };
        assert!(
            photosynthesis(&starved) < photosynthesis(&fed),
            "expected depleted nutrient to reduce photosynthesis: starved {} vs fed {}",
            photosynthesis(&starved),
            photosynthesis(&fed)
        );
    }

    // --- Whole-plant death --------------------------------------------------

    #[test]
    fn a_plant_dies_once_root_health_is_driven_to_zero_by_sustained_waterlogging() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            root_health: 0.0001,
            waterlogged_duration: config.soil.waterlog_grace_period + 1.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        assert_eq!(plant.stage, Stage::Dead, "expected total root loss to kill the plant");
        assert_eq!(plant.death_cause, Some(DeathCause::RootRot));
        assert_eq!(
            plant.leaves[0].senescence, 1.0,
            "expected death to force a fully dead visual appearance, not leave the leaf looking healthy"
        );
        assert_eq!(plant.leaves[0].droop, 1.0);
        assert_eq!(plant.stem_droop, config.plant.stem_droop_max_angle);
    }

    #[test]
    fn a_leafless_plant_with_no_carbon_eventually_dies_of_starvation() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: Vec::new(),
            carbon_pool: 0.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 0.0, ..Default::default() };
        let sun = midnight(&config);
        for _ in 0..(config.plant.starvation_death_threshold as usize + 100) {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        }
        assert_eq!(
            plant.stage,
            Stage::Dead,
            "expected sustained leaflessness with no carbon to eventually kill the plant"
        );
        assert_eq!(plant.death_cause, Some(DeathCause::Starvation));
    }

    // --- Cotyledons -----------------------------------------------------------

    #[test]
    fn cotyledon_fade_fraction_is_zero_during_seed_stage() {
        let config = config();
        let plant = Plant { stage: Stage::Seed, ..Plant::new() };
        assert_eq!(plant.cotyledon_fade_fraction(&config.plant), 0.0);
    }

    #[test]
    fn cotyledon_fade_fraction_is_full_strength_right_after_the_first_true_leaf() {
        let config = config();
        let mut plant = Plant { stage: Stage::Vegetative, ..Plant::new() };
        plant.spawn_leaf();
        assert!(plant.cotyledon_fade_fraction(&config.plant) < 1.0);
        assert!(plant.cotyledon_fade_fraction(&config.plant) > 0.0);
    }

    #[test]
    fn cotyledon_fade_fraction_reaches_zero_once_enough_true_leaves_have_ever_grown() {
        let config = config();
        let mut plant = Plant { stage: Stage::Vegetative, ..Plant::new() };
        for _ in 0..(config.plant.cotyledon_fade_over_leaves as usize) {
            plant.spawn_leaf();
        }
        assert_eq!(plant.cotyledon_fade_fraction(&config.plant), 0.0);
    }

    #[test]
    fn cotyledon_fade_fraction_never_climbs_back_up_after_leaves_are_later_lost() {
        let config = config();
        let mut plant = Plant { stage: Stage::Vegetative, ..Plant::new() };
        for _ in 0..(config.plant.cotyledon_fade_over_leaves as usize) {
            plant.spawn_leaf();
        }
        assert_eq!(plant.cotyledon_fade_fraction(&config.plant), 0.0);
        plant.leaves.clear();
        assert_eq!(
            plant.cotyledon_fade_fraction(&config.plant),
            0.0,
            "expected already-faded cotyledons to stay faded even after later losing every true leaf"
        );
    }

    // --- Scoring metrics ------------------------------------------------------

    #[test]
    fn spawning_a_leaf_increments_leaves_produced_total_and_never_decrements_it() {
        let mut plant = Plant::new();
        plant.spawn_leaf();
        plant.spawn_leaf();
        assert_eq!(plant.leaves_produced_total, 2);
        plant.leaves.clear(); // simulates pruning/senescence removing leaves
        assert_eq!(
            plant.leaves_produced_total, 2,
            "expected the lifetime total to survive leaves later being removed"
        );
    }

    #[test]
    fn spawn_due_leaves_fairly_increments_leaves_produced_total_for_every_leaf_it_spawns() {
        let config = config();
        let mut plant = Plant { height: 10.0, carbon_pool: 1000.0, branches: vec![Branch::new(0.0, Side::Left)], ..Plant::new() };
        plant.branches[0].height = 10.0;
        let spawned = plant.spawn_due_leaves_fairly(&config.plant);
        assert!(spawned, "expected a tall stem with ample carbon to actually grow at least one leaf");
        let total_leaves = plant.leaves.len() + plant.branches[0].leaves.len();
        assert_eq!(
            plant.leaves_produced_total, total_leaves as u32,
            "expected the lifetime counter to match exactly how many leaves this call actually spawned"
        );
    }

    #[test]
    fn max_leaves_at_once_records_a_high_water_mark_that_survives_losing_leaves_later() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left), mature_leaf(Side::Right), mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        assert_eq!(plant.max_leaves_at_once, 3);
        plant.leaves.truncate(1);
        plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        assert_eq!(
            plant.max_leaves_at_once, 3,
            "expected the high-water mark to survive losing leaves, not track the current count"
        );
    }

    #[test]
    fn max_height_reached_records_a_high_water_mark_that_survives_pruning() {
        let config = config();
        let mut plant =
            Plant { stage: Stage::Vegetative, height: 5.0, leaves: vec![mature_leaf(Side::Left)], ..Plant::new() };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        assert!(plant.max_height_reached >= 5.0);
        let peak = plant.max_height_reached;
        plant.height = 1.0; // simulates a prune/cutting shortening it
        plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        assert_eq!(
            plant.max_height_reached, peak,
            "expected the high-water mark to survive height dropping back down"
        );
    }

    #[test]
    fn alive_duration_freezes_at_death_while_total_time_keeps_accumulating() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            root_health: 0.0001,
            waterlogged_duration: config.soil.waterlog_grace_period + 1.0,
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        assert_eq!(plant.stage, Stage::Dead);
        let alive_at_death = plant.alive_duration;
        let total_at_death = plant.total_time;
        plant.step(10.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
        assert_eq!(
            plant.alive_duration, alive_at_death,
            "expected alive_duration to freeze once the plant is dead"
        );
        assert_eq!(
            plant.total_time,
            total_at_death + 10.0,
            "expected total_time to keep accumulating after death, unlike alive_duration"
        );
    }

    #[test]
    fn a_plant_with_leaves_never_dies_of_starvation_even_when_carbon_routinely_hits_zero_at_night() {
        // Regression test: carbon_pool sitting at its zero floor is a
        // completely routine sawtooth for a healthy plant (see module docs
        // — it banks up and gets spent in a lump, and drains every single
        // night regardless of health), not itself a sign of terminal
        // starvation. A first draft of the starvation timer treated any
        // zero-carbon tick as starving, which killed a perfectly healthy,
        // still-leaved germinating plant within its first real minute.
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        for i in 0..3000 {
            // Alternates every single tick — a deliberately worse-case sawtooth
            // than the real day/night cycle would ever actually produce.
            let sun = if i % 2 == 0 { noon(&config) } else { midnight(&config) };
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.water(0.001);
        }
        assert_ne!(
            plant.stage,
            Stage::Dead,
            "a plant that still has leaves shouldn't die just from carbon routinely bottoming out at night"
        );
    }

    // --- Pot-bound stress & repotting ---------------------------------------

    #[test]
    fn pot_bound_factor_drops_once_height_exceeds_capacity_and_repotting_raises_the_ceiling() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.initial_pot_capacity + config.plant.pot_bound_stress_range,
            ..Plant::new()
        };
        let bound_factor = plant.pot_bound_factor(&config.plant);
        assert!(
            bound_factor < 1.0,
            "expected a plant well past its pot capacity to be pot-bound, got {bound_factor}"
        );

        plant.repot(&config.plant);
        let factor_after_repot = plant.pot_bound_factor(&config.plant);
        assert!(
            factor_after_repot > bound_factor,
            "expected repotting to raise the ceiling and improve the factor: {factor_after_repot} vs {bound_factor}"
        );
    }

    #[test]
    fn repotting_restores_root_health_and_costs_a_growth_shock() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            root_health: 0.1,
            ..Plant::new()
        };
        assert!(plant.repot(&config.plant));
        assert!(plant.root_health >= config.plant.repot_root_health_restore);
        assert!(plant.growth_shock > 0.0, "expected repotting to cost a temporary setback");
    }

    #[test]
    fn repot_is_a_no_op_once_the_plant_is_dead() {
        let config = config();
        let mut plant = Plant { stage: Stage::Dead, ..Plant::new() };
        assert!(!plant.repot(&config.plant));
    }

    // --- Pruning -------------------------------------------------------------

    #[test]
    fn pruning_the_main_stem_cuts_height_sheds_leaves_above_the_cut_and_releases_branches() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            leaves: vec![
                Leaf { attach_height: 0.2, ..mature_leaf(Side::Left) },
                Leaf { attach_height: 1.9, ..mature_leaf(Side::Right) },
            ],
            ..Plant::new()
        };
        assert!(plant.branches.is_empty());
        assert!(plant.prune_main_stem(&config.plant));
        assert!(plant.height < 2.0, "expected pruning to cut the stem back");
        assert!(
            !plant.leaves.iter().any(|l| l.attach_height > plant.height),
            "expected leaves above the cut point to be shed"
        );
        assert!(!plant.branches.is_empty(), "expected pruning to immediately release branches");
        assert!(plant.growth_shock > 0.0, "expected pruning to cost a temporary setback");
    }

    #[test]
    fn pruning_below_the_minimum_height_is_a_no_op() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.prune_min_height * 0.5,
            ..Plant::new()
        };
        assert!(!plant.prune_main_stem(&config.plant));
        assert_eq!(plant.height, config.plant.prune_min_height * 0.5);
    }

    #[test]
    fn pruning_a_branch_cuts_its_height_and_sheds_leaves_above_the_cut() {
        let config = config();
        let mut branch = Branch::new(0.5, Side::Left);
        branch.height = 1.0;
        branch.leaves.push(Leaf { attach_height: 0.9, ..mature_leaf(Side::Left) });
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            branches: vec![branch],
            ..Plant::new()
        };
        assert!(plant.prune_branch(0, &config.plant));
        assert!(plant.branches[0].height < 1.0);
        assert!(
            plant.branches[0].leaves.is_empty(),
            "expected the leaf above the cut point to be shed"
        );
    }

    #[test]
    fn pruning_a_nonexistent_branch_index_is_a_no_op() {
        let config = config();
        let mut plant = Plant { stage: Stage::Vegetative, height: 2.0, ..Plant::new() };
        assert!(!plant.prune_branch(0, &config.plant));
    }

    #[test]
    fn cut_main_stem_at_cuts_to_the_exact_height_given_not_a_fixed_fraction() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            leaves: vec![
                Leaf { attach_height: 0.5, ..mature_leaf(Side::Left) },
                Leaf { attach_height: 1.5, ..mature_leaf(Side::Right) },
            ],
            ..Plant::new()
        };
        assert!(plant.cut_main_stem_at(1.0, &config.plant));
        assert_eq!(plant.height, 1.0, "should cut to exactly the given height, not a fraction of the old one");
        assert_eq!(plant.leaves.len(), 1, "expected the leaf above the cut point to be shed");
        assert!(!plant.branches.is_empty(), "expected the same apical-dominance branch release as prune_main_stem");
        assert!(plant.growth_shock > 0.0, "expected the same growth-shock cost as prune_main_stem");
    }

    #[test]
    fn cut_main_stem_at_a_height_at_or_above_the_tip_is_a_no_op() {
        let config = config();
        let mut plant = Plant { stage: Stage::Vegetative, height: 2.0, ..Plant::new() };
        assert!(!plant.cut_main_stem_at(2.0, &config.plant), "cutting exactly at the tip isn't a cut");
        assert!(!plant.cut_main_stem_at(5.0, &config.plant), "cutting above the tip isn't a cut");
        assert_eq!(plant.height, 2.0);
    }

    #[test]
    fn cutting_the_main_stem_also_removes_branches_attached_above_the_cut() {
        // Cutting also releases up to `PlantConfig::prune_branch_release_
        // count` *new* branches via apical dominance (see `prune_main_
        // stem`'s own doc comment) — so this checks specific attach heights
        // survive/vanish rather than a raw branch count, which the new
        // releases would otherwise throw off.
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            branches: vec![
                Branch::new(0.5, Side::Left),  // below the cut — should survive
                Branch::new(1.5, Side::Right), // above the cut — should be removed
            ],
            ..Plant::new()
        };
        assert!(plant.cut_main_stem_at(1.0, &config.plant));
        assert!(
            plant.branches.iter().any(|b| b.attach_height == 0.5),
            "the branch below the cut should still be there"
        );
        assert!(
            !plant.branches.iter().any(|b| b.attach_height == 1.5),
            "the branch above the cut should have been removed with it"
        );
    }

    #[test]
    fn cutting_the_main_stem_keeps_a_branch_attached_exactly_at_the_cut_height() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            branches: vec![Branch::new(1.0, Side::Left)],
            ..Plant::new()
        };
        assert!(plant.cut_main_stem_at(1.0, &config.plant));
        assert!(
            plant.branches.iter().any(|b| b.attach_height == 1.0),
            "a branch attached exactly at the cut point still has a stem to grow from"
        );
    }

    #[test]
    fn pruning_the_main_stem_by_fraction_also_removes_branches_above_the_cut() {
        // Same cascading-removal rule applies to the fixed-fraction "Prune
        // stem" button (prune_main_stem), not just the exact-height
        // click-to-prune tool (cut_main_stem_at) — both share cut_main_
        // stem_to's mechanics.
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            branches: vec![Branch::new(1.9, Side::Right)],
            ..Plant::new()
        };
        assert!(plant.prune_main_stem(&config.plant));
        assert!(plant.height < 1.9, "sanity check: the cut landed below this branch's attach height");
        assert!(plant.branches.iter().all(|b| b.attach_height <= plant.height), "no branch should be left stranded above the new height");
    }

    #[test]
    fn cut_main_stem_at_below_the_minimum_height_is_a_no_op() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.prune_min_height * 0.5,
            ..Plant::new()
        };
        assert!(!plant.cut_main_stem_at(0.01, &config.plant));
    }

    #[test]
    fn cut_branch_at_cuts_to_the_exact_height_given_not_a_fixed_fraction() {
        let config = config();
        let mut branch = Branch::new(0.5, Side::Left);
        branch.height = 1.0;
        branch.leaves.push(Leaf { attach_height: 0.3, ..mature_leaf(Side::Left) });
        branch.leaves.push(Leaf { attach_height: 0.9, ..mature_leaf(Side::Left) });
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            branches: vec![branch],
            ..Plant::new()
        };
        assert!(plant.cut_branch_at(0, 0.5, &config.plant));
        assert_eq!(plant.branches[0].height, 0.5);
        assert_eq!(plant.branches[0].leaves.len(), 1, "expected the leaf above the cut point to be shed");
    }

    #[test]
    fn cut_branch_at_a_height_at_or_above_its_tip_is_a_no_op() {
        let config = config();
        let mut branch = Branch::new(0.5, Side::Left);
        branch.height = 1.0;
        let mut plant = Plant { stage: Stage::Vegetative, height: 2.0, branches: vec![branch], ..Plant::new() };
        assert!(!plant.cut_branch_at(0, 1.0, &config.plant));
        assert_eq!(plant.branches[0].height, 1.0);
    }

    #[test]
    fn cut_branch_at_a_nonexistent_index_is_a_no_op() {
        let config = config();
        let mut plant = Plant { stage: Stage::Vegetative, height: 2.0, ..Plant::new() };
        assert!(!plant.cut_branch_at(0, 0.5, &config.plant));
    }

    #[test]
    fn prune_leaf_removes_a_main_stem_leaf_by_its_flat_slot() {
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            leaves: vec![
                Leaf { attach_height: 0.2, ..mature_leaf(Side::Left) },
                Leaf { attach_height: 0.5, ..mature_leaf(Side::Right) },
            ],
            ..Plant::new()
        };
        assert!(plant.prune_leaf(0));
        assert_eq!(plant.leaves.len(), 1);
        assert_eq!(plant.leaves[0].attach_height, 0.5, "expected the leaf at slot 0 specifically to be removed");
    }

    #[test]
    fn prune_leaf_reaches_into_branches_once_main_stem_slots_are_exhausted() {
        let mut branch = Branch::new(0.5, Side::Left);
        branch.height = 1.0;
        branch.leaves.push(Leaf { attach_height: 0.9, ..mature_leaf(Side::Left) });
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: 2.0,
            leaves: vec![Leaf { attach_height: 0.2, ..mature_leaf(Side::Left) }],
            branches: vec![branch],
            ..Plant::new()
        };
        // Slot 0 is the main stem's only leaf; slot 1 is the branch's only
        // leaf — mirrors `render::mod`'s main-stem-first-then-branches order.
        assert!(plant.prune_leaf(1));
        assert_eq!(plant.leaves.len(), 1, "the main stem leaf should be untouched");
        assert!(plant.branches[0].leaves.is_empty(), "the branch leaf at slot 1 should be gone");
    }

    #[test]
    fn prune_leaf_out_of_range_is_a_no_op() {
        let mut plant = Plant { stage: Stage::Vegetative, height: 2.0, ..Plant::new() };
        assert!(!plant.prune_leaf(0));
    }

    #[test]
    fn prune_leaf_is_a_no_op_once_dead() {
        let mut plant = Plant {
            stage: Stage::Dead,
            leaves: vec![Leaf { attach_height: 0.2, ..mature_leaf(Side::Left) }],
            ..Plant::new()
        };
        assert!(!plant.prune_leaf(0));
        assert_eq!(plant.leaves.len(), 1);
    }

    // --- Propagation (stem cuttings) ----------------------------------------

    #[test]
    fn taking_a_cutting_costs_the_parent_some_height_like_a_small_prune() {
        let config = config();
        let starting_height = config.plant.cutting_min_height + 1.0;
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: starting_height,
            stem_radius: 0.5,
            leaves: vec![mature_leaf(Side::Left); 5],
            carbon_pool: 10.0,
            ..Plant::new()
        };
        assert!(plant.take_cutting(&config.plant));
        assert_eq!(plant.stage, Stage::Vegetative, "the parent stays alive and vegetative, not reset");
        assert!(
            (plant.height - starting_height * (1.0 - config.plant.cutting_cost_height_fraction)).abs() < 1e-9,
            "expected the parent to lose exactly `cutting_cost_height_fraction` of its height, got {} from {starting_height}",
            plant.height
        );
        assert!(plant.height > 0.0, "a cutting shouldn't be able to zero out the parent's own height");
        assert!(plant.growth_shock > 0.0, "expected taking a cutting to cost a temporary setback, same as pruning");
    }

    #[test]
    fn taking_a_cutting_below_the_minimum_height_is_a_no_op() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.cutting_min_height * 0.5,
            ..Plant::new()
        };
        assert!(!plant.take_cutting(&config.plant));
    }

    #[test]
    fn taking_a_cutting_never_happens_once_dead() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Dead,
            height: config.plant.cutting_min_height + 1.0,
            ..Plant::new()
        };
        assert!(!plant.take_cutting(&config.plant));
    }

    #[test]
    fn from_cutting_at_full_vigor_produces_a_small_fresh_rooting_specimen() {
        let config = config();
        let fresh = Plant::from_cutting(&config.plant, 1.0);
        assert_eq!(fresh.stage, Stage::Rooting);
        assert_eq!(fresh.height, config.plant.cutting_start_height);
        assert_eq!(fresh.carbon_pool, config.plant.cutting_start_carbon);
        assert_eq!(fresh.leaves.len(), config.plant.cutting_start_leaves);
        assert_eq!(
            fresh.stem_radius, 0.0,
            "expected a fresh cutting to start with no accumulated stem thickness"
        );
        assert_eq!(
            fresh.max_height_reached, config.plant.cutting_start_height,
            "expected the propagated plant's own high-water mark to start from its own starting height, not zero"
        );
        assert_eq!(fresh.max_leaves_at_once, config.plant.cutting_start_leaves as u32);
        assert_eq!(
            fresh.leaves_produced_total, config.plant.cutting_start_leaves as u32,
            "expected the propagated plant's own lifetime counter to start from its own starting leaves, not the parent's"
        );
    }

    #[test]
    fn from_cutting_is_structurally_independent_of_whatever_parent_took_the_cutting() {
        let config = config();
        let mut parent = Plant {
            stage: Stage::Vegetative,
            height: config.plant.cutting_min_height + 5.0,
            leaves: vec![mature_leaf(Side::Left); 5],
            ..Plant::new()
        };
        assert!(parent.take_cutting(&config.plant));
        let propagated = Plant::from_cutting(&config.plant, parent.root_health);
        assert_ne!(
            propagated.height, parent.height,
            "the propagated plant should start fresh, not mirror whatever the parent looks like after the cut"
        );
        assert_eq!(propagated.height, config.plant.cutting_start_height);
    }

    #[test]
    fn from_cutting_scales_starting_reserves_with_parent_vigor() {
        let config = config();
        let weak = Plant::from_cutting(&config.plant, 0.5);
        let strong = Plant::from_cutting(&config.plant, 1.0);
        assert!(
            weak.carbon_pool < strong.carbon_pool,
            "a cutting off a half-healthy parent should start with less stored carbon, got {} vs {}",
            weak.carbon_pool,
            strong.carbon_pool
        );
        assert!(weak.leaves.len() >= 1, "even a weak cutting keeps at least one starter leaf");
    }

    #[test]
    fn basal_rosette_species_can_never_take_a_cutting() {
        let plant_config = PlantConfig::peace_lily();
        let plant = Plant {
            stage: Stage::Vegetative,
            height: plant_config.cutting_min_height + 5.0,
            leaves: vec![mature_leaf(Side::Left); 5],
            ..Plant::new()
        };
        assert!(!plant.is_propagatable(&plant_config));
    }

    #[test]
    fn vine_species_cannot_take_a_leafless_cutting() {
        let plant_config = PlantConfig::pothos();
        let plant = Plant {
            stage: Stage::Vegetative,
            height: plant_config.cutting_min_height + 5.0,
            leaves: vec![],
            ..Plant::new()
        };
        assert!(!plant.is_propagatable(&plant_config));
    }

    #[test]
    fn upright_cane_species_can_take_a_leafless_cutting() {
        let plant_config = PlantConfig::dracaena();
        let plant = Plant {
            stage: Stage::Vegetative,
            height: plant_config.cutting_min_height + 5.0,
            leaves: vec![],
            ..Plant::new()
        };
        assert!(plant.is_propagatable(&plant_config));
    }

    #[test]
    fn taking_a_cutting_from_a_root_damaged_parent_is_refused() {
        let config = config();
        let plant = Plant {
            stage: Stage::Vegetative,
            height: config.plant.cutting_min_height + 5.0,
            leaves: vec![mature_leaf(Side::Left); 5],
            root_health: config.plant.cutting_min_root_health * 0.5,
            ..Plant::new()
        };
        assert!(!plant.is_propagatable(&config.plant));
    }

    #[test]
    fn a_rooting_cutting_does_not_elongate_or_gain_leaves_while_establishing() {
        let config = config();
        let mut plant = Plant::from_cutting(&config.plant, 1.0);
        let height_before = plant.height;
        let leaves_before = plant.leaves.len();
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..100 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.moisture = 1.0;
        }
        assert_eq!(plant.stage, Stage::Rooting, "expected 100 physiology-seconds to still be within the arcade rooting duration");
        assert_eq!(plant.height, height_before, "a rooting cutting shouldn't elongate");
        assert_eq!(plant.leaves.len(), leaves_before, "a rooting cutting shouldn't gain new leaves");
    }

    #[test]
    fn a_rooting_cutting_transitions_to_vegetative_once_the_arcade_duration_elapses() {
        let config = config();
        let mut plant = Plant::from_cutting(&config.plant, 1.0);
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..(config.plant.rooting_duration as u32 + 10) {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.moisture = 1.0;
        }
        assert_eq!(plant.stage, Stage::Vegetative);
    }

    #[test]
    fn realistic_rooting_takes_far_longer_than_arcade_rooting() {
        let mut plant_config = PlantConfig::dracaena();
        plant_config.rooting_duration = 400.0;
        plant_config.rooting_duration_realistic = 16800.0;
        let mut plant = Plant::from_cutting(&plant_config, 1.0).with_realistic_scale(true);
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let config = GrowthConfig { plant: plant_config, ..GrowthConfig::default() };
        let sun = noon(&config);
        for _ in 0..1000 {
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 1.0, &config);
            soil.moisture = 1.0;
        }
        assert_eq!(plant.stage, Stage::Rooting, "expected realistic rooting to still be in progress after 1000s");
    }

    #[test]
    fn rooting_progress_reports_full_strength_once_not_rooting() {
        let config = config();
        let plant = Plant::new();
        assert_eq!(plant.stage, Stage::Seed);
        assert_eq!(plant.rooting_progress(&config.plant), 1.0);
    }

    #[test]
    fn a_young_rosette_is_not_yet_dividable() {
        let plant_config = PlantConfig::peace_lily();
        let plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left); plant_config.division_min_leaves - 1],
            ..Plant::new()
        };
        assert!(!plant.is_dividable(&plant_config));
    }

    #[test]
    fn caning_and_vining_species_can_never_be_divided() {
        let dracaena = PlantConfig::dracaena();
        let pothos = PlantConfig::pothos();
        let plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left); 20],
            ..Plant::new()
        };
        assert!(!plant.is_dividable(&dracaena));
        assert!(!plant.is_dividable(&pothos));
    }

    #[test]
    fn dividing_a_rosette_splits_leaves_and_carbon_between_parent_and_offshoot() {
        let plant_config = PlantConfig::peace_lily();
        let mut parent = Plant {
            stage: Stage::Vegetative,
            height: 0.1,
            carbon_pool: 10.0,
            leaves: vec![mature_leaf(Side::Left); 6],
            ..Plant::new()
        };
        let leaves_before = parent.leaves.len();
        let carbon_before = parent.carbon_pool;
        let offshoot = parent.divide(&plant_config).expect("expected a mature rosette to be dividable");
        assert_eq!(
            parent.leaves.len() + offshoot.leaves.len(),
            leaves_before,
            "every leaf should land on exactly one of the two plants, none lost or duplicated"
        );
        assert!(!offshoot.leaves.is_empty(), "the offshoot should keep some of the real, already-established leaves");
        assert!(
            (parent.carbon_pool + offshoot.carbon_pool - carbon_before).abs() < 1e-9,
            "carbon should be split, not created or destroyed"
        );
        assert!(offshoot.carbon_pool > 0.0);
    }

    #[test]
    fn dividing_produces_an_already_vegetative_offshoot_not_a_rooting_one() {
        let plant_config = PlantConfig::peace_lily();
        let mut parent = Plant {
            stage: Stage::Vegetative,
            height: 0.1,
            carbon_pool: 10.0,
            leaves: vec![mature_leaf(Side::Left); 6],
            ..Plant::new()
        };
        let offshoot = parent.divide(&plant_config).expect("expected a mature rosette to be dividable");
        assert_eq!(
            offshoot.stage,
            Stage::Vegetative,
            "a divided section already has its own roots, unlike a stem cutting"
        );
    }

    #[test]
    fn dividing_below_the_leaf_threshold_is_a_no_op() {
        let plant_config = PlantConfig::peace_lily();
        let mut parent = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left); plant_config.division_min_leaves - 1],
            ..Plant::new()
        };
        assert!(parent.divide(&plant_config).is_none());
    }

    // --- Dormancy ------------------------------------------------------------

    #[test]
    fn winter_dormancy_slows_elongation_relative_to_midsummer() {
        let config = config();
        let mut summer = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut winter = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            total_time: config.season.season_length_sim_seconds / 2.0,
            ..Plant::new()
        };
        let mut soil_a = Soil { moisture: 1.0, ..Default::default() };
        let mut soil_b = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..500 {
            summer.step(1.0, &sun, &neutral_climate(), &mut soil_a, 1.0, &config);
            winter.step(1.0, &sun, &neutral_climate(), &mut soil_b, 1.0, &config);
        }
        assert!(
            winter.height < summer.height,
            "expected winter dormancy to slow elongation: winter {} vs summer {}",
            winter.height,
            summer.height
        );
    }

    // --- Pests ---------------------------------------------------------------

    #[test]
    fn pest_infestation_grows_in_dry_air_and_taxes_photosynthesis() {
        let config = config();
        let mut plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut soil = Soil { moisture: 1.0, ..Default::default() };
        let sun = noon(&config);
        for _ in 0..3000 {
            // Bone-dry air — the worst case for pest pressure.
            plant.step(1.0, &sun, &neutral_climate(), &mut soil, 0.0, &config);
            soil.water(0.01);
        }
        assert!(
            plant.pest_infestation > 0.0,
            "expected sustained dry air to grow a pest infestation, got {}",
            plant.pest_infestation
        );
    }

    #[test]
    fn treating_pests_knocks_infestation_down() {
        let config = config();
        let mut plant = Plant { pest_infestation: 0.8, ..Plant::new() };
        plant.treat_pests(&config.pest);
        assert!(plant.pest_infestation < 0.8);
    }

    // --- Humidity / VPD --------------------------------------------------------

    #[test]
    fn hot_dry_air_draws_down_soil_moisture_faster_than_hot_humid_air() {
        let config = config();
        let mut dry_plant = Plant {
            stage: Stage::Vegetative,
            leaves: vec![mature_leaf(Side::Left)],
            ..Plant::new()
        };
        let mut humid_plant = dry_plant.clone();
        let mut dry_soil = Soil { moisture: 1.0, ..Default::default() };
        let mut humid_soil = Soil { moisture: 1.0, ..Default::default() };
        let hot_climate = ClimateState {
            temperature_c: config.humidity.vpd_reference_temperature_c + 15.0,
        };
        let sun = noon(&config);
        for _ in 0..500 {
            dry_plant.step(1.0, &sun, &hot_climate, &mut dry_soil, 0.05, &config);
            humid_plant.step(1.0, &sun, &hot_climate, &mut humid_soil, 0.95, &config);
        }
        assert!(
            dry_soil.moisture < humid_soil.moisture,
            "expected hot dry air to draw down soil moisture faster than hot humid air: dry {} vs humid {}",
            dry_soil.moisture,
            humid_soil.moisture
        );
    }
}
