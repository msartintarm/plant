//! All tunable numbers for the simulation, gathered as plain data separate
//! from the logic that uses them (`sun.rs`, `soil.rs`, `plant.rs`) — those
//! modules take a config as a parameter and stay pure functions/methods
//! over it, rather than reading scattered module-level constants baked into
//! their own bodies. Each field's rationale lives here, next to the number,
//! not spread across call sites.

/// Where the sun sits in the sky and what it looks like, as a function of
/// `day_progress` — see `sun::sun_state`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunConfig {
    /// Sunrise/sunset are placed a quarter cycle to either side of
    /// midnight, so solar noon lands exactly at `day_progress == 0.5`.
    pub sunrise: f64,
    pub sunset: f64,
    /// 0.0-1.0-per-channel tint at the horizon (dawn/dusk).
    pub dawn_color: [f32; 3],
    /// 0.0-1.0-per-channel tint at solar noon.
    pub noon_color: [f32; 3],
    /// How far below the horizon (in `elevation` units, -1..0) twilight
    /// persists — real dusk/dawn fade gradually via atmospheric scattering
    /// rather than cutting to black the instant the sun crosses the
    /// horizon.
    pub twilight_depth: f64,
    /// Residual light at the horizon itself (elevation == 0), decaying to 0
    /// by `twilight_depth` below it — dimmer than daylight, never as bright
    /// as noon.
    pub twilight_intensity: f64,
}

impl Default for SunConfig {
    fn default() -> Self {
        SunConfig {
            sunrise: 0.25,
            sunset: 0.75,
            dawn_color: [1.0, 0.55, 0.30],
            noon_color: [1.0, 1.0, 0.95],
            twilight_depth: 0.1,
            twilight_intensity: 0.15,
        }
    }
}

/// The moon's current appearance — see `moon::current_phase`/`appearance`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonConfig {
    /// 0.0..1.0 — where the real lunar cycle actually is right now (see
    /// `moon::phase_for_date`), used as the starting point for a fresh
    /// session. Illuminated fraction is the same everywhere; only rise/set
    /// timing (`moon::arc_position`) and crescent tilt (`observer_latitude_
    /// degrees`) depend on location.
    pub initial_phase: f64,
    /// Sim-seconds for one full 29.53-day synodic month, applied to the
    /// game's own (already-compressed) day unit — this engine has no live
    /// connection to the real calendar once a session starts, so
    /// "realistic" here means the correct *ratio* to the game's own day
    /// length, not an ongoing real-time sync.
    pub cycle_length_sim_seconds: f64,
    /// How much light (same 0.0..1.0 scale as `sun::SunState::intensity`) a
    /// *full* moon adds on top of the sun's own — see `moon::
    /// apply_moonlight`. Scales down toward 0 as the moon wanes toward new
    /// — real moonlight is astronomically far dimmer than this (reflected
    /// sunlight, not a light source of its own), but a value that small
    /// would be a no-op for gameplay purposes; this is tuned to be a real,
    /// noticeable-but-modest assist on a bright moonlit night, not a
    /// substitute for actual daylight.
    pub max_light_contribution: f64,
    /// Fixed observer latitude (degrees, +north/-south) for crescent tilt
    /// (`moon::crescent_tilt_angle`) — no live geolocation exists, so this
    /// defaults to San Francisco, matching `EngineCanvas.tsx`'s
    /// display-only `SEED_LOCATION`.
    pub observer_latitude_degrees: f64,
}

impl Default for MoonConfig {
    fn default() -> Self {
        MoonConfig {
            // A fixed fallback date, only actually reached by native tests
            // and anything else constructing `GrowthConfig::default()`
            // directly — the real wasm app always overrides this with
            // `moon::phase_for_date` grounded in the session's *actual*
            // start date instead (see `Simulation::create`'s own doc
            // comment for why the engine can't read "today" on its own).
            initial_phase: super::moon::phase_for_date(2026, 7, 20),
            cycle_length_sim_seconds: 29.530588853 * TimeConfig::default().day_length_sim_seconds,
            max_light_contribution: 0.05,
            observer_latitude_degrees: 37.7749,
        }
    }
}

/// Soil water-reservoir behavior — see `soil::Soil`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilConfig {
    /// Fraction of field capacity lost per second to surface evaporation at
    /// full light — scaled down toward night. Chosen so a well-watered pot
    /// dries out over several simulated days, not minutes.
    pub evaporation_rate_per_sec: f64,
    /// Below this fraction of field capacity, stomata increasingly close to
    /// conserve water — `water_factor` ramps linearly from 1.0 at this
    /// threshold down to 0.0 at bone dry.
    pub moisture_gate_threshold: f64,
    /// What a freshly potted plant starts at.
    pub initial_moisture: f64,
    /// The moisture floor a "self-watering" (wicking/reservoir) pot
    /// maintains — see `Soil::apply_auto_water`. Comfortably above
    /// `moisture_gate_threshold` (never stress-gated) but not saturated:
    /// real wicking planters keep soil evenly moist, not sopping wet.
    pub auto_water_floor: f64,

    // --- Overwatering / root rot (see `Soil::waterlog_stress`, `Plant::
    // root_health`) — waterlogged soil excludes oxygen from roots, which
    // suffocate and die back, distinct from (and the opposite failure
    // direction of) ordinary drought stress. ---
    /// Fraction of field capacity at/above which soil is oxygen-starved for
    /// roots — deliberately close to 1.0 (true field capacity itself isn't
    /// anoxic; only soil kept continuously re-saturated past it is), so a
    /// single watering dose that's then allowed to drain/evaporate normally
    /// never triggers this, only a pot kept artificially flooded (the
    /// player re-watering far more often than the plant can ever draw down)
    /// does.
    pub waterlogged_threshold: f64,
    /// How long (sim-seconds) soil has to stay continuously at/above
    /// `waterlogged_threshold` before root damage actually begins — real
    /// roots tolerate brief flooding (a heavy single dose draining down)
    /// without harm; only sustained anoxia actually suffocates them.
    pub waterlog_grace_period: f64,

    // --- Fertilizing (see `Soil::nutrient_factor`/`overfeed_stress`) — a
    // second Liebig-style resource alongside water, with its own two-sided
    // failure mode (starvation *and* overdose, like water's drought/root-rot
    // pair). ---
    /// What a typical fresh potting mix starts with — deliberately generous
    /// and slow-depleting (see `nutrient_uptake_coeff`/`nutrient_leach_rate_
    /// per_sec`) so an unfertilized plant isn't nutrient-starved almost
    /// immediately; fertilizing is a lever for sustaining *long* growth, not
    /// a requirement from the very first tick.
    pub initial_nutrient: f64,
    /// Below this, growth increasingly nitrogen/mineral-starves — same
    /// linear-ramp shape as `moisture_gate_threshold`.
    pub nutrient_gate_threshold: f64,
    /// Fraction of standing nutrient drawn down per unit of the plant's own
    /// water uptake — real nutrient uptake rides along with transpiration
    /// (mass flow), so it's driven by the same `uptake_rate` water usage
    /// already computes rather than tracked completely independently.
    pub nutrient_uptake_coeff: f64,
    /// Slow background depletion (leaching/microbial fixation) independent
    /// of plant uptake — real soil loses some nutrient content over time
    /// even from an unplanted pot.
    pub nutrient_leach_rate_per_sec: f64,
    /// Above this, exercise osmotic "fertilizer burn" — over-fertilizing
    /// raises soil salinity high enough to actively damage roots, the same
    /// real failure mode as overwatering (see `waterlogged_threshold`), just
    /// from the opposite direction (too much of a good thing, not too
    /// little). No matching hard ceiling the way water has field capacity —
    /// real fertilizer salts keep building up the more you add, which is
    /// exactly why overdosing is a real, easy mistake.
    pub overfeed_threshold: f64,
    /// A soft ceiling `Soil::fertilize` clamps to — not a hard physical
    /// limit like water's field capacity, just a sane bound so a
    /// misclicked/spammed fertilize action can't send the overfeed-stress
    /// ramp to an arbitrarily extreme value.
    pub max_nutrient: f64,
}

impl Default for SoilConfig {
    fn default() -> Self {
        SoilConfig {
            evaporation_rate_per_sec: 0.0003,
            moisture_gate_threshold: 0.35,
            initial_moisture: 0.75,
            auto_water_floor: 0.5,

            waterlogged_threshold: 0.97,
            waterlog_grace_period: 300.0,

            initial_nutrient: 1.0,
            nutrient_gate_threshold: 0.1,
            nutrient_uptake_coeff: 0.02,
            nutrient_leach_rate_per_sec: 0.000015,
            overfeed_threshold: 1.4,
            max_nutrient: 2.0,
        }
    }
}

/// Ambient temperature behavior — see `climate::climate_state`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClimateConfig {
    /// A comfortable indoor baseline — real rooms swing much less than
    /// outdoor air (`day_night_swing_c` below), but they aren't perfectly
    /// constant either.
    pub base_temperature_c: f64,
    /// How far above/below `base_temperature_c` the day/night cycle swings
    /// — modest on purpose (an indoor room, not an outdoor garden).
    pub day_night_swing_c: f64,
    /// How far past solar noon (as a fraction of a full day) peak
    /// temperature lags peak sunlight — real air/thermal mass takes time to
    /// heat up, so the warmest part of the day is mid-afternoon, not noon.
    pub temperature_peak_offset: f64,
}

impl Default for ClimateConfig {
    fn default() -> Self {
        ClimateConfig {
            base_temperature_c: 21.0,
            day_night_swing_c: 3.0,
            temperature_peak_offset: 0.1,
        }
    }
}

/// Where the pot sits relative to the window — a player-controlled tradeoff
/// distinct from `PlantConfig::window_light_zone_height`'s *vertical*
/// falloff (how tall the plant itself has grown): this is *horizontal*
/// placement, chosen once by the player rather than driven by growth. See
/// `room::apply_pot_position`. Real windowsills are the brightest spot in a
/// room but also the draftiest/coldest at night (single-pane glass radiates
/// heat away fast) — moving a plant back trades light for a steadier,
/// warmer microclimate, a real, common houseplant-placement decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoomConfig {
    /// Light multiplier at `position == 1.0` (as far from the window as this
    /// game models) — `1.0` at the window itself (`position == 0.0`), easing
    /// down to this floor. Distinct from (and multiplied together with)
    /// `PlantConfig::ambient_light_floor`.
    pub window_light_floor: f64,
    /// Maximum cold-draft penalty (°C) applied at `position == 0.0` (right
    /// at the glass), fading to zero by `position == 1.0`.
    pub window_draft_cold_c: f64,
}

impl Default for RoomConfig {
    fn default() -> Self {
        RoomConfig {
            window_light_floor: 0.35,
            window_draft_cold_c: 4.0,
        }
    }
}

/// Ambient air humidity — a separate reservoir from soil moisture (see
/// `humidity::Humidity`), representing dry indoor air (especially from
/// heating/AC) rather than the pot's own water content. Drives real vapor-
/// pressure-deficit-driven transpiration (hot *and* dry air pulls
/// dramatically more water out of leaves than either factor alone) and pest
/// pressure (spider mites specifically thrive in dry air) — see
/// `PestConfig`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumidityConfig {
    pub initial_humidity: f64,
    /// What indoor air settles toward absent misting — kept at/above
    /// `PestConfig::safe_humidity` so pests can't ratchet up from mere
    /// passive neglect (same principle as root rot needing active
    /// overwatering to trigger at all).
    pub dry_air_floor: f64,
    /// How fast humidity eases toward `dry_air_floor`, scaled up by heat
    /// (see `Humidity::update`) — real warm air holds (and loses) more
    /// moisture capacity than cool air.
    pub decay_rate_per_sec: f64,
    /// Reference temperature used only to scale the rate at which misted
    /// room air returns toward `dry_air_floor` (see `Humidity::update`).
    /// It is deliberately separate from VPD: dry air produces a vapor-
    /// pressure deficit at ordinary room temperatures too.
    pub vpd_reference_temperature_c: f64,
    /// VPD (kPa) at which `vpd_strength` applies. Around 1 kPa is a useful
    /// indoor baseline for a well-watered tropical houseplant.
    pub vpd_reference_kpa: f64,
    /// Additional transpiration multiplier at `vpd_reference_kpa`; scales
    /// linearly with the physically calculated vapor-pressure deficit. See
    /// `Humidity::vpd_factor`.
    pub vpd_strength: f64,
    /// VPD (kPa) at which stomata have completed half of their configurable
    /// closure response. This is a species-independent interim heuristic;
    /// species traits can replace it with calibrated curves later.
    pub stomatal_vpd_closure_kpa: f64,
    /// Residual fraction of a well-watered leaf's conductance at extreme
    /// VPD. Root-water limitation can still reduce conductance all the way
    /// to zero; this floor only prevents atmospheric closure from becoming
    /// an instantaneous hard switch.
    pub stomatal_min_conductance: f64,
}

impl Default for HumidityConfig {
    fn default() -> Self {
        HumidityConfig {
            initial_humidity: 0.5,
            dry_air_floor: 0.5,
            decay_rate_per_sec: 0.0004,
            vpd_reference_temperature_c: 24.0,
            vpd_reference_kpa: 1.0,
            vpd_strength: 0.15,
            // Deliberately broad until species-specific response curves are
            // calibrated: normal indoor VPD should trim conductance without
            // making the accelerated demo's baseline plant non-viable.
            stomatal_vpd_closure_kpa: 8.0,
            stomatal_min_conductance: 0.1,
        }
    }
}

/// Pest pressure (modeled on spider mites, the classic dry-indoor-air
/// houseplant pest) — see `pests::pest_growth_rate`. A threat orthogonal to
/// the water/light/nutrient economy, so a plant can't be kept alive purely
/// by "getting the resource dials right" once and walking away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PestConfig {
    /// Humidity at/above which pests never take hold at all — real spider
    /// mites specifically struggle in humid air.
    pub safe_humidity: f64,
    /// Growth rate per second at `humidity == 0.0`, scaling linearly to zero
    /// at `safe_humidity` — see `pests::pest_growth_rate`.
    pub growth_rate: f64,
    /// How much a full (1.0) infestation cuts photosynthesis — sap-sucking
    /// pests are a direct carbon-income tax, not just a stress signal.
    pub photosynthesis_penalty: f64,
    /// How much a full infestation multiplies leaf senescence, same idiom as
    /// `PlantConfig::leaf_stress_senescence_multiplier`.
    pub senescence_multiplier: f64,
    /// How much a single treatment action knocks infestation down by.
    pub treatment_reduction: f64,
}

impl Default for PestConfig {
    fn default() -> Self {
        PestConfig {
            safe_humidity: 0.5,
            growth_rate: 0.00015,
            photosynthesis_penalty: 0.6,
            senescence_multiplier: 2.5,
            treatment_reduction: 0.6,
        }
    }
}

/// A slow year-length cycle layered on top of the fast day/night cycle —
/// see `season::season_state`. Real houseplants (even indoors, insulated
/// from outdoor temperature swings) slow their growth in winter specifically
/// because of shorter days, a photoperiod response independent of
/// temperature — modeled the same "pure function of elapsed time" way
/// `climate::climate_state` is, for the same reason (see that module's doc
/// comment): no extra persistent state to thread through every call site
/// beyond the single `Plant::total_time` accumulator already needed to
/// evaluate it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeasonConfig {
    /// Sim-seconds for one full summer-to-summer cycle — deliberately long
    /// relative to a single day (itself `TimeConfig::day_length_sim_
    /// seconds`), so a season change reads as a real seasonal drift, not a
    /// day-to-day wobble.
    pub season_length_sim_seconds: f64,
    /// The lowest `day_length_factor` drops to at winter's peak — growth
    /// slows markedly but doesn't fully stop even in deep winter, matching
    /// how a real dormant houseplant still ticks over slowly rather than
    /// freezing solid.
    pub winter_floor: f64,
}

impl Default for SeasonConfig {
    fn default() -> Self {
        SeasonConfig {
            // Long enough that this demo's own fast validation pacing (a
            // handful of real minutes) barely samples a sliver of a full
            // year — see `season::season_state`'s tests for how the
            // dormancy effect itself is actually exercised (constructing a
            // `Plant` with `total_time` preset deep into winter directly,
            // not by running a multi-hour session).
            season_length_sim_seconds: 500_000.0,
            winter_floor: 0.4,
        }
    }
}

/// A species' above-ground silhouette — explicit rather than inferred from
/// `max_branches`/`trellis_height`. Drives leaf/stem placement in
/// `render::scene`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthHabit {
    /// Single elongating stem, optional crown branches — Dracaena.
    UprightCane,
    /// Leaves fan from a stemless crown at soil level — Peace Lily.
    BasalRosette,
    /// Climbs a support via aerial roots — Pothos.
    Vine,
}

/// Plant growth-model rates — see `plant::Plant`. Grouped by the same
/// mechanism headings used in `plant.rs`'s module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlantConfig {
    // --- Germination ---
    /// Real seeds won't imbibe and germinate in dry soil — requires water
    /// availability above this fraction of the stomatal-gating threshold.
    pub germination_water_factor: f64,
    /// Real seeds also won't germinate in cold soil, independent of how
    /// moist it is — set comfortably below any temperature a heated indoor
    /// room actually reaches (`ClimateConfig::base_temperature_c` minus its
    /// swing), so this is a real, testable gate that basically never
    /// actually blocks a houseplant's germination in practice, the same way
    /// it wouldn't for a plant that's never left sitting somewhere actually
    /// cold.
    pub germination_min_temperature_c: f64,
    /// Hypocotyl elongation rate while living off stored seed reserves —
    /// independent of light (a seedling underground has none to use yet).
    pub sprout_growth_rate: f64,
    /// Height at which the hypocotyl has "broken the surface" and
    /// cotyledons unfurl, entering `Stage::Vegetative`.
    pub sprout_height_threshold: f64,

    // --- Temperature response (see `climate::temperature_factor`/
    // `climate::q10_factor`) ---
    /// The temperature photosynthesis/elongation run best at — real
    /// enzyme-driven reactions have an optimum, not "faster is always
    /// better."
    pub optimal_temperature_c: f64,
    /// How wide that optimum's tolerance is — see `climate::
    /// temperature_factor`'s doc comment for the exact curve.
    pub temperature_tolerance_c: f64,
    /// Respiration's Q10 coefficient — canonically ~2 for plants (roughly
    /// doubles per 10°C rise), and unlike photosynthesis keeps climbing
    /// with heat rather than falling off past an optimum.
    pub respiration_q10: f64,
    /// The temperature `respiration_rate` itself was tuned assuming —
    /// `climate::q10_factor` is 1.0 (no adjustment) exactly here.
    pub respiration_reference_temperature_c: f64,
    /// Below this, cold stress accelerates leaf senescence (see
    /// `Leaf::senescence`) on top of whatever drought stress is doing —
    /// real tropical/subtropical houseplants often drop leaves after a
    /// cold snap (a draft, a forgotten windowsill in winter) independent of
    /// watering.
    pub cold_stress_threshold_c: f64,

    // --- Photosynthesis / carbon ---
    /// Photosynthetic yield per unit leaf area per unit light intensity.
    pub light_use_efficiency: f64,
    /// Cotyledons are photosynthetic organs themselves (in epigeal
    /// germination) — gives the seedling *some* carbon income immediately
    /// on sprouting, before its first true leaf exists.
    pub cotyledon_leaf_area: f64,
    /// Leaf area a single fully mature true leaf contributes.
    pub leaf_area_per_leaf: f64,
    // --- Light availability by height (spatial, distinct from the
    // day/night intensity cycle in sun.rs) — see `height_light_factor` ---
    /// Height up to which a plant gets full window light, regardless of
    /// time of day — real light doesn't just vary by time, it falls off
    /// with distance from the source, so a plant that's grown taller than
    /// its own window increasingly pushes new growth into a dim stretch of
    /// wall with no window at all. Deliberately kept as its own number
    /// here rather than derived from `render::config::SceneLayout`'s
    /// window geometry (`sim` stays independent of `render` everywhere
    /// else too) — tune by hand to visually match if the window's
    /// position/size ever changes.
    pub window_light_zone_height: f64,
    /// How much further above `window_light_zone_height` light keeps
    /// fading before bottoming out at `ambient_light_floor`.
    pub window_light_falloff_range: f64,
    /// This species' realistic mature height/branch-length — only enforced
    /// when `Plant::realistic_scale` is on (see `realistic_scale_taper`);
    /// with it off, growth stays today's default unbounded "gigantic
    /// plant" behavior. A real houseplant's practical ceiling, not a hard
    /// biological one — tuned per species (a trained Pothos vine trails
    /// much longer than a Peace Lily rosette ever gets tall).
    pub realistic_max_height: f64,
    /// Never quite zero even far from the window — real rooms have some
    /// ambient light bouncing around.
    pub ambient_light_floor: f64,
    /// Continuous maintenance respiration cost, scaled by plant size — real
    /// plants burn stored sugar around the clock, not just when
    /// photosynthesis is running.
    pub respiration_rate: f64,

    // --- Elongation ---
    /// Base turgor-driven elongation rate at full water availability and
    /// full carbon affordability.
    pub base_elongation_rate: f64,
    /// Carbon spent per unit of elongation — elongation is carbon-*limited*
    /// (scales down smoothly as the pool runs low), not gated by a hard
    /// on/off reserve threshold.
    pub elongation_carbon_cost: f64,
    /// How strongly low light pushes carbon allocation toward elongation
    /// over thickening (shade avoidance) — 0 would mean no etiolation.
    pub shade_avoidance_strength: f64,

    // --- Secondary thickening (pipe model) ---
    /// Target stem radius per unit sqrt(leaf area) — the "pipe model"
    /// coefficient: cross-section scales with the leaf area it supplies.
    pub pipe_model_coeff: f64,
    /// How quickly actual radius eases toward its pipe-model target per
    /// unit of water actually drawn through the stem. Scaled up alongside
    /// `transpiration_coeff` below (they were rebalanced together): when
    /// `uptake_rate` shrinks, this needs to grow by the same factor to keep
    /// the *pacing* of thickening the same, since it multiplies
    /// `uptake_rate` directly.
    pub thickening_rate_coeff: f64,
    /// Transpiration (and so water uptake) per unit leaf area per unit
    /// light and water availability. Real evapotranspiration studies put a
    /// leafy potted plant's water use at roughly 2-4x bare soil's own
    /// evaporation once it's reasonably established — *not* the 15-40x this
    /// used to work out to once a plant had more than a couple of leaves
    /// (found by simulating a played-through session: soil crashed from
    /// full to bone dry within about a real minute of the fast validation
    /// demo, well before a crown could ever form — see the top-level
    /// README). Tuned so a plant with leaf area ~8 (a modest multi-branch
    /// specimen) draws roughly 2x `SoilConfig::evaporation_rate_per_sec`,
    /// not 20x.
    pub transpiration_coeff: f64,

    // --- Leaf initiation & maturation ---
    /// How much the stem has to elongate between successive leaf
    /// primordia — a plastochron, in height units rather than real time
    /// (real shoot apical meristems produce nodes roughly per unit of
    /// extension, not on a wall-clock timer). Leaf initiation used to be
    /// purely carbon-cost-gated and would get starved out indefinitely by
    /// elongation's own (much cheaper, unconditional) claim on the same
    /// carbon pool once the stem passed `min_height_for_branching` and
    /// leaf-spawning was paused in favor of funding a branch — in practice
    /// that meant a stem could grow many multiples of its own branching
    /// height with only the one leaf it had before crossing the threshold,
    /// which read as "leaves only ever grow at the base." Gating leaf
    /// initiation by height-since-last-leaf instead (still carbon-limited,
    /// just no longer *exclusively* gated by racing a competing sink)
    /// guarantees new nodes keep appearing as the stem actually grows,
    /// independent of whether a branch happens to be mid-funding.
    pub plastochron_height_interval: f64,
    /// Carbon reserve required to initiate a new leaf primordium — a leaf
    /// only appears once both the height interval above has elapsed *and*
    /// this much sugar is banked, so a starved plant still delays new
    /// organs even though the interval itself is time/height-based.
    pub new_leaf_carbon_cost: f64,
    pub leaf_maturation_rate: f64,
    pub droop_response_rate: f64,
    pub max_carbon_pool: f64,

    // --- Leaf senescence & abscission (see `Leaf::age`/`Leaf::senescence`)
    // --- a leaf's life doesn't end at full maturity: real leaves age,
    // yellow, and eventually drop, faster under stress than in good
    // conditions. ---
    /// How long (sim-seconds) a leaf stays fully healthy after spawning
    /// before senescence begins ramping at all — comfortably longer than
    /// the time it actually takes to reach full `maturity`, so this reads
    /// as "old age," not "still unfurling."
    pub leaf_mature_lifespan: f64,
    /// Baseline rate senescence eases toward 1.0 once past
    /// `leaf_mature_lifespan`, absent any stress.
    pub leaf_senescence_rate: f64,
    /// How much drought or cold stress (whichever is worse this tick — see
    /// `step_vegetative`) multiplies the senescence rate above — real
    /// plants shed leaves faster under sustained stress specifically to cut
    /// their own transpiration/maintenance load, not just from passive
    /// aging.
    pub leaf_stress_senescence_multiplier: f64,
    /// Once senescence eases past this, the leaf abscises (is removed from
    /// the plant) — just under 1.0 since easing only asymptotically
    /// approaches its target and would otherwise never technically arrive.
    pub leaf_abscission_senescence_threshold: f64,

    // --- Wilting: stem/branch gravity droop (distinct from `Leaf::droop`,
    // which only tips leaf blades — see `plant.rs`'s module docs) ---
    /// A fully turgid-to-fully-collapsed stem's maximum physical sag under
    /// its own weight once hydrostatic (turgor) support is lost — real soft
    /// tissue stems visibly flop over when badly wilted, not just their
    /// leaves.
    pub stem_droop_max_angle: f64,
    /// How fast actual stem droop eases toward its water-stress target —
    /// slower than `droop_response_rate` (leaf lamina turgor loss is faster
    /// and more localized than a whole stem losing rigidity).
    pub stem_droop_response_rate: f64,
    /// Flexural stiffness grows sharply with stem radius in real stems
    /// (bending resistance scales with a high power of thickness) — this is
    /// the radius at which a stem droops at roughly half its bare-seedling
    /// rate, so a young thin stem flops dramatically while an established
    /// thick one barely bends, for the same water stress.
    pub stem_droop_reference_radius: f64,

    // --- Crown branching (modeled on Dracaena's habit of forking near its
    // growing tip once mature — see `plant.rs`'s module docs) ---
    /// A whole new growing point costs more banked carbon than a single
    /// leaf primordium — higher than `new_leaf_carbon_cost` on purpose.
    pub new_branch_carbon_cost: f64,
    /// Real crown branching only shows up once a stem has matured past a
    /// certain height — a two-leaf seedling doesn't branch yet.
    pub min_height_for_branching: f64,
    /// A believable crown is a handful of branches, not dozens — once
    /// reached, all further carbon goes to growing the existing structure
    /// rather than starting new branches.
    pub max_branches: usize,
    /// Lateral branches grow slower than the main leader (apical dominance
    /// is *reduced*, not fully gone) — this multiplies
    /// `base_elongation_rate` for branch growth specifically.
    pub branch_elongation_rate_factor: f64,

    /// See `GrowthHabit` — drives leaf/stem placement in `render/mod.rs`.
    pub growth_habit: GrowthHabit,
    /// Baked leaf mesh name — same per-frame repoint mechanism as
    /// `flower_mesh_name` below.
    pub leaf_mesh_name: &'static str,

    // --- Flowering (purely cosmetic — a terminal bloom that cycles open
    // and closed once the plant is mature enough, see `Plant::
    // bloom_intensity`/`render::scene::flower_transform` — doesn't feed
    // back into growth) ---
    /// Main-stem height at which the plant is considered mature enough to
    /// bloom *at all*. For a crown-branching habit this is tuned close to
    /// `min_height_for_branching` (real crown release is often triggered by
    /// — or coincides with — a terminal flower suppressing the tip's own
    /// apical dominance, per the module docs); for a basal-rosette habit
    /// (which never branches) it's simply "big enough rosette to flower."
    /// Reaching this height doesn't mean *permanently* in bloom — see
    /// `bloom_duration`/`bloom_rest_duration` for the actual cycle.
    pub flowering_height_threshold: f64,
    /// Which baked mesh (see `assets/svg/README.md`) this species' bloom
    /// draws as — real flower structure differs enough between species
    /// (Dracaena's wispy star-flowered panicle vs. Peace Lily's showy
    /// spathe-and-spadix) that one generic "flower" shape doesn't read as
    /// either accurately. `render/mod.rs` repoints the shared flower
    /// drawable to this name every frame — cheap, since `Drawable::mesh` is
    /// just a lookup key, not an owned resource.
    pub flower_mesh_name: &'static str,
    /// Sim-seconds the bloom stays open per cycle, once mature enough —
    /// see `bloom_rest_duration` for the interval between cycles. Real
    /// flowering plants don't stay in permanent bloom; they flush and then
    /// rest.
    pub bloom_duration: f64,
    /// Sim-seconds between the end of one bloom and the start of the
    /// next. Tuned per species to reflect real flowering frequency:
    /// Dracaena's famous bloom is rare enough to be a notable event when it
    /// happens, while an indoor Peace Lily commonly reblooms several times
    /// a year — a much shorter rest relative to its own bloom length.
    pub bloom_rest_duration: f64,
    /// How fast `Plant::bloom_intensity` eases toward its current phase's
    /// target (1.0 while in-cycle-open, 0.0 while resting) — a real flower
    /// opens/closes gradually over some days, it doesn't snap, same idiom
    /// as every other eased target in this module (`droop`, `helio_angle`,
    /// `fold`, `stem_droop`).
    pub bloom_response_rate: f64,

    /// Beer-Lambert light-extinction coefficient (Monsi & Saeki 1953 —
    /// still the standard basis for canopy photosynthesis models) for how
    /// strongly a grower's own newer, higher leaves shade its older, lower
    /// ones — distinct from (and stacked on top of) `window_light_zone_
    /// height`'s falloff, which is about whether that grower's *neighborhood
    /// in the room* gets light at all, not how crowded its own foliage is.
    /// See `self_shading_factors`: a leaf's own light factor decays
    /// exponentially with how much of this same grower's leaf area sits
    /// above it (attached more recently, hence physically higher). This is
    /// the main check on unbounded leaf count — every additional leaf's
    /// marginal photosynthetic income shrinks as the canopy thickens, while
    /// its maintenance respiration cost doesn't, so carbon income plateaus
    /// well before leaf count would otherwise run away, and heavily
    /// overtopped old leaves senesce faster (see `age_and_senesce_leaves`)
    /// — a real plant sheds its shaded lower leaves rather than
    /// indefinitely accumulating them.
    pub leaf_self_shading_coeff: f64,

    // --- Movement: phototropism, heliotropism, nyctinasty ---
    /// How fast the stem's phototropic lean accumulates per unit light
    /// intensity — one-directional (built from new tissue, doesn't relax
    /// back when light drops).
    pub lean_rate: f64,
    /// Real stems don't bend indefinitely toward one side.
    pub max_lean_angle: f64,
    /// How strongly a leaf's fast, reversible reorientation responds to the
    /// sun's position — distinct from (and much weaker/faster-acting than)
    /// the stem's slow permanent lean.
    pub helio_strength: f64,
    pub helio_response_rate: f64,
    /// Nyctinasty ("sleep movement") fold/reopen rate — same turgor-motor
    /// mechanism as heliotropism, driven here directly by light intensity
    /// rather than a separate circadian clock.
    pub fold_response_rate: f64,
    /// How much a stem/branch has to elongate before its currently-forming
    /// segment freezes at whatever `lean_angle` it has *right now* and a
    /// new segment starts forming — see `plant::MAX_STEM_SEGMENTS` and the
    /// module docs on why a stem's history is recorded piecewise like this
    /// rather than rendered as one rigid rotation: real stem tissue keeps
    /// whatever curvature it had when it stiffened, it doesn't retroactively
    /// straighten (or bend further) just because the growing tip keeps
    /// leaning more as the plant ages.
    pub stem_segment_height_interval: f64,

    /// `Some(height)` for a climbing/vining habit grown against a support
    /// (a moss pole/stake — modeled on *Epipremnum* (Pothos), see
    /// `PlantConfig::pothos`): while a grower's own height (main stem's
    /// `Plant::height`, or a branch's `attach_height + height`) is still
    /// within reach of the support, phototropic lean simply doesn't
    /// accumulate (see `Plant::step_vegetative`/`step_branch`) — a real
    /// aerial-root climber like Pothos is mechanically held flat against
    /// the support (see `aerial_root_height_interval`), not actively
    /// bending toward light, right up until it outgrows it and flops over
    /// reaching for light like any freestanding stem. Note this is *not*
    /// how every climbing plant climbs — twining vines (morning glory,
    /// wisteria) spiral their whole stem around a support instead, and
    /// tendril-bearers (peas, grapes) stay fairly straight themselves but
    /// send out separate coiling tendrils; this models the aerial-root
    /// strategy specifically, since that's what Pothos actually does.
    /// `None` for every non-climbing habit (Dracaena, Peace Lily) —
    /// ordinary phototropism applies from height zero, unconditionally.
    pub trellis_height: Option<f64>,
    /// Height interval between successive `AerialRoot`s the main stem puts
    /// out while still within reach of its support (`trellis_height`) — see
    /// that field's doc comment and `plant::AerialRoot`. Irrelevant (never
    /// spawns anything) when `trellis_height` is `None`, but still a plain
    /// `f64` rather than `Option<f64>` for the same reason `stem_segment_
    /// height_interval` is: it's a spacing, not a capability flag —
    /// `trellis_height` alone already gates whether it applies at all.
    pub aerial_root_height_interval: f64,

    // --- Root health: overwatering / fertilizer burn (see `Plant::
    // root_health`, `Soil::waterlog_stress`/`overfeed_stress`) — the two-
    // sided failure mode paired with drought/nutrient starvation. Distinct
    // from leaf senescence: this damages the *plant's ability to take up
    // water at all*, so a rotted-root plant can visibly wilt even though the
    // soil itself reads wet, the real, counterintuitive symptom that makes
    // overwatering a genuine mistake to diagnose rather than just "not
    // enough of a good thing." ---
    /// How fast `Plant::root_health` decays per second of sustained
    /// waterlogged/overfed stress (whichever is worse — same "worst, not
    /// summed" pattern as `leaf_stress_signal`).
    pub root_rot_rate: f64,
    /// How fast `root_health` recovers per second absent that stress — real
    /// mild root damage can partially recover once soil dries back out;
    /// slower than the damage rate, since roots take longer to regrow than
    /// they take to suffocate.
    pub root_recovery_rate: f64,

    // --- Whole-plant death (see `Stage::Dead`) — previously there was no
    // failure state at all: a starved or root-rotted plant just idled
    // forever as a bare, leafless cane. Real plants actually die, from
    // either total root loss (roots at zero can no longer take up any
    // water) or prolonged carbon starvation (an extended run with no net
    // photosynthetic income at all exhausts stored reserves). ---
    /// Sim-seconds `carbon_pool` has to stay pinned at (essentially) zero
    /// before starvation actually kills the plant — a grace period, not an
    /// instant kill, since a brief carbon deficit (one bad night) is normal
    /// and recoverable.
    pub starvation_death_threshold: f64,

    // --- Pot-bound stress & repotting (see `Plant::pot_capacity_
    // multiplier`, `Plant::repot`) — a real container caps how large a root
    // system (and so the whole plant) can get before it needs a bigger pot;
    // repotting raises that ceiling but costs a temporary setback
    // (`growth_shock`), the same real tradeoff as pruning. ---
    /// Height, at a fresh `pot_capacity_multiplier` of 1.0, past which the
    /// plant starts running out of room to root into.
    pub initial_pot_capacity: f64,
    /// How much further past `initial_pot_capacity` (in the same height
    /// units) the pot-bound penalty ramps down to its floor — a gradual
    /// squeeze, not a hard wall the instant the ceiling is reached.
    pub pot_bound_stress_range: f64,
    /// The most a badly pot-bound plant's effective root capacity ever
    /// drops to — stunted, not fully zero (a real overgrown houseplant
    /// keeps ticking over in too-small a pot, just slowly).
    pub pot_bound_floor: f64,
    /// Multiplies `pot_capacity_multiplier` each time `Plant::repot` is
    /// called — repotting into a meaningfully bigger container, not a
    /// token bump.
    pub repot_capacity_multiplier: f64,
    /// How much a repot restores `root_health` to, at minimum — real
    /// repotting lets a grower trim off rotted roots and refresh the soil,
    /// so it's a genuine (partial) fix for root rot, not just a size-cap
    /// reset.
    pub repot_root_health_restore: f64,

    // --- Growth shock (see `Plant::growth_shock`) — a shared, generic
    // "just had a physical setback" mechanic reused by both pruning (cut
    // tissue) and repotting (disturbed roots), the same "one shared
    // mechanism, not duplicated per-cause" pattern `stem_droop_target`
    // already uses for the main stem vs. branches. ---
    /// How fast `growth_shock` eases back down to zero once nothing new is
    /// adding to it.
    pub shock_recovery_rate: f64,
    /// At `growth_shock == 1.0`, how much elongation/photosynthesis is
    /// reduced (a fraction, not a hard stop — a shocked plant still ticks
    /// over, just slower).
    pub shock_growth_penalty: f64,

    // --- Pruning (see `Plant::prune_main_stem`/`prune_branch`) — a direct,
    // player-triggered version of the same apical-dominance release crown
    // branching already models automatically (see the module docs):
    // cutting the growing tip removes the bud that was suppressing the
    // lateral buds below it, releasing them *immediately* rather than
    // waiting for the plant to reach branching height/carbon on its own. ---
    /// Minimum height a stem must have before it can be pruned at all — cutting
    /// a two-leaf seedling isn't a meaningful shaping action.
    pub prune_min_height: f64,
    /// Fraction of current height removed by a single prune.
    pub prune_height_fraction: f64,
    /// How many new branches a main-stem prune releases at once (capped by
    /// however much room remains under `max_branches`) — real apical-
    /// dominance release frees *several* co-dominant buds together, not
    /// just one.
    pub prune_branch_release_count: usize,
    /// How much `growth_shock` a single prune adds.
    pub prune_shock_amount: f64,
    /// How much `growth_shock` a single repot adds.
    pub repot_shock_amount: f64,

    // --- Propagation (see `Plant::take_cutting`/`Plant::from_cutting`) — a
    // real stem cutting: snipping off a piece of the growing tip costs the
    // parent plant some height (`cutting_cost_height_fraction`, the same
    // "remove the tip, release lateral buds" mechanics `cut_main_stem_to`
    // already models for pruning) and produces a separate, storable item
    // (see `render::mod::CuttingItem`) a player can later plant into its own
    // pot/`PlantSlot` — a real second, independently-growing specimen, not
    // a substitute for the mother plant. ---
    /// Minimum height before a cutting can be taken — a cutting has to
    /// actually have a node/internode to root from.
    pub cutting_min_height: f64,
    /// Fraction of the parent's current height removed by taking a cutting —
    /// `Plant::take_cutting`'s own cost, applied via the same `cut_main_
    /// stem_to` mechanics `prune_main_stem` uses (shed leaves/aerial roots/
    /// branches above the cut, release lateral buds, growth shock). Smaller
    /// than `prune_height_fraction` by default: taking one cutting is a
    /// lighter touch than a full prune, even though the mechanism is
    /// identical.
    pub cutting_cost_height_fraction: f64,
    /// The propagated plant's starting height, immediately in `Stage::
    /// Vegetative` (a rooted cutting skips germination/sprouting entirely —
    /// it's already-differentiated tissue, not a seed) — see `Plant::from_
    /// cutting`.
    pub cutting_start_height: f64,
    /// The propagated plant's starting carbon reserve — real cuttings root
    /// using their own stored reserves before establishing enough leaf area
    /// to feed themselves, the same reasoning as `sprout_growth_rate`.
    pub cutting_start_carbon: f64,
    /// How many starter leaves the propagated cutting begins with.
    pub cutting_start_leaves: usize,

    // --- Dormancy (see `season::season_state`) ---
    /// How much winter's shorter days suppress elongation — multiplies
    /// `season_state.day_length_factor` directly into the elongation
    /// calculation, so `1.0` here would mean no dormancy effect at all.
    pub dormancy_elongation_sensitivity: f64,
}

impl PlantConfig {
    /// A caning, crown-branching habit — modeled on *Dracaena* (see
    /// `plant.rs`'s module docs). The default: what every existing tuning
    /// comment in this file was written against.
    pub fn dracaena() -> Self {
        PlantConfig {
            germination_water_factor: 0.5,
            germination_min_temperature_c: 15.0,
            sprout_growth_rate: 0.0006,
            sprout_height_threshold: 0.0015,

            optimal_temperature_c: 24.0,
            temperature_tolerance_c: 10.0,
            respiration_q10: 2.0,
            respiration_reference_temperature_c: 20.0,
            cold_stress_threshold_c: 12.0,

            // Calibrated to `render::config::SceneLayout`'s default window:
            // its own bottom/top edges sit at plant heights ~1.0/~4.9 (see
            // that struct's `window_offset`/`window_scale` doc comment), so
            // full light lasts through the window's own span and fades out
            // over a comparable further stretch above it.
            window_light_zone_height: 4.9,
            window_light_falloff_range: 4.0,
            // A real indoor Dracaena reaches ceiling height over years —
            // comparable to, or a bit past, its own window.
            realistic_max_height: 5.5,
            ambient_light_floor: 0.15,

            light_use_efficiency: 0.05,
            cotyledon_leaf_area: 0.4,
            leaf_area_per_leaf: 1.0,
            respiration_rate: 0.008,

            base_elongation_rate: 0.004,
            elongation_carbon_cost: 0.3,
            shade_avoidance_strength: 1.2,

            pipe_model_coeff: 0.006,
            // Rescaled together, ~7.5x from their original values, when
            // `transpiration_coeff` was cut to fix soil crashing to bone
            // dry within about a real minute of play (see that field's doc
            // comment) — `thickening_rate_coeff` multiplies the same
            // `uptake_rate` transpiration produces, so it has to grow by
            // the same factor to keep stem-thickening pacing unchanged.
            thickening_rate_coeff: 300.0,
            transpiration_coeff: 0.00008,

            plastochron_height_interval: 0.15,
            new_leaf_carbon_cost: 4.0,
            leaf_maturation_rate: 0.15,
            droop_response_rate: 0.4,
            max_carbon_pool: 20.0,

            // A leaf stays fully healthy for 6000 sim-seconds (15 sim-days
            // at the default `TimeConfig::day_length_sim_seconds`), then
            // eases toward abscission over roughly another 6000 (three
            // time constants of a 0.0005/sec ease covers ~95% of the way)
            // absent stress — a full leaf lifecycle of ~30 sim-days, which
            // plays out over a few real minutes at this demo's validation
            // pacing. Stress can cut that by up to 4x
            // (`leaf_stress_senescence_multiplier`).
            leaf_mature_lifespan: 6000.0,
            leaf_senescence_rate: 0.0005,
            leaf_stress_senescence_multiplier: 3.0,
            leaf_abscission_senescence_threshold: 0.95,

            // Tuned so a leaf with ~8-10 full-grown leaves' worth of area
            // above it (`leaf_area_per_leaf` = 1.0, so that's area_above ≈
            // 8-10) is already down to roughly a third of full light
            // (exp(-0.12*9) ≈ 0.33) — enough to noticeably slow a plant's
            // net carbon income well before it reaches an unrealistic
            // leaf count, without also snuffing out a young plant's first
            // handful of leaves before it even has a chance to establish.
            leaf_self_shading_coeff: 0.12,

            new_branch_carbon_cost: 10.0,
            min_height_for_branching: 0.6,
            max_branches: 4,
            branch_elongation_rate_factor: 0.6,

            growth_habit: GrowthHabit::UprightCane,
            leaf_mesh_name: "leaf",

            flowering_height_threshold: 0.6,
            flower_mesh_name: "flower_dracaena",
            // Real Dracaena flowering is rare enough in cultivation to be
            // a notable event, and each bloom itself is fairly short-lived
            // (the famous fragrance peaks for only a night or two) — a
            // long rest relative to a short bloom, roughly a 15:1 ratio.
            bloom_duration: 1000.0,
            bloom_rest_duration: 15000.0,
            bloom_response_rate: 0.01,

            stem_droop_max_angle: 0.9,
            stem_droop_response_rate: 0.15,
            stem_droop_reference_radius: 0.03,

            lean_rate: 0.00015,
            max_lean_angle: 0.5,
            helio_strength: 0.25,
            helio_response_rate: 0.05,
            fold_response_rate: 0.03,
            stem_segment_height_interval: 1.0,
            trellis_height: None,
            // Inert here (trellis_height is None) — see PlantConfig::pothos
            // for the value that actually matters.
            aerial_root_height_interval: 0.3,

            // Slower than leaf senescence rates — root rot is a background
            // process that takes sustained neglect (in either direction) to
            // actually kill a plant, not a same-tick punishment.
            root_rot_rate: 0.00015,
            root_recovery_rate: 0.00005,

            // ~2.5 sim-days of zero carbon income at this demo's fast
            // validation pacing — long enough that a single bad night never
            // kills the plant, short enough that sustained neglect
            // (starved dark corner, bone-dry soil for an extended stretch)
            // actually does within a realistic session.
            starvation_death_threshold: 1000.0,

            initial_pot_capacity: 15.0,
            pot_bound_stress_range: 6.0,
            pot_bound_floor: 0.3,
            repot_capacity_multiplier: 1.8,
            repot_root_health_restore: 0.6,

            shock_recovery_rate: 0.002,
            shock_growth_penalty: 0.6,

            prune_min_height: 0.3,
            prune_height_fraction: 0.35,
            prune_branch_release_count: 2,
            prune_shock_amount: 0.5,
            repot_shock_amount: 0.35,

            cutting_min_height: 1.0,
            cutting_cost_height_fraction: 0.15,
            cutting_start_height: 0.05,
            cutting_start_carbon: 2.0,
            cutting_start_leaves: 1,

            dormancy_elongation_sensitivity: 1.0,
        }
    }

    /// A basal-rosette habit — modeled on *Spathiphyllum* (Peace Lily) and
    /// similar houseplants (hostas, many ferns): leaves emerge in a tight
    /// cluster directly from a compressed crown near soil level, with no
    /// visible caning stem and no lateral branching at all. Derived from
    /// `dracaena()` by scaling `base_elongation_rate` *and*
    /// `plastochron_height_interval` down by the same ~13x factor —
    /// scaling both together keeps roughly the same *rate* of new leaves
    /// per unit time (rate ≈ elongation/interval, and that ratio is
    /// unchanged), while the plant's absolute height stays small throughout
    /// its life, matching the rosette habit's real lack of internode
    /// elongation. `max_branches: 0` rules out crown branching entirely —
    /// a rosette's growing points are basal offsets/pups in reality, not
    /// modeled here, not lateral canes partway up a stem.
    pub fn peace_lily() -> Self {
        PlantConfig {
            base_elongation_rate: 0.0003,
            plastochron_height_interval: 0.011,
            max_branches: 0,
            cutting_min_height: 0.08,
            cutting_start_height: 0.01,
            growth_habit: GrowthHabit::BasalRosette,
            leaf_mesh_name: "leaf_peace_lily",
            // A squat rosette's own "stem" (the compressed crown) never
            // gets tall enough for this to matter functionally once
            // `max_branches` is 0, but keep it consistent with the scaled-
            // down height range rather than leaving Dracaena's much taller
            // trigger in place unused.
            min_height_for_branching: 0.05,
            // A mature Peace Lily rosette stays low and compact — a small
            // fraction of Dracaena's own realistic ceiling.
            realistic_max_height: 0.25,
            // Blooms once it's grown a reasonably full rosette — tuned to
            // the same scaled-down height range as the fields above.
            flowering_height_threshold: 0.045,
            flower_mesh_name: "flower_peace_lily",
            // A real Peace Lily reblooms readily indoors — often several
            // times a year, each bloom lasting weeks — so unlike
            // Dracaena's rare/brief cycle, this blooms *more* than it
            // rests.
            bloom_duration: 3000.0,
            bloom_rest_duration: 2000.0,
            ..PlantConfig::dracaena()
        }
    }

    /// A climbing/vining habit trained against a support — modeled on
    /// *Epipremnum aureum* (Pothos), a fast-growing vine almost always
    /// grown up a moss pole/stake indoors rather than left to trail. See
    /// `trellis_height`'s doc comment for the mechanism; the other
    /// overrides here are the real differences between a trained climber
    /// and Dracaena's freestanding cane: pothos elongates noticeably
    /// faster, tolerates (and is usually grown in) lower light without
    /// stalling out, branches modestly at nodes rather than Dracaena's
    /// crown release, and essentially never flowers in indoor cultivation
    /// (real indoor pothos flowering is rare enough to be noteworthy when
    /// it happens) — reflected as a height threshold high enough that a
    /// typical session won't reach it, rather than removing flowering
    /// capability outright.
    pub fn pothos() -> Self {
        PlantConfig {
            trellis_height: Some(3.0),
            base_elongation_rate: 0.006,
            ambient_light_floor: 0.35,
            max_branches: 2,
            growth_habit: GrowthHabit::Vine,
            leaf_mesh_name: "leaf_pothos",
            // A real trained Pothos keeps trailing well past its own
            // support once it outgrows it (see `trellis_height: Some(3.0)`
            // above) — several times that support's own height.
            realistic_max_height: 12.0,
            new_branch_carbon_cost: 14.0,
            flowering_height_threshold: 50.0,
            // If a session ever did somehow reach that height, Pothos
            // (Epipremnum) is in the same family (Araceae) as Peace Lily
            // and shares its spathe/spadix flower structure — a real
            // Pothos bloom looks like a smaller version of a Peace Lily's,
            // not at all like Dracaena's star-flowered panicle.
            flower_mesh_name: "flower_peace_lily",
            ..PlantConfig::dracaena()
        }
    }
}

impl Default for PlantConfig {
    fn default() -> Self {
        Self::dracaena()
    }
}

/// Every selectable growth habit, by the name a UI (or a test) would use to
/// pick one — see `PlantConfig::dracaena`/`PlantConfig::peace_lily`. Kept as
/// a name-indexed lookup, not just the two constructors, so the wasm-facing
/// `Simulation::set_species` and native tests share one place that knows
/// what names are valid. Falls back to `dracaena` for anything unrecognized
/// rather than erroring — a typo'd species name shouldn't be able to crash
/// the sim.
pub fn plant_config_for_species(name: &str) -> PlantConfig {
    match name {
        "peace_lily" => PlantConfig::peace_lily(),
        "pothos" => PlantConfig::pothos(),
        _ => PlantConfig::dracaena(),
    }
}

#[cfg(test)]
mod species_tests {
    use super::*;

    #[test]
    fn unknown_species_names_fall_back_to_dracaena() {
        assert_eq!(plant_config_for_species("not-a-real-species"), PlantConfig::dracaena());
        assert_eq!(plant_config_for_species(""), PlantConfig::dracaena());
    }

    #[test]
    fn known_species_names_resolve_to_their_own_config() {
        assert_eq!(plant_config_for_species("dracaena"), PlantConfig::dracaena());
        assert_eq!(plant_config_for_species("peace_lily"), PlantConfig::peace_lily());
        assert_eq!(plant_config_for_species("pothos"), PlantConfig::pothos());
        assert_ne!(PlantConfig::dracaena(), PlantConfig::peace_lily());
        assert_ne!(PlantConfig::dracaena(), PlantConfig::pothos());
    }

    #[test]
    fn only_the_climbing_habit_sets_a_trellis_height() {
        assert_eq!(PlantConfig::dracaena().trellis_height, None);
        assert_eq!(PlantConfig::peace_lily().trellis_height, None);
        assert_eq!(PlantConfig::pothos().trellis_height, Some(3.0));
    }

    #[test]
    fn each_species_points_at_its_own_botanically_accurate_flower_mesh() {
        assert_eq!(PlantConfig::dracaena().flower_mesh_name, "flower_dracaena");
        assert_eq!(PlantConfig::peace_lily().flower_mesh_name, "flower_peace_lily");
        // Pothos shares Peace Lily's mesh (both Araceae, same spathe/spadix
        // structure) rather than getting a bespoke asset nobody would ever
        // realistically see — see PlantConfig::pothos's own doc comment.
        assert_eq!(PlantConfig::pothos().flower_mesh_name, "flower_peace_lily");
    }

    #[test]
    fn each_species_points_at_its_own_botanically_accurate_leaf_mesh_and_growth_habit() {
        let dracaena = PlantConfig::dracaena();
        let peace_lily = PlantConfig::peace_lily();
        let pothos = PlantConfig::pothos();
        assert_eq!(dracaena.leaf_mesh_name, "leaf");
        assert_eq!(peace_lily.leaf_mesh_name, "leaf_peace_lily");
        assert_eq!(pothos.leaf_mesh_name, "leaf_pothos");
        // Every species' own leaf mesh is distinct — no two share a shape.
        assert_ne!(dracaena.leaf_mesh_name, peace_lily.leaf_mesh_name);
        assert_ne!(dracaena.leaf_mesh_name, pothos.leaf_mesh_name);
        assert_ne!(peace_lily.leaf_mesh_name, pothos.leaf_mesh_name);

        assert_eq!(dracaena.growth_habit, GrowthHabit::UprightCane);
        assert_eq!(peace_lily.growth_habit, GrowthHabit::BasalRosette);
        assert_eq!(pothos.growth_habit, GrowthHabit::Vine);
    }

    #[test]
    fn each_species_has_its_own_distinct_realistic_max_height() {
        let dracaena = PlantConfig::dracaena();
        let peace_lily = PlantConfig::peace_lily();
        let pothos = PlantConfig::pothos();
        assert!(peace_lily.realistic_max_height < dracaena.realistic_max_height, "a rosette should stay far shorter than a cane");
        assert!(pothos.realistic_max_height > dracaena.realistic_max_height, "a trained vine should realistically trail longer than a cane grows tall");
    }

    #[test]
    fn dracaena_blooms_far_more_rarely_and_briefly_than_peace_lily() {
        let dracaena = PlantConfig::dracaena();
        let peace_lily = PlantConfig::peace_lily();
        assert!(
            dracaena.bloom_rest_duration > dracaena.bloom_duration,
            "a real Dracaena rests far longer than it blooms"
        );
        assert!(
            peace_lily.bloom_duration >= peace_lily.bloom_rest_duration,
            "a real Peace Lily reblooms readily, spending at least as much time in bloom as resting"
        );
        assert!(
            dracaena.bloom_rest_duration > peace_lily.bloom_rest_duration,
            "Dracaena's rest between blooms should be much longer than Peace Lily's"
        );
    }
}

/// Wall-clock-to-sim-time conversion — one dial that speeds up (or slows
/// down) growth, soil drying, and the day/night cycle together. Lives here
/// (not in `render::config`) so a real-time pacing scenario — "does a
/// branch actually appear within N minutes of real play" — can be
/// regression-tested with plain `cargo test`, without touching wasm/render
/// at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeConfig {
    /// Real seconds are multiplied by this to get "sim seconds", the units
    /// `SoilConfig`/`PlantConfig`'s rates and `day_length_sim_seconds` below
    /// are tuned in.
    pub sim_seconds_per_real_second: f64,
    /// How many sim-seconds make one full day/night cycle.
    pub day_length_sim_seconds: f64,
}

impl Default for TimeConfig {
    fn default() -> Self {
        // 400 sim-seconds/day at 80 sim-seconds per real second is a
        // 5-real-second day/night cycle. The only consumer of this default
        // today is the `web/` demo, whose job is validating the engine —
        // seeing germination, growth, day/night lighting, and crown
        // branching play out within a short session — not final gameplay
        // feel, so it's tuned aggressively fast on purpose. Revisit once
        // there's an actual gameplay-tuned config (and ideally a UI time
        // scale control) distinct from this validation default.
        TimeConfig {
            sim_seconds_per_real_second: 80.0,
            day_length_sim_seconds: 400.0,
        }
    }
}

impl TimeConfig {
    /// Sane bounds for a runtime speed control (e.g. a UI slider), relative
    /// to `Default::default()`'s own pace — low enough the sim doesn't
    /// effectively freeze, high enough it doesn't blow past every discrete
    /// event in a single frame's `dt` and become meaningless to watch.
    pub const MIN_SPEED_MULTIPLIER: f64 = 0.25;
    pub const MAX_SPEED_MULTIPLIER: f64 = 5.0;

    /// Clamps a requested speed multiplier into that range — used by
    /// `Simulation::set_time_scale` so a UI slider (or a malformed direct
    /// call) can't stall the sim or spin it fast enough to be meaningless.
    /// A non-finite input (NaN/±infinity — e.g. from a misread JS `Number`)
    /// falls back to `1.0` rather than being passed through: `f64::clamp`
    /// does *not* sanitize NaN (it returns NaN as-is, unclamped), and a NaN
    /// `sim_seconds_per_real_second` would poison every downstream
    /// computation (`day_progress`, then every `SunState` field) for the
    /// rest of the session.
    pub fn clamp_speed_multiplier(multiplier: f64) -> f64 {
        if !multiplier.is_finite() {
            return 1.0;
        }
        multiplier.clamp(Self::MIN_SPEED_MULTIPLIER, Self::MAX_SPEED_MULTIPLIER)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpiration_stays_within_a_realistic_multiple_of_bare_soil_evaporation() {
        // Real evapotranspiration studies put an established, leafy potted
        // plant's own water draw at roughly 2-5x bare soil's own
        // evaporation — not the 15-40x `transpiration_coeff` used to work
        // out to (found by simulating a played-through session: soil
        // crashed from full to bone dry within about a real minute of the
        // fast validation demo, before a leaf even matured on some runs).
        // Pinning the ratio at a believable "established" leaf area here
        // catches a future regression back toward that blowup even in a
        // change that never happens to run a long simulated session.
        let config = GrowthConfig::default();
        let established_leaf_area = 8.0;
        let uptake_at_full_sun_and_water = established_leaf_area * config.plant.transpiration_coeff;
        let evaporation = config.soil.evaporation_rate_per_sec;
        let total_to_evaporation_ratio = (uptake_at_full_sun_and_water + evaporation) / evaporation;
        assert!(
            (1.5..6.0).contains(&total_to_evaporation_ratio),
            "expected a modestly-leaved plant's total water draw to be within a believable multiple of bare-soil evaporation alone, got {total_to_evaporation_ratio}x"
        );
    }

    #[test]
    fn ambient_humidity_never_passively_drifts_into_pest_territory() {
        // dry_air_floor < safe_humidity would mean pests ratchet up forever
        // from default play alone, with no auto-mist to counter it (found
        // by a played-through session stuck at one leaf, starved to death).
        let config = GrowthConfig::default();
        assert!(config.humidity.dry_air_floor >= config.pest.safe_humidity);
    }

    #[test]
    fn clamp_speed_multiplier_passes_through_values_already_in_range() {
        assert_eq!(TimeConfig::clamp_speed_multiplier(1.0), 1.0);
        assert_eq!(TimeConfig::clamp_speed_multiplier(2.5), 2.5);
    }

    #[test]
    fn clamp_speed_multiplier_clamps_extremes() {
        assert_eq!(
            TimeConfig::clamp_speed_multiplier(0.0),
            TimeConfig::MIN_SPEED_MULTIPLIER
        );
        assert_eq!(
            TimeConfig::clamp_speed_multiplier(-5.0),
            TimeConfig::MIN_SPEED_MULTIPLIER
        );
        assert_eq!(
            TimeConfig::clamp_speed_multiplier(1000.0),
            TimeConfig::MAX_SPEED_MULTIPLIER
        );
    }

    #[test]
    fn clamp_speed_multiplier_falls_back_to_1x_for_non_finite_input() {
        // Found by writing this test, not by observing it in the browser:
        // plain `f64::clamp` does *not* sanitize NaN (NaN < / > anything is
        // false, so it passes through both bound checks unchanged) — it
        // would have silently poisoned every downstream time computation
        // instead of failing loudly or falling back sanely.
        assert_eq!(TimeConfig::clamp_speed_multiplier(f64::NAN), 1.0);
        assert_eq!(TimeConfig::clamp_speed_multiplier(f64::INFINITY), 1.0);
        assert_eq!(TimeConfig::clamp_speed_multiplier(f64::NEG_INFINITY), 1.0);
    }
}

/// Everything the growth simulation needs, bundled so callers pass one
/// value instead of four.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GrowthConfig {
    pub sun: SunConfig,
    pub soil: SoilConfig,
    pub plant: PlantConfig,
    pub time: TimeConfig,
    pub climate: ClimateConfig,
    pub room: RoomConfig,
    pub humidity: HumidityConfig,
    pub pest: PestConfig,
    pub season: SeasonConfig,
    pub moon: MoonConfig,
}
