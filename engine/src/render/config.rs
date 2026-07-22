//! Scene layout numbers — kept as data separate from `scene.rs`'s logic,
//! the same way `sim::config` separates the biology's numbers from
//! `sim::plant`/`sim::soil`. Nothing biological lives here; this is purely
//! "where things sit on screen." Wall-clock-to-sim-time pacing
//! (`TimeConfig`) lives in `sim::config` instead, alongside the rates it
//! paces, so a pacing scenario can be regression-tested with plain
//! `cargo test` — no wasm/render involved.

/// Where/how big each background (non-simulated) piece sits, and the
/// sim-space-to-clip-space conversion factors for the parts the growth
/// model actually drives (stem height/radius, leaf size/spread).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneLayout {
    /// The *closest-in* the camera ever gets — applied to everything except
    /// the wall (see `render/mod.rs`'s `render`, which special-cases it),
    /// multiplying every other drawable's offset and scale equally so the
    /// composition stays proportionally identical, just smaller/farther
    /// away. `scene::dynamic_zoom` pulls back *further* than this once the
    /// plant's own height would otherwise grow past
    /// `zoom_visible_half_height` — this value is only the floor (the most
    /// zoomed-in the camera is allowed to be), not the zoom used every
    /// frame. The wall is exempted from both because it's just an
    /// overscanned background fill with no in-world "size" of its own to
    /// preserve.
    pub zoom: f32,
    /// How far up clip space (`InstanceUniform`'s offset is in normalized
    /// device coordinates, so 1.0 is the very top edge) the plant is allowed
    /// to reach before `scene::dynamic_zoom` starts pulling the camera back
    /// further than `zoom`. Set a little under the true edge (1.0) so the
    /// tip visibly has a margin rather than touching the frame exactly.
    pub zoom_visible_half_height: f32,
    /// The same margin as `zoom_visible_half_height`, along the horizontal
    /// axis instead — how far right of center (in the same pre-`zoom` clip-
    /// space units `pot_anchor`/`plant_slot_spacing` use) the room's
    /// rightmost plant is allowed to sit before `scene::dynamic_zoom_for_
    /// room` starts pulling back further than `zoom`, the same way a single
    /// tall plant pulls back vertically. Only matters once more than one
    /// plant shares the room (see `plant_slot_spacing`) — a single plant at
    /// `pot_anchor` is never this close to the edge under this scene's
    /// default tuning.
    pub zoom_visible_half_width: f32,
    /// Shared by the pot, soil cap, seed, and stem/leaves' base — the point
    /// where "the plant" sits, in clip space (pre-`zoom`).
    pub pot_anchor: [f32; 2],
    /// Sideways step between one plant slot's own base anchor and the
    /// next — see `scene::plant_slot_base_anchor`. This side-profile scene
    /// has no real depth axis, so "several pots" can only be laid out side
    /// by side along the sill.
    pub plant_slot_spacing: f32,
    pub wall_scale: f32,
    pub window_offset: [f32; 2],
    pub window_scale: f32,
    /// The sun/moon disc's size and the local (pre-`window_scale`) extent it
    /// arcs across inside the window pane — see `scene::sky_object_transform`.
    pub sky_object_scale: f32,
    pub sky_object_local_x_range: [f32; 2],
    pub sky_object_local_y_range: [f32; 2],
    /// The dark disc `scene::moon_shadow_transform` overlaps the moon with
    /// to fake its current phase — the moon's own unlit surface, not quite
    /// black.
    pub moon_shadow_tint: [f32; 3],
    /// Max blend fraction toward the sky's own ambient color at full sun
    /// intensity — see `scene::daytime_fade`.
    pub moon_daytime_fade_strength: f32,
    /// Ambient tint applied to the wall/window at night (see
    /// `scene::ambient_tint`) — never fully black, a dim moonlit room
    /// rather than a blackout.
    pub night_ambient_color: [f32; 3],
    /// Distance-squared falloff coefficient for the GPU point light at the
    /// window (see `scene::SceneLightUniform`) — bigger falls off faster.
    pub scene_light_falloff: f32,
    /// Target on-screen thickness in *CSS* pixels (device-independent — see
    /// `render::mod`'s `device_pixel_ratio` handling) of the white halo
    /// drawn behind every plant-asset mesh — see `scene::outline_uniform`.
    /// Scales with the canvas's actual pixel width rather than with zoom or
    /// a mesh's own size, so it reads as the same thin line regardless of
    /// window size or how big the plant has grown. 3.0 is the requested
    /// "equivalent of 3px on a 4K display" — a CSS pixel is device-
    /// independent by definition, so that phrase and "3 CSS pixels" are the
    /// same target regardless of the viewer's actual resolution.
    pub outline_pixel_width: f32,
    pub outline_tint_day: [f32; 3],
    pub outline_tint_night: [f32; 3],

    /// Real GPU depth-buffer Z (see `InstanceUniform::from_transform`) for
    /// the wall/window/pot/soil/trellis/sun/moon layer — farther than
    /// anything the plant itself draws at, so the plant always wins the
    /// depth test against the room behind it regardless of draw order.
    pub background_depth: f32,
    /// Depth for the climbing-support pole — between the background and
    /// the plant itself (see `render`'s own doc comment on why the trellis
    /// draws where it does: behind the foliage, in front of the wall).
    pub trellis_depth: f32,
    /// Nominal middle-layer depth shared by the stem, roots, seed,
    /// cotyledons, aerial roots, and flower — these don't get individual
    /// depth variety (unlike leaves, see `leaf_depth_spread`), just one
    /// consistent "the plant's own layer" plane.
    pub plant_depth: f32,
    /// How far an individual leaf's own depth (see `scene::leaf_depth`) is
    /// allowed to scatter above/below `plant_depth` — the whole point of
    /// adding real depth: leaves spread front-to-back instead of every one
    /// sitting in exactly the same flat plane.
    pub leaf_depth_spread: f32,
    /// How much bigger a fully-near instance renders than a fully-far one
    /// (as a fraction of its own scale) — see `scene::apply_depth_look`. A
    /// cheap fake-perspective cue, not real projection.
    pub depth_scale_falloff: f32,
    /// How much dimmer a fully-far instance renders than a fully-near one
    /// (as a fraction of its own tint) — see `scene::apply_depth_look`.
    pub depth_dim_falloff: f32,
    /// How much farther an outline halo's own depth sits than its paired
    /// normal-colored mesh's — see `scene::outline_uniform`'s doc comment
    /// on why this has to be nonzero (so the normal fill reliably wins the
    /// depth test over its own halo, regardless of draw order).
    pub outline_depth_bias: f32,

    /// How much bigger the currently hover-picked leaf renders (see
    /// `scene::apply_hover_scale`) — the prune tool's "this is what you're
    /// about to cut" cue.
    pub hover_scale_multiplier: f32,
    /// Outline tint for the currently hover-picked leaf specifically (see
    /// `scene::outline_uniform`'s `tint` param) — red in place of the usual
    /// `outline_tint` white, same saturating-multiplier trick (see that
    /// field's own doc comment) so it reads as solid red regardless of
    /// time of day.
    pub hover_outline_tint: [f32; 3],

    /// Brightness of the second, cursor-tracking point light (see
    /// `scene::SceneLightUniform`) at its very center — small relative to
    /// the window light's own `SunConfig`-driven intensity, since this is
    /// meant as a local pool of light around the pointer, not a second sun.
    pub cursor_light_intensity: f32,
    /// Distance-squared falloff for the cursor light — much bigger than
    /// `scene_light_falloff` so it stays a small, local highlight rather
    /// than relighting the whole room the way the window light does.
    pub cursor_light_falloff: f32,
    /// Blinn-Phong shininess exponent for the cursor specular highlight on
    /// leaves specifically (see `scene::with_leaf_specular`) — higher is a
    /// smaller, sharper highlight; lower is a broader, softer one.
    pub leaf_shininess: f32,

    /// Fixed room position (top-left, like the window is top-right-ish) of
    /// a wall-mounted lamp — a third light source, always present but only
    /// actually *lit* at night (see `render/mod.rs`'s lamp-intensity calc).
    /// Gives leaves a specular sheen after dark the way the cursor light
    /// does by day, without needing the pointer over the canvas.
    pub lamp_offset: [f32; 2],
    pub lamp_scale: f32,
    /// Diffuse+specular brightness the lamp reaches at full night — scaled
    /// down toward 0 by day (see `render/mod.rs`).
    pub lamp_intensity_max: f32,
    /// Distance-squared falloff — same shape as `cursor_light_falloff`, a
    /// small local pool of light, not a second window.
    pub lamp_falloff: f32,
    /// The fixture's own tint fully off (daytime) vs. fully on (night) —
    /// same "fade the mesh itself, not just the light it casts" idiom
    /// `moon_shadow_tint`/`ambient_tint` already use elsewhere.
    pub lamp_off_tint: [f32; 3],
    pub lamp_on_tint: [f32; 3],

    pub pot_scale: f32,
    pub soil_scale: f32,
    pub seed_scale: f32,
    pub seed_min_swell_scale_fraction: f32,
    pub cotyledon_scale: f32,
    /// Outward tilt angle (radians) for each cotyledon, fanned symmetrically
    /// left/right from the stem's base — a fixed pose, cotyledons don't
    /// track light or fold the way true leaves do (see `scene.rs`).
    pub cotyledon_spread_angle: f32,
    /// The terminal bloom's mesh scale — see `scene::flower_transform` and
    /// `sim::config::PlantConfig::flowering_height_threshold`.
    pub flower_scale: f32,

    /// Width of the optional climbing-support pole/lattice (see
    /// `scene::trellis_transform`, `sim::config::PlantConfig::
    /// trellis_height`) — a fixed rigid prop, not pipe-model-thickened like
    /// a stem, so a single constant rather than a `_scale` multiplying a
    /// growing radius. Its *height* isn't a separate tunable at all — it
    /// reuses `stem_height_scale` directly so it renders exactly as tall
    /// as the plant's own stem would reach at that same height.
    pub trellis_width_scale: f32,
    /// Sideways offset from `pot_anchor` — a real climber drapes *beside*
    /// its stake, not exactly through its centerline, and without this the
    /// pole would end up fully hidden behind the stem once the stem grows
    /// thick enough to match its width (both would otherwise share the
    /// exact same x position).
    pub trellis_x_offset: f32,
    /// One-time base lean (radians) a `GrowthHabit::Vine` stem's first
    /// segment takes toward the trellis, so it reads as reaching for/
    /// hugging its support instead of growing straight up the pot's
    /// center. See `scene::stem_segment_angle`.
    pub vine_trellis_lean_angle: f32,
    /// Fixed size of each `AerialRoot` mark (see `scene::aerial_root_
    /// transform`) — cosmetic and small, not tied to any growing quantity,
    /// so a single constant rather than a `_scale` multiplying something
    /// that changes over time.
    pub aerial_root_scale: f32,

    /// `Plant::height` (dimensionless sim units) times this is the stem's
    /// clip-space scale.y. Shared by branches too (see
    /// `scene::stem_like_transform`) — a branch's own `height` is in the
    /// same units.
    pub stem_height_scale: f32,
    /// `Plant::stem_radius` times this is the stem's clip-space scale.x
    /// (before aspect correction). Shared by branches too.
    pub stem_radius_scale: f32,
    /// Fixed outward angle (radians) a branch diverges from whatever it's
    /// attached to — narrower than a leaf's spread (branches grow upward
    /// toward light, they don't splay flat like a blade), before its own
    /// independent phototropic lean adjusts it further.
    pub branch_spread_angle: f32,

    /// Leaf mesh scale at full maturity (before aspect correction).
    pub leaf_scale: f32,
    /// Outward spread angle (radians) at full turgor/daytime posture,
    /// before nyctinasty/wilt/heliotropism adjust it.
    pub leaf_base_spread_angle: f32,
    /// How far nyctinastic folding (0..1) pulls the spread angle back
    /// toward (and past) vertical.
    pub leaf_fold_max_angle: f32,
    /// How far wilting (0..1) droops the leaf, on top of folding.
    pub leaf_droop_max_angle: f32,
    /// How far heliotropic tracking (-1..1) nudges the leaf angle, applied
    /// in a fixed direction (toward the window) regardless of which side
    /// of the stem the leaf is on.
    pub leaf_helio_max_angle: f32,
    /// `GrowthHabit::BasalRosette` fan spread (radians) for a just-emerged
    /// leaf — see `scene::rosette_leaf_transform`.
    pub rosette_leaf_min_spread_angle: f32,
    /// Same, for a leaf at/past `rosette_leaf_splay_age`.
    pub rosette_leaf_max_spread_angle: f32,
    /// Sim-time age at which a rosette leaf reaches full splay.
    pub rosette_leaf_splay_age: f64,
    /// Multiplies (not replaces) the leaf mesh's own baked green at
    /// `Leaf::senescence == 0.5` — since `tint` multiplies rather than sets
    /// color, this biases the mesh's own green toward yellow rather than
    /// specifying yellow outright, so it still reads as "this leaf" tinted,
    /// not a different mesh swapped in. See `scene::leaf_transform_in_frame`.
    pub leaf_senescent_tint: [f32; 3],
    /// Same idea, at `Leaf::senescence == 1.0` (browned, about to be shed).
    pub leaf_dead_tint: [f32; 3],
    /// How much a fully senesced leaf shrinks relative to its healthy
    /// mature size — real dying leaves curl and shrivel, not just
    /// discolor.
    pub leaf_shrivel_max_fraction: f32,
    /// Dimmest a fully self-shaded leaf's tint gets multiplied down to (see
    /// `sim::plant::self_shading_factors`) — not zero, since even a buried
    /// leaf gets some bounced light.
    pub leaf_occlusion_min_brightness: f32,

    /// Total horizontal distance the pot travels across the full
    /// `Simulation::set_pot_position` range (0.0 = right at the window,
    /// 1.0 = as far back as this game models) — see `scene::pot_anchor_
    /// for_position`. Player-chosen placement wasn't visible in the scene
    /// at all before this (only its *effect* on light/temperature was
    /// simulated), which made the whole mechanic illegible — moving the
    /// slider now visibly slides the pot (and everything anchored to it:
    /// stem, leaves, flower, aerial roots, the light beam's far end)
    /// toward or away from the window.
    pub pot_position_x_travel: f32,
    /// Multiplies the *stem* mesh's own baked color at zero vitality
    /// (`Decision::Vegetative::effective_water_factor` — water availability
    /// after root health/pot-bound stress, see `render/mod.rs`'s call site)
    /// — `1.0` (`NO_TINT`) at full vitality. A distinct visual channel from
    /// leaf senescence (`leaf_senescent_tint`/`leaf_dead_tint`), which
    /// stays on its own yellow-to-brown axis meaning "this leaf is old or
    /// shaded" — the stem instead shows overall plant vigor, whatever the
    /// cause (dry soil, rotted roots, or a pot-bound plant). See
    /// `scene::stem_health_tint`.
    pub stem_unhealthy_tint: [f32; 3],
    /// Soil tint at bone-dry moisture — pale, the mesh's own baked color
    /// lightened rather than a wholly different color, same "multiply, don't
    /// replace" idiom as every other tint here. See `scene::soil_moisture_
    /// tint`.
    pub soil_dry_tint: [f32; 3],
    /// Soil tint at a healthy, well-watered moisture level — the mesh's own
    /// baked color, effectively `NO_TINT`, included explicitly so the
    /// dry/waterlogged lerp has a clear, named midpoint to aim at.
    pub soil_wet_tint: [f32; 3],
    /// Soil tint once moisture crosses `SoilConfig::waterlogged_threshold`
    /// — visibly darker/murkier, a direct "the soil itself looks wrong"
    /// warning that a player can act on (stop watering) *before* root
    /// damage has had time to show up in `Stats::root_health` at all, since
    /// root rot only starts after `SoilConfig::waterlog_grace_period` of
    /// sustained waterlogging.
    pub soil_waterlogged_tint: [f32; 3],

    /// Fixed size of the small root tendrils drawn peeking out from beneath
    /// the pot (see `roots.svg`) — cosmetic and small, same reasoning as
    /// `aerial_root_scale`. Tinted by the same `scene::stem_health_tint`
    /// the stem itself uses (see `render/mod.rs`), so a struggling plant's
    /// roots visibly look as unwell as its stem does, without needing a
    /// fully transparent pot (a much bigger change — real alpha blending
    /// isn't in this renderer's pipeline at all yet, everything draws
    /// opaque, back-to-front).
    pub roots_scale: f32,

    /// Wall tint at the year's summer quarter (`sim::season::Season::
    /// Summer`) — see `scene::seasonal_wall_tint`. Multiplies together with
    /// (not instead of) the day/night `ambient_tint` the window already
    /// gets, so both cycles are visible at once: the room brightens/dims
    /// through the day, and separately drifts through a slow year-long mood
    /// shift. A subtle multiplier close to `NO_TINT`, tuned against wall.svg's
    /// own baked brown — this is a background mood cue, not something that
    /// should fight with the plant itself for attention.
    pub season_summer_tint: [f32; 3],
    /// Wall tint at the year's autumn quarter.
    pub season_autumn_tint: [f32; 3],
    /// Wall tint at the year's winter quarter.
    pub season_winter_tint: [f32; 3],
    /// Wall tint at the year's spring quarter.
    pub season_spring_tint: [f32; 3],
}

impl Default for SceneLayout {
    fn default() -> Self {
        SceneLayout {
            // Pulled back further than a single-stem plant would need —
            // paired with `TimeConfig`'s validation-demo pacing (see its
            // doc comment), a lot more growth/branching happens in a short
            // session than a slower, final-gameplay pacing would produce,
            // so this needs real headroom to keep it all on-frame.
            zoom: 0.5,
            zoom_visible_half_height: 0.85,
            zoom_visible_half_width: 0.85,
            pot_anchor: [-0.15, -0.45],
            plant_slot_spacing: 0.45,
            wall_scale: 0.05,
            // Raised and enlarged for realistic proportions: a real window
            // sill sits well above where a floor/table plant's pot is, not
            // near it, and a window is roughly comparable in height to a
            // well-grown (not extreme) houseplant, not a small fraction of
            // it. With these numbers the plant reaches the window's own
            // bottom edge around height ~1 (roughly when it's just started
            // branching/flowering) and its top around height ~4.9 (a
            // sizeable, well-established specimen) — see
            // `sim::config::PlantConfig`'s window-light-zone fields, which
            // are deliberately calibrated to this same range so growth
            // past the window's height starts running low on light, the
            // same reason a real plant wouldn't keep extending upward past
            // its actual light source indefinitely.
            window_offset: [0.62, 0.6],
            window_scale: 0.02,
            sky_object_scale: 0.004,
            sky_object_local_x_range: [-15.0, 15.0],
            sky_object_local_y_range: [-10.0, 25.0],
            moon_shadow_tint: [0.08, 0.08, 0.12],
            moon_daytime_fade_strength: 0.85,
            night_ambient_color: [0.30, 0.34, 0.48],
            scene_light_falloff: 4.0,
            outline_pixel_width: 3.0,
            outline_tint_day: [3.0, 2.2, 0.7],
            outline_tint_night: [1.2, 2.6, 3.2],
            background_depth: 0.9,
            trellis_depth: 0.7,
            plant_depth: 0.5,
            leaf_depth_spread: 0.15,
            depth_scale_falloff: 0.1,
            depth_dim_falloff: 0.15,
            outline_depth_bias: 0.02,
            hover_scale_multiplier: 1.15,
            hover_outline_tint: [10.0, 0.0, 0.0],
            cursor_light_intensity: 0.35,
            cursor_light_falloff: 60.0,
            leaf_shininess: 20.0,
            lamp_offset: [-0.62, 0.75],
            lamp_scale: 0.012,
            // Deliberately brighter and further-reaching than strict
            // realism would call for (comparable falloff to the window
            // light itself) — a game readability choice, so the plant
            // stays clearly visible at night, not just a faint sliver near
            // the fixture.
            lamp_intensity_max: 0.6,
            lamp_falloff: 5.0,
            lamp_off_tint: [0.35, 0.32, 0.28],
            lamp_on_tint: [1.3, 1.15, 0.7],
            pot_scale: 0.01,
            soil_scale: 0.01,
            seed_scale: 0.01,
            seed_min_swell_scale_fraction: 0.55,
            cotyledon_scale: 0.012,
            cotyledon_spread_angle: 0.4,
            flower_scale: 0.009,
            trellis_width_scale: 0.004,
            trellis_x_offset: 0.03,
            vine_trellis_lean_angle: 0.45,
            aerial_root_scale: 0.006,

            stem_height_scale: 0.006,
            stem_radius_scale: 0.35,
            branch_spread_angle: 0.5,

            leaf_scale: 0.007,
            leaf_base_spread_angle: 0.6,
            leaf_fold_max_angle: 0.8,
            leaf_droop_max_angle: 0.9,
            leaf_helio_max_angle: 0.3,
            rosette_leaf_min_spread_angle: 0.15,
            rosette_leaf_max_spread_angle: 1.1,
            rosette_leaf_splay_age: 500.0,
            // Tuned against leaf.svg's own baked green (#5a9c4e ≈ [0.35,
            // 0.61, 0.31]): multiplying by these lands roughly on a
            // yellowing olive and a dry brown respectively, rather than
            // picking absolute colors that would fight with the mesh's own.
            leaf_senescent_tint: [2.2, 1.2, 0.8],
            leaf_dead_tint: [1.5, 0.55, 0.65],
            leaf_shrivel_max_fraction: 0.3,
            leaf_occlusion_min_brightness: 0.4,

            // A visually modest but clearly noticeable slide relative to
            // the room's own scale — enough to read as "closer to/farther
            // from the window," not so much it risks pushing a tall plant
            // out past the wall's overscanned edge.
            pot_position_x_travel: 0.3,
            // A dull, desaturated gray-brown — deliberately *not* on the
            // same green-to-yellow-to-brown axis `leaf_senescent_tint`/
            // `leaf_dead_tint` use, so a rotted stem reads as a distinct
            // kind of "wrong" from drought/age/pest-yellowed leaves, not a
            // darker version of the same signal.
            stem_unhealthy_tint: [0.5, 0.48, 0.45],
            // Real dry potting mix is pale and sandy; real moist mix is
            // noticeably darker. Tuned as multipliers against soil.svg's
            // own baked color the same way every other tint here is —
            // lightened at bone-dry, darkened at a healthy moist level.
            soil_dry_tint: [1.3, 1.15, 0.85],
            soil_wet_tint: [0.7, 0.62, 0.5],
            // Past `SoilConfig::waterlogged_threshold`: darker and
            // desaturated further, with a slight cool/green cast — a
            // stagnant, oxygen-starved look distinct from ordinary healthy
            // moist soil, so a player has a *leading* visual cue (the soil
            // itself looks wrong) well before `Stats::root_health` actually
            // starts dropping.
            soil_waterlogged_tint: [0.4, 0.42, 0.38],

            roots_scale: 0.006,

            // Deliberately subtle (all four hover close to NO_TINT's
            // [1,1,1]) — a background mood cue that multiplies on top of
            // the day/night ambient tint, not a competing light source.
            season_summer_tint: [1.05, 1.0, 0.85],
            season_autumn_tint: [1.1, 0.82, 0.55],
            season_winter_tint: [0.78, 0.82, 0.95],
            season_spring_tint: [0.88, 1.05, 0.85],
        }
    }
}
