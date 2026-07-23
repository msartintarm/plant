//! Turns live simulation state (`sim::plant::Plant`, `sim::sun::SunState`)
//! into instance transforms — pure functions of `(state, SceneLayout)`, no
//! rendering concerns of their own. `render/mod.rs` is responsible for
//! turning these into GPU buffer writes and deciding what to actually draw
//! each frame.

use bytemuck::{Pod, Zeroable};

use crate::render::config::SceneLayout;
use crate::sim::moon::MoonAppearance;
use crate::sim::plant::{AerialRoot, Branch, Leaf, Plant, Side};
use crate::sim::sun::SunState;

/// Matches `scene.wgsl`'s `Instance` uniform struct field-for-field
/// (including order — WGSL computes each member's offset from its
/// predecessors' natural alignment, so this struct has to declare fields in
/// the exact same sequence for the byte layout to line up). `_pad0`/`_pad1`
/// exist purely so this struct's actual byte size (64) matches what WGSL
/// itself already computes as `Instance`'s size: a struct containing a
/// `vec3<f32>` member has a 16-byte overall alignment by the language's own
/// layout rules, which rounds sizes up (and inserts a gap before `tint`
/// specifically, since it's the one `vec3` here) regardless of whether
/// anything is declared there — `#[repr(C)]` doesn't know to add that
/// padding itself, so the buffer this uploads would otherwise fall short of
/// (or misalign against) what wgpu validates. Lives here rather than in
/// `render::mod::wgpu_engine` (the only thing that actually uploads it to
/// the GPU) because the conversion it does — zoom and aspect correction —
/// is plain arithmetic with no wgpu dependency, and is exactly the
/// arithmetic responsible for a real past bug (the sun/moon disc drifting
/// outside the window on a non-square canvas); keeping it here means that
/// arithmetic is unit-tested natively instead of only reachable via the
/// wasm build.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct InstanceUniform {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    /// The mesh's own (max |x|, max |y|) among its baked vertices (see
    /// `meshes::MeshRegistry::local_half_extent`) — [1,1] (never zero, to
    /// avoid a division by zero in `scene.wgsl`'s `fs_main`) for anything
    /// that isn't specular-lit (`shininess` 0). See `with_leaf_specular`.
    pub local_extent: [f32; 2],
    pub rotation: f32,
    pub depth: f32,
    /// Blinn-Phong shininess exponent for the cursor specular highlight —
    /// 0 disables it entirely (every non-leaf mesh). See
    /// `with_leaf_specular`/`SceneLayout::leaf_shininess`.
    pub shininess: f32,
    /// Nonzero for a light-*transmitting* surface (currently only the
    /// window pane — see `with_transmissive`) — `fs_main` glows it with the
    /// room light's own color/intensity directly rather than lighting it
    /// *by* that light at a distance the way every opaque mesh is.
    pub transmissive: f32,
    pub _pad0: [f32; 2],
    pub tint: [f32; 3],
    pub _pad1: f32,
}

impl InstanceUniform {
    /// `aspect` (width / height) divides the x scale so meshes authored in
    /// plain SVG units aren't stretched by a non-square canvas. `zoom` (see
    /// `SceneLayout::zoom`) uniformly pulls the camera back — callers
    /// drawing the wall pass `1.0` (see `render::mod::wgpu_engine::render`),
    /// everything else passes `layout.zoom`. `depth` (0.0 nearest .. 1.0
    /// farthest) feeds the real GPU depth buffer — see `apply_depth_look`
    /// for this same value's effect on how a mesh actually *looks*, applied
    /// separately (to `t` itself) before it ever reaches here. `local_
    /// extent`/`shininess` default to "not specular-lit" — see `with_leaf_
    /// specular` for the one place that overrides them.
    pub fn from_transform(t: &Transform, aspect: f32, zoom: f32, depth: f32) -> Self {
        InstanceUniform {
            offset: [t.offset[0] * zoom, t.offset[1] * zoom],
            scale: [t.scale_x * zoom / aspect, t.scale_y * zoom],
            local_extent: [1.0, 1.0],
            rotation: t.rotation,
            depth: depth.clamp(0.0, 1.0),
            shininess: 0.0,
            transmissive: 0.0,
            _pad0: [0.0; 2],
            tint: t.tint,
            _pad1: 0.0,
        }
    }
}

/// Marks an `InstanceUniform` as light-transmitting (the window pane) —
/// see its own field doc and `fs_main`.
pub fn with_transmissive(mut uniform: InstanceUniform) -> InstanceUniform {
    uniform.transmissive = 1.0;
    uniform
}

/// Turns on the cursor specular highlight for one `InstanceUniform` — see
/// `scene.wgsl`'s `fs_main` for the actual fake-dome-normal/Blinn-Phong
/// math this feeds. `local_half_extent` should be the mesh's own real
/// extent (see `meshes::MeshRegistry::local_half_extent`), *not* the margin-
/// adjusted one `outline_uniform` computes — the specular highlight tracks
/// the leaf's real surface, not its outline halo.
pub fn with_leaf_specular(mut uniform: InstanceUniform, local_half_extent: (f32, f32), shininess: f32) -> InstanceUniform {
    uniform.local_extent = [local_half_extent.0.max(f32::EPSILON), local_half_extent.1.max(f32::EPSILON)];
    uniform.shininess = shininess;
    uniform
}

/// A cheap "closer looks slightly bigger and brighter, farther looks
/// slightly smaller and dimmer" cue applied to a `Transform` *before* it
/// becomes an `InstanceUniform` — real perspective/lighting falloff would
/// need actual 3D geometry and a projection matrix, which this flat-vector-
/// art pipeline doesn't have; this fakes just enough of the effect on the
/// existing 2D placement math to read as depth, at essentially no runtime
/// cost. Distinct from `depth` itself reaching the GPU depth buffer (see
/// `InstanceUniform::from_transform`) — that's what resolves *occlusion*
/// between overlapping instances correctly; this is purely the look.
pub fn apply_depth_look(t: &Transform, depth: f32, layout: &SceneLayout) -> Transform {
    let depth = depth.clamp(0.0, 1.0);
    let perspective_scale = 1.0 + layout.depth_scale_falloff * (0.5 - depth);
    let dim = 1.0 - layout.depth_dim_falloff * depth;
    Transform {
        offset: t.offset,
        scale_x: t.scale_x * perspective_scale,
        scale_y: t.scale_y * perspective_scale,
        rotation: t.rotation,
        tint: [t.tint[0] * dim, t.tint[1] * dim, t.tint[2] * dim],
    }
}

/// A leaf's own individual depth (0.0..1.0, see `InstanceUniform::
/// from_transform`) — scattered around `SceneLayout::plant_depth` by a
/// cheap, stable hash of `Leaf::attach_height` (fixed for the leaf's whole
/// life, so this never jitters frame to frame or reshuffles as new leaves
/// grow in). Gives leaves genuine front-to-back variety instead of every
/// one sitting in exactly the same flat plane — both cosmetically (see
/// `apply_depth_look`) and for correct occlusion between overlapping
/// leaves via the real GPU depth buffer, rather than relying only on
/// manual draw order.
pub fn leaf_depth(leaf: &Leaf, layout: &SceneLayout) -> f32 {
    let h = (leaf.attach_height * 12.9898).sin() * 43758.5453;
    let unit = (h - h.floor()) * 2.0 - 1.0; // -1.0..1.0, deterministic per leaf
    (layout.plant_depth + unit as f32 * layout.leaf_depth_spread).clamp(0.0, 1.0)
}

/// Grows a mesh in place (same center, same depth) — the hover-highlight
/// cue for whichever leaf or stem segment `render::mod`'s GPU pick pass
/// currently reports under the cursor (see `encode_pick_target`/
/// `decode_pick_target`).
pub fn apply_hover_scale(t: &Transform, layout: &SceneLayout) -> Transform {
    Transform {
        offset: t.offset,
        scale_x: t.scale_x * layout.hover_scale_multiplier,
        scale_y: t.scale_y * layout.hover_scale_multiplier,
        rotation: t.rotation,
        tint: t.tint,
    }
}

/// Generous upper bound on how many leaf-or-stem-segment slots a *single*
/// plant could ever claim in the shared pick ID space (see
/// `encode_pick_target`) — `MAX_LEAVES` (96) plus room comfortably past
/// `render::mod`'s own `MAX_STEM_SEGMENT_DRAWABLES` (360 as of writing).
/// Kept as a plain rounded constant here, rather than importing that exact
/// value, so `scene.rs` doesn't need a dependency on the wasm-only module
/// just to know its own ID-space stride — this only has to stay *larger*
/// than leaf-slots-per-plant + stem-segment-slots-per-plant combined, not
/// exactly equal to it.
const PICK_TARGET_STRIDE: usize = 500;

/// What a GPU pick readback resolved to — see `render::mod`'s pick pass.
/// All three fields share one flat ID space (see `encode_pick_target`) so a
/// single texel readback can distinguish "which plant, and a leaf vs. a
/// stem segment on it" without a second pass or a second texture. Slot
/// numbering matches whatever each variant's own consumer expects: `Leaf`
/// mirrors `sim::plant::Plant::prune_leaf`'s flat main-stem-then-branches
/// ordering (see `leaf_depth`'s doc comment); `StemSegment` is an index
/// into `render::mod`'s own per-frame, per-plant `stem_segment_targets`,
/// which is what actually maps a segment back to "which grower, what
/// height."
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PickTarget {
    Leaf { plant_index: usize, slot: usize },
    StemSegment { plant_index: usize, slot: usize },
}

/// Turns a `PickTarget` into a flat color the pick pass's fragment shader
/// can output verbatim and a later texel readback can decode exactly — one
/// byte per channel, split across the R/G channels (comfortably covering
/// far more slots than any plant/pool combination needs). Reserves raw ID 0
/// (color `[0,0,0]`) for "nothing was there": the pick pass's offscreen
/// target is cleared to exactly this color every frame, so reading it back
/// unambiguously means nothing on any plant claimed that pixel.
pub fn encode_pick_target(target: PickTarget) -> [f32; 3] {
    let (plant_index, local_id) = match target {
        PickTarget::Leaf { plant_index, slot } => (plant_index, slot),
        // Offset past every leaf slot so the two pools can never collide
        // within one plant's own share of the ID space, regardless of
        // either pool's own size.
        PickTarget::StemSegment { plant_index, slot } => (plant_index, MAX_LEAVES + slot),
    };
    let raw = (plant_index * PICK_TARGET_STRIDE + local_id + 1) as u32;
    [(raw & 0xFF) as f32 / 255.0, ((raw >> 8) & 0xFF) as f32 / 255.0, 0.0]
}

/// The exact inverse of `encode_pick_target`, given the raw bytes read back
/// from the pick texture (an `Rgba8Unorm` target, so each channel comes
/// back as a plain `u8`) — `None` for the reserved "nothing was there"
/// color.
pub fn decode_pick_target(rgba: [u8; 4]) -> Option<PickTarget> {
    let raw = rgba[0] as usize | ((rgba[1] as usize) << 8);
    if raw == 0 {
        return None;
    }
    let id = raw - 1;
    let plant_index = id / PICK_TARGET_STRIDE;
    let local_id = id % PICK_TARGET_STRIDE;
    if local_id < MAX_LEAVES {
        Some(PickTarget::Leaf { plant_index, slot: local_id })
    } else {
        Some(PickTarget::StemSegment { plant_index, slot: local_id - MAX_LEAVES })
    }
}

/// The white outline halo drawn behind a plant-asset mesh (see `render::
/// mod`, which draws this copy first, then the normal-tinted one on top so
/// only a rim shows). This pipeline tessellates solid fills only — no
/// stroke geometry — so a real stroke isn't an option; a scaled-up
/// duplicate silhouette stands in for one instead.
///
/// `local_half_extent` is the mesh's own (max |x|, max |y|) among its
/// vertices, in its native (pre-`Transform`) SVG units — see `meshes::
/// MeshRegistry::local_half_extent`. It's needed *per axis, not as one
/// shared radius* because `scale` multiplies those native units directly
/// (see `scene.wgsl`'s `vs_main`): a flat pixel margin added straight to
/// `scale` would inflate a small mesh (few native units across) far more
/// than a large one for the same nominal amount, and a single shared radius
/// breaks down further for a mesh whose two axes aren't close to equal
/// (`stem_segment.svg` is long and thin — see `local_half_extent`'s own
/// doc comment for the real visual bug this caused). Each axis's margin is
/// expressed in clip-space units and divided by *that axis's own* extent to
/// convert it into the right `scale` delta for it specifically.
/// `depth` is the *paired normal mesh's own* depth, not yet biased —
/// `SceneLayout::outline_depth_bias` is added on top here so the outline
/// always sits a hair farther away than its own normal-colored counterpart
/// (via the real GPU depth test), guaranteeing the normal fill wins over
/// its own halo regardless of which one happens to be drawn first, while
/// still letting a genuinely nearer *neighboring* instance's own draw beat
/// this one fairly. `tint` is a plain parameter, not something this
/// function decides on its own.
/// The `[x, y]` delta `outline_uniform` adds on top of a mesh's own `scale`
/// so its outline halo renders `SceneLayout::outline_pixel_width` pixels
/// bigger on every side than the mesh itself — factored out so `render::
/// mod`'s GPU pick pass can give a leaf/stem-segment's clickable hitbox this
/// *exact same* margin (see `render::mod::wgpu_engine::write_pick_
/// transform`), matching whichever outline (white idle halo or
/// `SceneLayout::hover_outline_tint`'s red hover one) is actually visible
/// around it, rather than a hitbox that stops at the plain mesh's own
/// (smaller, invisible) edge.
pub fn outline_scale_margin(local_half_extent: (f32, f32), aspect: f32, canvas_width_px: f32, layout: &SceneLayout) -> [f32; 2] {
    if canvas_width_px <= 0.0 {
        return [0.0, 0.0];
    }
    // Clip space spans -1..1 (2.0 total) across `canvas_width_px` physical
    // pixels in x; y's own per-pixel clip delta is `aspect` times bigger/
    // smaller than x's (see `InstanceUniform::from_transform`'s own aspect
    // handling), hence `margin_clip_y`.
    let margin_clip_x = 2.0 * layout.outline_pixel_width / canvas_width_px;
    let margin_clip_y = margin_clip_x * aspect;
    let margin_x = if local_half_extent.0 > 0.0 { margin_clip_x / local_half_extent.0 } else { 0.0 };
    let margin_y = if local_half_extent.1 > 0.0 { margin_clip_y / local_half_extent.1 } else { 0.0 };
    [margin_x, margin_y]
}

pub fn outline_uniform(
    t: &Transform,
    aspect: f32,
    zoom: f32,
    local_half_extent: (f32, f32),
    layout: &SceneLayout,
    canvas_width_px: f32,
    depth: f32,
    tint: [f32; 3],
) -> InstanceUniform {
    let looked = apply_depth_look(t, depth, layout);
    let mut uniform = InstanceUniform::from_transform(&looked, aspect, zoom, depth + layout.outline_depth_bias);
    let margin = outline_scale_margin(local_half_extent, aspect, canvas_width_px, layout);
    uniform.scale[0] += margin[0];
    uniform.scale[1] += margin[1];
    uniform.tint = tint;
    uniform
}

/// A shared, per-frame (not per-instance) uniform — bound at group(1), see
/// `scene.wgsl`'s matching `SceneLight` struct. Drives a GPU-computed point
/// light at the window: every fragment's brightness falls off with distance
/// from `pos`, replacing the old single light-beam mesh with something that
/// actually looks like light filling the room. `cursor_*` is a second,
/// much-tighter point light at the player's own cursor (see `render::mod`'s
/// `pointer_pixel`) — both the diffuse glow and the leaf specular highlight
/// (see `scene.wgsl`'s `fs_main`) key off this same position. Explicit
/// `_pad` fields make Rust's layout match WGSL's own vec3-aligns-to-16-bytes
/// rule, which `#[repr(C)]` doesn't insert automatically the way WGSL does.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SceneLightUniform {
    pub pos: [f32; 2],
    pub intensity: f32,
    pub falloff: f32,
    pub color: [f32; 3],
    pub _pad0: f32,
    pub ambient_floor: [f32; 3],
    pub _pad1: f32,
    pub cursor_pos: [f32; 2],
    pub cursor_intensity: f32,
    pub cursor_falloff: f32,
    /// A third, fixed-position light (a wall lamp — see `SceneLayout::
    /// lamp_offset`) that only actually lights anything at night (eased by
    /// `sun.intensity`, computed in `new` below — the same "how dark is it
    /// right now" input the window light itself already reads). Zoom- and
    /// pan-adjusted the same way `pos` is, unlike `cursor_pos`.
    pub lamp_pos: [f32; 2],
    pub lamp_intensity: f32,
    pub lamp_falloff: f32,
}

impl SceneLightUniform {
    /// `zoom`/`pan`-adjusted the same way `InstanceUniform::from_transform`
    /// adjusts every other offset, so distances computed against `pos`/
    /// `lamp_pos` in the shader use the same convention every mesh's own
    /// world position already does. `cursor_ndc` is already in that same
    /// clip-space convention (see `render::mod`'s conversion from canvas
    /// pixels) and is *not* further zoom/pan-adjusted — unlike `pos`/`lamp_
    /// pos` (world-anchored points that should visually shrink toward
    /// center as the camera zooms out, and shift as it pans, with
    /// everything else), the cursor is a screen-space concept: wherever the
    /// mouse sits on screen is exactly where the highlight should be,
    /// regardless of zoom/pan. `None` (pointer not over the canvas) zeroes
    /// `cursor_intensity`, turning the whole term off in the shader.
    pub fn new(sun: &SunState, layout: &SceneLayout, zoom: f32, pan: [f32; 2], cursor_ndc: Option<[f32; 2]>) -> Self {
        let (cursor_pos, cursor_intensity) = match cursor_ndc {
            Some(pos) => (pos, layout.cursor_light_intensity),
            None => ([0.0, 0.0], 0.0),
        };
        let lamp_intensity = layout.lamp_intensity_max * (1.0 - sun.intensity.clamp(0.0, 1.0)) as f32;
        SceneLightUniform {
            pos: [layout.window_offset[0] * zoom + pan[0], layout.window_offset[1] * zoom + pan[1]],
            intensity: sun.intensity as f32,
            falloff: layout.scene_light_falloff,
            color: sun.color,
            _pad0: 0.0,
            ambient_floor: layout.night_ambient_color,
            _pad1: 0.0,
            cursor_pos,
            cursor_intensity,
            cursor_falloff: layout.cursor_light_falloff,
            lamp_pos: [layout.lamp_offset[0] * zoom + pan[0], layout.lamp_offset[1] * zoom + pan[1]],
            lamp_intensity,
            lamp_falloff: layout.lamp_falloff,
        }
    }
}

/// How much the lamp fixture's own tint (not the light it casts — see
/// `SceneLightUniform::new`) should read as "on," 0.0 (day) ..= 1.0
/// (night) — same driver, kept in sync so the fixture visibly lights up
/// exactly when it starts actually casting light.
pub fn lamp_on_fraction(sun_intensity: f64) -> f32 {
    (1.0 - sun_intensity.clamp(0.0, 1.0)) as f32
}

/// The wall lamp's own fixed transform (see `SceneLayout::lamp_offset`) —
/// tint is the caller's job (`render/mod.rs`, based on how "on" it
/// currently is), matching how `sky_object_transform` leaves the sun/moon
/// disc's tint to its own callers.
pub fn lamp_transform(layout: &SceneLayout) -> Transform {
    Transform {
        offset: layout.lamp_offset,
        scale_x: layout.lamp_scale,
        scale_y: layout.lamp_scale,
        rotation: 0.0,
        tint: NO_TINT,
    }
}

/// Fixed pool size for leaf instances — `render/mod.rs` pre-allocates this
/// many GPU buffers once and only ever draws the first `plant.leaves.len()`
/// of them, so a growing leaf count never needs new buffers created
/// mid-game. Sized to comfortably cover the main stem's leaves *and* every
/// branch's own leaves combined.
pub const MAX_LEAVES: usize = 96;
/// Fixed pool size for branch instances — see `sim::config::PlantConfig::
/// max_branches`, which this must be at least as large as.
pub const MAX_BRANCHES: usize = 8;

/// `stem_segment.svg`'s own local vertical extent (anchor at y=0, its
/// polygon's top points at y=-60 — see that file) — the raw mesh vertex
/// coordinates the vertex shader scales by `Transform::scale_y` directly
/// (`world = rotated_local_position * scale + offset`, see `scene.wgsl`),
/// *not* normalized to a unit height. `stem_like_transform`'s `scale_y`
/// already accounts for this correctly (it just sets the scale; the shader
/// does the rest), but anything computing where a *point along the stem*
/// ends up in clip space by hand — a leaf/branch's attachment offset, or
/// how far up frame the tip reaches for `dynamic_zoom` — has to multiply
/// this in too, or it places things ~60x closer to the pot than the
/// stem mesh it's actually attached to visually reaches. (This was a real
/// bug: leaves correctly recorded higher and higher `attach_height`s in the
/// simulation data, per `sim::plant`'s plastochron fix, but still rendered
/// clustered right at the pot while the stem itself stretched far past
/// them, because this factor was missing here.)
const STEM_LOCAL_HEIGHT: f32 = 60.0;

/// A position + angle at one specific point — a stem/branch's base
/// (`stem_base_frame`), or anywhere along its curve (`frame_at_height`).
/// See `StemCurve` for how a whole stem/branch's *curve* (not just one
/// point on it) is represented and walked.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub offset: [f32; 2],
    pub angle: f32,
}

fn side_sign(side: Side) -> f32 {
    match side {
        Side::Left => 1.0,
        Side::Right => -1.0,
    }
}

/// Rotates a point that's `local_y` up a stem/branch (in already-scaled clip
/// units) by `frame`'s angle, then places it at `frame`'s offset — the same
/// rigid-rotation-about-the-base approximation used for the stem, any
/// branch, and every leaf.
fn rotate_and_place(local_y: f32, frame: Frame) -> [f32; 2] {
    let (sin_a, cos_a) = frame.angle.sin_cos();
    [
        frame.offset[0] - local_y * sin_a,
        frame.offset[1] + local_y * cos_a,
    ]
}

/// Offset/scale/rotation/tint for one instance — everything the shared
/// render pipeline needs to place and light one mesh; see
/// `render/mod.rs`'s `InstanceUniform` for how this gets aspect-corrected
/// and uploaded.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub offset: [f32; 2],
    pub scale_x: f32,
    pub scale_y: f32,
    pub rotation: f32,
    /// Multiplies the mesh's own baked vertex color — [1,1,1] leaves it
    /// unchanged. See `ambient_tint`.
    pub tint: [f32; 3],
}

const NO_TINT: [f32; 3] = [1.0, 1.0, 1.0];

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub struct BackgroundSpec {
    pub mesh: &'static str,
    pub transform: Transform,
}

/// The wall/window — one room, shared by every pot in it (see
/// `plant_slot_base_anchor`), so these are built and drawn exactly once
/// regardless of how many plants share the room, unlike `pot_background`
/// below. Fixed for the whole session aside from resize's aspect recompute
/// and the wall/window's ambient tint (see `render/mod.rs`, which rewrites
/// every drawable's uniform every frame regardless, so "fixed" only means
/// the pre-tint/pre-aspect values here never change).
pub fn room_background(layout: &SceneLayout) -> Vec<BackgroundSpec> {
    vec![
        BackgroundSpec {
            mesh: "wall",
            transform: Transform {
                offset: [0.0, 0.0],
                scale_x: layout.wall_scale,
                scale_y: layout.wall_scale,
                rotation: 0.0,
                tint: NO_TINT,
            },
        },
        BackgroundSpec {
            mesh: "window_frame",
            transform: Transform {
                offset: layout.window_offset,
                scale_x: layout.window_scale,
                scale_y: layout.window_scale,
                rotation: 0.0,
                tint: NO_TINT,
            },
        },
        BackgroundSpec {
            mesh: "window_pane",
            transform: Transform {
                offset: layout.window_offset,
                scale_x: layout.window_scale,
                scale_y: layout.window_scale,
                rotation: 0.0,
                tint: NO_TINT,
            },
        },
    ]
}

/// One pot + its soil cap — unlike `room_background`, every plant gets its
/// own (see `PlantSlot::background_specs`), anchored at `layout.pot_anchor`
/// here and re-anchored per frame to that specific plant's own effective
/// position (see `plant_slot_base_anchor`/`pot_anchor_for_position`) by
/// `render`, the same way it already re-anchors for `Simulation::set_pot_
/// position`.
pub fn pot_background(layout: &SceneLayout) -> Vec<BackgroundSpec> {
    vec![
        BackgroundSpec {
            mesh: "pot",
            transform: Transform {
                offset: layout.pot_anchor,
                scale_x: layout.pot_scale,
                scale_y: layout.pot_scale,
                rotation: 0.0,
                tint: NO_TINT,
            },
        },
        BackgroundSpec {
            mesh: "soil",
            transform: Transform {
                offset: layout.pot_anchor,
                scale_x: layout.soil_scale,
                scale_y: layout.soil_scale,
                rotation: 0.0,
                tint: NO_TINT,
            },
        },
    ]
}

/// How the day/night cycle tints the wall/window — the most "in context"
/// way to show the light's current intensity/color without a HUD gauge:
/// the room itself visibly brightens/warms and dims/cools. Never fully
/// black at night (`SceneLayout::night_ambient_color` is a dim floor, not
/// zero) — a moonlit room, not a blackout.
pub fn ambient_tint(sun: &SunState, layout: &SceneLayout) -> [f32; 3] {
    let t = sun.intensity.clamp(0.0, 1.0) as f32;
    let night = layout.night_ambient_color;
    [
        lerp(night[0], sun.color[0], t),
        lerp(night[1], sun.color[1], t),
        lerp(night[2], sun.color[2], t),
    ]
}

pub fn outline_tint_for_sun(sun_intensity: f64, layout: &SceneLayout) -> [f32; 3] {
    let t = sun_intensity.clamp(0.0, 1.0) as f32;
    let night = layout.outline_tint_night;
    let day = layout.outline_tint_day;
    [lerp(night[0], day[0], t), lerp(night[1], day[1], t), lerp(night[2], day[2], t)]
}

/// Where the sun (or moon) sits inside the window pane — the most "in
/// context" way to show the light's current *angle*, alongside the tint
/// above for its intensity/color. Shares one arc shape for both bodies (not
/// a separate lunar model — a deliberate simplification): whichever is up
/// traces a low-to-high path across the window as its own "day" progresses.
/// Caller decides whether to actually draw the sun/moon based on each
/// body's own elevation (see `render/mod.rs`).
///
/// Takes `aspect` because, unlike every other transform here (which just
/// hands `scale_x`/`scale_y` to `InstanceUniform::from_transform` and lets
/// *that* divide x by aspect), this position is computed by hand in local
/// window-pane units — it has to redo the same x-aspect-correction the
/// window mesh's own vertices get, or it drifts off-pane on non-square
/// canvases (this was a real bug: without it, the disc's x offset was
/// computed as if the canvas were square, and it visibly poked outside the
/// window frame on the actual ~16:10 canvas).
pub fn sky_object_transform(sun: &SunState, layout: &SceneLayout, aspect: f32) -> Transform {
    let elevation_factor = sun.elevation.abs().clamp(0.0, 1.0) as f32;
    let local_x = lerp(
        layout.sky_object_local_x_range[0],
        layout.sky_object_local_x_range[1],
        sun.azimuth as f32,
    );
    let local_y = lerp(
        layout.sky_object_local_y_range[0],
        layout.sky_object_local_y_range[1],
        elevation_factor,
    );
    Transform {
        offset: [
            layout.window_offset[0] + local_x * layout.window_scale / aspect,
            layout.window_offset[1] + local_y * layout.window_scale,
        ],
        scale_x: layout.sky_object_scale,
        scale_y: layout.sky_object_scale,
        rotation: 0.0,
        tint: NO_TINT,
    }
}

/// The moon's own on-screen position — diagonally opposite the sun's
/// general direction rather than tracing an independent arc that could
/// visually coincide with it. Built from `day_progress` directly (not
/// `SunState::azimuth`, which holds flat outside daytime and jumps at the
/// day/night wrap) so this sweeps continuously across the full 24-hour
/// cycle with no freeze or snap. `sun_elevation` (the sun's own, which
/// unlike its azimuth never holds) still drives how high the mirrored
/// position sits.
pub fn moon_position_opposite_sun(day_progress: f64, sun_elevation: f64) -> SunState {
    SunState {
        elevation: 1.0 - sun_elevation.abs().clamp(0.0, 1.0),
        azimuth: (day_progress + 0.5).rem_euclid(1.0),
        intensity: 0.0,
        color: NO_TINT,
    }
}

/// A second "shadow" disc, same size and position as the moon's own (see
/// `sky_object_transform`), shifted sideways by `appearance.illuminated_
/// fraction` — drawn on top of the moon disc, this is the classic flat-icon
/// two-circle trick for a crescent/gibbous moon: at illuminated_fraction 0
/// the shadow disc exactly covers the moon (fully dark), at 1.0 it's shifted
/// a full diameter away (moon fully lit), in between it eats into one side.
/// Shifts right while waxing, left while waning, rotated by `tilt_angle`
/// (see `sim::moon::crescent_tilt_angle`) — 0.0 reproduces the old
/// pure-horizontal shift exactly. Only the x component needs the `aspect`
/// divide (see `sky_object_transform`'s own offset handling — offsets
/// aren't aspect-corrected automatically the way scale is).
pub fn moon_shadow_transform(
    sky_transform: &Transform,
    appearance: &MoonAppearance,
    tilt_angle: f32,
    layout: &SceneLayout,
    aspect: f32,
) -> Transform {
    let side_sign = if appearance.waxing { 1.0 } else { -1.0 };
    let magnitude = 2.0 * layout.sky_object_scale * appearance.illuminated_fraction as f32 * side_sign;
    let (sin_t, cos_t) = tilt_angle.sin_cos();
    Transform {
        offset: [
            sky_transform.offset[0] + magnitude * cos_t / aspect,
            sky_transform.offset[1] + magnitude * sin_t,
        ],
        scale_x: sky_transform.scale_x,
        scale_y: sky_transform.scale_y,
        rotation: 0.0,
        tint: layout.moon_shadow_tint,
    }
}

/// Fades a night-sky tint toward the window's own ambient sky color as
/// `sun_intensity` rises — a real daytime moon isn't literally darker, it
/// just loses contrast against a much brighter scattering sky, which is
/// why it's hard to spot. `max_fade` caps the blend short of 1.0 so it
/// stays at least faintly visible at noon rather than vanishing outright.
pub fn daytime_fade(base: [f32; 3], ambient: [f32; 3], sun_intensity: f64, max_fade: f32) -> [f32; 3] {
    let t = sun_intensity.clamp(0.0, 1.0) as f32 * max_fade;
    [lerp(base[0], ambient[0], t), lerp(base[1], ambient[1], t), lerp(base[2], ambient[2], t)]
}

/// Only actually drawn pre-germination (`Stage::Seed`) — see `render/mod.rs`.
pub fn seed_transform(layout: &SceneLayout, swell_fraction: f64) -> Transform {
    let t = swell_fraction.clamp(0.0, 1.0) as f32;
    let scale = layout.seed_scale * lerp(layout.seed_min_swell_scale_fraction, 1.0, t);
    Transform {
        offset: layout.pot_anchor,
        scale_x: scale,
        scale_y: scale,
        rotation: 0.0,
        tint: NO_TINT,
    }
}

/// A fixed pose — cotyledons don't track light or fold like true leaves do
/// (a deliberate scope simplification). Only drawn from `Stage::Sprout`
/// onward.
pub fn cotyledon_transform(layout: &SceneLayout, side: Side) -> Transform {
    Transform {
        offset: layout.pot_anchor,
        scale_x: layout.cotyledon_scale,
        scale_y: layout.cotyledon_scale,
        rotation: side_sign(side) * layout.cotyledon_spread_angle,
        tint: NO_TINT,
    }
}

/// The zoom to actually use this frame: `layout.zoom` (the closest-in the
/// camera ever gets) unless some plant's own current reach — the tallest
/// main stem height or branch tip (its own attach height plus its own
/// length) across every plant sharing the room, or the rightmost plant's
/// own pot position (see `plant_slot_base_anchor`) — would poke past
/// `layout.zoom_visible_half_height`/`zoom_visible_half_width` at that zoom,
/// in which case the camera pulls back just enough to keep everything in
/// frame. Never zooms in *tighter* than `layout.zoom`, so a small seedling
/// (alone in the room) keeps the same framing the old fixed zoom always
/// gave it.
pub fn dynamic_zoom_for_room<'a>(plants: impl Iterator<Item = &'a Plant>, layout: &SceneLayout) -> f32 {
    let mut plant_count = 0;
    let mut tallest_reach = 0.0_f64;
    for plant in plants {
        plant_count += 1;
        let reach = plant
            .branches
            .iter()
            .map(|b| b.attach_height + b.height)
            .fold(plant.height, f64::max);
        tallest_reach = tallest_reach.max(reach);
    }

    let top_y_unzoomed =
        layout.pot_anchor[1] + tallest_reach as f32 * layout.stem_height_scale * STEM_LOCAL_HEIGHT;
    let vertical_zoom = if top_y_unzoomed <= 0.0 {
        layout.zoom
    } else {
        layout.zoom.min(layout.zoom_visible_half_height / top_y_unzoomed)
    };

    if plant_count <= 1 {
        return vertical_zoom;
    }
    // Only the rightmost pot's own anchor is checked (see `plant_slot_
    // base_anchor` — slots only ever step further right, never left), the
    // same "just the one edge that can actually run off frame" simplification
    // `dynamic_zoom` already makes for the vertical case above.
    let rightmost_anchor_x = layout.pot_anchor[0] + (plant_count - 1) as f32 * layout.plant_slot_spacing;
    if rightmost_anchor_x <= 0.0 {
        return vertical_zoom;
    }
    let horizontal_zoom = layout.zoom.min(layout.zoom_visible_half_width / rightmost_anchor_x);
    vertical_zoom.min(horizontal_zoom)
}

/// The main stem's own base reference — everything along its curve extends
/// from here (see `StemCurve`). Unlike a plain `Frame`, this carries no
/// lean/droop of its own: those are now spread across the stem's recorded
/// segment history instead of being one rigid whole-stem rotation — see
/// `sim::plant`'s module docs and `StemCurve`'s own doc comment.
pub fn stem_base_frame(layout: &SceneLayout) -> Frame {
    Frame { offset: layout.pot_anchor, angle: 0.0 }
}

/// Everything needed to walk a stem or branch's own curve — bundled so
/// `frame_at_height`/`stem_segment_transforms`/`leaf_transform_in_frame`
/// take one value instead of five. A real stem doesn't bend as one rigid
/// rotation about its base: older, already-stiffened tissue keeps whatever
/// curvature it had when it formed, while only the still-*growing* tip is
/// flexible enough to respond to the plant's *current* lean/droop. This is
/// modeled by walking `segment_history` — `lean_angle` frozen at the moment
/// each fixed-height segment stopped being that growing tip (see
/// `sim::plant::record_stem_segments`) — and only falling back to
/// `current_lean_angle`/`current_extra_angle` (stem_droop — turgor loss
/// only affects currently-soft, still-growing tissue, not
/// already-lignified older segments) once the walk runs past the end of
/// that recorded history, for whatever's still actively growing.
#[derive(Debug, Clone, Copy)]
pub struct StemCurve<'a> {
    pub base: Frame,
    pub segment_history: &'a [f64],
    pub current_lean_angle: f64,
    pub current_extra_angle: f64,
    pub segment_height_interval: f64,
    /// `sim::plant::Plant::lean_angle` (and every recorded history value)
    /// is an unsigned magnitude that just grows toward the light — it has
    /// no opinion on which screen direction that means, since `sim` has no
    /// notion of where the window is rendered. `rotate_and_place`'s
    /// rotation convention bends a *positive* angle toward -x (left,
    /// standard counter-clockwise rotation); this flips that to +1.0 or
    /// -1.0 so the composed angle actually leans toward wherever
    /// `render/mod.rs` says the window is. Real bug this fixed: with this
    /// always implicitly +1.0 (no flip), the stem leaned away from a
    /// window placed at positive x, the opposite of real phototropism.
    pub lean_sign: f32,
    /// `GrowthHabit::Vine` only — one-time lean its first segment takes
    /// toward the trellis (see `SceneLayout::vine_trellis_lean_angle`), 0.0
    /// for every other habit.
    pub vine_base_lean_angle: f32,
}

/// Which sign to apply to an unsigned `lean_angle` magnitude (see
/// `StemCurve::lean_sign`) so it actually leans toward the window: -1.0 if
/// the window sits at or to the right of the pot (matching
/// `rotate_and_place`'s convention, where a positive angle bends toward
/// -x), +1.0 if it's to the left.
pub fn lean_sign_toward_window(window_offset: [f32; 2], pot_anchor: [f32; 2]) -> f32 {
    if window_offset[0] >= pot_anchor[0] {
        -1.0
    } else {
        1.0
    }
}

/// One segment's own angle, by index along the curve — a recorded
/// historical value if it's already been frozen, otherwise (this is the
/// still-growing tip) today's live lean plus whatever extra (stem_droop)
/// only applies to actively-growing tissue. `lean_sign` orients the result
/// toward the actual window direction — see `StemCurve::lean_sign`.
fn stem_segment_angle(curve: &StemCurve, segment_index: usize) -> f32 {
    let lean_and_extra = if segment_index < curve.segment_history.len() {
        curve.segment_history[segment_index] as f32
    } else {
        (curve.current_lean_angle + curve.current_extra_angle) as f32
    };
    let vine_lean = if segment_index == 0 { curve.vine_base_lean_angle } else { 0.0 };
    curve.base.angle + curve.lean_sign * lean_and_extra + vine_lean
}

/// Walks `curve` from its base up to (at most) `target_height`, returning
/// the frame at exactly that point — used to place anything that attaches
/// at a specific height along it: a leaf, a branch's own attachment point,
/// or the flower. See `stem_segment_transforms` for rendering the curve
/// itself (same per-segment angle logic, but keeping every intermediate
/// frame instead of just the last).
pub fn frame_at_height(curve: &StemCurve, target_height: f64, layout: &SceneLayout) -> Frame {
    let mut offset = curve.base.offset;
    let mut angle = curve.base.angle;
    let mut height_walked = 0.0_f64;
    let mut index = 0;
    while height_walked < target_height - 1e-9 && curve.segment_height_interval > 0.0 {
        angle = stem_segment_angle(curve, index);
        let this_height = curve.segment_height_interval.min(target_height - height_walked);
        let local_y = this_height as f32 * layout.stem_height_scale * STEM_LOCAL_HEIGHT;
        offset = rotate_and_place(local_y, Frame { offset, angle });
        height_walked += this_height;
        index += 1;
    }
    Frame { offset, angle }
}

/// Every segment's own transform along `curve`, from its base up to
/// `total_height` — each one its own `stem_segment` mesh instance covering
/// just that segment's own portion of the height (the fixed
/// `segment_height_interval` for a completed one, whatever's left for the
/// still-growing tip), chained end to end. This is what makes the whole
/// stem read as a gentle sweep — straighter down low (formed early, before
/// much lean had accumulated), more bent up high (recent growth, under
/// whatever lean is current *now*) — instead of one rigid line pivoting
/// from the base. See `frame_at_height` for placing a single *point* along
/// the same curve instead of rendering the curve itself.
pub fn stem_segment_transforms(
    curve: &StemCurve,
    total_height: f64,
    stem_radius: f64,
    tint: [f32; 3],
    layout: &SceneLayout,
) -> Vec<Transform> {
    let mut transforms = Vec::new();
    let mut offset = curve.base.offset;
    let mut height_walked = 0.0_f64;
    let mut index = 0;
    while height_walked < total_height - 1e-9 && curve.segment_height_interval > 0.0 {
        let angle = stem_segment_angle(curve, index);
        let this_height = curve.segment_height_interval.min(total_height - height_walked);
        let frame = Frame { offset, angle };
        transforms.push(Transform {
            offset: frame.offset,
            scale_x: (layout.stem_radius_scale * stem_radius as f32).max(0.0),
            scale_y: (layout.stem_height_scale * this_height as f32).max(0.0),
            rotation: frame.angle,
            tint,
        });
        let local_y = this_height as f32 * layout.stem_height_scale * STEM_LOCAL_HEIGHT;
        offset = rotate_and_place(local_y, frame);
        height_walked += this_height;
        index += 1;
    }
    transforms
}

/// Where the pot sits along the floor between the window and the far side
/// of the room, as a fraction of `SceneLayout::pot_position_x_travel` — see
/// `sim::room::PotPosition` for the same 0.0 (at the window)..1.0 (far from
/// it) convention the simulation side already uses. `position == 0.5`
/// exactly reproduces `base_anchor` unchanged (the room's originally-tuned
/// layout), so the *default* `Simulation::set_pot_position` value renders
/// identically to before this mechanic existed; moving the slider off that
/// midpoint is what actually shifts anything.
pub fn pot_anchor_for_position(base_anchor: [f32; 2], position: f64, travel: f32) -> [f32; 2] {
    // 0.0 must be a no-op here, matching room::apply_pot_position's own
    // neutral point (full light, no draft) — a mismatched "neutral" between
    // the two (this used to be 0.5) meant the default pot_position silently
    // cut every session's light and temperature from frame one.
    let position = position.clamp(0.0, 1.0) as f32;
    let shift = -travel * position;
    [base_anchor[0] + shift, base_anchor[1]]
}

/// Where the *n*-th plant slot's own base anchor sits, before `pot_anchor_
/// for_position`'s own per-plant "how far from the window" adjustment is
/// layered on top. Slot 0 always reproduces `layout.pot_anchor` exactly
/// (matching how `position == 0.0` is a no-op for `pot_anchor_for_
/// position`), so a single-plant session renders pixel-identical to before
/// multi-plant support existed. Additional slots step sideways along the
/// windowsill by `layout.plant_slot_spacing` each — this is a real side-
/// profile scene with no true depth axis to place several pots at once, so
/// "several pots side by side" is the only spatial arrangement this flat
/// art style can actually depict; each one's own `pot_position` still
/// separately nudges it toward/away from the window from there.
pub fn plant_slot_base_anchor(layout: &SceneLayout, plant_index: usize) -> [f32; 2] {
    [layout.pot_anchor[0] + plant_index as f32 * layout.plant_slot_spacing, layout.pot_anchor[1]]
}

/// A specific plant slot's actual world-space pot anchor — `plant_slot_
/// base_anchor` stepped sideways for `plant_index`, then shifted by that
/// plant's own `pot_position` slider (see `pot_anchor_for_position`).
/// Doesn't include camera pan or the `* zoom` `InstanceUniform::from_
/// transform` applies — see `render::mod::wgpu_engine::render`'s own
/// `effective_pot_anchor`/`pan` handling for that, which this exists to be
/// the single source of truth for (both the actual render loop and
/// `Simulation::plant_pot_hud`, which projects this same anchor into CSS-
/// pixel space for the per-pot water gauge, call this instead of each
/// separately re-deriving it).
pub fn plant_pot_world_anchor(layout: &SceneLayout, plant_index: usize, pot_position: f64) -> [f32; 2] {
    let base = plant_slot_base_anchor(layout, plant_index);
    pot_anchor_for_position(base, pot_position, layout.pot_position_x_travel)
}

/// Multiplies the stem mesh's own baked color as `Plant::root_health` drops
/// — a distinct visual channel from leaf senescence (see `SceneLayout::
/// stem_unhealthy_tint`'s doc comment on why root damage needs its own
/// signal rather than reusing the leaf yellow-to-brown axis). `NO_TINT`
/// (unchanged) at full vitality, easing toward `stem_unhealthy_tint` as it
/// drops to zero. Takes `vitality` as a plain 0.0..1.0 fraction rather than
/// reading `Plant::root_health` directly — the caller passes in `Decision::
/// Vegetative::effective_water_factor` (water availability *after*
/// discounting root health and pot-bound stress), not raw root health alone:
/// root rot is only one of several ways a plant can be failing (plain
/// drought/neglect is the far more common one), and all of them should
/// visibly show up here, not just the overwatering-specific case. See
/// `render/mod.rs`'s call site.
pub fn stem_health_tint(vitality: f64, layout: &SceneLayout) -> [f32; 3] {
    let t = 1.0 - vitality.clamp(0.0, 1.0) as f32;
    [
        lerp(NO_TINT[0], layout.stem_unhealthy_tint[0], t),
        lerp(NO_TINT[1], layout.stem_unhealthy_tint[1], t),
        lerp(NO_TINT[2], layout.stem_unhealthy_tint[2], t),
    ]
}

/// Tints the soil surface by how wet it currently is — the most "in
/// context" leading indicator of overwatering there is: this responds to
/// raw moisture immediately, well before sustained waterlogging has had
/// time to actually damage `Plant::root_health` (see `SoilConfig::
/// waterlog_grace_period`), the same "show it in the scene, not just a HUD
/// gauge" philosophy `ambient_tint`/`senescence_tint` already use. Three-
/// stage lerp (dry → wet → waterlogged) rather than a single dry-to-
/// waterlogged line, since "healthy moist soil" and "actively waterlogged"
/// are visually and functionally distinct, not just two points on one
/// scale.
pub fn soil_moisture_tint(moisture: f64, waterlogged_threshold: f64, layout: &SceneLayout) -> [f32; 3] {
    let moisture = moisture.clamp(0.0, 1.0);
    if moisture <= waterlogged_threshold {
        let range = waterlogged_threshold.max(1e-9);
        let t = (moisture / range) as f32;
        [
            lerp(layout.soil_dry_tint[0], layout.soil_wet_tint[0], t),
            lerp(layout.soil_dry_tint[1], layout.soil_wet_tint[1], t),
            lerp(layout.soil_dry_tint[2], layout.soil_wet_tint[2], t),
        ]
    } else {
        let range = (1.0 - waterlogged_threshold).max(1e-9);
        let t = ((moisture - waterlogged_threshold) / range) as f32;
        [
            lerp(layout.soil_wet_tint[0], layout.soil_waterlogged_tint[0], t),
            lerp(layout.soil_wet_tint[1], layout.soil_waterlogged_tint[1], t),
            lerp(layout.soil_wet_tint[2], layout.soil_waterlogged_tint[2], t),
        ]
    }
}

/// Tints the wall by where `phase` (see `sim::season::SeasonState`) falls
/// in the year — a genuinely "wall-integrated" way to show the current
/// season (as the user asked for), rather than a HUD readout: the room's
/// own base color slowly drifts summer → autumn → winter → spring → summer
/// over the course of a year, the same "show it in the scene, not a gauge"
/// idiom `ambient_tint` already uses for the day/night cycle. Continuous
/// (driven directly by `phase`, not the discrete `Season` label), so the
/// wall eases between quarters rather than snapping at each boundary — four
/// keyframes, each lerped against its neighbor.
pub fn seasonal_wall_tint(phase: f64, layout: &SceneLayout) -> [f32; 3] {
    let phase = phase.rem_euclid(1.0);
    let keyframes = [
        layout.season_summer_tint,
        layout.season_autumn_tint,
        layout.season_winter_tint,
        layout.season_spring_tint,
        layout.season_summer_tint,
    ];
    let segment = (phase * 4.0) as f32;
    let index = (segment.floor() as usize).min(3);
    let t = segment - index as f32;
    let a = keyframes[index];
    let b = keyframes[index + 1];
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

/// The optional climbing-support pole/lattice a `PlantConfig::trellis_
/// height` species is trained against (see that field's doc comment) —
/// `None` for a freestanding habit (`trellis_height: None`). Unlike a
/// stem, it's a single rigid instance with no `StemCurve`/lean of its own
/// (a real stake doesn't bend) — just anchored at the pot and scaled, via
/// the same `stem_height_scale`/`STEM_LOCAL_HEIGHT` conversion a stem
/// segment uses, to reach exactly as tall as the plant's own stem would at
/// that same height (see trellis.svg's own doc comment).
pub fn trellis_transform(trellis_height: Option<f64>, layout: &SceneLayout) -> Option<Transform> {
    let trellis_height = trellis_height?;
    Some(Transform {
        offset: [layout.pot_anchor[0] + layout.trellis_x_offset, layout.pot_anchor[1]],
        scale_x: layout.trellis_width_scale,
        scale_y: (layout.stem_height_scale * trellis_height as f32).max(0.0),
        rotation: 0.0,
        tint: NO_TINT,
    })
}

/// One `AerialRoot`'s own transform — anchored at its recorded height
/// along the main stem (via `frame_at_height`; every aerial root forms
/// while `leans_freely` is false, i.e. while the stem's own angle is
/// exactly 0 the whole time, so this is always a straight-up walk, never a
/// curved one) and reaching sideways toward whichever side the trellis
/// actually sits on — `aerial_root.svg` is authored reaching in +x, so
/// this only ever needs a fixed 0 or a 180° flip, never an arbitrary
/// angle, unlike a stem/leaf's own rotation.
pub fn aerial_root_transform(root: &AerialRoot, main_curve: &StemCurve, layout: &SceneLayout) -> Transform {
    let attach = frame_at_height(main_curve, root.attach_height, layout);
    Transform {
        offset: attach.offset,
        scale_x: layout.aerial_root_scale,
        scale_y: layout.aerial_root_scale,
        rotation: if layout.trellis_x_offset >= 0.0 { 0.0 } else { std::f32::consts::PI },
        tint: NO_TINT,
    }
}

/// Derives a branch's own curve from where it attaches along the main
/// stem's — its base is wherever `main_stem`'s curve actually is at
/// `branch.attach_height` (not extrapolated from a single rigid angle, so a
/// branch attaches at the *true* point on a curved main stem), rotated by
/// its fixed outward spread; from there it grows its own curve using its
/// own recorded history/live lean/droop, entirely independent of the main
/// stem's beyond that one shared starting point.
pub fn branch_curve<'a>(main_stem: &StemCurve, branch: &'a Branch, layout: &SceneLayout) -> StemCurve<'a> {
    let attach_frame = frame_at_height(main_stem, branch.attach_height, layout);
    StemCurve {
        base: Frame {
            offset: attach_frame.offset,
            angle: attach_frame.angle + side_sign(branch.side) * layout.branch_spread_angle,
        },
        segment_history: &branch.segment_history,
        current_lean_angle: branch.lean_angle,
        current_extra_angle: branch.droop,
        segment_height_interval: main_stem.segment_height_interval,
        lean_sign: main_stem.lean_sign,
        vine_base_lean_angle: main_stem.vine_base_lean_angle,
    }
}

/// A terminal bloom at the *main stem's* own tip — see
/// `sim::config::PlantConfig::flowering_height_threshold` for when the
/// caller should actually draw this (this always computes a position;
/// whether it's mature enough to show one is the caller's decision, not
/// this function's, matching how `render/mod.rs` already decides sun vs.
/// moon rather than `scene.rs` doing it). Sits at whatever the curve's own
/// angle is at the very tip, same as a leaf attached at the stem's current
/// full height would.
///
/// `bloom_intensity` (0.0..1.0, see `Plant::bloom_intensity`) scales the
/// whole bloom directly — a real flower opens/closes gradually in size,
/// not just fades in place. Floored at `layout.bud_min_intensity` once
/// `mature_enough` (the plant has reached `PlantConfig::
/// flowering_height_threshold` at least once) rather than left to reach
/// exactly 0.0 between bloom flushes — see that field's own doc comment:
/// a real flowering-age plant keeps a small closed bud visible while
/// resting, it doesn't disappear and reappear from nothing every cycle.
/// Still exactly zero pre-maturity (`!mature_enough`) — a plant that's
/// never reached flowering height hasn't formed any bud yet to show, the
/// one case this doesn't just duplicate `bloom_intensity_target` already
/// being 0 there (rather than reading `flowering_height_threshold` itself,
/// which lives in `PlantConfig`, not this render-only `SceneLayout`).
pub fn flower_transform(curve: &StemCurve, total_height: f64, bloom_intensity: f64, layout: &SceneLayout) -> Transform {
    let frame = frame_at_height(curve, total_height, layout);
    let scale = layout.flower_scale * bloom_intensity.clamp(0.0, 1.0) as f32;
    Transform {
        offset: frame.offset,
        scale_x: scale,
        scale_y: scale,
        rotation: frame.angle,
        tint: NO_TINT,
    }
}

/// A leaf's final angle composes four independent things: its fixed
/// phyllotactic spread (which side it's on), nyctinastic fold and drought
/// droop (both pull it back toward/past vertical), heliotropic tracking (a
/// small bias toward the window, same direction regardless of side), and
/// whatever it's attached to's own angle at that exact point along the
/// curve (a leaf tips with whatever segment it's actually on, not the
/// stem/branch's base or its current tip). Position likewise walks
/// `curve` out to `leaf.attach_height` — works identically for a
/// main-stem leaf or a branch's leaf, whichever `curve` is passed.
/// `shade_factor` is this leaf's own entry from `sim::plant::
/// self_shading_factors` (1.0 = nothing of this grower's own canopy above
/// it, lower = increasingly buried) — darkens the tint on top of (not
/// instead of) senescence's color shift, so a leaf can read as both "old"
/// and "shaded" at once rather than one hiding the other.
pub fn leaf_transform_in_frame(curve: &StemCurve, leaf: &Leaf, shade_factor: f64, layout: &SceneLayout) -> Transform {
    let frame = frame_at_height(curve, leaf.attach_height, layout);

    let spread = layout.leaf_base_spread_angle
        - layout.leaf_fold_max_angle * leaf.fold as f32
        - layout.leaf_droop_max_angle * leaf.droop as f32;
    let helio = layout.leaf_helio_max_angle * leaf.helio_angle as f32;
    let rotation = side_sign(leaf.side) * spread + helio + frame.angle;

    let (scale, tint) = leaf_visual(leaf, shade_factor, layout);
    Transform { offset: frame.offset, scale_x: scale, scale_y: scale, rotation, tint }
}

/// Scale/tint shared by every leaf placement function — maturity growth,
/// senescence shrivel/color, and shade occlusion. See `leaf_transform_in_
/// frame`'s own doc comment for why "smaller wins" rather than averaging
/// maturity and shrivel together.
fn leaf_visual(leaf: &Leaf, shade_factor: f64, layout: &SceneLayout) -> (f32, [f32; 3]) {
    let maturity = (leaf.maturity as f32).max(0.05);
    let shrivel = 1.0 - layout.leaf_shrivel_max_fraction * leaf.senescence as f32;
    let scale = layout.leaf_scale * maturity * shrivel;
    let occlusion = lerp(layout.leaf_occlusion_min_brightness, 1.0, shade_factor.clamp(0.0, 1.0) as f32);
    let base_tint = senescence_tint(leaf.senescence as f32, layout);
    (scale, [base_tint[0] * occlusion, base_tint[1] * occlusion, base_tint[2] * occlusion])
}

/// `GrowthHabit::BasalRosette` leaf placement — no stem to walk (leaves
/// attach directly at `base.offset`), fan spread widens with `leaf.age`
/// (young leaves upright/central, older ones splayed outward) instead of
/// the fixed left/right split an upright cane's leaves use.
pub fn rosette_leaf_transform(base: Frame, leaf: &Leaf, shade_factor: f64, layout: &SceneLayout) -> Transform {
    let age_t = ((leaf.age / layout.rosette_leaf_splay_age) as f32).clamp(0.0, 1.0);
    let spread = lerp(layout.rosette_leaf_min_spread_angle, layout.rosette_leaf_max_spread_angle, age_t)
        - layout.leaf_fold_max_angle * leaf.fold as f32
        - layout.leaf_droop_max_angle * leaf.droop as f32;
    let helio = layout.leaf_helio_max_angle * leaf.helio_angle as f32;
    let rotation = side_sign(leaf.side) * spread + helio;

    let (scale, tint) = leaf_visual(leaf, shade_factor, layout);
    Transform { offset: base.offset, scale_x: scale, scale_y: scale, rotation, tint }
}

/// Green (healthy) → yellowing olive → brown, as `Leaf::senescence` rises —
/// the most "in context" way to show an individual leaf's health without a
/// per-leaf HUD gauge, same philosophy as `ambient_tint` for the day/night
/// cycle. Two-stage lerp (via the 0.5 midpoint) rather than one straight
/// green-to-brown line, since real chlorophyll breakdown visibly passes
/// through yellow on its way to brown, not a direct blend of the two.
fn senescence_tint(senescence: f32, layout: &SceneLayout) -> [f32; 3] {
    let senescence = senescence.clamp(0.0, 1.0);
    if senescence <= 0.5 {
        let t = senescence / 0.5;
        [
            lerp(NO_TINT[0], layout.leaf_senescent_tint[0], t),
            lerp(NO_TINT[1], layout.leaf_senescent_tint[1], t),
            lerp(NO_TINT[2], layout.leaf_senescent_tint[2], t),
        ]
    } else {
        let t = (senescence - 0.5) / 0.5;
        [
            lerp(layout.leaf_senescent_tint[0], layout.leaf_dead_tint[0], t),
            lerp(layout.leaf_senescent_tint[1], layout.leaf_dead_tint[1], t),
            lerp(layout.leaf_senescent_tint[2], layout.leaf_dead_tint[2], t),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Local (pre-scale) half-extents baked from the actual SVG assets —
    // see engine/assets/svg/{sun,moon,window_frame,wall}.svg. These are the
    // numbers a human would otherwise have to hold in their head while
    // eyeballing a screenshot ("is the disc still inside the window pane?
    // does the wall still cover the corners?"); keeping them here as
    // explicit constants is what lets that same check run as `cargo test`.
    // Update these if the corresponding SVG's dimensions ever change.
    const SUN_MOON_LOCAL_RADIUS: f32 = 15.0; // sun.svg r=15; moon.svg r=12, so this is the more conservative of the two
    const WINDOW_PANE_LOCAL_HALF_WIDTH: f32 = 25.0; // window_frame.svg inner pane rect, x=-25 w=50
    const WINDOW_PANE_LOCAL_HALF_HEIGHT: f32 = 35.0; // window_frame.svg inner pane rect, y=-35 h=70
    const WALL_LOCAL_HALF_EXTENT: f32 = 50.0; // wall.svg rect, x=-50 y=-50 w=100 h=100

    fn sun(azimuth: f64, elevation: f64) -> SunState {
        SunState {
            elevation,
            azimuth,
            intensity: elevation.max(0.0),
            color: [1.0, 1.0, 1.0],
        }
    }

    /// (min, max) clip-space bounds of a disc of `local_radius` placed by
    /// `transform`, replicating the aspect/zoom math `InstanceUniform::
    /// from_transform` applies in `render/mod.rs` — duplicated here (rather
    /// than shared) because that type only exists on wasm32; the formula
    /// itself (`scale_x / aspect`, offset untouched by aspect) is simple
    /// enough to not be worth restructuring the wasm-only code to export.
    /// Final clip-space bounds of a disc/box of `local_half_extent`, going
    /// through the *real* `InstanceUniform::from_transform` (zoom fixed at
    /// 1.0 — see that fn's doc comment on why zoom doesn't affect
    /// containment) rather than a hand-reimplemented copy of its math, so
    /// these tests can't silently drift from what actually gets uploaded to
    /// the GPU.
    fn clip_bounds(transform: &Transform, local_half_extent: (f32, f32), aspect: f32) -> ([f32; 2], [f32; 2]) {
        let uniform = InstanceUniform::from_transform(transform, aspect, 1.0, 0.5);
        let half_w = uniform.scale[0] * local_half_extent.0;
        let half_h = uniform.scale[1] * local_half_extent.1;
        (
            [uniform.offset[0] - half_w, uniform.offset[0] + half_w],
            [uniform.offset[1] - half_h, uniform.offset[1] + half_h],
        )
    }

    fn contains(outer: ([f32; 2], [f32; 2]), inner: ([f32; 2], [f32; 2])) -> bool {
        inner.0[0] >= outer.0[0] && inner.0[1] <= outer.0[1] && inner.1[0] >= outer.1[0] && inner.1[1] <= outer.1[1]
    }

    fn test_leaf(side: Side) -> Leaf {
        Leaf {
            attach_height: 0.0,
            side,
            maturity: 0.0,
            droop: 0.0,
            helio_angle: 0.0,
            fold: 0.0,
            age: 0.0,
            senescence: 0.0,
        }
    }

    /// A trivial curve sitting at the origin with no lean/droop/history at
    /// all — for tests that only care about a leaf's own rotation/scale/
    /// tint (fold, droop, helio, maturity, senescence), not where it sits
    /// along a stem. Since these all use `attach_height: 0.0`, walking any
    /// curve up to that height is a no-op (the loop never executes), so the
    /// specific field values here beyond `base` don't actually matter for
    /// those tests — only `base` (the origin, unrotated) does.
    fn curve_at_origin() -> StemCurve<'static> {
        StemCurve {
            base: Frame { offset: [0.0, 0.0], angle: 0.0 },
            segment_history: &[],
            current_lean_angle: 0.0,
            current_extra_angle: 0.0,
            segment_height_interval: 1.0,
            lean_sign: 1.0,
            vine_base_lean_angle: 0.0,
        }
    }

    #[test]
    fn moon_position_opposite_sun_mirrors_elevation_and_offsets_azimuth_by_half_a_cycle() {
        let moon = moon_position_opposite_sun(0.2, 0.3);
        assert!((moon.azimuth - 0.7).abs() < 1e-6);
        assert!((moon.elevation - 0.7).abs() < 1e-6);
    }

    #[test]
    fn moon_position_opposite_sun_azimuth_never_freezes_or_jumps_across_the_day_progress_wrap() {
        let just_before_wrap = moon_position_opposite_sun(0.999, 0.0);
        let just_after_wrap = moon_position_opposite_sun(0.001, 0.0);
        let delta = (just_after_wrap.azimuth - just_before_wrap.azimuth).rem_euclid(1.0);
        assert!(delta < 0.01 || delta > 0.99, "azimuth should sweep smoothly across the wrap, not jump: {just_before_wrap:?} -> {just_after_wrap:?}");
    }

    #[test]
    fn moon_position_opposite_sun_stays_within_the_same_valid_range_at_the_extremes() {
        for (elevation, day_progress) in [(0.0, 0.0), (1.0, 1.0), (-1.0, 0.0), (0.5, 0.5)] {
            let moon = moon_position_opposite_sun(day_progress, elevation);
            assert!((0.0..=1.0).contains(&moon.elevation));
            assert!((0.0..=1.0).contains(&moon.azimuth));
        }
    }

    #[test]
    fn daytime_fade_leaves_the_tint_untouched_at_night() {
        let base = [1.0, 1.0, 1.0];
        let ambient = [0.9, 0.7, 0.4];
        assert_eq!(daytime_fade(base, ambient, 0.0, 0.85), base);
    }

    #[test]
    fn daytime_fade_blends_toward_ambient_but_never_fully_at_full_sun() {
        let base = [1.0, 1.0, 1.0];
        let ambient = [0.0, 0.0, 0.0];
        let faded = daytime_fade(base, ambient, 1.0, 0.85);
        assert!((faded[0] - 0.15).abs() < 1e-6, "expected a 0.85 blend toward ambient, got {faded:?}");
        assert!(faded[0] > 0.0, "should stay at least faintly distinct from the sky, not fully vanish");
    }

    #[test]
    fn moon_shadow_fully_covers_the_moon_at_new_moon() {
        let layout = SceneLayout::default();
        let sky_transform = sky_object_transform(&sun(0.0, -1.0), &layout, 1.0);
        let shadow = moon_shadow_transform(
            &sky_transform,
            &MoonAppearance { illuminated_fraction: 0.0, waxing: true },
            0.0,
            &layout,
            1.0,
        );
        assert_eq!(shadow.offset, sky_transform.offset, "expected zero shift at illuminated_fraction 0");
        assert_eq!(shadow.tint, layout.moon_shadow_tint);
    }

    #[test]
    fn moon_shadow_shifts_a_full_diameter_away_at_full_moon() {
        let layout = SceneLayout::default();
        let sky_transform = sky_object_transform(&sun(0.0, -1.0), &layout, 1.0);
        let shadow = moon_shadow_transform(
            &sky_transform,
            &MoonAppearance { illuminated_fraction: 1.0, waxing: true },
            0.0,
            &layout,
            1.0,
        );
        let shift = (shadow.offset[0] - sky_transform.offset[0]).abs();
        assert!(
            (shift - 2.0 * layout.sky_object_scale).abs() < 1e-6,
            "expected the shadow to clear the moon by a full diameter, got shift {shift}"
        );
        assert_eq!(shadow.offset[1], sky_transform.offset[1], "expected no vertical shift");
    }

    #[test]
    fn moon_shadow_shifts_opposite_directions_waxing_vs_waning() {
        let layout = SceneLayout::default();
        let sky_transform = sky_object_transform(&sun(0.0, -1.0), &layout, 1.0);
        let waxing = moon_shadow_transform(
            &sky_transform,
            &MoonAppearance { illuminated_fraction: 0.5, waxing: true },
            0.0,
            &layout,
            1.0,
        );
        let waning = moon_shadow_transform(
            &sky_transform,
            &MoonAppearance { illuminated_fraction: 0.5, waxing: false },
            0.0,
            &layout,
            1.0,
        );
        assert!((waxing.offset[0] - sky_transform.offset[0]) > 0.0);
        assert!((waning.offset[0] - sky_transform.offset[0]) < 0.0);
    }

    #[test]
    fn sky_object_stays_within_the_window_pane_across_aspect_ratios_and_positions() {
        let layout = SceneLayout::default();
        // Portrait phone-ish, square, and landscape wider than this app's
        // own ~16:10 canvas — the bug this guards against (disc drifting
        // outside the pane) only showed up on a non-square canvas, so 1.0
        // alone wouldn't have caught it.
        for aspect in [0.5, 1.0, 1.6, 2.5] {
            let pane = clip_bounds(
                &Transform {
                    offset: layout.window_offset,
                    scale_x: layout.window_scale,
                    scale_y: layout.window_scale,
                    rotation: 0.0,
                    tint: NO_TINT,
                },
                (WINDOW_PANE_LOCAL_HALF_WIDTH, WINDOW_PANE_LOCAL_HALF_HEIGHT),
                aspect,
            );
            // Corners and midpoints of the sun's whole arc: sunrise/sunset
            // extremes and solar noon, at each end.
            for azimuth in [0.0, 0.5, 1.0] {
                for elevation in [0.001, 0.5, 1.0] {
                    let transform = sky_object_transform(&sun(azimuth, elevation), &layout, aspect);
                    let disc = clip_bounds(
                        &transform,
                        (SUN_MOON_LOCAL_RADIUS, SUN_MOON_LOCAL_RADIUS),
                        aspect,
                    );
                    assert!(
                        contains(pane, disc),
                        "sun disc escaped the window pane at aspect={aspect}, azimuth={azimuth}, elevation={elevation}: pane={pane:?} disc={disc:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn wall_covers_the_full_canvas_across_aspect_ratios() {
        let layout = SceneLayout::default();
        for aspect in [0.4, 1.0, 1.6, 2.5] {
            let wall = Transform {
                offset: [0.0, 0.0],
                scale_x: layout.wall_scale,
                scale_y: layout.wall_scale,
                rotation: 0.0,
                tint: NO_TINT,
            };
            // The wall is exempt from zoom (render/mod.rs always passes 1.0
            // for it), so this checks the same clip-space extent the wall
            // actually renders at.
            let (x_range, y_range) = clip_bounds(&wall, (WALL_LOCAL_HALF_EXTENT, WALL_LOCAL_HALF_EXTENT), aspect);
            assert!(
                x_range[0] <= -1.0 && x_range[1] >= 1.0,
                "wall doesn't cover full canvas width at aspect={aspect}: {x_range:?}"
            );
            assert!(
                y_range[0] <= -1.0 && y_range[1] >= 1.0,
                "wall doesn't cover full canvas height at aspect={aspect}: {y_range:?}"
            );
        }
    }

    #[test]
    fn left_and_right_cotyledons_are_mirror_images() {
        let layout = SceneLayout::default();
        let left = cotyledon_transform(&layout, Side::Left);
        let right = cotyledon_transform(&layout, Side::Right);
        assert_eq!(left.rotation, -right.rotation);
        assert_eq!(left.offset, right.offset, "both attach at the same point on the stem");
    }

    /// A curve reflecting `plant`'s current state — no history recorded
    /// yet, so the whole thing (whatever height it's asked about) uses
    /// `plant`'s *live* lean/droop, same as a freshly-sprouted plant with
    /// no frozen segments behind it yet.
    fn curve_for_plant(plant: &Plant) -> StemCurve<'_> {
        StemCurve {
            base: Frame { offset: [0.0, 0.0], angle: 0.0 },
            segment_history: &plant.stem_segment_history,
            current_lean_angle: plant.lean_angle,
            current_extra_angle: plant.stem_droop,
            segment_height_interval: 1.0,
            lean_sign: 1.0,
            vine_base_lean_angle: 0.0,
        }
    }

    #[test]
    fn lean_sign_points_toward_a_window_on_either_side() {
        let pot = [-0.15, -0.45];
        assert_eq!(lean_sign_toward_window([0.62, 0.6], pot), -1.0, "window to the right");
        assert_eq!(lean_sign_toward_window([-0.9, 0.6], pot), 1.0, "window to the left");
    }

    #[test]
    fn stem_leans_toward_a_window_on_the_right_not_away_from_it() {
        // Regression test for a real bug: `rotate_and_place`'s rotation
        // convention bends a *positive* angle toward -x (left) — with the
        // default layout's window on the *right* (positive x, same side as
        // the actual `SceneLayout::default()`), an uncorrected positive
        // `lean_angle` would visibly bend the stem away from the window,
        // the opposite of real phototropism.
        let layout = SceneLayout::default();
        assert!(layout.window_offset[0] >= layout.pot_anchor[0], "test assumes the default window is to the right");

        let mut plant = Plant::new();
        plant.lean_angle = 0.3;
        let curve = StemCurve {
            base: stem_base_frame(&layout),
            segment_history: &plant.stem_segment_history,
            current_lean_angle: plant.lean_angle,
            current_extra_angle: plant.stem_droop,
            segment_height_interval: 1.0,
            lean_sign: lean_sign_toward_window(layout.window_offset, layout.pot_anchor),
            vine_base_lean_angle: 0.0,
        };

        let base_x = curve.base.offset[0];
        let tip = frame_at_height(&curve, curve.segment_height_interval, &layout);
        assert!(
            tip.offset[0] > base_x,
            "expected the leaning stem to shift toward the window's side (+x): tip {} vs base {base_x}",
            tip.offset[0]
        );
    }

    #[test]
    fn branch_curve_attaches_at_the_true_point_along_a_leaning_main_stem() {
        let layout = SceneLayout::default();
        let mut plant = Plant::new();
        plant.lean_angle = 0.3;
        let main_curve = curve_for_plant(&plant);
        let stem_base = main_curve.base;

        let branch = Branch::new(1.0, Side::Left);
        let bcurve = branch_curve(&main_curve, &branch, &layout);

        // A leaning stem shifts a higher attachment point sideways (this is
        // the rigid-rotation-about-the-base approximation, still true for
        // any *one* segment) — with a positive lean and the shader's
        // rotation convention, that's a negative x shift.
        assert!(
            bcurve.base.offset[0] < stem_base.offset[0],
            "expected the attachment point to shift sideways with stem lean: {} vs {}",
            bcurve.base.offset[0],
            stem_base.offset[0]
        );
        // Higher on a leaning stem also means higher up on screen than the
        // stem's own base, not lower.
        assert!(bcurve.base.offset[1] > stem_base.offset[1]);

        // The branch's own angle includes the stem's lean plus its side
        // spread — a Right branch should end up with a smaller (or more
        // negative) angle than an otherwise-identical Left one.
        let right_branch = Branch::new(1.0, Side::Right);
        let right_curve = branch_curve(&main_curve, &right_branch, &layout);
        assert!(right_curve.base.angle < bcurve.base.angle);
    }

    #[test]
    fn frame_at_height_uses_the_live_lean_and_droop_when_no_history_is_recorded_yet() {
        let layout = SceneLayout::default();
        let mut plant = Plant::new();
        plant.lean_angle = 0.3;
        plant.stem_droop = 0.2;
        let curve = curve_for_plant(&plant);

        // No segments recorded yet, so *any* height along the curve (here,
        // one full segment's worth) falls back to today's live lean+droop.
        let frame = frame_at_height(&curve, curve.segment_height_interval, &layout);

        assert!(
            (frame.angle - 0.5).abs() < 1e-6,
            "expected the still-growing tip's angle to be lean + droop, got {}",
            frame.angle
        );
    }

    #[test]
    fn a_completed_segment_stays_frozen_at_its_recorded_angle_even_as_current_lean_keeps_growing() {
        let layout = SceneLayout::default();
        let mut plant = Plant::new();
        // One segment already recorded (frozen at 0.1), but the plant has
        // since kept leaning further — the *recorded* segment shouldn't
        // care about that.
        plant.stem_segment_history = vec![0.1];
        plant.lean_angle = 0.4;
        let curve = curve_for_plant(&plant);

        let frame = frame_at_height(&curve, curve.segment_height_interval, &layout);
        assert!(
            (frame.angle - 0.1).abs() < 1e-6,
            "expected the completed segment to stay at its recorded angle (0.1), not today's live lean (0.4): got {}",
            frame.angle
        );

        // But asking for a point *past* that one recorded segment reaches
        // the still-growing tip, which *does* use the live lean.
        let tip_frame = frame_at_height(&curve, curve.segment_height_interval * 1.5, &layout);
        assert!(
            (tip_frame.angle - 0.4).abs() < 1e-6,
            "expected the still-growing tip beyond the recorded history to use live lean (0.4): got {}",
            tip_frame.angle
        );
    }

    #[test]
    fn branch_curve_angle_includes_the_branchs_own_droop_on_top_of_everything_else() {
        let layout = SceneLayout::default();
        let plant = Plant::new();
        let main_curve = curve_for_plant(&plant);

        let mut upright_branch = Branch::new(1.0, Side::Left);
        let upright_curve = branch_curve(&main_curve, &upright_branch, &layout);
        let upright_frame = frame_at_height(&upright_curve, upright_curve.segment_height_interval, &layout);

        upright_branch.droop = 0.15;
        let drooped_curve = branch_curve(&main_curve, &upright_branch, &layout);
        let drooped_frame = frame_at_height(&drooped_curve, drooped_curve.segment_height_interval, &layout);

        assert!(
            (drooped_frame.angle - upright_frame.angle - 0.15).abs() < 1e-6,
            "expected the branch's own droop to add directly onto its angle: {} vs {}",
            drooped_frame.angle,
            upright_frame.angle
        );
    }

    #[test]
    fn leaf_fold_and_droop_pull_rotation_back_toward_and_past_vertical() {
        let layout = SceneLayout::default();
        let curve = curve_at_origin();
        let mut leaf = test_leaf(Side::Left);
        leaf.maturity = 1.0;

        let open = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);

        leaf.fold = 1.0;
        let folded = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);
        assert!(
            folded.rotation < open.rotation,
            "fully folded should rotate back toward vertical relative to fully open: {} vs {}",
            folded.rotation,
            open.rotation
        );

        leaf.fold = 0.0;
        leaf.droop = 1.0;
        let wilted = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);
        assert!(wilted.rotation < open.rotation, "fully wilted should also droop back from fully open");
    }

    #[test]
    fn leaf_helio_bias_applies_the_same_regardless_of_side() {
        let layout = SceneLayout::default();
        let curve = curve_at_origin();
        let mut left = test_leaf(Side::Left);
        let mut right = test_leaf(Side::Right);
        left.maturity = 1.0;
        right.maturity = 1.0;

        left.helio_angle = 0.0;
        right.helio_angle = 0.0;
        let left_base = leaf_transform_in_frame(&curve, &left, 1.0, &layout).rotation;
        let right_base = leaf_transform_in_frame(&curve, &right, 1.0, &layout).rotation;

        left.helio_angle = 1.0;
        right.helio_angle = 1.0;
        let left_biased = leaf_transform_in_frame(&curve, &left, 1.0, &layout).rotation;
        let right_biased = leaf_transform_in_frame(&curve, &right, 1.0, &layout).rotation;

        // Helio tracks the window's fixed direction, not "outward" — so the
        // same helio_angle shifts both sides' rotation by the same amount,
        // unlike fold/droop/spread which are mirrored per side.
        assert!((left_biased - left_base - (right_biased - right_base)).abs() < 1e-6);
    }

    #[test]
    fn a_just_budded_leaf_still_has_a_nonzero_visible_size() {
        let layout = SceneLayout::default();
        let curve = curve_at_origin();
        let mut leaf = test_leaf(Side::Left);
        leaf.maturity = 0.0;
        let transform = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);
        assert!(transform.scale_x > 0.0, "a maturity-0 leaf should still render as a tiny bud, not vanish");
    }

    #[test]
    fn rosette_leaf_transform_anchors_at_base_regardless_of_attach_height() {
        let layout = SceneLayout::default();
        let base = Frame { offset: [0.1, -0.2], angle: 0.0 };
        let mut leaf = test_leaf(Side::Left);
        leaf.attach_height = 5.0;
        let transform = rosette_leaf_transform(base, &leaf, 1.0, &layout);
        assert_eq!(transform.offset, base.offset);
    }

    #[test]
    fn rosette_leaf_transform_spread_widens_with_age() {
        let layout = SceneLayout::default();
        let base = Frame { offset: [0.0, 0.0], angle: 0.0 };
        let mut young = test_leaf(Side::Left);
        young.age = 0.0;
        let mut old = test_leaf(Side::Left);
        old.age = layout.rosette_leaf_splay_age;
        let young_rotation = rosette_leaf_transform(base, &young, 1.0, &layout).rotation;
        let old_rotation = rosette_leaf_transform(base, &old, 1.0, &layout).rotation;
        assert!(old_rotation.abs() > young_rotation.abs(), "an older rosette leaf should splay out further than a young one");
    }

    #[test]
    fn trellis_transform_is_absent_for_a_freestanding_habit() {
        let layout = SceneLayout::default();
        assert!(trellis_transform(None, &layout).is_none());
    }

    #[test]
    fn trellis_transform_reaches_exactly_as_tall_as_a_stem_would_at_the_same_height() {
        let layout = SceneLayout::default();
        let trellis_height = 3.0;
        let transform = trellis_transform(Some(trellis_height), &layout).expect("climbing habit");
        assert_eq!(
            transform.offset,
            [layout.pot_anchor[0] + layout.trellis_x_offset, layout.pot_anchor[1]],
            "should be planted right beside the stem's own anchor, not through its centerline"
        );
        assert_eq!(transform.rotation, 0.0, "a rigid support doesn't lean");

        // Same STEM_LOCAL_HEIGHT-based conversion `stem_segment_transforms`
        // uses for the stem itself — walking a stem curve straight up
        // (angle 0) by `trellis_height` should land at the same *height*
        // (y) the trellis's own far end renders at (x differs deliberately
        // by `trellis_x_offset` — see the assertion above).
        let curve = StemCurve {
            base: Frame { offset: layout.pot_anchor, angle: 0.0 },
            segment_history: &[],
            current_lean_angle: 0.0,
            current_extra_angle: 0.0,
            segment_height_interval: 1.0,
            lean_sign: 1.0,
            vine_base_lean_angle: 0.0,
        };
        let stem_tip = frame_at_height(&curve, trellis_height, &layout).offset;
        let trellis_top_y = transform.offset[1] + transform.scale_y * STEM_LOCAL_HEIGHT;
        assert!(
            (trellis_top_y - stem_tip[1]).abs() < 1e-4,
            "expected the trellis's own top to reach the same height a straight stem's tip would: trellis {trellis_top_y} vs stem {}",
            stem_tip[1]
        );
    }

    #[test]
    fn aerial_root_transform_sits_at_its_recorded_height_along_a_straight_stem() {
        let layout = SceneLayout::default();
        let curve = StemCurve {
            base: Frame { offset: layout.pot_anchor, angle: 0.0 },
            segment_history: &[],
            current_lean_angle: 0.0,
            current_extra_angle: 0.0,
            segment_height_interval: 1.0,
            lean_sign: 1.0,
            vine_base_lean_angle: 0.0,
        };
        let root = AerialRoot { attach_height: 1.5 };
        let transform = aerial_root_transform(&root, &curve, &layout);
        let expected_offset = frame_at_height(&curve, root.attach_height, &layout).offset;
        assert_eq!(transform.offset, expected_offset);
    }

    #[test]
    fn aerial_root_transform_reaches_toward_whichever_side_the_trellis_is_actually_on() {
        let mut layout = SceneLayout::default();
        let curve = StemCurve {
            base: Frame { offset: layout.pot_anchor, angle: 0.0 },
            segment_history: &[],
            current_lean_angle: 0.0,
            current_extra_angle: 0.0,
            segment_height_interval: 1.0,
            lean_sign: 1.0,
            vine_base_lean_angle: 0.0,
        };
        let root = AerialRoot { attach_height: 1.0 };

        layout.trellis_x_offset = 0.03;
        let toward_right = aerial_root_transform(&root, &curve, &layout);
        assert_eq!(toward_right.rotation, 0.0, "reaches +x (unrotated) when the trellis is on the +x side");

        layout.trellis_x_offset = -0.03;
        let toward_left = aerial_root_transform(&root, &curve, &layout);
        assert_eq!(
            toward_left.rotation,
            std::f32::consts::PI,
            "should flip 180° to reach -x when the trellis is on that side instead"
        );
    }

    #[test]
    fn a_healthy_leaf_renders_with_no_tint_at_all() {
        let layout = SceneLayout::default();
        let curve = curve_at_origin();
        let mut leaf = test_leaf(Side::Left);
        leaf.maturity = 1.0;
        leaf.senescence = 0.0;
        let transform = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);
        assert_eq!(transform.tint, NO_TINT, "a fresh, healthy leaf shouldn't be tinted at all");
    }

    #[test]
    fn senescence_tints_a_leaf_toward_yellow_then_brown_and_shrinks_it() {
        let layout = SceneLayout::default();
        let curve = curve_at_origin();
        let mut leaf = test_leaf(Side::Left);
        leaf.maturity = 1.0;

        leaf.senescence = 0.0;
        let healthy = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);
        leaf.senescence = 0.5;
        let yellowing = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);
        leaf.senescence = 1.0;
        let dead = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);

        assert_eq!(yellowing.tint, layout.leaf_senescent_tint, "expected the midpoint to land exactly on the configured senescent tint");
        assert_eq!(dead.tint, layout.leaf_dead_tint, "expected fully senesced to land exactly on the configured dead tint");

        // Shrinks monotonically as senescence rises, on top of maturity.
        assert!(dead.scale_x < yellowing.scale_x);
        assert!(yellowing.scale_x < healthy.scale_x);
        assert!(
            (dead.scale_x - layout.leaf_scale * (1.0 - layout.leaf_shrivel_max_fraction)).abs() < 1e-6,
            "expected a fully senesced, fully mature leaf to shrink by exactly leaf_shrivel_max_fraction"
        );
    }

    #[test]
    fn a_fully_occluded_leaf_is_darker_than_an_unshaded_one_but_not_black() {
        let layout = SceneLayout::default();
        let curve = curve_at_origin();
        let leaf = test_leaf(Side::Left);

        let unshaded = leaf_transform_in_frame(&curve, &leaf, 1.0, &layout);
        let occluded = leaf_transform_in_frame(&curve, &leaf, 0.0, &layout);

        assert_eq!(unshaded.tint, senescence_tint(leaf.senescence as f32, &layout));
        for c in 0..3 {
            assert!(occluded.tint[c] < unshaded.tint[c], "expected occlusion to darken channel {c}");
            assert!(occluded.tint[c] > 0.0, "expected a floor, not pitch black");
        }
    }

    #[test]
    fn stem_health_tint_is_untinted_at_full_health_and_matches_the_configured_tint_at_zero() {
        let layout = SceneLayout::default();
        let full_health = stem_health_tint(1.0, &layout);
        let zero_health = stem_health_tint(0.0, &layout);
        for channel in 0..3 {
            assert!((full_health[channel] - NO_TINT[channel]).abs() < 1e-5);
            assert!((zero_health[channel] - layout.stem_unhealthy_tint[channel]).abs() < 1e-5);
        }
    }

    #[test]
    fn stem_health_tint_worsens_monotonically_as_vitality_drops() {
        let layout = SceneLayout::default();
        let healthy = stem_health_tint(0.9, &layout);
        let damaged = stem_health_tint(0.4, &layout);
        let rotted = stem_health_tint(0.0, &layout);
        // Every channel of `stem_unhealthy_tint` in the default layout is
        // below 1.0 (`NO_TINT`), so "worse" means monotonically decreasing
        // here — this would need to flip if that default ever changed to a
        // brightening tint instead.
        assert!(healthy[0] > damaged[0] && damaged[0] > rotted[0]);
    }

    #[test]
    fn soil_moisture_tint_lands_on_the_configured_dry_wet_and_waterlogged_tints_exactly() {
        let layout = SceneLayout::default();
        let threshold = 0.9;
        assert_eq!(soil_moisture_tint(0.0, threshold, &layout), layout.soil_dry_tint);
        assert_eq!(soil_moisture_tint(threshold, threshold, &layout), layout.soil_wet_tint);
        assert_eq!(soil_moisture_tint(1.0, threshold, &layout), layout.soil_waterlogged_tint);
    }

    #[test]
    fn soil_moisture_tint_is_a_distinct_visual_stage_past_the_waterlogged_threshold() {
        let layout = SceneLayout::default();
        let threshold = 0.9;
        let healthy_moist = soil_moisture_tint(threshold * 0.9, threshold, &layout);
        let waterlogged = soil_moisture_tint(1.0, threshold, &layout);
        assert_ne!(
            healthy_moist, waterlogged,
            "expected waterlogged soil to look visibly different from ordinary healthy moist soil"
        );
    }

    #[test]
    fn seasonal_wall_tint_lands_exactly_on_each_configured_keyframe_at_its_quarter_boundary() {
        let layout = SceneLayout::default();
        assert_eq!(seasonal_wall_tint(0.0, &layout), layout.season_summer_tint);
        assert_eq!(seasonal_wall_tint(0.25, &layout), layout.season_autumn_tint);
        assert_eq!(seasonal_wall_tint(0.5, &layout), layout.season_winter_tint);
        assert_eq!(seasonal_wall_tint(0.75, &layout), layout.season_spring_tint);
    }

    #[test]
    fn seasonal_wall_tint_wraps_back_to_summer_at_a_full_year() {
        let layout = SceneLayout::default();
        assert_eq!(seasonal_wall_tint(1.0, &layout), layout.season_summer_tint);
    }

    #[test]
    fn seasonal_wall_tint_eases_between_keyframes_rather_than_snapping() {
        let layout = SceneLayout::default();
        let just_after_summer = seasonal_wall_tint(0.01, &layout);
        // Nowhere near either pure keyframe — a genuine blend, not a jump.
        assert_ne!(just_after_summer, layout.season_summer_tint);
        assert_ne!(just_after_summer, layout.season_autumn_tint);
    }

    #[test]
    fn pot_anchor_for_position_is_a_no_op_at_zero() {
        let base = [-0.15, -0.45];
        assert_eq!(pot_anchor_for_position(base, 0.0, 0.3), base);
    }

    #[test]
    fn pot_anchor_for_position_shifts_away_from_the_window_as_position_rises() {
        let base = [-0.15, -0.45];
        let far = pot_anchor_for_position(base, 1.0, 0.3);
        // The default layout's window sits at positive x relative to the
        // pot, so moving away from it means decreasing x.
        assert!(far[0] < base[0]);
        // Only the horizontal position changes — the pot doesn't float up
        // or sink into the floor as it slides.
        assert_eq!(far[1], base[1]);
    }

    #[test]
    fn pot_anchor_for_position_clamps_out_of_range_input() {
        let base = [-0.15, -0.45];
        let over = pot_anchor_for_position(base, 5.0, 0.3);
        let far = pot_anchor_for_position(base, 1.0, 0.3);
        assert_eq!(over, far, "expected out-of-range position to clamp rather than extrapolate");
    }

    #[test]
    fn plant_slot_base_anchor_reproduces_the_layouts_own_pot_anchor_at_slot_zero() {
        let layout = SceneLayout::default();
        assert_eq!(plant_slot_base_anchor(&layout, 0), layout.pot_anchor);
    }

    #[test]
    fn plant_slot_base_anchor_steps_sideways_for_each_additional_slot() {
        let layout = SceneLayout::default();
        let slot0 = plant_slot_base_anchor(&layout, 0);
        let slot1 = plant_slot_base_anchor(&layout, 1);
        let slot2 = plant_slot_base_anchor(&layout, 2);
        assert!((slot1[0] - slot0[0] - layout.plant_slot_spacing).abs() < 1e-6);
        assert!((slot2[0] - slot1[0] - layout.plant_slot_spacing).abs() < 1e-6);
        // Slots don't drift vertically — only sideways along the sill.
        assert_eq!(slot0[1], slot1[1]);
        assert_eq!(slot1[1], slot2[1]);
    }

    #[test]
    fn plant_pot_world_anchor_matches_slot_base_anchor_when_pot_position_is_neutral() {
        let layout = SceneLayout::default();
        assert_eq!(plant_pot_world_anchor(&layout, 1, 0.0), plant_slot_base_anchor(&layout, 1));
    }

    #[test]
    fn plant_pot_world_anchor_combines_the_slots_own_sideways_step_with_its_pot_position_shift() {
        let layout = SceneLayout::default();
        let combined = plant_pot_world_anchor(&layout, 2, 1.0);
        let expected = pot_anchor_for_position(plant_slot_base_anchor(&layout, 2), 1.0, layout.pot_position_x_travel);
        assert_eq!(combined, expected);
    }

    #[test]
    fn ambient_tint_matches_night_floor_and_sun_color_at_the_extremes() {
        let layout = SceneLayout::default();
        let night = ambient_tint(&sun(0.5, -1.0), &layout);
        assert_eq!(night, layout.night_ambient_color);

        let noon_color = [1.0, 1.0, 0.95];
        let noon = ambient_tint(
            &SunState { elevation: 1.0, azimuth: 0.5, intensity: 1.0, color: noon_color },
            &layout,
        );
        assert_eq!(noon, noon_color);
    }

    #[test]
    fn outline_tint_for_sun_matches_the_configured_day_and_night_extremes() {
        let layout = SceneLayout::default();
        let night = outline_tint_for_sun(0.0, &layout);
        let day = outline_tint_for_sun(1.0, &layout);
        for i in 0..3 {
            assert!((night[i] - layout.outline_tint_night[i]).abs() < 1e-5);
            assert!((day[i] - layout.outline_tint_day[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn room_and_pot_background_together_include_every_expected_piece_exactly_once() {
        let layout = SceneLayout::default();
        let meshes: Vec<&str> = room_background(&layout)
            .iter()
            .chain(pot_background(&layout).iter())
            .map(|s| s.mesh)
            .collect();
        for expected in ["wall", "window_frame", "window_pane", "pot", "soil"] {
            assert_eq!(
                meshes.iter().filter(|m| **m == expected).count(),
                1,
                "expected exactly one {expected:?} in the background, got {meshes:?}"
            );
        }
    }

    #[test]
    fn room_background_holds_only_the_wall_and_window_not_the_pot() {
        let layout = SceneLayout::default();
        let meshes: Vec<&str> = room_background(&layout).iter().map(|s| s.mesh).collect();
        assert_eq!(meshes, vec!["wall", "window_frame", "window_pane"]);
    }

    #[test]
    fn pot_background_holds_only_the_pot_and_soil() {
        let layout = SceneLayout::default();
        let meshes: Vec<&str> = pot_background(&layout).iter().map(|s| s.mesh).collect();
        assert_eq!(meshes, vec!["pot", "soil"]);
    }

    #[test]
    fn dynamic_zoom_for_room_matches_the_configured_floor_for_a_small_plant() {
        let layout = SceneLayout::default();
        let mut plant = Plant::new();
        plant.height = 0.1; // tiny seedling, nowhere near the visible edge
        assert_eq!(dynamic_zoom_for_room(std::iter::once(&plant), &layout), layout.zoom);
    }

    #[test]
    fn dynamic_zoom_for_room_pulls_back_further_once_a_solo_plants_stem_would_run_off_the_top_of_frame() {
        let layout = SceneLayout::default();
        // Tall enough that `layout.zoom` alone would push the tip well past
        // `zoom_visible_half_height`.
        let unzoomed_top =
            layout.pot_anchor[1] + 500.0 * layout.stem_height_scale * STEM_LOCAL_HEIGHT;
        assert!(
            unzoomed_top * layout.zoom > layout.zoom_visible_half_height,
            "test setup should actually exceed the visible area at the default zoom"
        );

        let mut plant = Plant::new();
        plant.height = 500.0;
        let zoom = dynamic_zoom_for_room(std::iter::once(&plant), &layout);

        assert!(zoom < layout.zoom, "expected the camera to pull back further than the floor, got {zoom}");
        assert!(
            (unzoomed_top * zoom - layout.zoom_visible_half_height).abs() < 1e-4,
            "expected the tip to land almost exactly on the visible edge, got tip at {}",
            unzoomed_top * zoom
        );
    }

    #[test]
    fn dynamic_zoom_for_room_accounts_for_a_branchs_own_reach_not_just_the_main_stem() {
        let layout = SceneLayout::default();
        let mut short_main_stem = Plant::new();
        short_main_stem.height = 1.0;
        short_main_stem
            .branches
            .push(Branch::new(0.9, Side::Left));
        short_main_stem.branches[0].height = 800.0; // towers well past the main stem

        let zoom_with_tall_branch = dynamic_zoom_for_room(std::iter::once(&short_main_stem), &layout);
        let mut no_branch = Plant::new();
        no_branch.height = 1.0;
        let zoom_without_branch = dynamic_zoom_for_room(std::iter::once(&no_branch), &layout);

        assert!(
            zoom_with_tall_branch < zoom_without_branch,
            "expected a towering branch to pull the camera back further than the short main stem alone would: {zoom_with_tall_branch} vs {zoom_without_branch}"
        );
    }

    #[test]
    fn dynamic_zoom_for_room_fits_the_tallest_plant_even_if_its_not_the_first() {
        let layout = SceneLayout::default();
        let mut short = Plant::new();
        short.height = 1.0;
        let mut tall = Plant::new();
        tall.height = 500.0;

        let room_tall_first = dynamic_zoom_for_room([&tall, &short].into_iter(), &layout);
        let room_tall_second = dynamic_zoom_for_room([&short, &tall].into_iter(), &layout);
        let solo_tall = dynamic_zoom_for_room(std::iter::once(&tall), &layout);

        assert_eq!(room_tall_first, solo_tall);
        assert_eq!(room_tall_second, solo_tall);
    }

    #[test]
    fn dynamic_zoom_for_room_pulls_back_further_for_several_plants_side_by_side_than_for_one() {
        let layout = SceneLayout::default();
        // Short enough that the vertical fit alone never kicks in — only
        // the horizontal, several-pots-along-the-sill fit should be able to
        // pull the camera back here.
        let mut plant = Plant::new();
        plant.height = 1.0;

        let one_plant = dynamic_zoom_for_room(std::iter::once(&plant), &layout);
        // MAX_PLANTS worth of pots (see `render::mod::MAX_PLANTS`) — enough
        // side-by-side spacing at the default layout to actually force a
        // pull-back past the zoom floor.
        let plants: Vec<Plant> = (0..6).map(|_| Plant::new()).collect();
        let many_plants = dynamic_zoom_for_room(plants.iter(), &layout);

        assert!(
            many_plants < one_plant,
            "expected several side-by-side plants to need a wider view than just one: {many_plants} vs {one_plant}"
        );
    }

    #[test]
    fn dynamic_zoom_for_room_never_zooms_in_tighter_than_the_configured_floor() {
        let layout = SceneLayout::default();
        let plant = Plant::new();
        assert!(dynamic_zoom_for_room(std::iter::once(&plant), &layout) <= layout.zoom);
    }

    #[test]
    fn vine_base_lean_only_tilts_the_first_segment() {
        let layout = SceneLayout::default();
        let mut plant = Plant::new();
        plant.height = 3.0;
        let curve = StemCurve {
            base: stem_base_frame(&layout),
            segment_history: &plant.stem_segment_history,
            current_lean_angle: plant.lean_angle,
            current_extra_angle: plant.stem_droop,
            segment_height_interval: 1.0,
            lean_sign: 1.0,
            vine_base_lean_angle: 0.4,
        };
        let segments = stem_segment_transforms(&curve, plant.height, plant.stem_radius, NO_TINT, &layout);
        assert!((segments[0].rotation - 0.4).abs() < 1e-6, "first segment should carry the vine lean");
        assert!((segments[1].rotation - 0.0).abs() < 1e-6, "later segments shouldn't inherit it");
    }

    #[test]
    fn zero_vine_base_lean_leaves_the_stem_perfectly_straight() {
        let layout = SceneLayout::default();
        let mut plant = Plant::new();
        plant.height = 2.0;
        let curve = StemCurve {
            base: stem_base_frame(&layout),
            segment_history: &plant.stem_segment_history,
            current_lean_angle: plant.lean_angle,
            current_extra_angle: plant.stem_droop,
            segment_height_interval: 1.0,
            lean_sign: 1.0,
            vine_base_lean_angle: 0.0,
        };
        let segments = stem_segment_transforms(&curve, plant.height, plant.stem_radius, NO_TINT, &layout);
        assert!(segments.iter().all(|s| s.rotation == 0.0));
    }

    #[test]
    fn dynamic_zoom_for_room_with_no_plants_is_a_no_op_at_the_floor() {
        let layout = SceneLayout::default();
        let none: [&Plant; 0] = [];
        assert_eq!(dynamic_zoom_for_room(none.into_iter(), &layout), layout.zoom);
    }

    #[test]
    fn seed_transform_swells_toward_full_scale_as_the_fraction_rises() {
        let layout = SceneLayout::default();
        let dry = seed_transform(&layout, 0.0);
        let swollen = seed_transform(&layout, 1.0);
        assert!((dry.scale_x - layout.seed_scale * layout.seed_min_swell_scale_fraction).abs() < 1e-6);
        assert!((swollen.scale_x - layout.seed_scale).abs() < 1e-6);
        assert!(swollen.scale_x > dry.scale_x);
    }

    #[test]
    fn seed_and_stem_and_branch_transforms_sit_at_their_shared_anchor() {
        let layout = SceneLayout::default();
        assert_eq!(seed_transform(&layout, 1.0).offset, layout.pot_anchor);

        let mut plant = Plant::new();
        plant.height = 2.0;
        plant.stem_radius = 0.5;
        let main_curve = StemCurve {
            base: stem_base_frame(&layout),
            segment_history: &plant.stem_segment_history,
            current_lean_angle: plant.lean_angle,
            current_extra_angle: plant.stem_droop,
            segment_height_interval: 10.0, // taller than plant.height: renders as one segment
            lean_sign: 1.0,
            vine_base_lean_angle: 0.0,
        };
        let segments = stem_segment_transforms(&main_curve, plant.height, plant.stem_radius, NO_TINT, &layout);
        assert_eq!(segments.len(), 1, "expected one segment covering the whole (still-growing) stem");
        let stem = segments[0];
        assert_eq!(stem.offset, layout.pot_anchor);
        assert!((stem.scale_y - layout.stem_height_scale * 2.0).abs() < 1e-6);
        assert!((stem.scale_x - layout.stem_radius_scale * 0.5).abs() < 1e-6);

        let branch = Branch::new(1.0, Side::Left);
        let bcurve = branch_curve(&main_curve, &branch, &layout);
        assert_eq!(
            bcurve.base.offset,
            frame_at_height(&main_curve, 1.0, &layout).offset,
            "a branch's own curve starts exactly where it attaches on the main stem's"
        );
    }

    #[test]
    fn flower_sits_at_the_same_point_a_leaf_attached_at_the_stems_full_height_would() {
        let layout = SceneLayout::default();
        let mut plant = Plant::new();
        plant.height = 3.0;
        let curve = curve_for_plant(&plant);

        let flower = flower_transform(&curve, plant.height, 1.0, &layout);
        let leaf_at_tip = leaf_transform_in_frame(
            &curve,
            &Leaf {
                attach_height: plant.height,
                side: Side::Left,
                maturity: 1.0,
                droop: 0.0,
                helio_angle: 0.0,
                fold: 0.0,
                age: 0.0,
                senescence: 0.0,
            },
            1.0,
            &layout,
        );

        // Both walk the same curve to the same height, so they land at
        // exactly the same point — only `scale`/`rotation` legitimately
        // differ (a flower isn't a leaf-shaped mesh, and doesn't carry a
        // leaf's own fold/droop/helio rotation terms).
        assert_eq!(
            flower.offset, leaf_at_tip.offset,
            "the flower should sit at exactly the point along the stem a leaf attached at the same height would"
        );
    }

    #[test]
    fn flower_tips_along_with_the_stems_live_lean_and_droop() {
        let layout = SceneLayout::default();
        let mut plant = Plant::new();
        plant.height = 3.0;
        plant.lean_angle = 0.2;
        plant.stem_droop = 0.1;
        let curve = curve_for_plant(&plant);

        // No history recorded yet, so the tip (where the flower sits) uses
        // today's live lean + droop.
        let flower = flower_transform(&curve, plant.height, 1.0, &layout);
        assert!((flower.rotation - 0.3).abs() < 1e-6);
    }

    #[test]
    fn flower_transform_scales_directly_with_bloom_intensity() {
        let layout = SceneLayout::default();
        let curve = curve_at_origin();
        let closed = flower_transform(&curve, 1.0, 0.0, &layout);
        let half_open = flower_transform(&curve, 1.0, 0.5, &layout);
        let fully_open = flower_transform(&curve, 1.0, 1.0, &layout);
        assert_eq!(closed.scale_x, 0.0, "fully closed should render as zero-size (invisible)");
        assert_eq!(fully_open.scale_x, layout.flower_scale);
        assert!((half_open.scale_x - layout.flower_scale * 0.5).abs() < 1e-6);
    }

    #[test]
    fn instance_uniform_divides_x_scale_by_aspect_but_not_offset_or_y_scale() {
        let transform = Transform {
            offset: [0.4, 0.2],
            scale_x: 0.1,
            scale_y: 0.1,
            rotation: 0.5,
            tint: [0.9, 0.8, 0.7],
        };
        let uniform = InstanceUniform::from_transform(&transform, 2.0, 1.0, 0.5);
        assert_eq!(uniform.offset, [0.4, 0.2], "offset is never aspect-corrected");
        assert!((uniform.scale[0] - 0.05).abs() < 1e-6, "x scale divides by aspect");
        assert_eq!(uniform.scale[1], 0.1, "y scale is untouched by aspect");
        assert_eq!(uniform.rotation, transform.rotation);
        assert_eq!(uniform.tint, transform.tint);
    }

    #[test]
    fn instance_uniform_clamps_depth_into_range() {
        let transform = Transform { offset: [0.0, 0.0], scale_x: 1.0, scale_y: 1.0, rotation: 0.0, tint: NO_TINT };
        assert_eq!(InstanceUniform::from_transform(&transform, 1.0, 1.0, -0.5).depth, 0.0);
        assert_eq!(InstanceUniform::from_transform(&transform, 1.0, 1.0, 1.5).depth, 1.0);
        assert_eq!(InstanceUniform::from_transform(&transform, 1.0, 1.0, 0.3).depth, 0.3);
    }

    #[test]
    fn instance_uniform_zoom_scales_offset_and_both_axes_equally() {
        let transform = Transform {
            offset: [0.4, -0.2],
            scale_x: 0.1,
            scale_y: 0.2,
            rotation: 0.0,
            tint: NO_TINT,
        };
        let uniform = InstanceUniform::from_transform(&transform, 1.0, 0.5, 0.5);
        assert_eq!(uniform.offset, [0.2, -0.1], "zoom halves the offset");
        assert_eq!(uniform.scale, [0.05, 0.1], "zoom halves both scale axes equally (before aspect)");
    }

    #[test]
    fn outline_uniform_grows_both_scale_axes_and_saturates_the_tint() {
        let layout = SceneLayout::default();
        let transform = Transform {
            offset: [0.1, 0.2],
            scale_x: 0.3,
            scale_y: 0.4,
            rotation: 0.5,
            tint: [0.2, 0.6, 0.2],
        };
        let base = InstanceUniform::from_transform(&transform, 1.5, 1.0, 0.5);
        let outline = outline_uniform(&transform, 1.5, 1.0, (10.0, 10.0), &layout, 800.0, 0.5, layout.outline_tint_day);
        assert!(outline.scale[0] > base.scale[0], "outline should be larger than the base mesh in x");
        assert!(outline.scale[1] > base.scale[1], "outline should be larger than the base mesh in y");
        assert_eq!(outline.offset, base.offset, "outline shares the same center as the base mesh");
        assert_eq!(outline.tint, layout.outline_tint_day, "outline discards the mesh's own tint entirely");
    }

    #[test]
    fn outline_uniform_margin_shrinks_as_the_canvas_gets_wider() {
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.0, 0.0], scale_x: 0.3, scale_y: 0.3, rotation: 0.0, tint: NO_TINT };
        let narrow = outline_uniform(&transform, 1.0, 1.0, (10.0, 10.0), &layout, 400.0, 0.5, layout.outline_tint_day);
        let wide = outline_uniform(&transform, 1.0, 1.0, (10.0, 10.0), &layout, 1600.0, 0.5, layout.outline_tint_day);
        let narrow_margin = narrow.scale[0] - 0.3;
        let wide_margin = wide.scale[0] - 0.3;
        assert!(narrow_margin > wide_margin, "a narrower canvas needs a bigger scale delta for the same pixel width");
        assert!((narrow_margin - wide_margin * 4.0).abs() < 1e-6, "margin is exactly inversely proportional to canvas width");
    }

    #[test]
    fn outline_uniform_margin_shrinks_as_the_mesh_local_extent_grows() {
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.0, 0.0], scale_x: 0.3, scale_y: 0.3, rotation: 0.0, tint: NO_TINT };
        let small_mesh = outline_uniform(&transform, 1.0, 1.0, (5.0, 5.0), &layout, 800.0, 0.5, layout.outline_tint_day);
        let big_mesh = outline_uniform(&transform, 1.0, 1.0, (50.0, 50.0), &layout, 800.0, 0.5, layout.outline_tint_day);
        assert!(
            small_mesh.scale[0] - 0.3 > big_mesh.scale[0] - 0.3,
            "a mesh with a smaller native extent needs a bigger scale delta for the same on-screen pixel width"
        );
    }

    #[test]
    fn outline_uniform_margin_is_computed_independently_per_axis() {
        // A long, thin mesh (like stem_segment.svg — see `MeshRegistry::
        // local_half_extent`'s doc comment) should get a *big* margin on
        // its short axis and a *small* one on its long axis, not the same
        // margin on both the way a single shared radius would produce.
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.0, 0.0], scale_x: 0.3, scale_y: 0.3, rotation: 0.0, tint: NO_TINT };
        let long_thin = outline_uniform(&transform, 1.0, 1.0, (5.0, 60.0), &layout, 800.0, 0.5, layout.outline_tint_day);
        let margin_x = long_thin.scale[0] - 0.3;
        let margin_y = long_thin.scale[1] - 0.3;
        assert!(margin_x > margin_y, "the short axis (small local extent) should get the bigger margin");
    }

    #[test]
    fn outline_uniform_is_a_no_op_scale_change_for_a_zero_extent_mesh() {
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.0, 0.0], scale_x: 0.3, scale_y: 0.3, rotation: 0.0, tint: NO_TINT };
        let outline = outline_uniform(&transform, 1.0, 1.0, (0.0, 0.0), &layout, 800.0, 0.5, layout.outline_tint_day);
        assert_eq!(outline.scale, [0.3, 0.3], "no extent to divide by, so no margin can be computed");
    }

    #[test]
    fn outline_uniform_sits_farther_away_than_its_own_paired_normal() {
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.0, 0.0], scale_x: 0.3, scale_y: 0.3, rotation: 0.0, tint: NO_TINT };
        let outline = outline_uniform(&transform, 1.0, 1.0, (10.0, 10.0), &layout, 800.0, 0.4, layout.outline_tint_day);
        assert!((outline.depth - (0.4 + layout.outline_depth_bias)).abs() < 1e-6);
    }

    #[test]
    fn outline_uniform_uses_whatever_tint_its_caller_passes() {
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.0, 0.0], scale_x: 0.3, scale_y: 0.3, rotation: 0.0, tint: NO_TINT };
        let hovered =
            outline_uniform(&transform, 1.0, 1.0, (10.0, 10.0), &layout, 800.0, 0.5, layout.hover_outline_tint);
        assert_eq!(hovered.tint, layout.hover_outline_tint);
    }

    #[test]
    fn apply_hover_scale_grows_scale_in_place_without_moving_the_center() {
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.3, -0.2], scale_x: 0.4, scale_y: 0.5, rotation: 0.1, tint: [0.5, 0.5, 0.5] };
        let hovered = apply_hover_scale(&transform, &layout);
        assert!(hovered.scale_x > transform.scale_x);
        assert!(hovered.scale_y > transform.scale_y);
        assert_eq!(hovered.offset, transform.offset);
        assert_eq!(hovered.rotation, transform.rotation);
    }

    fn rgba_from_tint(tint: [f32; 3]) -> [u8; 4] {
        [
            (tint[0] * 255.0).round() as u8,
            (tint[1] * 255.0).round() as u8,
            (tint[2] * 255.0).round() as u8,
            255,
        ]
    }

    #[test]
    fn leaf_pick_targets_round_trip_through_encode_and_decode() {
        for slot in [0usize, 1, 42, 95] {
            let target = PickTarget::Leaf { plant_index: 0, slot };
            let rgba = rgba_from_tint(encode_pick_target(target));
            assert_eq!(decode_pick_target(rgba), Some(target), "leaf slot {slot} didn't round-trip");
        }
    }

    #[test]
    fn stem_segment_pick_targets_round_trip_through_encode_and_decode() {
        for slot in [0usize, 1, 42, 300] {
            let target = PickTarget::StemSegment { plant_index: 0, slot };
            let rgba = rgba_from_tint(encode_pick_target(target));
            assert_eq!(decode_pick_target(rgba), Some(target), "stem segment slot {slot} didn't round-trip");
        }
    }

    #[test]
    fn leaf_and_stem_segment_id_spaces_never_collide_within_one_plant() {
        // Every leaf slot up to MAX_LEAVES shares one flat ID space with
        // every stem segment slot (see `encode_pick_target`'s doc comment)
        // — a leaf and a stem segment must never decode to the same color.
        let leaf = encode_pick_target(PickTarget::Leaf { plant_index: 0, slot: MAX_LEAVES - 1 });
        let stem_segment = encode_pick_target(PickTarget::StemSegment { plant_index: 0, slot: 0 });
        assert_ne!(leaf, stem_segment);
        assert_eq!(
            decode_pick_target(rgba_from_tint(leaf)),
            Some(PickTarget::Leaf { plant_index: 0, slot: MAX_LEAVES - 1 })
        );
        assert_eq!(
            decode_pick_target(rgba_from_tint(stem_segment)),
            Some(PickTarget::StemSegment { plant_index: 0, slot: 0 })
        );
    }

    #[test]
    fn different_plants_never_collide_even_at_the_same_slot() {
        for slot in [0usize, MAX_LEAVES - 1] {
            let plant0 = encode_pick_target(PickTarget::Leaf { plant_index: 0, slot });
            let plant1 = encode_pick_target(PickTarget::Leaf { plant_index: 1, slot });
            assert_ne!(plant0, plant1, "the same leaf slot on two different plants must not collide");
            assert_eq!(
                decode_pick_target(rgba_from_tint(plant1)),
                Some(PickTarget::Leaf { plant_index: 1, slot }),
                "should decode back to plant 1, not plant 0"
            );
        }
    }

    #[test]
    fn decode_pick_target_reads_the_reserved_none_color_as_nothing_hovered() {
        assert_eq!(decode_pick_target([0, 0, 0, 0]), None);
    }

    #[test]
    fn distinct_targets_never_encode_to_the_same_color() {
        let a = encode_pick_target(PickTarget::Leaf { plant_index: 0, slot: 10 });
        let b = encode_pick_target(PickTarget::Leaf { plant_index: 0, slot: 11 });
        assert_ne!(a, b);
    }

    #[test]
    fn apply_depth_look_shrinks_and_dims_farther_instances() {
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.1, 0.1], scale_x: 0.3, scale_y: 0.3, rotation: 0.0, tint: [1.0, 1.0, 1.0] };
        let near = apply_depth_look(&transform, 0.0, &layout);
        let far = apply_depth_look(&transform, 1.0, &layout);
        assert!(near.scale_x > far.scale_x, "nearer instances should render larger");
        assert!(near.tint[0] > far.tint[0], "nearer instances should render brighter");
        assert_eq!(near.offset, transform.offset, "depth never moves the instance's own position");
    }

    #[test]
    fn leaf_depth_is_stable_and_scattered_around_the_plant_depth() {
        fn leaf_at(attach_height: f64) -> Leaf {
            Leaf {
                attach_height,
                side: Side::Left,
                maturity: 1.0,
                droop: 0.0,
                helio_angle: 0.0,
                fold: 0.0,
                age: 0.0,
                senescence: 0.0,
            }
        }
        let layout = SceneLayout::default();
        let leaf_a = leaf_at(10.0);
        let leaf_b = leaf_at(23.0);
        assert_eq!(leaf_depth(&leaf_a, &layout), leaf_depth(&leaf_a, &layout), "same leaf, same depth every time");
        assert!(
            (leaf_depth(&leaf_a, &layout) - layout.plant_depth).abs() <= layout.leaf_depth_spread + 1e-6,
            "a leaf's depth should stay within its configured scatter range"
        );
        assert_ne!(leaf_depth(&leaf_a, &layout), leaf_depth(&leaf_b, &layout), "different leaves should generally scatter to different depths");
    }

    #[test]
    fn scene_light_uniform_scales_position_by_zoom_and_pan_like_every_other_offset() {
        let layout = SceneLayout::default();
        let sun = SunState { elevation: 1.0, azimuth: 0.5, intensity: 0.8, color: [1.0, 0.9, 0.7] };
        let light = SceneLightUniform::new(&sun, &layout, 0.5, [0.1, -0.05], None);
        assert_eq!(light.pos, [layout.window_offset[0] * 0.5 + 0.1, layout.window_offset[1] * 0.5 - 0.05]);
        assert_eq!(light.intensity, 0.8);
        assert_eq!(light.color, [1.0, 0.9, 0.7]);
        assert_eq!(light.ambient_floor, layout.night_ambient_color);
    }

    #[test]
    fn scene_light_uniform_cursor_is_off_when_the_pointer_is_not_over_the_canvas() {
        let layout = SceneLayout::default();
        let sun = SunState { elevation: 1.0, azimuth: 0.5, intensity: 0.8, color: [1.0, 0.9, 0.7] };
        let light = SceneLightUniform::new(&sun, &layout, 1.0, [0.0, 0.0], None);
        assert_eq!(light.cursor_intensity, 0.0);
    }

    #[test]
    fn scene_light_uniform_cursor_position_is_not_zoom_or_pan_adjusted_unlike_the_window_light() {
        let layout = SceneLayout::default();
        let sun = SunState { elevation: 1.0, azimuth: 0.5, intensity: 0.8, color: [1.0, 0.9, 0.7] };
        let light = SceneLightUniform::new(&sun, &layout, 0.5, [0.2, 0.2], Some([0.3, -0.2]));
        assert_eq!(light.cursor_pos, [0.3, -0.2], "cursor is screen-space, zoom/pan shouldn't move it");
        assert_eq!(light.cursor_intensity, layout.cursor_light_intensity);
    }

    #[test]
    fn scene_light_uniform_lamp_is_off_by_day_and_on_at_night() {
        let layout = SceneLayout::default();
        let day = SunState { elevation: 1.0, azimuth: 0.5, intensity: 1.0, color: [1.0, 1.0, 1.0] };
        let night = SunState { elevation: -0.5, azimuth: 1.0, intensity: 0.0, color: [1.0, 1.0, 1.0] };
        let day_light = SceneLightUniform::new(&day, &layout, 0.5, [0.0, 0.0], None);
        let night_light = SceneLightUniform::new(&night, &layout, 0.5, [0.0, 0.0], None);
        assert_eq!(day_light.lamp_intensity, 0.0);
        assert_eq!(night_light.lamp_intensity, layout.lamp_intensity_max);
        assert_eq!(night_light.lamp_pos, [layout.lamp_offset[0] * 0.5, layout.lamp_offset[1] * 0.5]);
        assert_eq!(night_light.lamp_falloff, layout.lamp_falloff);
    }

    #[test]
    fn lamp_on_fraction_matches_the_darkness_of_the_sun() {
        assert_eq!(lamp_on_fraction(1.0), 0.0);
        assert_eq!(lamp_on_fraction(0.0), 1.0);
        assert!((lamp_on_fraction(0.7) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn with_leaf_specular_sets_extent_and_shininess_without_touching_anything_else() {
        let layout = SceneLayout::default();
        let transform = Transform { offset: [0.1, 0.2], scale_x: 0.3, scale_y: 0.4, rotation: 0.5, tint: [0.6, 0.7, 0.8] };
        let base = InstanceUniform::from_transform(&transform, 1.2, 1.0, 0.5);
        let specular = with_leaf_specular(base, (12.0, 8.0), 24.0);
        assert_eq!(specular.local_extent, [12.0, 8.0]);
        assert_eq!(specular.shininess, 24.0);
        assert_eq!(specular.offset, base.offset);
        assert_eq!(specular.scale, base.scale);
        assert_eq!(specular.tint, base.tint);
    }

    #[test]
    fn instance_uniform_defaults_to_no_specular() {
        let transform = Transform { offset: [0.0, 0.0], scale_x: 1.0, scale_y: 1.0, rotation: 0.0, tint: NO_TINT };
        let uniform = InstanceUniform::from_transform(&transform, 1.0, 1.0, 0.5);
        assert_eq!(uniform.shininess, 0.0);
        assert_eq!(uniform.local_extent, [1.0, 1.0]);
    }
}
