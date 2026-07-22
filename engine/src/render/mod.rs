//! Owns the wgpu surface/device, holds the live simulation state, and
//! drives its own `requestAnimationFrame` loop — JS constructs a
//! `Simulation`, calls `start()`, and otherwise stays out of the way. Each
//! frame advances `sim::plant::Plant`/`sim::soil::Soil` by real elapsed time
//! (scaled — see `config::TimeConfig`) and redraws the scene built from
//! their current state (`scene.rs`).
//!
//! `config` and `scene` are pure math/data with no wgpu/wasm-bindgen
//! dependency of their own, so they're declared unconditionally here and
//! compile natively — `cargo test` exercises the exact placement geometry
//! (sun/moon position inside the window, leaf/branch frames, wall
//! coverage) that would otherwise only be checkable by eyeballing a
//! rendered screenshot. `meshes` and everything below (`wgpu_engine`)
//! genuinely need wgpu/web-sys/wasm-bindgen, which only exist as
//! dependencies on `wasm32` (see Cargo.toml's target-gated dependency
//! block), so those stay gated.

pub mod config;
#[cfg(target_arch = "wasm32")]
mod meshes;
pub mod scene;

#[cfg(target_arch = "wasm32")]
pub use wgpu_engine::Simulation;

#[cfg(target_arch = "wasm32")]
mod wgpu_engine {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use bytemuck::bytes_of;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use web_sys::HtmlCanvasElement;
    use wgpu::util::DeviceExt;

    use super::config::SceneLayout;
    use super::meshes::{self, MeshRegistry};
    use super::scene::{self, BackgroundSpec, StemCurve, Transform, MAX_BRANCHES, MAX_LEAVES};
    use crate::sim::climate;
    use crate::sim::config::{plant_config_for_species, GrowthConfig, GrowthHabit, PlantConfig, TimeConfig};
    use crate::sim::humidity::Humidity;
    use crate::sim::plant::{
        self_shading_factors, Decision, DeathCause, Plant, Side, Stage, MAX_AERIAL_ROOTS, MAX_STEM_SEGMENTS,
    };
    use crate::sim::moon;
    use crate::sim::room;
    use crate::sim::season;
    use crate::sim::soil::Soil;
    use crate::sim::sun;

    use super::scene::InstanceUniform;

    struct Drawable {
        mesh: &'static str,
        uniform_buffer: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
    }

    /// Real GPU depth-buffer format backing `InstanceUniform::depth` (see
    /// `scene.rs`) — lets overlapping instances (two leaves, a leaf and its
    /// own outline halo) resolve correctly via the depth test instead of
    /// relying entirely on manual back-to-front draw order.
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

    fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth-texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Backs the GPU leaf-picking pass (see `pick_pipeline`'s own doc
    /// comment) — canvas-sized so its pixels line up 1:1 with the cursor's
    /// own canvas-pixel position, even though only a single scissored-in
    /// texel ever actually gets shaded/read back each frame.
    const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// Returns the texture itself alongside its view — unlike `depth_view`,
    /// `render` needs the raw texture too (`copy_texture_to_buffer`'s source
    /// takes a texture, not a view) for the pick readback.
    fn create_pick_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick-texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn write_transform(
        queue: &wgpu::Queue,
        drawable: &Drawable,
        transform: &Transform,
        aspect: f32,
        zoom: f32,
        depth: f32,
        layout: &SceneLayout,
    ) {
        let looked = scene::apply_depth_look(transform, depth, layout);
        let uniform = InstanceUniform::from_transform(&looked, aspect, zoom, depth);
        queue.write_buffer(&drawable.uniform_buffer, 0, bytes_of(&uniform));
    }

    fn write_transmissive_transform(
        queue: &wgpu::Queue,
        drawable: &Drawable,
        transform: &Transform,
        aspect: f32,
        zoom: f32,
        depth: f32,
        layout: &SceneLayout,
    ) {
        let looked = scene::apply_depth_look(transform, depth, layout);
        let uniform = InstanceUniform::from_transform(&looked, aspect, zoom, depth);
        let transmissive = scene::with_transmissive(uniform);
        queue.write_buffer(&drawable.uniform_buffer, 0, bytes_of(&transmissive));
    }

    /// Like `write_transform`, but also turns on the cursor specular
    /// highlight (see `scene::with_leaf_specular`) — leaves only, since
    /// that's the one mesh this fake-dome-normal trick actually reads as a
    /// glossy surface rather than a flat cutout.
    fn write_leaf_transform(
        queue: &wgpu::Queue,
        meshes: &meshes::MeshRegistry,
        drawable: &Drawable,
        transform: &Transform,
        aspect: f32,
        zoom: f32,
        depth: f32,
        layout: &SceneLayout,
    ) {
        let looked = scene::apply_depth_look(transform, depth, layout);
        let uniform = InstanceUniform::from_transform(&looked, aspect, zoom, depth);
        let local_half_extent = meshes.local_half_extent(drawable.mesh);
        let specular = scene::with_leaf_specular(uniform, local_half_extent, layout.leaf_shininess);
        queue.write_buffer(&drawable.uniform_buffer, 0, bytes_of(&specular));
    }

    /// Writes the white halo `Drawable` that sits behind (drawn before) a
    /// plant-asset mesh's own normal-tinted `Drawable` — see `scene::
    /// outline_uniform`. A separate function (not just another `write_
    /// transform` call) because it needs the mesh's own baked local radius
    /// and the canvas's current pixel width, neither of which the plain
    /// per-instance path cares about.
    fn write_outline_transform(
        queue: &wgpu::Queue,
        meshes: &meshes::MeshRegistry,
        drawable: &Drawable,
        transform: &Transform,
        aspect: f32,
        zoom: f32,
        canvas_width_px: f32,
        depth: f32,
        tint: [f32; 3],
        layout: &SceneLayout,
    ) {
        let local_half_extent = meshes.local_half_extent(drawable.mesh);
        let uniform = scene::outline_uniform(
            transform,
            aspect,
            zoom,
            local_half_extent,
            layout,
            canvas_width_px,
            depth,
            tint,
        );
        queue.write_buffer(&drawable.uniform_buffer, 0, bytes_of(&uniform));
    }

    /// Writes the GPU hit-testing `Drawable` for one leaf or stem segment —
    /// see `pick_pipeline`'s own doc comment. Same transform/depth as the
    /// real draw (so the pick pass's depth test against the already-
    /// populated depth buffer resolves occlusion identically), tinted with
    /// its flat target-ID color instead of a real color, and — like
    /// `write_outline_transform` — grown by `scene::outline_scale_margin`
    /// on top of whatever `transform` already is, so the clickable hitbox
    /// extends all the way to the visible outline halo's own edge (white
    /// while idle, `SceneLayout::hover_outline_tint`'s red once hovered —
    /// `transform` should already be `scene::apply_hover_scale`d when this
    /// target is the current `hovered_target`, the same `display_transform`
    /// passed to the real draw/outline, so the hitbox and the highlight
    /// grow together instead of the hitbox staying pinned at the plain
    /// mesh's own smaller, now visually-inaccurate edge).
    fn write_pick_transform(
        queue: &wgpu::Queue,
        meshes: &meshes::MeshRegistry,
        drawable: &Drawable,
        transform: &Transform,
        aspect: f32,
        zoom: f32,
        canvas_width_px: f32,
        depth: f32,
        layout: &SceneLayout,
        target: scene::PickTarget,
    ) {
        let id_tint = Transform { tint: scene::encode_pick_target(target), ..*transform };
        let mut uniform = InstanceUniform::from_transform(&id_tint, aspect, zoom, depth);
        let local_half_extent = meshes.local_half_extent(drawable.mesh);
        let margin = scene::outline_scale_margin(local_half_extent, aspect, canvas_width_px, layout);
        uniform.scale[0] += margin[0];
        uniform.scale[1] += margin[1];
        queue.write_buffer(&drawable.uniform_buffer, 0, bytes_of(&uniform));
    }

    /// Pool size for `GpuState::stem_segment_drawables` — the main stem's
    /// own segments (up to `MAX_STEM_SEGMENTS`) plus every branch's own (up
    /// to `MAX_STEM_SEGMENTS` each), combined the same way `leaf_drawables`
    /// already covers the main stem's leaves plus every branch's own.
    const MAX_STEM_SEGMENT_DRAWABLES: usize = MAX_STEM_SEGMENTS * (1 + MAX_BRANCHES);

    /// Fixed cap on simultaneously-held plants — see `GpuState::plants` and
    /// `Simulation::plant_cutting`. Same "fixed pool, only draw/use the first N
    /// that actually exist" pattern this pipeline already uses everywhere
    /// else (leaves, stem segments, aerial roots); a small number since
    /// each additional plant is a *full* duplicate drawable-pool set (see
    /// `build_plant_slot`), not a cheap instance.
    const MAX_PLANTS: usize = 6;

    /// Clamp on `GpuState::camera_pan`, in NDC units (each axis -1..1 is
    /// the visible canvas) — keeps the wall's own edge (see `SceneLayout::
    /// wall_scale`'s overscan) from ever panning into view; the room has
    /// no art beyond that edge to reveal.
    const MAX_CAMERA_PAN: f32 = 1.3;

    /// Everything genuinely *per plant* — one pot, one growing thing, its
    /// own full set of GPU drawable pools. `GpuState::plants` holds one of
    /// these per pot in the room (see `scene::plant_slot_base_anchor` for
    /// how each one's own windowsill position is derived from its index);
    /// every plant simulates *and* renders every frame regardless of
    /// selection (see `render`), while `GpuState::selected_plant_index`
    /// decides which one the HUD and player actions (water, prune, repot,
    /// etc.) actually target.
    struct PlantSlot {
        plant: Plant,
        soil: Soil,
        /// This specific plant's own species/growth-habit tuning (see
        /// `sim::config::plant_config_for_species`) — every other
        /// `GrowthConfig` sub-config (soil physics, pests, climate, the
        /// room, the season/moon cycles) is genuinely shared/global and
        /// stays on `GpuState::growth_config`; only species varies plant to
        /// plant. Wherever a `&GrowthConfig` is needed for this specific
        /// plant (its own `Plant::step`, prune/repot/cutting actions), it's
        /// built fresh as `GrowthConfig { plant: plant_config, ..growth_
        /// config }` rather than this slot carrying a whole redundant copy
        /// of the shared fields too.
        plant_config: PlantConfig,
        /// The canonical species name (see `sim::config::plant_config_for_
        /// species`) that produced `plant_config` — kept alongside it since
        /// `PlantConfig` itself doesn't round-trip back to a name. Needed
        /// so a stem cutting taken from this plant (see `Simulation::take_
        /// cutting`/`InventoryItem`) remembers which species to grow once
        /// planted, and so the inventory/species UI can show which species
        /// each plant/cutting actually is.
        species_name: String,
        /// Where the pot sits relative to the window (see `sim::room`) —
        /// 0.0 (right at the sill) ..= 1.0 (far across the room). Player-
        /// controlled via `Simulation::set_pot_position`.
        pot_position: f64,
        /// Whether `pot_position` should actually be applied yet. There's no
        /// single position that's a no-op for *both* light and draft (moving
        /// toward the window trades one for the other, by design) — a
        /// default position value can only be a no-op for one axis, at the
        /// cost of unconditionally taxing every session on the other. So
        /// this stays off (full light, no draft — matching the room's
        /// original, pre-this-mechanic tuning) until `set_pot_position` is
        /// actually called, the same "inert until the player engages with
        /// it" rule every other new mechanic this session follows.
        pot_position_active: bool,
        /// Self-watering-pot mode (see `Soil::apply_auto_water`) — per
        /// plant, not a room-global setting (a room can hold several
        /// plants with very different water needs). Off by default for
        /// every newly created plant, same "inert until the player
        /// actually engages with it" rule `pot_position_active` follows.
        auto_water_enabled: bool,

        /// This plant's own pot + soil — see `scene::pot_background`. The
        /// wall/window are room-level instead (see `GpuState::room_
        /// background_drawables`), not duplicated per plant.
        background_specs: Vec<BackgroundSpec>,
        background_drawables: Vec<Drawable>,
        /// A climbing-support pole/lattice, only actually present/drawn
        /// once this slot's own `plant_config.trellis_height` is `Some` —
        /// see `scene::trellis_transform`. Allocated unconditionally (like
        /// `flower_drawable`) so switching species at runtime (`set_
        /// species`) never needs a new GPU buffer created mid-session.
        trellis_drawable: Drawable,
        /// A fibrous root mass drawn over the soil, visible through
        /// `pot.svg`'s hollow outline (see that file's own doc comment —
        /// `pot` is a rim + wall-strip silhouette with an open middle, not
        /// a filled shape, specifically so this and `soil` show through
        /// it), tinted by `Plant::root_health`. The cheap version of "see
        /// into the pot": the render pipeline draws every mesh fully
        /// opaque, back-to-front, with no alpha blending at all (see
        /// `RenderPipelineDescriptor`'s `BlendState::REPLACE` in `GpuState
        /// ::new`), so genuine see-through transparency isn't available —
        /// hollowing the pot mesh out and layering soil/roots underneath
        /// achieves the same visible result without needing it. Always
        /// present once germinated, same reasoning as `trellis_drawable`
        /// for why it's allocated unconditionally.
        roots_drawable: Drawable,
        /// Fixed-size pool (see `sim::plant::MAX_AERIAL_ROOTS`) covering
        /// `plant.aerial_roots` — main stem only, see that field's own doc
        /// comment. Only the first N (however many actually exist this
        /// frame) are ever drawn, same pattern as `leaf_drawables`.
        aerial_root_drawables: Vec<Drawable>,
        seed_drawable: Drawable,
        cotyledon_drawables: [Drawable; 2],
        /// Repointed to `PlantConfig::flower_mesh_name` every frame — see
        /// `render`. Always drawn; `Plant::bloom_intensity` (0 whenever not
        /// currently in bloom) scales it to invisible on its own.
        flower_drawable: Drawable,
        /// Fixed-size pool covering every segment of the main stem's own
        /// curve *and* every branch's own curve combined (see
        /// `scene::stem_segment_transforms`/`sim::plant::MAX_STEM_SEGMENTS`),
        /// filled main-stem-first then branch-by-branch each frame (see
        /// `render`) — same pattern as `leaf_drawables`. A stem no longer
        /// renders as one rigid rotated mesh; it's a chain of these, each
        /// covering just its own portion of the height, so the whole thing
        /// reads as a gentle sweep instead of a straight pivoted line.
        stem_segment_drawables: Vec<Drawable>,
        /// Fixed-size pool (see `scene::MAX_LEAVES`) — covers the main stem's
        /// leaves *and* every branch's own leaves, filled main-stem-first then
        /// branch-by-branch each frame (see `render`). Only the first N (however
        /// many actually exist this frame) are ever drawn; growing the plant
        /// never needs a new GPU buffer created mid-frame.
        leaf_drawables: Vec<Drawable>,

        // Everything below mirrors one of the pools above 1:1 (same mesh,
        // same fill order, same pool size) but draws `scene::outline_
        // uniform`'s enlarged/white-tinted version instead — see `render`,
        // which writes/draws each pair back-to-back. A wholly separate pool
        // rather than drawing each `Drawable` above twice, because wgpu
        // buffer writes only take effect by GPU-execution time: writing the
        // same uniform buffer twice before either draw actually executes
        // just leaves it holding its *last* written value for both draws
        // (a real bug hit while building this), not two different ones.
        roots_outline_drawable: Drawable,
        aerial_root_outline_drawables: Vec<Drawable>,
        seed_outline_drawable: Drawable,
        cotyledon_outline_drawables: [Drawable; 2],
        /// Repointed alongside `flower_drawable` every frame — see `render`.
        flower_outline_drawable: Drawable,
        stem_segment_outline_drawables: Vec<Drawable>,
        leaf_outline_drawables: Vec<Drawable>,
        /// One per leaf slot, tinted with that slot's flat pick-ID color
        /// each frame — see `pick_pipeline`'s doc comment. A separate pool
        /// (not a repurposed `leaf_drawables`/`leaf_outline_drawables`
        /// buffer) for the same reason every other outline pool is
        /// separate: a buffer only holds its *last* written value by the
        /// time either draw actually executes, so drawing the same leaf
        /// with two different tints this frame needs two different
        /// buffers.
        leaf_pick_drawables: Vec<Drawable>,
        /// One per stem-segment pool slot (main stem then branches, same
        /// pool/ordering as `stem_segment_drawables`), tinted with that
        /// slot's flat pick-ID color each frame — see `pick_pipeline`'s doc
        /// comment and `stem_segment_targets`, which is what actually maps
        /// a hovered slot back to "which grower, what height to cut."
        stem_segment_pick_drawables: Vec<Drawable>,
        /// Per-frame: for each stem-segment pool slot actually drawn this
        /// frame, which grower it belongs to (`None` = main stem, `Some(i)`
        /// = branch `i`) and the cut height a click on it should apply
        /// (that segment's own base height along that grower's stem) — see
        /// `Simulation::prune_hovered`, which looks this up once a stem-
        /// segment `PickTarget` comes back from a readback. Rebuilt fresh
        /// every `render()` call (same lifetime as the drawables
        /// themselves), not persisted across frames.
        stem_segment_targets: Vec<(Option<usize>, f64)>,
        /// How many of `leaf_drawables`/`stem_segment_drawables`/`aerial_
        /// root_drawables` (and their outline/pick twins) actually hold
        /// real, this-frame content vs. stale/zeroed leftovers from a
        /// smaller previous frame — the render pass's draw loop uses these
        /// to draw only the first N of each pool. Set once per `render()`
        /// call by the write phase, read back by the separate draw-pass
        /// loop that runs after every plant's writes are done; whether the
        /// trellis is present this frame at all follows the same "written
        /// during the write phase, read during the draw phase" pattern.
        leaves_drawn: usize,
        segments_drawn: usize,
        aerial_roots_drawn: usize,
        trellis_active: bool,
    }

    /// Builds a brand-new `PlantSlot` — everything `GpuState::new` needs for
    /// the session's first plant, and what `Simulation::plant_cutting`/`plant_
    /// cutting` need to bring another one into existence at runtime. A free
    /// function (not a `&self` method) specifically so `GpuState::new` can
    /// call it *before* a `GpuState` exists to be `self`. `initial_plant`
    /// lets callers start it already-grown (`Plant::from_cutting`, for
    /// `plant_cutting`) instead of always a fresh seed (`Plant::new`, for
    /// `GpuState::new`/`plant_cutting`).
    fn build_plant_slot(
        device: &wgpu::Device,
        instance_bind_group_layout: &wgpu::BindGroupLayout,
        aspect: f32,
        layout: &SceneLayout,
        growth_config: &GrowthConfig,
        plant_config: PlantConfig,
        initial_plant: Plant,
        species_name: String,
    ) -> PlantSlot {
        let make_drawable = |mesh: &'static str, transform: &Transform| -> Drawable {
            // The wall is exempt from zoom (see `SceneLayout::zoom`'s doc
            // comment) — same special case `render()` applies every frame.
            let zoom = if mesh == "wall" { 1.0 } else { layout.zoom };
            // Depth doesn't matter here — every drawable's uniform gets
            // fully rewritten (including its real depth) the first time
            // `render()` runs, this just needs to produce a validly-sized
            // initial buffer.
            let uniform = InstanceUniform::from_transform(transform, aspect, zoom, 0.5);
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(mesh),
                contents: bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(mesh),
                layout: instance_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            Drawable { mesh, uniform_buffer, bind_group }
        };

        let background_specs = scene::pot_background(layout);
        let background_drawables =
            background_specs.iter().map(|spec| make_drawable(spec.mesh, &spec.transform)).collect();

        let seed_drawable = make_drawable("seed", &scene::seed_transform(layout, 0.0));
        let cotyledon_drawables = [
            make_drawable("cotyledon", &scene::cotyledon_transform(layout, Side::Left)),
            make_drawable("cotyledon", &scene::cotyledon_transform(layout, Side::Right)),
        ];

        let plant = initial_plant;
        let zero_transform =
            Transform { offset: [0.0, 0.0], scale_x: 0.0, scale_y: 0.0, rotation: 0.0, tint: [1.0, 1.0, 1.0] };
        let flower_drawable = make_drawable(plant_config.flower_mesh_name, &zero_transform);
        let trellis_drawable = make_drawable(
            "trellis",
            &scene::trellis_transform(plant_config.trellis_height, layout).unwrap_or(zero_transform),
        );
        let roots_drawable = make_drawable("roots", &zero_transform);
        let aerial_root_drawables =
            (0..MAX_AERIAL_ROOTS).map(|_| make_drawable("aerial_root", &zero_transform)).collect();
        let stem_segment_drawables =
            (0..MAX_STEM_SEGMENT_DRAWABLES).map(|_| make_drawable("stem_segment", &zero_transform)).collect();
        let leaf_drawables = (0..MAX_LEAVES).map(|_| make_drawable("leaf", &zero_transform)).collect();

        let seed_outline_drawable = make_drawable("seed", &scene::seed_transform(layout, 0.0));
        let cotyledon_outline_drawables = [
            make_drawable("cotyledon", &scene::cotyledon_transform(layout, Side::Left)),
            make_drawable("cotyledon", &scene::cotyledon_transform(layout, Side::Right)),
        ];
        let flower_outline_drawable = make_drawable(plant_config.flower_mesh_name, &zero_transform);
        let roots_outline_drawable = make_drawable("roots", &zero_transform);
        let aerial_root_outline_drawables =
            (0..MAX_AERIAL_ROOTS).map(|_| make_drawable("aerial_root", &zero_transform)).collect();
        let stem_segment_outline_drawables =
            (0..MAX_STEM_SEGMENT_DRAWABLES).map(|_| make_drawable("stem_segment", &zero_transform)).collect();
        let leaf_outline_drawables = (0..MAX_LEAVES).map(|_| make_drawable("leaf", &zero_transform)).collect();
        let leaf_pick_drawables = (0..MAX_LEAVES).map(|_| make_drawable("leaf", &zero_transform)).collect();
        let stem_segment_pick_drawables =
            (0..MAX_STEM_SEGMENT_DRAWABLES).map(|_| make_drawable("stem_segment", &zero_transform)).collect();

        PlantSlot {
            plant,
            soil: Soil::new(&growth_config.soil),
            plant_config,
            species_name,
            pot_position: 0.0,
            pot_position_active: false,
            auto_water_enabled: false,
            background_specs,
            background_drawables,
            trellis_drawable,
            roots_drawable,
            aerial_root_drawables,
            seed_drawable,
            cotyledon_drawables,
            flower_drawable,
            stem_segment_drawables,
            leaf_drawables,
            roots_outline_drawable,
            aerial_root_outline_drawables,
            seed_outline_drawable,
            cotyledon_outline_drawables,
            flower_outline_drawable,
            stem_segment_outline_drawables,
            leaf_outline_drawables,
            leaf_pick_drawables,
            stem_segment_pick_drawables,
            stem_segment_targets: Vec::new(),
            leaves_drawn: 0,
            segments_drawn: 0,
            aerial_roots_drawn: 0,
            trellis_active: false,
        }
    }

    /// Which kind of geometry a click acts on — see `GpuState::active_tool`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Tool {
        /// Removes a whole leaf — see `Plant::prune_leaf`.
        Prune,
        /// Cuts the main stem or a branch at a clicked point — see
        /// `Plant::cut_main_stem_at`/`cut_branch_at`.
        Trim,
    }

    struct GpuState {
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        pipeline: wgpu::RenderPipeline,
        /// Group(0)'s layout — stored (not just a constructor-local, like it
        /// used to be) so `Simulation::plant_cutting` can build a new plant's
        /// worth of `Drawable`s at runtime via `build_plant_slot`, the same
        /// way `GpuState::new` builds the first one.
        instance_bind_group_layout: wgpu::BindGroupLayout,
        /// Backs `InstanceUniform::depth` — see `DEPTH_FORMAT`. Recreated in
        /// `resize` to match the surface's new size.
        depth_view: wgpu::TextureView,
        meshes: MeshRegistry,
        /// Group(1) uniform — see `scene::SceneLightUniform`. Rewritten once
        /// per frame (not per-instance, unlike everything in
        /// `background_drawables`/etc.) and bound once before the draw loop.
        scene_light_buffer: wgpu::Buffer,
        scene_light_bind_group: wgpu::BindGroup,

        /// Which drawn objects show the sun/moon's current angle — see
        /// `scene::sky_object_transform`. Only one is ever drawn per frame,
        /// chosen by `sun.elevation`'s sign. Genuinely room-global (there's
        /// one sky regardless of how many plants sit in front of it), so
        /// these live here rather than in `PlantSlot`.
        sun_drawable: Drawable,
        moon_drawable: Drawable,
        /// The dark disc drawn on top of `moon_drawable` to fake its current
        /// phase — see `scene::moon_shadow_transform`. Reuses the "moon"
        /// mesh itself (just tinted dark and shifted), no new asset needed.
        moon_shadow_drawable: Drawable,
        /// The wall lamp fixture — see `scene::lamp_transform`. Room-global
        /// like the sky drawables above, always drawn (its own tint reads
        /// as "off" by day).
        lamp_drawable: Drawable,
        table_drawable: Drawable,

        /// The wall/window pane — see `scene::room_background`. Built once
        /// (unlike `PlantSlot::background_specs`/`background_drawables`,
        /// which each plant carries its own copy of for its pot/soil),
        /// since there's only one room regardless of how many plants share
        /// it.
        room_background_specs: Vec<BackgroundSpec>,
        room_background_drawables: Vec<Drawable>,
        /// One entry per pot in the room — see `PlantSlot`'s own doc
        /// comment. Capped at `MAX_PLANTS`; grown by `Simulation::
        /// plant_cutting`.
        plants: Vec<PlantSlot>,
        /// Which `plants` entry the whole per-plant HUD/render/action
        /// surface currently targets — see `render`'s own doc comment on
        /// why only this one actually renders each frame (every plant
        /// still *simulates* regardless of selection). Defaults to 0 (the
        /// session's first plant); `Simulation::set_selected_plant` is the
        /// only thing that ever changes it.
        selected_plant_index: usize,

        pick_pipeline: wgpu::RenderPipeline,
        /// Recreated in `resize` alongside `depth_view`. `pick_texture`
        /// itself is needed for `copy_texture_to_buffer`'s source; `pick_
        /// view` is what the pick render pass actually draws into.
        pick_texture: wgpu::Texture,
        pick_view: wgpu::TextureView,
        /// Where the single-texel `copy_texture_to_buffer` in `render`
        /// lands — mapped and read back asynchronously (see `hovered_
        /// target`/`pick_pending`). Fixed 256-byte size regardless of
        /// canvas resolution (see its own construction), never recreated.
        pick_readback_buffer: wgpu::Buffer,
        /// The prune tool's current hover target, if any — `None` means
        /// either nothing is under the cursor or no readback has completed
        /// yet. Shared (`Rc<Cell<_>>`, not a plain field) because the async
        /// `map_async` callback that actually updates it runs later, in a
        /// separate invocation outside of `render`'s own borrow of `self`
        /// entirely — it needs its own independent handle into the same
        /// cell. Lagging the mouse by a frame or two (until a pick readback
        /// resolves) is a deliberate tradeoff for keeping this off the CPU
        /// entirely — see `render`'s own doc comment on the pick pass.
        hovered_target: Rc<Cell<Option<scene::PickTarget>>>,
        /// Which kind of geometry the pick pass actually hit-tests against
        /// right now (see `render`'s pick pass and `Simulation::set_active_
        /// tool`) — Prune only ever picks leaves, Trim only ever picks stem
        /// segments, so a click always acts on the tool the player actually
        /// chose rather than whichever happened to be nearest the cursor.
        active_tool: Option<Tool>,
        /// Guards against issuing a second pick readback while one is
        /// already in flight — `map_async`'s callback flips this back to
        /// `false` once it resolves. Shared for the same reason `hovered_
        /// target` is.
        pick_pending: Rc<Cell<bool>>,
        /// The pointer's last-known position over the canvas, in CSS
        /// pixels (see `Simulation::set_pointer_position`) — `None` while
        /// the pointer isn't over the canvas at all, in which case no pick
        /// pass runs and `hovered_target` is cleared. Converted to device
        /// pixels (via `device_pixel_ratio`) only at the point of use in
        /// `render`, matching how every other CSS-pixel input this engine
        /// takes is handled.
        pointer_pixel: Option<(f32, f32)>,

        /// Camera pan offset, in the same final NDC units `cursor_ndc`
        /// above uses — accumulated by drag gestures (see `Simulation::
        /// pan_camera`), clamped to `MAX_CAMERA_PAN` so the wall's own
        /// (overscanned but still finite) edge never comes into view.
        /// Added to each room-level anchor's own pre-zoom offset, divided
        /// by that frame's `zoom` first so the same pan always reads as
        /// the same on-screen distance regardless of how zoomed out the
        /// room currently is (the wall itself is exempt from zoom, so it
        /// alone gets the raw, undivided value — see `render`).
        camera_pan: [f32; 2],

        /// Ambient air humidity — see `sim::humidity::Humidity`. One shared
        /// value for the whole room (misting affects the room's air, not
        /// one specific plant), advanced once per frame (`Humidity::
        /// update`) using the room-position-adjusted temperature, same
        /// place each plant's own soil update happens (inside `Plant::
        /// step`).
        humidity: Humidity,
        /// `window.devicePixelRatio` at last resize — `canvas.width/height`
        /// (what `config.width`/`config.height` hold) are already scaled by
        /// this for crisp hi-DPI output (see `EngineCanvas.tsx`'s
        /// `syncCanvasBackingSize`), so anything meant to look a fixed *CSS*
        /// pixel size on screen — currently just the outline halo's target
        /// width, see `render`'s `css_width_px` — has to divide it back out
        /// first. Without this, the halo was sized against the raw device-
        /// pixel canvas width, making it genuinely sub-CSS-pixel-thin (and
        /// prone to disappearing between rasterized pixels, with no MSAA to
        /// soften it) on any hi-DPI display.
        device_pixel_ratio: f32,
        growth_config: GrowthConfig,
        scene_layout: SceneLayout,
        /// 0.0..1.0, wraps — advanced each frame by real elapsed time scaled
        /// through `growth_config.time`.
        day_progress: f64,
        /// Cumulative sim-seconds since this *session* started — distinct
        /// from any individual `Plant::total_time` (which resets whenever
        /// that specific plant restarts/is replaced). Drives the moon phase
        /// (see `sim::moon`'s own doc comment: a genuinely ongoing
        /// astronomical cycle grounded in the real calendar date the
        /// session started on) and, since multiple plants can exist and
        /// restart independently, also the room's shared season/day-count
        /// display (see `Stats::days_elapsed`/`season`) — neither makes
        /// sense tied to any one specific plant's own lifecycle once more
        /// than one can exist. The moon was the first thing this fixed:
        /// reading `plant.total_time()` like everything else used to meant
        /// restarting reset it to 0, snapping the moon back to its
        /// session-start phase too, even though a fresh phase computation
        /// from scratch was always mathematically correct. Never reset by
        /// `reset_plant`.
        session_time: f64,
        last_timestamp: Option<f64>,
        /// Stem cuttings taken (see `Simulation::take_cutting`) but not yet
        /// planted (see `Simulation::plant_cutting`) — session-level, like
        /// `session_time`: a cutting sitting in inventory isn't attached to
        /// any specific `PlantSlot`, so it survives that plant later dying,
        /// being pruned further, or `reset_plant` replacing it entirely.
        inventory: Vec<InventoryItem>,
    }

    struct InventoryItem {
        species_name: String,
        from_seed: bool,
    }

    impl GpuState {
        async fn new(
            canvas: HtmlCanvasElement,
            dpr: f32,
            seed_year: i32,
            seed_month: u32,
            seed_day: u32,
        ) -> Result<Self, JsValue> {
            let width = canvas.width().max(1);
            let height = canvas.height().max(1);

            // Prefers WebGPU, but only if `is_browser_webgpu_supported()` (an
            // async check — a browser can define `navigator.gpu` yet still fail
            // to produce an adapter) actually confirms it; otherwise transparently
            // falls back to the WebGL2 backend. See wgpu::util's doc comment for
            // why this has to be async rather than a plain `Instance::new`.
            let instance =
                wgpu::util::new_instance_with_webgpu_detection(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                })
                .await;

            let surface = instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                .map_err(|e| JsValue::from_str(&format!("create_surface failed: {e}")))?;

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|e| JsValue::from_str(&format!("request_adapter failed: {e}")))?;

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("plant-device"),
                    required_features: wgpu::Features::empty(),
                    // WebGL2's limits are stricter than wgpu's defaults (which
                    // assume a more capable backend) — downlevel_webgl2_defaults
                    // keeps this working on the WebGL2 fallback path, not just
                    // WebGPU.
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    ..Default::default()
                })
                .await
                .map_err(|e| JsValue::from_str(&format!("request_device failed: {e}")))?;

            let config = surface
                .get_default_config(&adapter, width, height)
                .ok_or_else(|| JsValue::from_str("surface unsupported by this adapter"))?;
            surface.configure(&device, &config);

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()),
            });

            let instance_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("instance-bind-group-layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        // Used to be VERTEX-only — correct back when only
                        // `vs_main` ever read `instance` fields directly
                        // (the fragment stage only saw whatever `vs_main`
                        // had already baked into its `color`/`world_pos`
                        // varyings). `fs_main`/`fs_pick` now read `instance.
                        // tint`/`shininess`/`local_extent` directly too (the
                        // cursor specular highlight and the pick pass's ID
                        // color), so the layout has to declare the binding
                        // visible to both stages or pipeline creation itself
                        // fails validation — which doesn't just skip the
                        // effect, it invalidates the *entire* pipeline (and
                        // therefore every command buffer built against it),
                        // which is what actually caused a real regression:
                        // the whole scene going black, not just the new
                        // lighting failing to show up.
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            // Shared across every draw this frame (see `scene::
            // SceneLightUniform`) — visible to both stages, unlike the
            // per-instance uniform above (vertex passes world position
            // through, fragment does the actual distance-based lighting).
            let scene_light_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("scene-light-bind-group-layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
            let scene_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene-light"),
                contents: bytes_of(&scene::SceneLightUniform {
                    pos: [0.0, 0.0],
                    intensity: 0.0,
                    falloff: 0.0,
                    color: [1.0, 1.0, 1.0],
                    _pad0: 0.0,
                    ambient_floor: [0.0, 0.0, 0.0],
                    _pad1: 0.0,
                    cursor_pos: [0.0, 0.0],
                    cursor_intensity: 0.0,
                    cursor_falloff: 0.0,
                    lamp_pos: [0.0, 0.0],
                    lamp_intensity: 0.0,
                    lamp_falloff: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let scene_light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene-light-bind-group"),
                layout: &scene_light_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_light_buffer.as_entire_binding(),
                }],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene-pipeline-layout"),
                bind_group_layouts: &[Some(&instance_bind_group_layout), Some(&scene_light_bind_group_layout)],
                immediate_size: 0,
            });

            let vertex_layout = wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<meshes::Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 8,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            };

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("scene-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(vertex_layout.clone())],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    // `LessEqual`, not `Less` — several background pieces
                    // (wall, window, pot, soil, sun/moon) intentionally
                    // share the exact same nominal `background_depth`
                    // (there's no real 3D position for them, just "the
                    // room" as one flat layer). With strict `Less`, a tied
                    // depth always *fails* against whatever already wrote
                    // that same value, so only the first same-depth piece
                    // drawn each frame would ever show — which is exactly
                    // what made the window/sun/moon disappear entirely.
                    // `LessEqual` keeps ties resolving by draw order (the
                    // painter's-algorithm behavior this pipeline already
                    // relied on before depth existed) while still letting
                    // genuinely different depths (leaves vs. stem, plant
                    // vs. background, an outline vs. its own paired normal)
                    // resolve correctly via the real depth test.
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            // GPU hit-testing for the prune tool (see `Simulation::set_
            // pointer_position`/`prune_hovered`): every leaf and stem
            // segment is drawn a second time, flat-tinted with its own
            // `scene::PickTarget` ID color instead of a real color, into
            // `pick_view`. Reuses the *same* depth buffer the main pass
            // just populated this frame (see `render`) with writes
            // disabled and `LessEqual` — so a leaf or segment hidden behind
            // something nearer correctly never wins the pick, without this
            // pass ever needing to redraw any of that occluding geometry
            // itself. `render` then reads back just the one texel under the
            // cursor (via a scissor rect, so the fragment-shading cost is
            // one pixel regardless of canvas resolution) asynchronously —
            // see `hovered_target`/`pick_pending`'s own doc comments for
            // why a frame or two of latency here is an acceptable,
            // deliberate tradeoff for keeping this entirely off the CPU.
            let pick_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pick-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(vertex_layout)],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_pick"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: PICK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let depth_view = create_depth_view(&device, width, height);
            let (pick_texture, pick_view) = create_pick_texture(&device, width, height);
            let pick_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pick-readback"),
                // One texel's worth, padded to `COPY_BYTES_PER_ROW_ALIGNMENT`
                // (256) — wgpu requires buffer-side bytes-per-row to be a
                // multiple of that regardless of how little real data (4
                // bytes) is actually being copied.
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let meshes = MeshRegistry::load_all(&device);
            let scene_layout = SceneLayout::default();
            let mut growth_config = GrowthConfig::default();
            // Grounds the moon's starting phase in the real date this
            // session actually started on (see `Simulation::create`'s own
            // doc comment) rather than `MoonConfig::default`'s hardcoded
            // fallback date.
            growth_config.moon.initial_phase = moon::phase_for_date(seed_year, seed_month, seed_day);
            let layout = &scene_layout;
            let aspect = width as f32 / height as f32;

            // Only sun/moon still need a local closure here — genuinely
            // room-global, not part of `PlantSlot` (see `build_plant_slot`
            // for everything that is).
            let make_sky_drawable = |mesh: &'static str, transform: &Transform| -> Drawable {
                let uniform = InstanceUniform::from_transform(transform, aspect, layout.zoom, 0.5);
                let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(mesh),
                    contents: bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(mesh),
                    layout: &instance_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    }],
                });
                Drawable { mesh, uniform_buffer, bind_group }
            };

            let initial_sun = sun::sun_state(0.0, &growth_config.sun);
            let sun_drawable =
                make_sky_drawable("sun", &scene::sky_object_transform(&initial_sun, layout, aspect));
            let moon_drawable =
                make_sky_drawable("moon", &scene::sky_object_transform(&initial_sun, layout, aspect));
            let moon_shadow_drawable =
                make_sky_drawable("moon", &scene::sky_object_transform(&initial_sun, layout, aspect));
            let lamp_drawable = make_sky_drawable("wall_lamp", &scene::lamp_transform(layout));
            let table_drawable = make_sky_drawable(
                "table",
                &Transform { offset: [0.0, 0.0], scale_x: 0.0, scale_y: 0.0, rotation: 0.0, tint: [1.0, 1.0, 1.0] },
            );

            let room_background_specs = scene::room_background(layout);
            let room_background_drawables =
                room_background_specs.iter().map(|spec| make_sky_drawable(spec.mesh, &spec.transform)).collect();

            Ok(GpuState {
                surface,
                device,
                queue,
                config,
                pipeline,
                instance_bind_group_layout,
                depth_view,
                meshes,
                scene_light_buffer,
                scene_light_bind_group,
                sun_drawable,
                moon_drawable,
                moon_shadow_drawable,
                lamp_drawable,
                table_drawable,
                room_background_specs,
                room_background_drawables,
                plants: Vec::new(),
                selected_plant_index: 0,
                pick_pipeline,
                pick_texture,
                pick_view,
                pick_readback_buffer,
                hovered_target: Rc::new(Cell::new(None)),
                active_tool: None,
                pick_pending: Rc::new(Cell::new(false)),
                pointer_pixel: None,
                camera_pan: [0.0, 0.0],
                humidity: Humidity::new(&growth_config.humidity),
                device_pixel_ratio: dpr,
                growth_config,
                scene_layout,
                day_progress: 0.0,
                session_time: 0.0,
                last_timestamp: None,
                inventory: vec![InventoryItem { species_name: "dracaena".to_string(), from_seed: true }],
            })
        }

        /// Starts a brand-new seed/soil/humidity under whatever species is
        /// currently set in the selected plant's own `plant_config` —
        /// shared by `set_species` (which first swaps that config) and
        /// `restart` (which reuses whatever's already there). Deliberately
        /// leaves `pot_position` alone — where the player physically placed
        /// the pot isn't something a fresh plant/soil/species change should
        /// reset. `auto_water_enabled` *does* reset to off, same "inert
        /// until re-engaged" rule every other per-plant toggle gets on a
        /// genuinely new plant. `realistic_scale` is the caller's call
        /// (`set_species` re-asks, `restart` reuses the outgoing plant's
        /// own choice — see their own doc comments).
        fn reset_plant(&mut self, realistic_scale: bool) {
            let slot = &mut self.plants[self.selected_plant_index];
            slot.plant = Plant::new().with_realistic_scale(realistic_scale);
            slot.soil = Soil::new(&self.growth_config.soil);
            slot.auto_water_enabled = false;
            // Room humidity, not the selected plant's own — resetting one
            // plant among several shouldn't reset the whole room's air.
            if self.plants.len() == 1 {
                self.humidity = Humidity::new(&self.growth_config.humidity);
            }
        }

        fn resize(&mut self, width: u32, height: u32, dpr: f32) {
            if width == 0 || height == 0 {
                return;
            }
            self.config.width = width;
            self.config.height = height;
            self.device_pixel_ratio = dpr;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth_view(&self.device, width, height);
            let (pick_texture, pick_view) = create_pick_texture(&self.device, width, height);
            self.pick_texture = pick_texture;
            self.pick_view = pick_view;
            // Every drawable's uniform is rewritten unconditionally in the next
            // render() (see there) using the fresh aspect ratio, so nothing
            // else needs updating here.
        }

        fn render(&mut self, timestamp_ms: f64) {
            let dt_real = match self.last_timestamp {
                // Clamped so a backgrounded/throttled tab resuming doesn't feed
                // the sim one giant catch-up step.
                Some(prev) => ((timestamp_ms - prev) / 1000.0).clamp(0.0, 0.25),
                None => 0.0,
            };
            self.last_timestamp = Some(timestamp_ms);

            let time = &self.growth_config.time;
            let sim_dt = dt_real * time.sim_seconds_per_real_second;
            self.day_progress =
                (self.day_progress + sim_dt / time.day_length_sim_seconds).rem_euclid(1.0);
            self.session_time += sim_dt;
            let sun_state = sun::sun_state(self.day_progress, &self.growth_config.sun);
            let climate_state = climate::climate_state(self.day_progress, &self.growth_config.climate);
            // The moon runs on its own, much longer cycle (see `sim::moon`)
            // independent of the daily sun — `lit_sun_state` is what
            // everything *light-related* reads from here on. `moon_elevation`
            // (phase-linked rise/set, see `moon::arc_position`) gates both
            // the moonlight contribution and whether it's drawn at all below
            // — its on-screen *position* is derived from the sun's own
            // instead (see `scene::moon_position_opposite_sun`), so the
            // azimuth this arc computes isn't needed here.
            let moon_phase = moon::current_phase(self.session_time, &self.growth_config.moon);
            let moon_appearance = moon::appearance(moon_phase);
            let (_, moon_elevation) = moon::arc_position(self.day_progress, moon_phase, &self.growth_config.sun);
            let lit_sun_state = moon::apply_moonlight(sun_state, moon_appearance, moon_elevation, &self.growth_config.moon);
            // The window's own light/room temperature stay as computed above
            // (rendering the sky/window pane itself) — `plant_sun`/`plant_
            // climate` are what a plant *actually experiences*, adjusted
            // for how far its own pot sits from the window (see `sim::
            // room`), once the player has actually chosen a position for it
            // (see `pot_position_active`'s doc comment).
            //
            // Room humidity is one shared value every plant's own `step`
            // reads from (see `PlantSlot::plant`/`GpuState::humidity`'s own
            // doc comments) — driven by the room's raw `climate_state`, not
            // any one plant's pot-adjusted `plant_climate`, the same "room-
            // level rendering reads room-level state" rule `SceneLightUniform`
            // below already follows for light (misting affects the whole
            // room's air, not wherever one specific pot happens to sit).
            self.humidity.update(
                time.physiology_dt(sim_dt),
                climate_state.temperature_c,
                &self.growth_config.humidity,
            );
            // Every plant in the room keeps growing every frame, not just
            // whichever one is currently selected/rendered below — "several
            // simultaneously alive plants" is the whole point of holding
            // more than one `PlantSlot`, and a plant a player isn't
            // currently looking at shouldn't just freeze.
            for plant_slot in self.plants.iter_mut() {
                let (plant_sun, plant_climate) = if plant_slot.pot_position_active {
                    room::apply_pot_position(
                        lit_sun_state,
                        climate_state,
                        plant_slot.pot_position,
                        &self.growth_config.room,
                    )
                } else {
                    (lit_sun_state, climate_state)
                };
                let plant_growth_config =
                    GrowthConfig { plant: plant_slot.plant_config, ..self.growth_config };
                plant_slot.plant.step(
                    sim_dt,
                    &plant_sun,
                    &plant_climate,
                    &mut plant_slot.soil,
                    self.humidity.level,
                    &plant_growth_config,
                );
                plant_slot.soil.apply_auto_water(plant_slot.auto_water_enabled, &self.growth_config.soil);
            }

            // Every plant sharing the room renders simultaneously now (see
            // `PlantSlot`'s own doc comment and `scene::plant_slot_base_
            // anchor`) — `self.selected_plant_index` still decides which
            // one the HUD/action surface targets (`Stats`, `prune_main_
            // stem`, etc.), but every plant in `self.plants` gets its own
            // full write+draw pass below, not just the selected one.
            let aspect = self.config.width as f32 / self.config.height as f32;
            // CSS pixels, not the (devicePixelRatio-scaled) backing buffer
            // width — see `device_pixel_ratio`'s own doc comment for why
            // `scene::outline_uniform` needs this rather than the raw
            // `config.width`.
            let canvas_width_px = self.config.width as f32 / self.device_pixel_ratio;
            // See `scene::dynamic_zoom_for_room` — one shared zoom for the
            // whole room (pulling back further than `self.scene_layout.
            // zoom` once the tallest plant or the widest spread of pots
            // would otherwise run off frame), recomputed every frame since
            // it depends on every plant's current height/branches/count.
            let zoom = scene::dynamic_zoom_for_room(self.plants.iter().map(|p| &p.plant), &self.scene_layout);
            // `camera_pan` (see its own field doc) is in already-final NDC
            // units; dividing by `zoom` here cancels the `*zoom` every
            // instance's own offset still gets in `InstanceUniform::from_
            // transform`, so a dragged pan always reads as the same on-
            // screen distance regardless of how zoomed out the room is.
            let pan = [self.camera_pan[0] / zoom, self.camera_pan[1] / zoom];

            // The cursor's own position in the same clip-space convention
            // `world_pos` uses in the shader — reuses `pointer_pixel` (see
            // its own doc comment), already tracked off real `pointermove`
            // events for the prune tool's GPU pick pass, rather than
            // introducing a second, separate way to watch the cursor. `y`
            // flips (pixel space is top-down, clip space is bottom-up/
            // y-up) but `x` doesn't need the aspect correction every other
            // offset gets — NDC is already screen-space on both axes, no
            // per-mesh local-to-world conversion involved here.
            let cursor_ndc = self.pointer_pixel.map(|(css_x, css_y)| {
                let px = css_x * self.device_pixel_ratio;
                let py = css_y * self.device_pixel_ratio;
                let ndc_x = (px / self.config.width as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (py / self.config.height as f32) * 2.0;
                [ndc_x, ndc_y]
            });
            // Uses the room's own `lit_sun_state` (not `plant_sun`) — the
            // window itself doesn't dim just because a pot happens to sit
            // far from it; that's a separate, plant-specific effect already
            // handled inside `Plant::step`. Moonlight is folded in here too
            // (see `lit_sun_state`), so a full-moon night actually reads as
            // a bit brighter than a new-moon one, not just numerically.
            // Uses the room's own original, unshifted `scene_layout` — the
            // window (what this light represents) sits in one fixed place
            // regardless of any one plant's own pot position. `pan` is
            // still applied on top (see its own doc comment) — dragging the
            // room's view drags where its light comes from too.
            self.queue.write_buffer(
                &self.scene_light_buffer,
                0,
                bytes_of(&scene::SceneLightUniform::new(&lit_sun_state, &self.scene_layout, zoom, self.camera_pan, cursor_ndc)),
            );

            // The room's own light — the most "in context" way to show the
            // sun's current intensity/color is tinting what it's actually
            // lighting, not a HUD gauge. Applied to wall/window only, not any
            // plant itself, so a plant's own state (droop, fold, lean — each
            // already a visible signal in its own right) stays legible even at
            // night.
            let ambient = scene::ambient_tint(&lit_sun_state, &self.scene_layout);
            let base_outline_tint = scene::outline_tint_for_sun(sun_state.intensity, &self.scene_layout);
            // The slow year-long cycle, layered on the wall itself — a
            // genuinely "wall-integrated" way to show the current season
            // (distinct from `ambient`'s day/night tint, which stays on the
            // window only) — see `scene::seasonal_wall_tint`.
            let season_state = season::season_state(self.session_time, &self.growth_config.season);
            let seasonal_tint = scene::seasonal_wall_tint(season_state.phase, &self.scene_layout);
            // The wall/window — drawn exactly once regardless of how many
            // plants share the room (see `room_background_drawables`'s own
            // doc comment), unlike every plant's own pot/soil below.
            for (spec, drawable) in self.room_background_specs.iter().zip(&self.room_background_drawables) {
                let mut transform = spec.transform;
                if drawable.mesh == "wall" {
                    transform.tint = seasonal_tint;
                }
                let (zoom, drawable_pan) =
                    if drawable.mesh == "wall" { (1.0, self.camera_pan) } else { (zoom, pan) };
                transform.offset[0] += drawable_pan[0];
                transform.offset[1] += drawable_pan[1];
                if drawable.mesh == "window_pane" {
                    write_transmissive_transform(&self.queue, drawable, &transform, aspect, zoom, self.scene_layout.background_depth, &self.scene_layout);
                } else {
                    write_transform(&self.queue, drawable, &transform, aspect, zoom, self.scene_layout.background_depth, &self.scene_layout);
                }
            }
            // The sun's position inside the window is the other half of
            // showing light "in context" — its angle, not just its color.
            let mut sky_transform = scene::sky_object_transform(&sun_state, &self.scene_layout, aspect);
            sky_transform.offset[0] += pan[0];
            sky_transform.offset[1] += pan[1];
            write_transform(&self.queue, &self.sun_drawable, &sky_transform, aspect, zoom, self.scene_layout.background_depth, &self.scene_layout);
            // Shown diagonally opposite wherever the sun currently sits
            // (see `scene::moon_position_opposite_sun`) rather than tracing
            // its own independent arc, which could otherwise visually
            // coincide with the sun now that both can be up at once (see
            // `moon::arc_position`'s phase-linked timing). `moon_elevation`
            // (from that phase-linked arc, computed above) still gates
            // whether it's shown/contributes light at all — just not where
            // it's drawn.
            let moon_position = scene::moon_position_opposite_sun(&sun_state);
            let mut moon_sky_transform = scene::sky_object_transform(&moon_position, &self.scene_layout, aspect);
            moon_sky_transform.offset[0] += pan[0];
            moon_sky_transform.offset[1] += pan[1];
            // A real daytime moon isn't darker, it just loses contrast
            // against a much brighter sky — faded toward the same ambient
            // sky tint the window itself uses, not tinted independently.
            moon_sky_transform.tint =
                scene::daytime_fade(moon_sky_transform.tint, ambient, lit_sun_state.intensity, self.scene_layout.moon_daytime_fade_strength);
            write_transform(&self.queue, &self.moon_drawable, &moon_sky_transform, aspect, zoom, self.scene_layout.background_depth, &self.scene_layout);
            let crescent_tilt = moon::crescent_tilt_angle(
                (moon_sky_transform.offset[0] as f64, moon_sky_transform.offset[1] as f64),
                (sky_transform.offset[0] as f64, sky_transform.offset[1] as f64),
                self.growth_config.moon.observer_latitude_degrees,
            ) as f32;
            let mut shadow_transform =
                scene::moon_shadow_transform(&moon_sky_transform, &moon_appearance, crescent_tilt, &self.scene_layout, aspect);
            shadow_transform.tint =
                scene::daytime_fade(shadow_transform.tint, ambient, lit_sun_state.intensity, self.scene_layout.moon_daytime_fade_strength);
            write_transform(
                &self.queue,
                &self.moon_shadow_drawable,
                &shadow_transform,
                aspect,
                zoom,
                self.scene_layout.background_depth,
                &self.scene_layout,
            );

            // A wall-mounted lamp, always present but only visibly "on"
            // (and only actually casting light — see the `SceneLightUniform`
            // write above) at night.
            let mut lamp_transform = scene::lamp_transform(&self.scene_layout);
            lamp_transform.offset[0] += pan[0];
            lamp_transform.offset[1] += pan[1];
            let lamp_on = scene::lamp_on_fraction(lit_sun_state.intensity);
            lamp_transform.tint = [
                scene::lerp(self.scene_layout.lamp_off_tint[0], self.scene_layout.lamp_on_tint[0], lamp_on),
                scene::lerp(self.scene_layout.lamp_off_tint[1], self.scene_layout.lamp_on_tint[1], lamp_on),
                scene::lerp(self.scene_layout.lamp_off_tint[2], self.scene_layout.lamp_on_tint[2], lamp_on),
            ];
            write_transform(&self.queue, &self.lamp_drawable, &lamp_transform, aspect, zoom, self.scene_layout.background_depth, &self.scene_layout);

            let pot_half_height = self.meshes.local_half_extent("pot").1 * self.scene_layout.pot_scale;
            let table_top_y = self.scene_layout.pot_anchor[1] - pot_half_height;
            let table_transform = Transform {
                offset: [pan[0], table_top_y + pan[1]],
                scale_x: self.scene_layout.table_scale,
                scale_y: self.scene_layout.table_scale,
                rotation: 0.0,
                tint: [1.0, 1.0, 1.0],
            };
            write_transform(&self.queue, &self.table_drawable, &table_transform, aspect, zoom, self.scene_layout.background_depth, &self.scene_layout);

            for (plant_index, slot) in self.plants.iter_mut().enumerate() {
                // Where this plant's own pot actually sits this frame —
                // `plant_pot_world_anchor` combines this slot's sideways step
                // along the sill with this specific plant's own `Simulation::
                // set_pot_position` slider (see its own doc comment).
                // `position == 0.5` reproduces the slot's own base anchor
                // exactly, so this is a pure addition, not a retuning. Also
                // the single source of truth `Simulation::plant_pot_hud`
                // projects into CSS-pixel space for the per-pot water gauge.
                let unpanned_pot_anchor =
                    scene::plant_pot_world_anchor(&self.scene_layout, plant_index, slot.pot_position);
                let effective_pot_anchor = [unpanned_pot_anchor[0] + pan[0], unpanned_pot_anchor[1] + pan[1]];
                let effective_layout = SceneLayout { pot_anchor: effective_pot_anchor, ..self.scene_layout };
                let layout = &effective_layout;

                // How wet the soil itself looks right now — see `scene::
                // soil_moisture_tint`'s doc comment: this is a *leading*
                // overwatering indicator (responds to raw moisture
                // immediately), distinct from `Plant::root_health` (a
                // *lagging* one, since root damage only starts after
                // `SoilConfig::waterlog_grace_period` of sustained
                // waterlogging).
                let soil_tint = scene::soil_moisture_tint(
                    slot.soil.moisture,
                    self.growth_config.soil.waterlogged_threshold,
                    layout,
                );
                for (spec, drawable) in slot.background_specs.iter().zip(&slot.background_drawables) {
                    let mut transform = spec.transform;
                    transform.offset = effective_pot_anchor;
                    if drawable.mesh == "soil" {
                        transform.tint = soil_tint;
                    }
                    write_transform(&self.queue, drawable, &transform, aspect, zoom, layout.background_depth, layout);
                }

                // A climbing habit's support pole — present from the very
                // start (a real gardener plants the stake at the same time
                // as the seed, not once it's already climbing), so this is
                // written outside the seed/vegetative stage split below.
                let trellis_transform = scene::trellis_transform(slot.plant_config.trellis_height, layout);
                slot.trellis_active = trellis_transform.is_some();
                if let Some(transform) = &trellis_transform {
                    write_transform(&self.queue, &slot.trellis_drawable, transform, aspect, zoom, layout.trellis_depth, layout);
                }

                let seed_swell = match slot.plant.last_decision {
                    Some(Decision::Seed { water_factor, threshold, .. }) if threshold > 0.0 => water_factor / threshold,
                    _ => 0.0,
                };
                let seed_transform = scene::seed_transform(layout, seed_swell);
                write_transform(&self.queue, &slot.seed_drawable, &seed_transform, aspect, zoom, layout.plant_depth, layout);
                write_outline_transform(
                    &self.queue,
                    &self.meshes,
                    &slot.seed_outline_drawable,
                    &seed_transform,
                    aspect,
                    zoom,
                    canvas_width_px,
                    layout.plant_depth,
                    base_outline_tint,
                    layout,
                );
                let cotyledon_transforms =
                    [scene::cotyledon_transform(layout, Side::Left), scene::cotyledon_transform(layout, Side::Right)];
                for i in 0..2 {
                    write_transform(&self.queue, &slot.cotyledon_drawables[i], &cotyledon_transforms[i], aspect, zoom, layout.plant_depth, layout);
                    write_outline_transform(
                        &self.queue,
                        &self.meshes,
                        &slot.cotyledon_outline_drawables[i],
                        &cotyledon_transforms[i],
                        aspect,
                        zoom,
                        canvas_width_px,
                        layout.plant_depth,
                        base_outline_tint,
                        layout,
                    );
                }

                // The root mass, visible through the pot's hollow outline —
                // tinted by `root_health` specifically (not the stem's
                // broader `vitality` signal below), so this is the one
                // place a player can tell "the roots themselves are rotted"
                // apart from "the plant is generally unwell" — see `roots_
                // drawable`'s field doc.
                let roots_transform = Transform {
                    offset: effective_pot_anchor,
                    scale_x: layout.roots_scale,
                    scale_y: layout.roots_scale,
                    rotation: 0.0,
                    tint: scene::stem_health_tint(slot.plant.root_health, layout),
                };
                write_transform(&self.queue, &slot.roots_drawable, &roots_transform, aspect, zoom, layout.plant_depth, layout);
                write_outline_transform(
                    &self.queue,
                    &self.meshes,
                    &slot.roots_outline_drawable,
                    &roots_transform,
                    aspect,
                    zoom,
                    canvas_width_px,
                    layout.plant_depth,
                    base_outline_tint,
                    layout,
                );

                // `sim::plant::Plant::lean_angle` is an unsigned magnitude
                // (sim has no notion of where the window is rendered) —
                // this derives which screen direction "toward the window"
                // actually means from the window's real position, so
                // phototropism keeps pointing at it correctly if that
                // position is ever retuned. See `StemCurve::lean_sign`'s
                // doc comment for the rotation convention this is
                // correcting for.
                let lean_sign = scene::lean_sign_toward_window(layout.window_offset, layout.pot_anchor);

                // The main stem's own curve — see `StemCurve`'s doc comment
                // on why a stem isn't one rigid rotation: older, already-
                // stiffened segments stay frozen at whatever lean they had
                // when they formed (`stem_segment_history`), only the
                // still-growing tip uses today's live lean/droop.
                let vine_base_lean_angle = if slot.plant_config.growth_habit == GrowthHabit::Vine {
                    if layout.trellis_x_offset >= 0.0 { layout.vine_trellis_lean_angle } else { -layout.vine_trellis_lean_angle }
                } else {
                    0.0
                };
                let main_curve = StemCurve {
                    base: scene::stem_base_frame(layout),
                    segment_history: &slot.plant.stem_segment_history,
                    current_lean_angle: slot.plant.lean_angle,
                    current_extra_angle: slot.plant.stem_droop,
                    segment_height_interval: slot.plant_config.stem_segment_height_interval,
                    lean_sign,
                    vine_base_lean_angle,
                };

                // Overall plant vitality right now — one whole-plant value
                // shared by the main stem and every branch (there's one
                // shared root system, not one per branch). Deliberately
                // reads `Decision::Vegetative::effective_water_factor`,
                // *not* `Plant::root_health` alone: root rot is only one of
                // several ways a plant can be failing (plain drought/
                // neglect is the far more common one under default settings
                // with no player input), and `effective_water_factor`
                // already folds root health, raw soil moisture, *and* pot-
                // bound stress into one number, so this tints the stem
                // whenever *any* of those is dragging the plant down — not
                // just the overwatering-specific case. Falls back to fully
                // healthy (`1.0`) outside `Stage::Vegetative` (a sprouting
                // seedling on stored reserves isn't meaningfully "thirsty"
                // yet in this sense). Once `Stage::Dead` freezes `last_
                // decision`, this stays pinned at whatever it was the
                // instant it died — for `DeathCause::RootRot` that's
                // reliably at or near zero (root health, which multiplies
                // directly into this, just hit zero), but a `DeathCause::
                // Starvation` death can in principle happen with plenty of
                // water still available (every leaf lost to cold or pests
                // despite a fully watered pot) — the frozen stem tint alone
                // won't distinguish that case, which is exactly why `Stats::
                // death_cause` exists as its own explicit signal rather
                // than something a player has to infer purely from the
                // stem's final color.
                let vitality = match slot.plant.last_decision {
                    Some(Decision::Vegetative { effective_water_factor, .. }) => effective_water_factor,
                    _ => 1.0,
                };
                let stem_tint = scene::stem_health_tint(vitality, layout);

                // Stem segments fill the shared pool main-stem-first, then
                // branch by branch — see the `stem_segment_drawables` field
                // doc (same pattern as `leaf_drawables` below). `stem_
                // segment_targets` is rebuilt fresh alongside — see its own
                // field doc.
                slot.stem_segment_targets.clear();
                let mut segment_slot = 0;
                // A basal rosette has no visible above-ground stem — the
                // crown the leaves fan from sits under the soil the pot
                // already renders.
                let main_segments = if slot.plant_config.growth_habit == GrowthHabit::BasalRosette {
                    Vec::new()
                } else {
                    scene::stem_segment_transforms(&main_curve, slot.plant.height, slot.plant.stem_radius, stem_tint, layout)
                };
                for (local_index, transform) in main_segments.iter().enumerate() {
                    if segment_slot >= MAX_STEM_SEGMENT_DRAWABLES {
                        break;
                    }
                    let base_height = local_index as f64 * slot.plant_config.stem_segment_height_interval;
                    let hovered = self.hovered_target.get() == Some(scene::PickTarget::StemSegment { plant_index, slot: segment_slot });
                    let display_transform =
                        if hovered { scene::apply_hover_scale(transform, layout) } else { *transform };
                    let outline_tint = if hovered { layout.hover_outline_tint } else { base_outline_tint };
                    write_transform(&self.queue, &slot.stem_segment_drawables[segment_slot], &display_transform, aspect, zoom, layout.plant_depth, layout);
                    write_outline_transform(
                        &self.queue,
                        &self.meshes,
                        &slot.stem_segment_outline_drawables[segment_slot],
                        &display_transform,
                        aspect,
                        zoom,
                        canvas_width_px,
                        layout.plant_depth,
                        outline_tint,
                        layout,
                    );
                    write_pick_transform(
                        &self.queue,
                        &self.meshes,
                        &slot.stem_segment_pick_drawables[segment_slot],
                        &display_transform,
                        aspect,
                        zoom,
                        canvas_width_px,
                        layout.plant_depth,
                        layout,
                        scene::PickTarget::StemSegment { plant_index, slot: segment_slot },
                    );
                    slot.stem_segment_targets.push((None, base_height));
                    segment_slot += 1;
                }

                // Main stem only (see `aerial_root_drawables`'s field doc) —
                // fills from the front of the pool; only the first N
                // (however many actually exist this frame) are ever drawn.
                let mut aerial_root_slot = 0;
                for root in &slot.plant.aerial_roots {
                    if aerial_root_slot >= MAX_AERIAL_ROOTS {
                        break;
                    }
                    let transform = scene::aerial_root_transform(root, &main_curve, layout);
                    write_transform(&self.queue, &slot.aerial_root_drawables[aerial_root_slot], &transform, aspect, zoom, layout.plant_depth, layout);
                    write_outline_transform(
                        &self.queue,
                        &self.meshes,
                        &slot.aerial_root_outline_drawables[aerial_root_slot],
                        &transform,
                        aspect,
                        zoom,
                        canvas_width_px,
                        layout.plant_depth,
                        base_outline_tint,
                        layout,
                    );
                    aerial_root_slot += 1;
                }
                slot.aerial_roots_drawn = aerial_root_slot;

                // Repointed every frame to whichever mesh this species' own
                // bloom actually looks like (see `PlantConfig::flower_mesh_
                // name`) — cheap, since `Drawable::mesh` is just a lookup
                // key, not an owned GPU resource, so switching species at
                // runtime (`set_species`) never needs a new buffer. Always
                // drawn (see the render pass below) — `flower_transform`
                // itself scales it to exactly zero size pre-maturity, and to
                // its own `SceneLayout::bud_min_intensity` floor (not zero)
                // once mature but currently resting between bloom flushes.
                slot.flower_drawable.mesh = slot.plant_config.flower_mesh_name;
                slot.flower_outline_drawable.mesh = slot.plant_config.flower_mesh_name;
                let mature_enough_to_have_a_bud = slot.plant.height >= slot.plant_config.flowering_height_threshold;
                let flower_transform = scene::flower_transform(
                    &main_curve,
                    slot.plant.height,
                    slot.plant.bloom_intensity,
                    mature_enough_to_have_a_bud,
                    layout,
                );
                write_transform(&self.queue, &slot.flower_drawable, &flower_transform, aspect, zoom, layout.plant_depth, layout);
                write_outline_transform(
                    &self.queue,
                    &self.meshes,
                    &slot.flower_outline_drawable,
                    &flower_transform,
                    aspect,
                    zoom,
                    canvas_width_px,
                    layout.plant_depth,
                    base_outline_tint,
                    layout,
                );

                // Leaves fill the shared pool main-stem-first, then branch
                // by branch — see the `leaf_drawables` field doc. `shade_
                // factors` darkens each leaf by how much of this same
                // grower's own canopy sits above it (see `scene::leaf_
                // transform_in_frame`).
                let main_shade_factors = self_shading_factors(&slot.plant.leaves, &slot.plant_config);
                let mut leaf_slot = 0;
                for (leaf, &shade_factor) in slot.plant.leaves.iter().zip(&main_shade_factors) {
                    if leaf_slot >= MAX_LEAVES {
                        break;
                    }
                    let transform = if slot.plant_config.growth_habit == GrowthHabit::BasalRosette {
                        scene::rosette_leaf_transform(main_curve.base, leaf, shade_factor, layout)
                    } else {
                        scene::leaf_transform_in_frame(&main_curve, leaf, shade_factor, layout)
                    };
                    let depth = scene::leaf_depth(leaf, layout);
                    let hovered = self.hovered_target.get() == Some(scene::PickTarget::Leaf { plant_index, slot: leaf_slot });
                    let display_transform =
                        if hovered { scene::apply_hover_scale(&transform, layout) } else { transform };
                    let outline_tint = if hovered { layout.hover_outline_tint } else { base_outline_tint };
                    slot.leaf_drawables[leaf_slot].mesh = slot.plant_config.leaf_mesh_name;
                    slot.leaf_outline_drawables[leaf_slot].mesh = slot.plant_config.leaf_mesh_name;
                    slot.leaf_pick_drawables[leaf_slot].mesh = slot.plant_config.leaf_mesh_name;
                    write_leaf_transform(
                        &self.queue,
                        &self.meshes,
                        &slot.leaf_drawables[leaf_slot],
                        &display_transform,
                        aspect,
                        zoom,
                        depth,
                        layout,
                    );
                    write_outline_transform(
                        &self.queue,
                        &self.meshes,
                        &slot.leaf_outline_drawables[leaf_slot],
                        &display_transform,
                        aspect,
                        zoom,
                        canvas_width_px,
                        depth,
                        outline_tint,
                        layout,
                    );
                    write_pick_transform(
                        &self.queue,
                        &self.meshes,
                        &slot.leaf_pick_drawables[leaf_slot],
                        &display_transform,
                        aspect,
                        zoom,
                        canvas_width_px,
                        depth,
                        layout,
                        scene::PickTarget::Leaf { plant_index, slot: leaf_slot },
                    );
                    leaf_slot += 1;
                }

                let visible_branches = slot.plant.branches.len().min(MAX_BRANCHES);
                for i in 0..visible_branches {
                    let branch = &slot.plant.branches[i];
                    let bcurve = scene::branch_curve(&main_curve, branch, layout);
                    let branch_segments = scene::stem_segment_transforms(
                        &bcurve,
                        branch.height,
                        branch.stem_radius,
                        stem_tint,
                        layout,
                    );
                    for (local_index, transform) in branch_segments.iter().enumerate() {
                        if segment_slot >= MAX_STEM_SEGMENT_DRAWABLES {
                            break;
                        }
                        let base_height = local_index as f64 * slot.plant_config.stem_segment_height_interval;
                        let hovered = self.hovered_target.get() == Some(scene::PickTarget::StemSegment { plant_index, slot: segment_slot });
                        let display_transform =
                            if hovered { scene::apply_hover_scale(transform, layout) } else { *transform };
                        let outline_tint = if hovered { layout.hover_outline_tint } else { base_outline_tint };
                        write_transform(&self.queue, &slot.stem_segment_drawables[segment_slot], &display_transform, aspect, zoom, layout.plant_depth, layout);
                        write_outline_transform(
                            &self.queue,
                            &self.meshes,
                            &slot.stem_segment_outline_drawables[segment_slot],
                            &display_transform,
                            aspect,
                            zoom,
                            canvas_width_px,
                            layout.plant_depth,
                            outline_tint,
                            layout,
                        );
                        write_pick_transform(
                            &self.queue,
                            &self.meshes,
                            &slot.stem_segment_pick_drawables[segment_slot],
                            &display_transform,
                            aspect,
                            zoom,
                            canvas_width_px,
                            layout.plant_depth,
                            layout,
                            scene::PickTarget::StemSegment { plant_index, slot: segment_slot },
                        );
                        slot.stem_segment_targets.push((Some(i), base_height));
                        segment_slot += 1;
                    }
                    let branch_shade_factors = self_shading_factors(&branch.leaves, &slot.plant_config);
                    for (leaf, &shade_factor) in branch.leaves.iter().zip(&branch_shade_factors) {
                        if leaf_slot >= MAX_LEAVES {
                            break;
                        }
                        let leaf_transform = scene::leaf_transform_in_frame(&bcurve, leaf, shade_factor, layout);
                        let depth = scene::leaf_depth(leaf, layout);
                        let hovered = self.hovered_target.get() == Some(scene::PickTarget::Leaf { plant_index, slot: leaf_slot });
                        let display_transform =
                            if hovered { scene::apply_hover_scale(&leaf_transform, layout) } else { leaf_transform };
                        let outline_tint = if hovered { layout.hover_outline_tint } else { base_outline_tint };
                        slot.leaf_drawables[leaf_slot].mesh = slot.plant_config.leaf_mesh_name;
                        slot.leaf_outline_drawables[leaf_slot].mesh = slot.plant_config.leaf_mesh_name;
                        slot.leaf_pick_drawables[leaf_slot].mesh = slot.plant_config.leaf_mesh_name;
                        write_leaf_transform(
                            &self.queue,
                            &self.meshes,
                            &slot.leaf_drawables[leaf_slot],
                            &display_transform,
                            aspect,
                            zoom,
                            depth,
                            layout,
                        );
                        write_outline_transform(
                            &self.queue,
                            &self.meshes,
                            &slot.leaf_outline_drawables[leaf_slot],
                            &display_transform,
                            aspect,
                            zoom,
                            canvas_width_px,
                            depth,
                            outline_tint,
                            layout,
                        );
                        write_pick_transform(
                            &self.queue,
                            &self.meshes,
                            &slot.leaf_pick_drawables[leaf_slot],
                            &display_transform,
                            aspect,
                            zoom,
                            canvas_width_px,
                            depth,
                            layout,
                            scene::PickTarget::Leaf { plant_index, slot: leaf_slot },
                        );
                        leaf_slot += 1;
                    }
                }
                slot.leaves_drawn = leaf_slot;
                slot.segments_drawn = segment_slot;
            }

            let frame = match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => t,
                wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                    // Still presentable this frame; reconfigure so the *next*
                    // frame is optimal again.
                    self.surface.configure(&self.device, &self.config);
                    t
                }
                // Reconfiguring is the documented recovery for both — skip this
                // frame's draw, the next one will use the refreshed config.
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(&self.device, &self.config);
                    return;
                }
                wgpu::CurrentSurfaceTexture::Timeout
                | wgpu::CurrentSurfaceTexture::Occluded
                | wgpu::CurrentSurfaceTexture::Validation => return,
            };

            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // The `wall` mesh covers the whole canvas, so this
                            // is just a fallback for any gap, not the real
                            // background.
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.03,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(1, &self.scene_light_bind_group, &[]);

                let mut draw = |d: &Drawable| {
                    let mesh = self.meshes.get(d.mesh);
                    pass.set_bind_group(0, &d.bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                };

                // The wall/window — once for the whole room (see `room_
                // background_drawables`'s own doc comment), not per plant.
                for d in &self.room_background_drawables {
                    draw(d);
                }
                draw(&self.lamp_drawable);
                draw(&self.table_drawable);
                // Each drawn independently by its own elevation (phase-
                // linked, see `moon::arc_position`) — both up (daytime
                // moon) or neither (moonless night near new moon) are both
                // possible now, not just "sun by day, moon by night."
                if sun_state.elevation > 0.0 {
                    draw(&self.sun_drawable);
                }
                if moon_elevation > 0.0 {
                    draw(&self.moon_drawable);
                    draw(&self.moon_shadow_drawable);
                }
                // Every plant in the room draws its own full pot/stem/leaf
                // stack in turn — see `PlantSlot`'s own doc comment. Side by
                // side (see `scene::plant_slot_base_anchor`), so drawing
                // them in any consistent order never causes one to occlude
                // another incorrectly; the shared depth buffer only matters
                // *within* one plant's own layered pieces.
                for slot in &self.plants {
                    for d in &slot.background_drawables {
                        draw(d);
                    }
                    // Behind the stem/leaves (drawn next) but in front of
                    // the background — a real support pole sits among the
                    // foliage, not floating in front of or fully hidden
                    // behind it.
                    if slot.trellis_active {
                        draw(&slot.trellis_drawable);
                    }
                    // Each plant-asset mesh below draws its own outline
                    // `Drawable` (see that field's own doc comment)
                    // immediately before itself, so the normal-tinted
                    // draw's opaque fill covers everything but a thin white
                    // rim at the edges.
                    if slot.plant.stage == Stage::Seed {
                        draw(&slot.seed_outline_drawable);
                        draw(&slot.seed_drawable);
                    } else {
                        draw(&slot.cotyledon_outline_drawables[0]);
                        draw(&slot.cotyledon_drawables[0]);
                        draw(&slot.cotyledon_outline_drawables[1]);
                        draw(&slot.cotyledon_drawables[1]);
                        // Drawn right after the pot/soil (see the
                        // background loop above) and before the stem —
                        // real roots sit at the base, they don't float in
                        // front of the foliage.
                        draw(&slot.roots_outline_drawable);
                        draw(&slot.roots_drawable);
                        for i in 0..slot.segments_drawn {
                            draw(&slot.stem_segment_outline_drawables[i]);
                            draw(&slot.stem_segment_drawables[i]);
                        }
                        for i in 0..slot.aerial_roots_drawn {
                            draw(&slot.aerial_root_outline_drawables[i]);
                            draw(&slot.aerial_root_drawables[i]);
                        }
                        // Always drawn — `bloom_intensity` already scales
                        // it to zero size (invisible) whenever not in
                        // bloom.
                        draw(&slot.flower_outline_drawable);
                        draw(&slot.flower_drawable);
                    }
                    for i in 0..slot.leaves_drawn {
                        draw(&slot.leaf_outline_drawables[i]);
                        draw(&slot.leaf_drawables[i]);
                    }
                }
            }

            // GPU hit-testing for the prune tool — see `pick_pipeline`'s own
            // doc comment. Skipped entirely whenever the pointer isn't over
            // the canvas, or a previous readback hasn't resolved yet (see
            // `pick_pending`), so this never queues up more than one
            // in-flight request.
            let mut just_requested_pick = false;
            if let (Some((css_x, css_y)), Some(active_tool)) = (self.pointer_pixel, self.active_tool) {
                if !self.pick_pending.get() {
                    let px = ((css_x * self.device_pixel_ratio).round() as i32)
                        .clamp(0, self.config.width as i32 - 1) as u32;
                    let py = ((css_y * self.device_pixel_ratio).round() as i32)
                        .clamp(0, self.config.height as i32 - 1) as u32;
                    {
                        let mut pick_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("pick-pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &self.pick_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    // Solid black — the reserved "nothing
                                    // was there" color `scene::
                                    // decode_pick_target` treats as `None`.
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            // Prune reuses (`Load`s) the depth buffer the
                            // main pass above just populated this same
                            // frame, so a leaf actually hidden behind the
                            // opaque stem correctly never wins the pick (see
                            // `pick_pipeline`'s doc comment). Trim instead
                            // gets a fresh (`Clear`d) depth buffer: it only
                            // ever draws stem-segment pick geometry (see the
                            // `match self.active_tool` below), so reusing
                            // the main buffer would let a leaf drawn in
                            // front of a stem — real geometry Trim doesn't
                            // even consider a pick candidate — block picking
                            // that stem anyway, even though the player can
                            // still see (and mean to click) the stem right
                            // through/behind it.
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &self.depth_view,
                                depth_ops: Some(wgpu::Operations {
                                    load: match active_tool {
                                        Tool::Prune => wgpu::LoadOp::Load,
                                        Tool::Trim => wgpu::LoadOp::Clear(1.0),
                                    },
                                    store: wgpu::StoreOp::Discard,
                                }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pick_pass.set_pipeline(&self.pick_pipeline);
                        pick_pass.set_bind_group(1, &self.scene_light_bind_group, &[]);
                        // Restricts actual fragment shading to the one texel
                        // under the cursor — the target is canvas-sized
                        // (see `pick_view`'s own doc comment) purely so its
                        // pixels line up 1:1 with the cursor's own position,
                        // not because this pass does canvas-sized work.
                        pick_pass.set_scissor_rect(px, py, 1, 1);
                        let mut draw_pick = |d: &Drawable| {
                            let mesh = self.meshes.get(d.mesh);
                            pick_pass.set_bind_group(0, &d.bind_group, &[]);
                            pick_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            pick_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                            pick_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        };
                        // Every plant's own pick pool, not just the
                        // selected one — a hovered leaf/segment on *any*
                        // pot in the room should resolve (see `PickTarget`'s
                        // own `plant_index`, which is exactly what lets a
                        // readback map back to the right plant). Only the
                        // *active tool's* own geometry is ever drawn into
                        // the pick pass at all — Prune can only ever hit-
                        // test leaves, Trim only stem segments, so a click
                        // always acts on the kind of thing the player
                        // actually selected, never whichever geometry
                        // happens to be nearest the cursor.
                        for slot in &self.plants {
                            match active_tool {
                                Tool::Prune => {
                                    for i in 0..slot.leaves_drawn {
                                        draw_pick(&slot.leaf_pick_drawables[i]);
                                    }
                                }
                                Tool::Trim => {
                                    for i in 0..slot.segments_drawn {
                                        draw_pick(&slot.stem_segment_pick_drawables[i]);
                                    }
                                }
                            }
                        }
                    }
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.pick_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: &self.pick_readback_buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(256),
                                rows_per_image: Some(1),
                            },
                        },
                        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    );
                    self.pick_pending.set(true);
                    just_requested_pick = true;
                }
            } else {
                self.hovered_target.set(None);
            }

            self.queue.submit(Some(encoder.finish()));
            self.queue.present(frame);

            if just_requested_pick {
                let hovered_target = self.hovered_target.clone();
                let pick_pending = self.pick_pending.clone();
                let buffer = self.pick_readback_buffer.clone();
                self.pick_readback_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            let rgba = [view[0], view[1], view[2], view[3]];
                            hovered_target.set(scene::decode_pick_target(rgba));
                        }
                        buffer.unmap();
                    }
                    pick_pending.set(false);
                });
                // No explicit `device.poll()` here — on WebGPU it's
                // documented to have no effect at all (callbacks run off
                // the browser's own event loop), and calling it anyway,
                // synchronously, once per pick request right after
                // `queue.present`, was the actual cause of an intermittent
                // black-frame flicker: something about polling the device
                // mid-frame-loop on a hot path this codebase never
                // exercised before interfered with the surface's own
                // presentation on at least one backend. The tradeoff is
                // purely a little extra latency on the WebGL2 fallback
                // path before a pick resolves, which was already an
                // accepted cost of doing this off the CPU.
                //
                // `Device::poll` itself is still perfectly usable — the bug
                // was specifically *this* mouse-driven, once-per-frame call
                // site, not polling in general. A future feature that
                // genuinely needs a synchronous readback (blocking until a
                // specific submission's callbacks have actually run, e.g.
                // `wgpu::PollType::wait_indefinitely()`) should call it from
                // its own rare, deliberately-triggered action — not from
                // anywhere on the per-frame render path, mouse-driven or
                // otherwise.
            }
        }
    }

    /// A snapshot of plant/soil/day-cycle state for a UI HUD to poll — kept
    /// separate from (and much coarser-grained than) the per-frame render
    /// data; nothing here needs to be read more than a few times a second.
    #[wasm_bindgen(getter_with_clone)]
    pub struct Stats {
        /// 0.0..1.0, wraps at midnight.
        pub day_progress: f64,
        pub is_daytime: bool,
        /// "Seed" | "Sprout" | "Vegetative" | "Dead".
        pub stage: String,
        pub height: f64,
        /// Every leaf on the plant, main stem or branch.
        pub leaf_count: u32,
        pub branch_count: u32,
        pub stem_segment_count: u32,
        /// 0.0 (bone dry) ..= 1.0 (fully watered).
        pub water_level: f64,
        pub temperature_c: f64,
        /// 0.0 (depleted) ..= 1.0+ (well-fed, can exceed 1.0 — see
        /// `SoilConfig::max_nutrient`).
        pub nutrient_level: f64,
        /// 0.0 (bone dry indoor air) ..= 1.0 (humid).
        pub humidity_level: f64,
        /// 1.0 (fully healthy) ..= 0.0 (totally rotted — see
        /// `Plant::root_health`). The single clearest overwatering gauge:
        /// drops while soil stays waterlogged, independent of how wet the
        /// soil currently reads.
        pub root_health: f64,
        /// 0.0 (pest-free) ..= 1.0 (severe infestation).
        pub pest_infestation: f64,
        /// 1.0 at midsummer, dropping toward winter's floor — see
        /// `sim::season::season_state`.
        pub day_length_factor: f64,
        /// Where the pot currently sits relative to the window (see
        /// `Simulation::set_pot_position`) — echoed back so the HUD doesn't
        /// need to keep its own separate copy of player-set state.
        pub pot_position: f64,
        /// Whether *this* plant's self-watering mode is on (see
        /// `Simulation::set_auto_water`) — per plant, so the HUD's checkbox
        /// reflects whichever plant is actually selected rather than one
        /// shared guess, same reasoning as `pot_position` above.
        pub auto_water_enabled: bool,
        /// Whether *this* plant was grown with a realistic mature-height
        /// cap (see `Plant::realistic_scale`) — set once at creation
        /// (`Simulation::plant_cutting`/`set_species`), read-only afterward.
        pub realistic_scale: bool,
        /// "" while alive; once `stage` reads `"Dead"`, one of "Root rot"
        /// (sustained overwatering/fertilizer burn — see `Plant::
        /// root_health`) or "Starvation" (lost every leaf and never earned
        /// enough carbon back to grow a new one) — see `sim::plant::
        /// DeathCause`. Surfaced explicitly rather than left for a player to
        /// infer from the plant's final appearance, since the two calamities
        /// look similar but call for opposite corrective action.
        pub death_cause: String,
        /// "Spring" | "Summer" | "Autumn" | "Winter" — see `sim::season::
        /// Season`. Room-level (driven by `session_time`), not tied to any
        /// one plant's own lifecycle — it doesn't reset just because a
        /// specific plant restarts or gets replaced, the same reasoning
        /// `sim::moon` already uses.
        pub season: String,
        /// Whole sim-days elapsed since this *session* started — `floor
        /// (session_time / TimeConfig::day_length_sim_seconds)`. Room-
        /// level for the same reason `season` is; a specific plant's own
        /// age is `height`/`stage` plus whatever the player can infer from
        /// watching it grow, not this.
        pub days_elapsed: u32,
        /// Whether the prune tool currently has a leaf or stem segment
        /// hover-picked — see `Simulation::has_hover_target`. Mirrored here
        /// (rather than requiring a separate call) since the UI already
        /// polls `Stats` at a steady cadence for a cursor-style toggle.
        pub hover_active: bool,
        /// 0.0 (new moon) ..= 1.0 (full moon) — see `sim::moon::appearance`.
        /// Driven by `session_time`, not the current plant's own life (see
        /// that field's doc comment for the real bug this distinction
        /// fixes), so restarting/switching species never moves this
        /// backward.
        pub moon_illuminated_fraction: f64,
        /// Tallest `height` this plant has ever reached — see `Plant::
        /// max_height_reached`. A scoring metric: survives `height` itself
        /// dropping back down from a prune or cutting.
        pub max_height_reached: f64,
        /// Highest concurrent leaf count (main stem + every branch) this
        /// plant has ever held at once — see `Plant::max_leaves_at_once`. A
        /// scoring metric, distinct from `leaf_count`'s current reading.
        pub max_leaves_at_once: u32,
        /// Cumulative leaves ever spawned over this plant's life, including
        /// ones since shed or pruned — see `Plant::leaves_produced_total`.
        /// A scoring metric, distinct from `leaf_count`'s current reading.
        pub leaves_produced_total: u32,
        /// Sim-seconds this plant has spent alive — see `Plant::
        /// alive_duration`. A scoring metric: freezes once `stage` reads
        /// `"Dead"`, unlike `days_elapsed` (which is room-level session
        /// time, not this plant's own lifespan).
        pub alive_duration: f64,
        /// `alive_duration` converted to whole/fractional plant-days (same
        /// `TimeConfig::day_length_sim_seconds` divisor `days_elapsed` uses)
        /// — a UI-ready "how many days has it lived" reading so the HUD
        /// doesn't need its own copy of that constant to format
        /// `alive_duration` sensibly.
        pub alive_days: f64,
        pub bloom_intensity: f64,
    }

    /// A per-pot overlay reading for one plant — see `Simulation::
    /// plant_pot_hud`. Every plant renders simultaneously (see `render`'s
    /// own doc comment), so unlike `Stats` (which only ever describes the
    /// *selected* plant) a UI wanting a small water gauge/button under
    /// *each* pot needs one of these per plant, not just one shared reading.
    #[wasm_bindgen(getter_with_clone)]
    pub struct PlantPotHud {
        /// This plant's pot anchor projected into canvas CSS pixels (see
        /// `scene::plant_pot_world_anchor`) — where a UI element should be
        /// centered/anchored to sit right at this specific pot, accounting
        /// for the room's current zoom, camera pan, and this plant's own
        /// `pot_position` slider.
        pub x: f32,
        pub y: f32,
        /// 0.0 (bone dry) ..= 1.0 (fully watered) — same reading as
        /// `Stats::water_level`, just for a specific plant regardless of
        /// which one is currently selected.
        pub water_level: f64,
        /// Whether this plant's own self-watering mode is on — a per-pot
        /// water button should stay disabled while this is true, the same
        /// as the settings HUD's own Water button already does.
        pub auto_water_enabled: bool,
        pub is_dead: bool,
    }

    #[wasm_bindgen(getter_with_clone)]
    pub struct ScreenPosition {
        pub x: f32,
        pub y: f32,
    }

    #[wasm_bindgen]
    pub struct Simulation {
        inner: Rc<RefCell<GpuState>>,
        running: Rc<Cell<bool>>,
    }

    #[wasm_bindgen]
    impl Simulation {
        /// Async because acquiring a WebGPU/WebGL2 adapter+device is a JS
        /// Promise under the hood — wasm-bindgen constructors can't be async,
        /// so this is a plain static factory (`Simulation.create(canvas)` from
        /// JS) instead of `#[wasm_bindgen(constructor)]`.
        ///
        /// `seed_year`/`seed_month`/`seed_day` are the real calendar date
        /// (JS's own `Date`, at the moment the session starts) the moon's
        /// starting phase is grounded in — see `moon::phase_for_date`. The
        /// engine has no live clock of its own to read "today" from, so this
        /// is passed in rather than hardcoded (which is what a previous
        /// version of this did, freezing every session's moon at whatever
        /// date happened to be current when that code was written).
        pub async fn create(
            canvas: HtmlCanvasElement,
            device_pixel_ratio: f32,
            seed_year: i32,
            seed_month: u32,
            seed_day: u32,
        ) -> Result<Simulation, JsValue> {
            console_error_panic_hook::set_once();
            let state = GpuState::new(canvas, device_pixel_ratio, seed_year, seed_month, seed_day).await?;
            Ok(Simulation {
                inner: Rc::new(RefCell::new(state)),
                running: Rc::new(Cell::new(false)),
            })
        }

        /// Schedules the self-rescheduling `requestAnimationFrame` loop. Safe to
        /// call more than once — a second call is a no-op while already running.
        pub fn start(&self) {
            if self.running.replace(true) {
                return;
            }
            spawn_frame_loop(self.inner.clone(), self.running.clone());
        }

        /// Stops the frame loop after its current iteration. Needed because
        /// `Rc`-cloning `inner`/`running` into the recursive rAF closure keeps
        /// it (and the GPU resources it holds) alive independent of this
        /// wrapper's own lifetime — dropping `Simulation` alone would not stop a
        /// running loop. React's StrictMode double-invoked effects are exactly
        /// why JS needs this: call `stop()` on cleanup before the component's
        /// second (real) mount creates a fresh `Simulation`.
        pub fn stop(&self) {
            self.running.set(false);
        }

        pub fn resize(&self, width: u32, height: u32, device_pixel_ratio: f32) {
            self.inner.borrow_mut().resize(width, height, device_pixel_ratio);
        }

        /// Adds water to the soil (0.0-1.0, fraction of field capacity).
        /// Watering far more often than the plant can draw the pot back
        /// down keeps soil continuously saturated, which — sustained long
        /// enough — starts real root damage (see `SoilConfig::
        /// waterlogged_threshold`/`Plant::root_health`); this is
        /// deliberately *not* clamped/rate-limited here, the same way a
        /// real watering can doesn't stop a player from overdoing it.
        pub fn water(&self, amount: f64) {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return };
            slot.soil.water(amount);
        }

        /// Like `water`, but targets a specific plant by index regardless of
        /// which one is currently selected — for the per-pot water button
        /// (drawn under every pot at once, see `plant_pot_hud`), which
        /// shouldn't also switch the settings HUD's own selection just
        /// because the player watered a plant they weren't already looking
        /// at. A no-op if `index` is out of range.
        pub fn water_plant(&self, index: u32, amount: f64) {
            let mut state = self.inner.borrow_mut();
            let Some(slot) = state.plants.get_mut(index as usize) else { return };
            slot.soil.water(amount);
        }

        /// Adds fertilizer (see `Soil::fertilize`) — a second resource with
        /// its own two-sided failure mode: too little starves growth (see
        /// `SoilConfig::nutrient_gate_threshold`), too much causes real
        /// osmotic "fertilizer burn" root damage (see `SoilConfig::
        /// overfeed_threshold`), the same mechanism overwatering damages
        /// roots through.
        pub fn fertilize(&self, amount: f64) {
            let mut state = self.inner.borrow_mut();
            let soil_cfg = state.growth_config.soil;
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return };
            slot.soil.fertilize(amount, &soil_cfg);
        }

        /// Mists the air (see `Humidity::mist`) — raises ambient humidity,
        /// which both slows vapor-pressure-deficit-driven transpiration in
        /// hot rooms and suppresses pest growth (see `PestConfig::
        /// safe_humidity`).
        pub fn mist(&self, amount: f64) {
            self.inner.borrow_mut().humidity.mist(amount);
        }

        /// Treats a pest infestation — see `Plant::treat_pests`.
        pub fn treat_pests(&self) {
            let mut state = self.inner.borrow_mut();
            let pest_cfg = state.growth_config.pest;
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return };
            slot.plant.treat_pests(&pest_cfg);
        }

        /// Prunes the main stem back — see `Plant::prune_main_stem`. Returns
        /// whether it actually happened (a no-op if too short, or dead).
        pub fn prune_main_stem(&self) -> bool {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return false };
            let plant_cfg = slot.plant_config;
            slot.plant.prune_main_stem(&plant_cfg)
        }

        /// Prunes one branch back — see `Plant::prune_branch`. `index` is
        /// into the same order `Stats`/the branch count already implies
        /// (creation order). Returns whether it actually happened.
        pub fn prune_branch(&self, index: u32) -> bool {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return false };
            let plant_cfg = slot.plant_config;
            slot.plant.prune_branch(index as usize, &plant_cfg)
        }

        /// The prune tool's own hover tracking — `x`/`y` are canvas-relative
        /// *CSS* pixels (see `EngineCanvas.tsx`'s pointermove handler), not
        /// the devicePixelRatio-scaled backing-buffer pixels `resize` takes;
        /// `render` converts internally (see `device_pixel_ratio`). Called
        /// on every `pointermove` over the canvas; `clear_pointer_position`
        /// on `pointerleave`, since `None` (not a stale last-known position)
        /// is what actually stops the pick pass from running and clears any
        /// currently-hovered leaf once the cursor leaves the canvas.
        pub fn set_pointer_position(&self, x: f32, y: f32) {
            self.inner.borrow_mut().pointer_pixel = Some((x, y));
        }

        /// See `set_pointer_position`.
        pub fn clear_pointer_position(&self) {
            self.inner.borrow_mut().pointer_pixel = None;
        }

        /// Drags the whole room's view — `dx`/`dy` are a *CSS* pixel delta
        /// (this drag event's movement since the last one), same convention
        /// `set_pointer_position` uses for an absolute position. Converted
        /// to NDC and accumulated into `camera_pan`, clamped there.
        pub fn pan_camera(&self, dx: f32, dy: f32) {
            let mut state = self.inner.borrow_mut();
            let px_dx = dx * state.device_pixel_ratio;
            let px_dy = dy * state.device_pixel_ratio;
            let ndc_dx = (px_dx / state.config.width as f32) * 2.0;
            let ndc_dy = -(px_dy / state.config.height as f32) * 2.0;
            state.camera_pan[0] = (state.camera_pan[0] + ndc_dx).clamp(-MAX_CAMERA_PAN, MAX_CAMERA_PAN);
            state.camera_pan[1] = (state.camera_pan[1] + ndc_dy).clamp(-MAX_CAMERA_PAN, MAX_CAMERA_PAN);
        }

        /// Recenters the room's view — see `pan_camera`.
        pub fn reset_camera_pan(&self) {
            self.inner.borrow_mut().camera_pan = [0.0, 0.0];
        }

        /// Whether the currently active tool has a leaf or stem segment
        /// hover-picked — for the UI to show a different cursor (see
        /// `Stats::hover_active`, which mirrors this at the same coarse
        /// poll rate rather than needing its own separate call).
        pub fn has_hover_target(&self) -> bool {
            self.inner.borrow().hovered_target.get().is_some()
        }

        pub fn set_active_tool(&self, tool: &str) {
            let mut state = self.inner.borrow_mut();
            state.active_tool = match tool {
                "prune" => Some(Tool::Prune),
                "trim" => Some(Tool::Trim),
                _ => None,
            };
            state.hovered_target.set(None);
        }

        /// Acts on whichever leaf or stem segment the prune tool currently
        /// has hover-picked (see `set_pointer_position`) — the prune
        /// tool's click action. A leaf hover prunes that exact leaf (see
        /// `Plant::prune_leaf`); a stem-segment hover cuts the main stem or
        /// that branch at exactly the clicked segment's own base height
        /// (see `Plant::cut_main_stem_at`/`cut_branch_at` and `render`'s
        /// `stem_segment_targets`, which is what maps a segment slot back
        /// to "which grower, what height"). Returns whether anything
        /// actually happened — a no-op if nothing's hovered, or the pick
        /// readback just hasn't resolved yet (see `hovered_target`'s own
        /// doc comment on that latency).
        pub fn prune_hovered(&self) -> bool {
            let mut state = self.inner.borrow_mut();
            match state.hovered_target.get() {
                Some(scene::PickTarget::Leaf { plant_index, slot }) => {
                    let Some(plant_slot) = state.plants.get_mut(plant_index) else {
                        return false;
                    };
                    plant_slot.plant.prune_leaf(slot)
                }
                Some(scene::PickTarget::StemSegment { plant_index, slot }) => {
                    let Some(plant_slot) = state.plants.get_mut(plant_index) else {
                        return false;
                    };
                    let plant_cfg = plant_slot.plant_config;
                    let Some(&(grower, height)) = plant_slot.stem_segment_targets.get(slot) else {
                        return false;
                    };
                    match grower {
                        None => plant_slot.plant.cut_main_stem_at(height, &plant_cfg),
                        Some(branch_index) => plant_slot.plant.cut_branch_at(branch_index, height, &plant_cfg),
                    }
                }
                None => false,
            }
        }

        /// Moves the plant into a bigger pot — see `Plant::repot`. Returns
        /// whether it actually happened.
        pub fn repot(&self) -> bool {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return false };
            let plant_cfg = slot.plant_config;
            slot.plant.repot(&plant_cfg)
        }

        /// Takes a stem cutting from the selected plant — see `Plant::take_
        /// cutting`'s own doc comment: costs that plant some height, like a
        /// small prune, rather than resetting or replacing it. On success,
        /// adds a `InventoryItem` to the room's shared inventory (see `plant_
        /// cutting`, which later spends it on an actual new plant). Returns
        /// whether it actually happened (a no-op if too short, or dead).
        pub fn take_cutting(&self) -> bool {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return false };
            let plant_cfg = slot.plant_config;
            let took_cutting = slot.plant.take_cutting(&plant_cfg);
            if took_cutting {
                let species_name = slot.species_name.clone();
                state.inventory.push(InventoryItem { species_name, from_seed: false });
            }
            took_cutting
        }

        /// How many stem cuttings are currently sitting in inventory,
        /// waiting to be planted — see `take_cutting`/`plant_cutting`.
        pub fn inventory_count(&self) -> u32 {
            self.inner.borrow().inventory.len() as u32
        }

        pub fn grant_seed(&self, species: &str) {
            let mut state = self.inner.borrow_mut();
            state.inventory.push(InventoryItem { species_name: species.to_string(), from_seed: true });
        }

        /// Which species a given inventory slot's cutting is, for an
        /// inventory UI to label each one — "" if `index` is out of range.
        pub fn inventory_species(&self, index: u32) -> String {
            self.inner
                .borrow()
                .inventory
                .get(index as usize)
                .map(|item| item.species_name.clone())
                .unwrap_or_default()
        }

        /// Spends one inventory item to grow a brand-new, independent plant
        /// in its own pot along the windowsill (see `scene::plant_slot_
        /// base_anchor`) — a no-op if `index` is out of range or the room
        /// is already at `MAX_PLANTS`. A seed-sourced item (the room's
        /// starting stock — see `GpuState::new`) germinates from scratch
        /// (`Plant::new`); a cutting-sourced item (see `take_cutting`)
        /// starts already-rooted (`Plant::from_cutting`), skipping
        /// germination. Returns whether it actually happened; on success,
        /// removes that item from inventory and selects the new plant.
        pub fn plant_cutting(&self, index: u32, realistic_scale: bool) -> bool {
            let mut state = self.inner.borrow_mut();
            if state.plants.len() >= MAX_PLANTS {
                return false;
            }
            let Some(item) = state.inventory.get(index as usize) else {
                return false;
            };
            let species_name = item.species_name.clone();
            let from_seed = item.from_seed;
            let plant_config = plant_config_for_species(&species_name);
            let plant = if from_seed {
                Plant::new().with_realistic_scale(realistic_scale)
            } else {
                Plant::from_cutting(&plant_config).with_realistic_scale(realistic_scale)
            };
            let aspect = state.config.width as f32 / state.config.height as f32;
            let new_slot = build_plant_slot(
                &state.device,
                &state.instance_bind_group_layout,
                aspect,
                &state.scene_layout,
                &state.growth_config,
                plant_config,
                plant,
                species_name,
            );
            state.plants.push(new_slot);
            state.selected_plant_index = state.plants.len() - 1;
            state.inventory.remove(index as usize);
            true
        }

        /// Moves the pot relative to the window — see `sim::room`. 0.0 is
        /// right at the sill (brightest, but drafty at night); 1.0 is as far
        /// back into the room as this game models (dim, but climate-stable).
        pub fn set_pot_position(&self, position: f64) {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return };
            slot.pot_position = position.clamp(0.0, 1.0);
            slot.pot_position_active = true;
        }

        /// Toggles a self-watering-pot mode for the *currently selected*
        /// plant only — see `Soil::apply_auto_water`/`PlantSlot::auto_
        /// water_enabled`. While enabled, moisture never drops below
        /// `SoilConfig::auto_water_floor` on its own, so a player doesn't
        /// have to keep clicking Water to keep a fast-growing plant alive.
        /// Note this no longer makes manual watering strictly obsolete: the
        /// floor it maintains is comfortably below `SoilConfig::
        /// waterlogged_threshold`, so auto-water itself never causes root
        /// rot, but it also can't push moisture up for a thirsty plant the
        /// way a deliberate watering dose can.
        pub fn set_auto_water(&self, enabled: bool) {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return };
            slot.auto_water_enabled = enabled;
        }

        /// Switches growth habit (see `sim::config::plant_config_for_species`
        /// for valid names — anything unrecognized falls back to
        /// `"dracaena"`) and starts a *fresh* plant/soil/humidity under it.
        /// Different species aren't just re-tunings of the same growing
        /// plant (a caning, crown-branching habit vs. a basal rosette are
        /// different shapes from the very first true leaf onward), so this
        /// discards whatever was growing rather than trying to convert it.
        pub fn set_species(&self, species: &str, realistic_scale: bool) {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(slot) = state.plants.get_mut(selected) else { return };
            slot.plant_config = plant_config_for_species(species);
            slot.species_name = species.to_string();
            state.reset_plant(realistic_scale);
        }

        /// Starts over with a fresh seed under the *current* species (and
        /// this same plant's own existing realistic-scale choice, unlike
        /// `set_species` which re-asks) — the player-facing "restart"
        /// action, most relevantly once `Stats::stage` reads `"Dead"`
        /// (there's otherwise no way back from that stage; `step` is a
        /// deliberate no-op there — see `Stage::Dead`).
        pub fn restart(&self) {
            let mut state = self.inner.borrow_mut();
            let selected = state.selected_plant_index;
            let Some(realistic_scale) = state.plants.get(selected).map(|s| s.plant.realistic_scale) else { return };
            state.reset_plant(realistic_scale);
        }

        /// How many plants currently exist in the room — see `GpuState::
        /// plants`.
        pub fn plant_count(&self) -> u32 {
            self.inner.borrow().plants.len() as u32
        }

        /// Which species a specific plant slot is — for a plant-selector UI
        /// to label each one without needing to select it first (every
        /// plant can be a different species — see `PlantSlot::species_
        /// name`). "" if `index` is out of range.
        pub fn plant_species(&self, index: u32) -> String {
            self.inner
                .borrow()
                .plants
                .get(index as usize)
                .map(|slot| slot.species_name.clone())
                .unwrap_or_default()
        }

        /// Which plant every per-plant action/HUD reading currently
        /// targets — see `Simulation::set_selected_plant`.
        pub fn selected_plant_index(&self) -> u32 {
            self.inner.borrow().selected_plant_index as u32
        }

        /// A small per-pot overlay reading for one plant — the on-screen
        /// position (in canvas CSS pixels, matching what `Simulation::
        /// set_pointer_position` already expects) just below this pot's own
        /// rendered bottom edge, plus enough state (water level, auto-water,
        /// dead) for a "water this pot" button/gauge to sit there,
        /// regardless of which plant is currently selected. `None` if
        /// `index` is out of range.
        ///
        /// Reproduces the exact same projection `render` uses to actually
        /// place this pot's own drawables — `scene::plant_pot_world_anchor`
        /// for the world-space anchor (the pot mesh's own local origin, see
        /// its own doc comment), offset down by the "pot" mesh's own baked
        /// half-height so this tracks the pot's actual bottom edge rather
        /// than its center, then the same zoom/pan/NDC math `render`'s own
        /// `cursor_ndc` uses, just inverted (NDC → CSS pixels instead of CSS
        /// pixels → NDC).
        pub fn plant_pot_hud(&self, index: u32) -> Option<PlantPotHud> {
            let state = self.inner.borrow();
            let slot = state.plants.get(index as usize)?;
            let zoom = scene::dynamic_zoom_for_room(state.plants.iter().map(|p| &p.plant), &state.scene_layout);
            let world_anchor = scene::plant_pot_world_anchor(&state.scene_layout, index as usize, slot.pot_position);
            // `world_anchor` is where the pot mesh's own local origin sits
            // (see `scene::pot_background` — the pot is drawn centered on
            // it, same as the soil cap/seed/stem base), not its bottom edge
            // — offsetting by the mesh's own baked half-height (in the same
            // pre-zoom units `pot_scale` converts from) moves this down to
            // just below the pot itself, the same "local vertex, then
            // scale, then zoom" order `vs_main` applies to every real pot
            // vertex, so this tracks the pot's actual rendered bottom edge
            // at any zoom level rather than a fixed guess.
            let pot_half_height = state.meshes.local_half_extent("pot").1 * state.scene_layout.pot_scale;
            let below_pot_anchor = [world_anchor[0], world_anchor[1] - pot_half_height];
            let ndc_x = below_pot_anchor[0] * zoom + state.camera_pan[0];
            let ndc_y = below_pot_anchor[1] * zoom + state.camera_pan[1];
            let width_px = state.config.width as f32 / state.device_pixel_ratio;
            let height_px = state.config.height as f32 / state.device_pixel_ratio;
            Some(PlantPotHud {
                x: (ndc_x + 1.0) / 2.0 * width_px,
                y: (1.0 - ndc_y) / 2.0 * height_px,
                water_level: slot.soil.moisture,
                auto_water_enabled: slot.auto_water_enabled,
                is_dead: slot.plant.stage == Stage::Dead,
            })
        }

        pub fn lamp_screen_position(&self) -> ScreenPosition {
            let state = self.inner.borrow();
            let zoom = scene::dynamic_zoom_for_room(state.plants.iter().map(|p| &p.plant), &state.scene_layout);
            let ndc_x = state.scene_layout.lamp_offset[0] * zoom + state.camera_pan[0];
            let ndc_y = state.scene_layout.lamp_offset[1] * zoom + state.camera_pan[1];
            let width_px = state.config.width as f32 / state.device_pixel_ratio;
            let height_px = state.config.height as f32 / state.device_pixel_ratio;
            ScreenPosition {
                x: (ndc_x + 1.0) / 2.0 * width_px,
                y: (1.0 - ndc_y) / 2.0 * height_px,
            }
        }

        /// Switches which plant every per-plant action (`water`,
        /// `fertilize`, `prune_main_stem`, `stats`, ...) targets — every
        /// plant renders and simulates regardless of selection (see
        /// `render`'s own doc comment), this only changes which one the HUD
        /// and player actions actually reach. A no-op if `index` is out of
        /// range.
        pub fn set_selected_plant(&self, index: u32) {
            let mut state = self.inner.borrow_mut();
            if (index as usize) < state.plants.len() {
                state.selected_plant_index = index as usize;
            }
        }

        /// A relative speed multiplier (see `TimeConfig::clamp_speed_
        /// multiplier` for the valid range and why out-of-range/non-finite
        /// input is clamped/sanitized rather than passed through) applied
        /// on top of `TimeConfig::default()`'s own pace, so a UI slider's
        /// "1x" always means "today's validation-demo default speed"
        /// regardless of what that default happens to be tuned to.
        pub fn set_time_scale(&self, multiplier: f64) {
            let mut state = self.inner.borrow_mut();
            let base = TimeConfig::default().sim_seconds_per_real_second;
            state.growth_config.time.sim_seconds_per_real_second =
                base * TimeConfig::clamp_speed_multiplier(multiplier);
        }

        /// A coarse-grained snapshot for a UI HUD — see `Stats`. `None`
        /// while the room has no plants at all (before the first `add_
        /// plant`).
        pub fn stats(&self) -> Option<Stats> {
            let state = self.inner.borrow();
            let slot = state.plants.get(state.selected_plant_index)?;
            let stage = match slot.plant.stage {
                Stage::Seed => "Seed",
                Stage::Sprout => "Sprout",
                Stage::Vegetative => "Vegetative",
                Stage::Dead => "Dead",
            };
            let branch_leaf_count: usize = slot.plant.branches.iter().map(|b| b.leaves.len()).sum();
            let sun_state = sun::sun_state(state.day_progress, &state.growth_config.sun);
            let climate_state = climate::climate_state(state.day_progress, &state.growth_config.climate);
            let season_state = season::season_state(state.session_time, &state.growth_config.season);
            let death_cause = match slot.plant.death_cause {
                Some(DeathCause::RootRot) => "Root rot",
                Some(DeathCause::Starvation) => "Starvation",
                None => "",
            };
            let days_elapsed =
                (state.session_time / state.growth_config.time.day_length_sim_seconds).floor() as u32;
            let moon_phase = moon::current_phase(state.session_time, &state.growth_config.moon);
            Some(Stats {
                day_progress: state.day_progress,
                is_daytime: sun_state.intensity > 0.0,
                stage: stage.to_string(),
                height: slot.plant.height,
                leaf_count: (slot.plant.leaves.len() + branch_leaf_count) as u32,
                branch_count: slot.plant.branches.len() as u32,
                stem_segment_count: slot.segments_drawn as u32,
                water_level: slot.soil.moisture,
                temperature_c: climate_state.temperature_c,
                nutrient_level: slot.soil.nutrient,
                humidity_level: state.humidity.level,
                root_health: slot.plant.root_health,
                pest_infestation: slot.plant.pest_infestation,
                day_length_factor: season_state.day_length_factor,
                pot_position: slot.pot_position,
                auto_water_enabled: slot.auto_water_enabled,
                realistic_scale: slot.plant.realistic_scale,
                death_cause: death_cause.to_string(),
                season: season_state.season.name().to_string(),
                days_elapsed,
                hover_active: state.hovered_target.get().is_some(),
                moon_illuminated_fraction: moon::appearance(moon_phase).illuminated_fraction,
                max_height_reached: slot.plant.max_height_reached,
                max_leaves_at_once: slot.plant.max_leaves_at_once,
                leaves_produced_total: slot.plant.leaves_produced_total,
                alive_duration: slot.plant.alive_duration,
                alive_days: slot.plant.alive_duration / state.growth_config.time.day_length_sim_seconds,
                bloom_intensity: slot.plant.bloom_intensity,
            })
        }
    }

    fn spawn_frame_loop(state: Rc<RefCell<GpuState>>, running: Rc<Cell<bool>>) {
        // The classic wasm-bindgen self-referential rAF idiom: the closure needs
        // a handle to itself to reschedule, so it's stored behind a `Rc<RefCell<
        // Option<Closure>>>` that the closure itself captures a clone of.
        let slot: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
        let slot_for_first_call = slot.clone();

        *slot_for_first_call.borrow_mut() = Some(Closure::new(move |timestamp_ms: f64| {
            if !running.get() {
                // Drop our own reference — once no other clone of `slot`
                // remains, this closure (and the `state`/`Rc<GpuState>` it
                // holds) actually gets freed instead of leaking forever.
                slot.borrow_mut().take();
                return;
            }
            state.borrow_mut().render(timestamp_ms);
            request_animation_frame(slot.borrow().as_ref().unwrap());
        }));
        request_animation_frame(slot_for_first_call.borrow().as_ref().unwrap());
    }

    fn request_animation_frame(closure: &Closure<dyn FnMut(f64)>) {
        web_sys::window()
            .expect("no global `window`")
            .request_animation_frame(closure.as_ref().unchecked_ref())
            .expect("requestAnimationFrame failed");
    }
}
