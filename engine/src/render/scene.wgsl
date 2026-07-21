// Per-instance transform + color tint applied to a mesh's local
// (SVG-baked) vertex positions/colors to place and light it in the scene.
struct Instance {
    offset: vec2<f32>,
    scale: vec2<f32>,
    // The mesh's own (max |x|, max |y|) among its baked vertices (see
    // `meshes::MeshRegistry::local_half_extent`) — reused here (not just on
    // the CPU, for the outline margin) to normalize `local_pos` below into
    // a consistent -1..1 "dome" coordinate regardless of a mesh's native
    // SVG size. [1,1] (never zero — would divide by zero in `fs_main`) for
    // any instance that isn't specular-lit (`shininess` 0).
    local_extent: vec2<f32>,
    rotation: f32,
    // 0.0 (nearest camera) .. 1.0 (farthest) — real GPU depth-buffer Z, so
    // overlapping instances (two leaves, a leaf and its own outline halo)
    // resolve correctly regardless of draw order, instead of relying
    // entirely on manual back-to-front sequencing. See `scene::
    // apply_depth_look` for this same value's *cosmetic* (size/brightness)
    // effect, computed CPU-side rather than here.
    depth: f32,
    // Blinn-Phong shininess exponent — 0 disables the cursor specular
    // highlight entirely (see `fs_main`), which is how every non-leaf mesh
    // opts out without an extra branch/uniform. See `SceneLayout::
    // leaf_shininess`.
    shininess: f32,
    // Multiplied against the mesh's own baked vertex color — how the
    // day/night light level shows up on background pieces (see
    // `scene::ambient_tint`). [1,1,1] leaves a mesh's own color unchanged.
    tint: vec3<f32>,
};

@group(0) @binding(0) var<uniform> instance: Instance;

// Shared across every instance this frame — see `scene::SceneLightUniform`.
// A GPU-computed point light at the window: every fragment's brightness
// falls off with distance from `pos`, so the whole scene reads as lit from
// the window rather than each mesh carrying its own baked-in brightness.
// `cursor_*` is the same idea, second light: the player's own cursor,
// tracked the cheap way (see `render::mod`'s `pointer_pixel`, updated only
// on real `pointermove` events, not polled) rather than anything running
// per-frame CPU-side hit-testing of its own.
struct SceneLight {
    pos: vec2<f32>,
    intensity: f32,
    falloff: f32,
    color: vec3<f32>,
    ambient_floor: vec3<f32>,
    cursor_pos: vec2<f32>,
    cursor_intensity: f32,
    cursor_falloff: f32,
};

@group(1) @binding(0) var<uniform> light: SceneLight;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec2<f32>,
    // The vertex's own local (pre-transform) position, interpolated across
    // each triangle — `fs_main` turns this into a per-*fragment* fake
    // surface normal for the cursor specular highlight (see there), which
    // a per-instance value alone couldn't do (a highlight has to move
    // smoothly across a single leaf's own surface, not just jump instance
    // to instance).
    @location(2) local_pos: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Scale in the mesh's own local axes *first*, then rotate the
    // already-correctly-proportioned shape rigidly. Doing it the other way
    // around (rotate, then multiply by a non-uniform `scale` — scale_x and
    // scale_y genuinely differ for a stem/branch, driven by radius and
    // height respectively) shears the shape instead of just resizing it:
    // a non-uniform scale doesn't commute with rotation, so its visual long
    // axis drifts away from `instance.rotation` as the angle grows, even
    // though every *other* placement (a leaf, a branch, the flower — all
    // uniformly scaled, so order never mattered for them) still trusts that
    // same angle exactly. This was a real, visible bug: the flower (placed
    // by plain trig, unaffected) stopped lining up with the main stem's own
    // rendered tip once lean/droop grew large enough for the shear to be
    // obvious.
    let scaled = in.position * instance.scale;
    let c = cos(instance.rotation);
    let s = sin(instance.rotation);
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    );
    let world = rotated + instance.offset;

    var out: VertexOutput;
    out.position = vec4<f32>(world, instance.depth, 1.0);
    out.color = in.color * instance.tint;
    out.world_pos = world;
    out.local_pos = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = distance(light.pos, in.world_pos);
    let falloff_term = light.intensity / (1.0 + light.falloff * dist * dist);
    let cursor_dist = distance(light.cursor_pos, in.world_pos);
    // Same inverse-square-ish falloff shape as the window light, just a
    // much tighter `cursor_falloff` (see `SceneLayout`) so it reads as a
    // small, local pool of light around the pointer rather than relighting
    // the whole room.
    let cursor_term = light.cursor_intensity / (1.0 + light.cursor_falloff * cursor_dist * cursor_dist);
    let lit = light.ambient_floor + light.color * falloff_term + vec3<f32>(1.0, 0.98, 0.9) * cursor_term;

    var specular = vec3<f32>(0.0, 0.0, 0.0);
    if (instance.shininess > 0.0) {
        // A cheap "fake dome" normal — this pipeline has no real 3D
        // geometry or per-vertex normals (flat tessellated SVG fills
        // only), so a genuine Blinn-Phong specular isn't possible as such.
        // This instead treats a leaf as if it gently domed up out of the
        // page: its local (pre-transform) position, normalized by the
        // mesh's own extent into a -1..1 disc, gives an analytic normal
        // (steepest at the rim, straight up at the center) with no extra
        // per-vertex data needed beyond what's already on the GPU.
        let uv = in.local_pos / instance.local_extent;
        let r2 = clamp(dot(uv, uv), 0.0, 1.0);
        let normal = normalize(vec3<f32>(uv, sqrt(1.0 - r2)));
        // The cursor light is given a small notional height above the flat
        // scene plane (0.5) purely so its direction isn't perfectly
        // in-plane — with no real depth position of its own, a highlight
        // still has to come from *somewhere* not directly on the surface.
        let light_dir = normalize(vec3<f32>(light.cursor_pos - in.world_pos, 0.5));
        let view_dir = vec3<f32>(0.0, 0.0, 1.0);
        let half_dir = normalize(light_dir + view_dir);
        let spec_term = pow(max(dot(normal, half_dir), 0.0), instance.shininess);
        specular = vec3<f32>(1.0, 1.0, 0.95) * spec_term * cursor_term;
    }

    return vec4<f32>(in.color * lit + specular, 1.0);
}

// GPU hit-testing pass (see `render::mod`'s pick pipeline/texture and
// `scene::encode_pick_id`/`decode_pick_id`): outputs `instance.tint`
// completely unlit and untouched by the mesh's own baked vertex color,
// since here it's carrying a flat leaf-slot-ID color, not a real tint. The
// vertex stage (and its real depth) is shared with `fs_main` — drawn with
// the same transforms/depth-test as the main pass, against that same
// frame's already-resolved depth buffer, so a leaf hidden behind the stem
// or another nearer leaf correctly never wins the pick even though this
// pass never draws that occluding geometry itself.
@fragment
fn fs_pick(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(instance.tint, 1.0);
}
