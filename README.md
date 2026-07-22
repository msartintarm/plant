# plant

[msartintarm.github.io/plant](https://msartintarm.github.io/plant)

A houseplant growth simulation: a sprout grows into a plant, moving and
reorienting its leaves toward sunlight from a window over an accelerated
day/night cycle. Rendered in 2D side-profile.

Unlike a typical Rust/WASM web game (state-in-Rust, drawing-in-JS), the
engine here owns rendering too: it opens its own WebGPU (falling back to
WebGL2) surface on the page's `<canvas>` and drives its own frame loop —
closer to how a native engine like Unity owns its render pipeline. All art
is authored as SVG and tessellated into GPU triangle meshes at build time
(no runtime rasterization, no textures) so everything stays vector/scalable.

## Layout

- `engine/` — Rust, compiled to WebAssembly via `wasm-pack`
  - `src/sim/` — pure simulation logic, no wgpu/DOM dependency, unit- and
    integration-tested with plain `cargo test` (97 tests)
    - `sun.rs` — sun elevation/azimuth/intensity/color from `day_progress`
    - `climate.rs` — ambient temperature from `day_progress` (phase-lagged
      behind peak sunlight, like real thermal mass), plus the general
      temperature-response curves (`temperature_factor`'s bell curve,
      `q10_factor`'s respiration relationship) plant growth uses
    - `soil.rs` — soil moisture reservoir (evaporation, uptake, watering,
      an optional self-watering-pot floor)
    - `plant.rs` — the growth model: germination → sprout → vegetative
      growth (stem *and* crown branches, see below), gated by light,
      water, *and* temperature together; leaves have a full lifecycle
      (bud → mature → senescing → shed, faster under drought/cold stress),
      not just a fixed size. See its module docs for the specific
      plant-physiology mechanisms each part is modeled on (Liebig's law of
      the minimum, the Lockhart equation, the "pipe model," etiolation,
      phototropism/heliotropism/nyctinasty, apical-dominance-release
      crown branching modeled on *Dracaena*). Records a
      `Plant::last_decision` trace (inputs + outputs of the most recent
      step) so tests can assert on *why* something happened on a single
      tick instead of only inferring it from long aggregate runs.
    - `config.rs` — every tunable rate/threshold as plain data
      (`SunConfig`/`SoilConfig`/`PlantConfig`/`TimeConfig`/`ClimateConfig`),
      passed into the above rather than hardcoded as module constants —
      includes the three selectable growth-habit presets
      (`PlantConfig::dracaena`/`peace_lily`/`pothos`)
    - `playthrough_tests.rs` — steps `Plant`/`Soil` using the *same*
      real-time-to-sim-time pacing the live render loop uses, to answer
      "what does a player actually see within N minutes of real play"
      directly via `cargo test` rather than a browser/screenshots — this is
      how the crown-branching allocation bug below was actually found
  - `src/render/` — wgpu setup, mesh registry, the live scene, the frame
    loop. Only `mod.rs`'s `wgpu_engine` submodule and `meshes.rs` are
    wasm32-gated (they need wgpu/web-sys/wasm-bindgen, only available as
    dependencies on that target); `scene.rs` and `config.rs` are pure
    math/data with no such dependency and compile — and are unit-tested —
    natively, alongside `sim/` (33 tests)
    - `scene.rs` — pure functions turning `Plant`/`SunState` into instance
      transforms (position/scale/rotation/tint), generalized around a
      `Frame` (offset + angle) so the same math places a leaf on the main
      stem *or* on a branch; no GPU calls of its own. Its tests capture
      exactly what a human would otherwise check by eyeballing a
      screenshot — does the sun/moon disc stay inside the window pane
      across aspect ratios, does the wall still cover the full canvas,
      are left/right cotyledons mirror images, does fold/droop/helio
      compose the way the module docs say
    - `config.rs` — scene layout (`SceneLayout`) as data
    - `mod.rs` — owns the wgpu device/surface, steps the simulation each
      frame by real elapsed time, writes the resulting transforms into
      fixed pools of GPU buffers (see `scene::MAX_LEAVES`/`MAX_BRANCHES`,
      `sim::plant::MAX_STEM_SEGMENTS`) and draws them
  - `assets/svg/` — hand-authored vector art (leaf, cotyledon, seed, stem,
    pot, soil, sun, moon, window/wall, light beam, climbing trellis/aerial
    root, and one flower mesh per species — see its README for the anchor
    convention that makes swapping in real art a drop-in file replacement
  - `build.rs` — parses `assets/svg/*.svg` (via `usvg`) and tessellates them
    (via `lyon`) into static vertex/index data baked into the wasm binary;
    `usvg`/`lyon` are build-dependencies only and never ship in the binary
- `web/` — Next.js app (static export, deployed to GitHub Pages)
  - `src/components/EngineCanvas.tsx` — loads the wasm module, hosts the
    canvas the engine renders into
  - `tests/` — Playwright e2e; `src/**/*.test.ts` — Vitest unit tests

## Run it

```
cd web
npm install
npm run dev             # predev runs wasm:build automatically
npm run test:e2e        # first run: npm run test:e2e:install
npm run check           # tsc + eslint + vitest
```

`cargo test` in `engine/` is native and fast for iterating on sim logic;
`npm run wasm:build` (or `npm run dev` via `predev`) rebuilds the wasm the
browser actually loads.

## Current state

The full loop is live: a seed germinates (gated by soil moisture), pushes
up a hypocotyl on stored reserves, unfurls cotyledons, then grows true
leaves/stem/thickness driven by light- and water-gated photosynthesis, and
— once mature enough — sprouts crown branches that are themselves smaller
growing points with their own leaves/thickness/lean, each fairly sharing the
plant's one carbon pool (see "Non-obvious design points" below). Leaves
appear on a height-based plastochron rather than a pure carbon race, so they
keep appearing up the stem and on every branch as it grows, not just near
the base. Water-stressed stems/branches physically sag under their own
weight (reversible on rewatering), distinct from the older leaf-droop/fold
mechanisms. The camera pulls back dynamically as the plant grows tall enough
to otherwise run off the top of frame. Growth habits are selectable
(`sim::config::plant_config_for_species`) — Dracaena's caning/
crown-branching habit and Peace Lily's basal rosette (no branching,
squat, leaf-dense — see `PlantConfig::peace_lily`'s doc comment on how it's
derived from the caning habit's numbers) to start; see below for the third,
Pothos. Each blooms with its own species-specific terminal flower once
mature (`PlantConfig::flowering_height_threshold`, see below). Ambient
temperature (`sim::climate`) follows the day/night cycle (warmest a couple
hours after solar noon, like real thermal lag) and feeds back into growth:
photosynthesis and elongation both have a temperature optimum (a bell
curve, not "hotter is always better"), respiration follows the Q10
relationship (roughly doubling per 10°C, so a hot plant's own upkeep costs
more even as its income falls), and germination needs warmth as well as
moisture. Leaves have a full lifecycle, not just a fixed size: they expand
from a bud to full size, stay healthy for a while, then age — yellowing,
browning, shrinking, and finally abscising (removed from the plant
entirely), faster under drought or cold stress, same as a real plant
shedding leaves to cut its own losses. Stems and branches don't render as a
single rigid rotated line either: each records its own phototropic lean as
a history of frozen segments (`sim::plant::record_stem_segments`) — already-
stiffened growth keeps whatever curvature it had when it formed, only the
still-growing tip bends with *today's* lean/droop — so a long-lived plant
reads as a gentle sweep (straighter low down, more bent up high), the way a
real stem that's kept leaning toward a window over time actually looks.

The window is sized and positioned to real-world proportions (a sill well
above the pot, roughly as tall as a well-grown houseplant, not a small
fraction of one — see `render::config::SceneLayout::window_offset`'s doc
comment), and light availability now falls off with height, not just
time of day: `sim::plant::height_light_factor` keeps a grower fully lit
within the window's own height range, easing down to a dim ambient floor
over a further stretch above it — the real reason a real houseplant
doesn't keep growing indefinitely past its own light source. A light beam
(`scene::light_beam_transform`) visibly renders from the window onto the
plant, widening and warming with how much light is actually reaching it
right now (the product of the day/night cycle and that same height
falloff) rather than requiring a HUD gauge. Phototropic lean is verified to
bend *toward* wherever the window actually is (`scene::lean_sign_toward_
window`), not just in a fixed screen direction — a real sign bug here
(the rotation convention `rotate_and_place` uses bends positive angles
toward -x, but the window sits at +x) went undetected through the initial
curved-stem implementation until checked with an explicit numerical test,
not a screenshot.

Leaves also don't accumulate without limit: `sim::plant::self_shading_
factors` discounts each leaf's photosynthesis by how much of the *same
plant's own* leaf area sits above it (a Beer-Lambert canopy-light-
extinction model — Monsi & Saeki 1953), and a heavily overtopped leaf
senesces immediately rather than waiting out its full natural lifespan
(`age_and_senesce_leaves`) — together these keep standing leaf count
self-limiting (bounded over an arbitrarily long session, not growing
without limit) and produce the real "bare cane, leafy crown near the top"
silhouette of a mature Dracaena, rather than an unrealistic pile-up of
dozens of leaves.

A third growth habit, Pothos (`PlantConfig::pothos`), models a climbing/
vining aerial-root strategy distinct from Dracaena's freestanding cane and
Peace Lily's rosette: while its own height is still within reach of a
support (`PlantConfig::trellis_height`, rendered as `trellis.svg`, a
ladder-style stake), phototropic lean is suppressed entirely — a real
aerial-root climber like Pothos is mechanically held flat against its
support, not bending toward light — and it grows small anchoring
`AerialRoot`s into the support at intervals (real, distinct from a
twining vine's spiral wrap or a tendril-bearer's coiling appendages, which
model different real climbing strategies this doesn't attempt). Once it
outgrows the support's height, it flops over and leans toward light like
any freestanding stem, matching a real overgrown pothos vine trailing past
the top of its moss pole.

Blooming is now species-specific and cyclical, not a single generic shape
drawn permanently once mature: each species points at its own botanically
distinct flower mesh (`PlantConfig::flower_mesh_name` — Dracaena's wispy
star-flowered panicle vs. Peace Lily's spathe-and-spadix, which Pothos also
shares, being in the same family), and `Plant::bloom_intensity` eases
open and closed on a repeating cycle (`bloom_duration`/`bloom_rest_
duration`) tuned per species to reflect real flowering frequency — Dracaena
rests roughly 15x longer than it blooms (real Dracaena flowering is rare
enough in cultivation to be a notable event), while Peace Lily reblooms
readily and spends at least as much time in bloom as resting.

All rendered in real time from `wgpu`, with the sun/moon visibly arcing through
the window and the room's ambient tint shifting with the day/night cycle.

Verified several ways: `cargo test` (130 tests — 97 in `sim/`, including
`sim::playthrough_tests`, which replay the actual real-time-to-sim-time
pacing the render loop uses and is how several real bugs were found, see
below; 33 in `render::scene`, covering exactly the placement/geometry a
human would otherwise check by eyeballing a screenshot — including the
light beam's and trellis's own direction/scale math, verified by
replicating `scene.wgsl`'s actual vertex-shader formula in the test itself
rather than trusting a screenshot, which is what caught a genuine 180°
sign error in an earlier attempt at the light beam's angle); `vitest` (15
tests for `web/src/lib/formatStats.ts`, the HUD's pure display-formatting
logic); Playwright (5 end-to-end tests covering wasm load, live HUD state,
the water/time-scale controls, the auto-water toggle, and species
switching); and, for the rendering pipeline itself, browser screenshots
confirming it matches what the tests predict (germination, cotyledon
unfurl, growth/thickening, nyctinastic fold, sun/moon position and tinting,
leaves distributed up a visibly long stem, dynamic zoom-out, a terminal
flower at the stem's tip lining up exactly with the curved stem beneath it,
all three species' distinct silhouettes including Pothos visibly climbing
straight alongside its trellis with aerial roots peeking out beside the
stem and then flopping over to lean toward light once it outgrows the
support, the light beam shining from window to plant and dimming at night,
each species' own flower mesh opening during its bloom phase, leaves
visibly yellowing/browning with age, HUD overlay including temperature).

The `web/` demo's pacing (`sim::config::TimeConfig`) is tuned aggressively —
a 5-real-second day/night cycle — specifically so this demo is useful for
*validating the engine* (seeing germination, growth, lighting, and
branching play out in a short session), not for how a shipped game should
feel. Camera zoom (`render::config::SceneLayout::zoom`/
`zoom_visible_half_height`) no longer needs the same manual retuning as the
plant grows, since `scene::dynamic_zoom` pulls back on its own — but the
*pacing* is still a placeholder default. Revisit once there's an actual
gameplay-tuned config to compare against.

Known gaps / not yet built:

- UI controls exist: `Simulation::stats()` (a `Stats` snapshot — day
  progress, day/night, stage, height, leaf/branch counts, water level,
  temperature) backs a HUD polled every 250ms, a Water button calls
  `Simulation::water`, an
  auto-water checkbox calls `Simulation::set_auto_water` (models a
  self-watering/wicking pot maintaining a moisture floor — see
  `Soil::apply_auto_water`), and a range slider calls
  `Simulation::set_time_scale` (clamped via
  `TimeConfig::clamp_speed_multiplier`, sanitizing non-finite input rather
  than passing it through — see `sim/config.rs`).
- No persistence — reloading the page starts a fresh seed.
- Cotyledons are a fixed pose (no heliotropism/nyctinasty) and never shed,
  a deliberate scope simplification (see `scene.rs`).
- Pacing (`sim::config::TimeConfig`, currently a 5-real-second day, tuned
  for validating the engine quickly — see above) is still a placeholder
  default, not tuned for how long an actual play session should feel.
- Three growth habits exist (Dracaena-style caning/crown-branching,
  Peace-Lily-style basal rosette, Pothos-style aerial-root climbing); all
  share every other biological parameter (photosynthesis, transpiration,
  thickening) largely unchanged, varying mainly the elongation/plastochron/
  branching/trellis numbers — a habit with a genuinely different water-use
  profile is a natural follow-on.
- Only the aerial-root climbing strategy (Pothos/Monstera-style) is
  modeled. Twining vines (morning glory, wisteria — the whole stem spirals
  around a support) and tendril-bearers (peas, grapes — a separate coiling
  appendage while the main stem stays straight) are real, different
  climbing strategies not yet built; a natural follow-on species rather
  than a retrofit of Pothos's own mechanism.
- Aerial roots and the trellis mechanism apply to the main stem only, not
  branches — a deliberate scope simplification (Pothos's own lateral shoots
  typically hang free off an already-anchored vine rather than
  independently re-rooting).
- Leaf self-shading (`sim::plant::self_shading_factors`) only accounts for
  a leaf's position/age along its own grower (main stem or branch), not
  which side (`Side::Left`/`Right`) it's on — real phyllotaxis spreads new
  leaves in a spiral specifically so they don't stack directly over older
  ones on the *same* side, so an opposite-side leaf currently gets shaded
  as much as a same-side one, which isn't quite accurate. A same-side-
  weighted refinement is a cheap, 2D-compatible follow-on, not something
  that needs a 3D model.
- Transpiration is deliberately *not* temperature-scaled yet (real
  transpiration rises with heat via vapor pressure deficit) — see
  `step_vegetative`'s comment on why that's a scope decision, not an
  oversight: the water balance was only just recalibrated after a real
  crash-to-bone-dry bug, and layering in another multiplicative factor
  deserves its own dedicated pass, not an incidental addition here.

## Non-obvious design points

- **Why sim/ and render/ are split by compile target**: `sim/` has to stay
  plain, dependency-free Rust so `cargo test` (which runs natively, not on
  `wasm32-unknown-unknown`) can exercise plant-growth/lighting logic
  directly. `render/` depends on `wgpu`/`web-sys`/`wasm-bindgen`, which are
  gated to the `wasm32` target in `Cargo.toml` so the native test build
  never has to link them.
- **Why usvg/lyon are build-dependencies, not dependencies**: tessellation
  happens once, at build time, on the host — the shipped wasm binary only
  ever sees plain `&[f32]`/`&[u16]` mesh data, not an SVG parser.
- **SVG anchor correction**: usvg bakes the viewBox-to-viewport transform
  into every path's coordinates, which would otherwise silently shift each
  asset's anchor depending on its viewBox bounds. `build.rs` cancels this
  out per file (using the first shape's transform — see assets/svg/
  README.md rule 6 on why files must stay flat, no nested transforms), so
  "path coordinate `(0, 0)` is the anchor" holds regardless of viewBox.
- **basePath**: `NEXT_PUBLIC_BASE_PATH` (set in `deploy.yml`, `/plant` —
  confirmed against the actual repo, `github.com/msartintarm/plant`) prefixes
  any raw absolute path built by hand, like the wasm-pkg import in
  `EngineCanvas.tsx`. `next/link`/`next/image` handle it automatically;
  hand-built paths don't.
- **Playwright's dev port**: `playwright.config.ts` runs its own server on
  `:3100`, not the default `:3000` — avoids silently attaching to an
  unrelated project's dev server already running on this machine.
- **Config is data, not constants**: every tunable number (biology in
  `sim::config`, layout/pacing in `render::config`) lives in a plain
  `Default`-derived struct passed into the logic that uses it, rather than
  as module-level `const`s. `Plant::step`/`Soil::update` etc. are pure
  functions *of* a config, not hardcoded to one — tests construct their own
  where they need something other than the default.
- **`Plant::last_decision`**: every `step` records its own inputs (light,
  water factor, leaf area, carbon before) and outputs (photosynthesis,
  elongation, whether it was carbon-limited, whether a leaf spawned,
  movement targets) as a `Decision` enum. Cheap single-tick tests assert
  directly on this instead of running thousands of ticks and inferring the
  mechanism from the aggregate result — see `sim::plant`'s
  `single_tick_*`/`regression_*` tests.
- **Manual clip-space math must redo aspect correction by hand**: most
  transforms just hand `scale_x`/`scale_y` to `InstanceUniform`, which
  divides x by aspect for them. `scene::sky_object_transform` computes its
  offset *by hand* (the sun/moon's position inside the window, in local
  pane units) — this was a real bug: without also dividing that local-x
  term by aspect, the disc drifted outside the window frame on the actual
  (non-square) canvas, even though the math looked right against a mental
  model of a square canvas.
- **Light shown "in context," not as a HUD gauge**: the sun/moon's position
  inside the window shows its angle; tinting the wall/window (never fully
  black — a moonlit-room floor, not a blackout) shows its intensity/color.
  Deliberately not applied to the plant itself, so its own reactions
  (droop, fold, lean) stay legible at any time of day.
- **Leaf initiation used to make branches (nearly) impossible, and vice
  versa — now it's a plastochron, not a carbon race**: the original design
  paused *all* main-stem leaf initiation once a stem became branch-eligible,
  so cheap leaves couldn't perpetually outcompete a branch's higher carbon
  threshold. That worked for funding branches, but meant a stem could grow
  many multiples of its own branching height with only the single leaf it
  had *before* crossing the threshold — which read, quite literally, as
  "leaves only ever grow at the base" (a real player-reported bug). Fixed by
  making leaf initiation height-gated instead (a plastochron — see
  `PlantConfig::plastochron_height_interval`): a leaf becomes due every fixed
  amount of stem elongation, independent of whether a branch is also being
  funded, so new nodes keep appearing as the stem actually grows. Branch
  funding stays its own, separately-gated, rarer event layered on top.
- **...and that still opened the door to a second, subtler starvation bug
  between siblings**: naively processing "the main stem, then each branch in
  creation order, each fully catching up its own leaf backlog before moving
  to the next" let whichever grower ran first monopolize 100% of a tick's
  carbon on itself. Over a long enough session this reliably left the
  *newest* branch completely leafless forever, no matter how tall it grew —
  found by simulating a 10-real-minute session and printing each branch's
  leaf count, not by eyeballing a render. A naive "give everyone one leaf
  per round, round-robin" fix wasn't enough either: with a *fixed* starting
  slot each tick, the last slot in iteration order still lost whenever
  carbon ran out mid-round, every single tick. The real fix
  (`Plant::spawn_due_leaves_fairly`) persists the round-robin's "whose turn
  is next" pointer *across* ticks, not just within one — most ticks only
  afford a handful of leaf-spawns total, so a rotation reset every call
  never advanced past its first slot in practice.
- **Why respiration scales with leaf area, not stem height**: it used to
  include a direct `+ height` term. Once branches could pause main-stem leaf
  production (superseded by the plastochron rewrite above, but the
  respiration fix itself is still load-bearing), that created a runaway
  feedback — a stem kept growing taller (and so more respiration-expensive)
  while its leaf area, and therefore its income, stayed flat, making it
  progressively *less* able to ever afford a branch the taller it got.
  Respiration now scales with living/metabolically-active tissue (leaf area)
  instead, which is both more realistic (a bare cane's upkeep is cheap; a
  leafy crown's isn't) and avoids the feedback loop entirely.
- **Transpiration used to make soil crash to bone dry within about a real
  minute of play**: `transpiration_coeff` was tuned so a plant with even a
  couple of leaves drew water 15-40x faster than bare-soil evaporation alone
  — real evapotranspiration for an established leafy houseplant is more
  like 2-5x. Recalibrated (see that field's doc comment) and
  `thickening_rate_coeff` rescaled by the same factor alongside it (it
  multiplies the same `uptake_rate`, so it has to move with it to keep the
  pipe-model's *pacing* unchanged). A pinned test
  (`transpiration_stays_within_a_realistic_multiple_of_bare_soil_evaporation`)
  guards the ratio directly, independent of any simulated session. An
  auto-water toggle (`Simulation::set_auto_water`/`Soil::apply_auto_water`,
  modeling a self-watering/wicking pot's moisture floor) also now exists so
  a fast-growing plant's ever-increasing water draw doesn't require
  babysitting the Water button.
- **The rendering-side twin of the "leaves only at the base" bug**: even
  after the simulation correctly recorded higher and higher `attach_height`s
  per leaf, they still *rendered* clustered right at the pot. Cause: the
  stem mesh (`stem_segment.svg`) has a real local vertical extent (60 units,
  anchor at y=0 to its tip at y=-60) that `stem_like_transform`'s `scale_y`
  correctly multiplies via the vertex shader (`world = local_position *
  scale + offset` — see `scene.wgsl`) — but the *offset* formulas placing a
  leaf or branch along that same stem (`attach_height * stem_height_scale`)
  never accounted for that same 60x mesh-extent factor, positioning them
  ~60x closer to the pot than the visibly-scaled stem mesh they're supposed
  to sit on. Fixed by multiplying in the same `STEM_LOCAL_HEIGHT` constant
  wherever a point along the stem is placed by hand (leaf/branch offsets,
  and `scene::dynamic_zoom`'s reach calculation, below) — the same class of
  "manual clip-space math has to redo a correction the mesh pipeline already
  applies elsewhere" mistake as the sun/moon aspect-ratio bug above, just
  with the mesh's own local extent instead of the canvas's aspect ratio.
- **Dynamic camera zoom-out**: `scene::dynamic_zoom` computes the actual
  zoom used each frame — `SceneLayout::zoom` (the closest-in the camera ever
  gets) unless the plant's tallest reach (main stem or any branch's own tip)
  would otherwise poke past `zoom_visible_half_height`, in which case it
  pulls back just enough to keep it in frame. Deliberately conservative
  (ignores how much lean/droop rotation shortens a leaning stem's true
  vertical reach) since overestimating reach can only zoom out slightly
  more than strictly necessary, never clip the plant.
- **`render::scene`/`render::config` aren't wasm32-gated, unlike the rest of
  `render`**: they're pure math over `Transform`/`SceneLayout`/`Plant`/
  `SunState`, no wgpu/web-sys dependency of their own, so there's no reason
  they couldn't compile — and be unit-tested — natively. The wgpu/
  wasm-bindgen orchestration that genuinely needs those dependencies lives
  in `render::mod::wgpu_engine`, a nested module gated on its own. This
  split is what let the sun/moon-inside-the-window-pane and
  wall-covers-the-canvas checks (both previously only verifiable by
  eyeballing a screenshot at a few aspect ratios) become plain `cargo test`
  assertions across a whole grid of aspect ratios and sun positions instead.
- **`PlantConfig::peace_lily()` scales two numbers together, not one**:
  `base_elongation_rate` and `plastochron_height_interval` are both cut by
  the same ~13x factor from the caning habit. Cutting only the elongation
  rate (to get a squat rosette) would have starved leaf production too,
  since leaves are gated by height crossing a fixed interval — the *rate*
  of new leaves depends on the *ratio* of the two, so scaling them together
  keeps a comparable leaf-production rate while the absolute height stays
  low throughout the plant's life, matching a real rosette's lack of
  internode elongation rather than just "the same plant, but stunted."
- **The flower reuses the same stem-tip math as everything else along the
  stem**: `scene::flower_transform` calls `frame_at_height` (the same
  `STEM_LOCAL_HEIGHT`-corrected curve walk — see the rendering-scale bug
  above — every leaf/branch placement also uses), just evaluated at
  `plant.height` itself rather than some leaf's `attach_height` — so it
  always sits exactly where a leaf attached at the very tip would, with no
  separate position logic of its own to keep in sync.
- **Temperature is a pure function of `day_progress`, not stateful "eased"
  climate**: real room temperature lags peak sunlight by a couple hours
  (thermal mass), which could have been modeled as a value that eases
  toward a light-driven target over time — but that requires persistent
  state threaded through every call site the same way `Soil` is. Modeling
  the lag as a fixed phase offset on the same cosine curve `sun_state`
  itself uses (`climate::climate_state`) keeps temperature exactly as cheap
  and stateless as sun position already is: `Plant::step` gained one new
  `&ClimateState` parameter (computed by the caller, same as `&SunState`),
  not a new piece of mutable simulation state to manage.
- **`neutral_climate()` exists so ~40 pre-existing tests didn't have to
  become temperature tests**: every one of `plant.rs`'s tests written
  before temperature existed calls `.step(...)` with a climate fixed at
  `PlantConfig::dracaena()`'s own optimum, so `temperature_factor`/
  `q10_factor` both evaluate to their neutral (1.0) value — each test still
  isolates whatever mechanism it was actually written to check, rather than
  every one of them incidentally also asserting "and this holds at 24°C."
  Tests that *are* about temperature construct their own `ClimateState`
  explicitly.
- **Leaf senescence/abscission reuses the same "shared free function, not
  per-branch duplication" pattern as `spawn_one_due_leaf`/
  `stem_droop_target`**: `age_and_senesce_leaves` is called once for the
  main stem's `leaves` and once per branch's own, both drawing on one
  `stress_signal` computed a single time per tick (whichever of drought or
  cold is currently worse — not summed, since a plant shedding leaves over
  one bad condition isn't shedding twice as fast just because a second
  condition also happens to be present).
- **`scene.wgsl` scales in local space *before* rotating, not after**: a
  real bug — the stem/branch mesh (non-uniform `scale`: x from radius, y
  from height) visually sheared away from its own `rotation` once lean/
  droop grew large, because non-uniform scale doesn't commute with
  rotation. Every *other* rotated mesh (leaves, the flower) uses a uniform
  scale, so this never showed up there — the flower (placed by plain trig,
  never affected) simply stopped lining up with where the sheared stem mesh
  visually pointed. Same underlying lesson as the sun/moon aspect-ratio bug
  and the `STEM_LOCAL_HEIGHT` bug above: a manual transform has to match
  the *exact* order/assumptions the mesh pipeline uses, or the two drift
  apart in ways that only become obvious at extreme values (large angles,
  tall stems) rather than immediately.
- **A stem's curve is recorded history, not computed from current state**:
  once the shear bug above was fixed, the flower lined up with the stem —
  but the *whole stem* still rotated as one rigid line from the pot,
  which isn't how a real, long-lived, still-leaning stem looks (older
  tissue keeps whatever curvature it had when it stiffened; only the
  actively-growing tip responds to *today's* lean). Fixing that properly
  needed actual history, not a smarter formula: `Plant`/`Branch` each
  record `lean_angle` into a `segment_history` every fixed height interval
  as they grow (`sim::plant::record_stem_segments`), capped at
  `MAX_STEM_SEGMENTS` (bounded memory; past the cap, the still-growing tip
  just keeps extending instead of subdividing further — a real lower trunk
  is usually the straightest, most lignified part anyway, so this reads as
  a reasonable simplification rather than a visible seam). `render::scene::
  StemCurve` walks that history — a completed segment renders at its frozen
  angle, the one segment beyond the recorded history (still growing) uses
  the *live* `lean_angle`/`stem_droop` — and every other placement (a leaf,
  a branch's own attachment point, the flower) walks the *same* curve to
  its own height rather than assuming the stem is a single straight line,
  which is what keeps them all landing exactly on it regardless of how much
  it bends. This replaced the single `stem_drawable`/`branch_drawables`
  fields with one shared instance pool (mirroring how `leaf_drawables`
  already covers the main stem's leaves and every branch's own combined),
  since a stem is now potentially many mesh instances chained end to end,
  not one.
- **Phototropism had a silent 180° sign bug, caught by direct calculation,
  not eyeballing**: `rotate_and_place`'s rotation convention bends a
  positive angle toward -x, but the window sits at +x — so the whole
  curved-stem feature had been leaning the plant *away* from the window the
  entire time it existed. Fixed by deriving a `lean_sign` from the actual
  window/pot positions (`scene::lean_sign_toward_window`) rather than
  hardcoding a direction, so it stays correct if the window is ever
  repositioned again — verified with a regression test that walks a curve
  one segment and asserts the tip moves toward the *actual* window, not an
  assumed screen direction.
- **The light beam's angle formula had the same class of bug, caught the
  same way**: an early version of `scene::light_beam_transform` derived its
  angle from `rotate_and_place`'s convention but got the sign backwards
  (the beam pointed *away* from the plant, into the window), and the first
  test written for it happened to use a matching wrong sign for the mesh's
  own local axis, so it passed anyway despite the bug being real — caught
  only once the test was rewritten to replicate `scene.wgsl`'s actual
  vertex-shader formula directly (scale, rotate, translate the mesh's own
  baked far-point) rather than routing through a helper that could drift
  out of sync with the shader's real behavior. The broader lesson: a test
  that re-derives ground truth from a *different* abstraction than the one
  it's checking can be fooled by both sharing the same wrong assumption;
  replicating the actual shader math is what makes it trustworthy.
- **Self-shading was first miswired onto transpiration too, and that
  backfired**: the first implementation discounted *both* photosynthesis
  and transpiration by the same self-shading factor, on the reasoning that
  a shaded leaf's stomata are less active either way. In practice this
  created a confounding feedback — less transpiration meant more water
  conserved in the soil, which (in a test using only a trickle top-up) more
  than compensated for the lost photosynthesis, so the plant ended up
  *taller and leafier* than before, the opposite of the intended effect.
  Caught by a direct native-test probe of leaf count over time, not a
  screenshot. Fixed by keeping transpiration on the room-position-only
  `light_weighted_leaf_area` (unchanged) and adding a separate
  `photosynthesis_leaf_area` for the self-shaded figure, used only by the
  photosynthesis term — the real mechanism (Monsi & Saeki's canopy light
  extinction) is specifically about light capture for sugar production, not
  stomatal water loss, so conflating the two modeled an effect that wasn't
  actually there.
- **Shade-driven senescence is deliberately *not* age-gated the same way
  drought/cold stress is**: `age_and_senesce_leaves` only accelerates aging
  from drought/cold once a leaf is already past `leaf_mature_lifespan`, but
  applies shade-driven pressure immediately, regardless of age. This was a
  deliberate fix, not the original design — gating shade the same way as
  age/stress let a fast-growing plant's initial burst of new leaves pile up
  for an entire `leaf_mature_lifespan` before any could be shed, which is
  exactly the unrealistic runaway-leaf-count behavior self-shading was
  built to prevent. Real shade-induced leaf drop isn't a slower version of
  ordinary old-age senescence — a leaf rapidly overtopped by vigorous new
  growth can yellow and drop within weeks, long before anything like its
  full potential lifespan, because it's already a net carbon liability from
  the moment it's buried.
- **The climbing-suppression predicate is one pure function, not duplicated
  inline**: whether a grower (main stem or a branch, using its own
  `attach_height + height`) is currently "on" its trellis and held straight,
  or leaning freely, used to be computed inline at both call sites with
  near-identical expressions. Extracted into `plant::leans_freely(height,
  trellis_height)` — a single named, independently unit-tested function
  with no `Plant`/`Soil`/`SunState` involved at all — reused by both
  `step_vegetative` and `step_branch`. `spawn_due_aerial_roots` (the aerial-
  root spawning logic) follows the same discipline from the start: every
  input explicit, no hidden state, testable with a bare `Vec` and a couple
  of `f64`s. This is the same pattern `record_stem_segments`/
  `height_light_factor`/`self_shading_factors` already established — pure
  functions of explicit config + state, injected by the caller, is what
  lets nearly this entire codebase's growth logic run under plain
  `cargo test` with no mocking anywhere.
- **The shared flower drawable is repointed to a different mesh at
  runtime, not duplicated per species**: `Drawable::mesh` is just a
  `&'static str` lookup key into `MeshRegistry`, not an owned GPU resource,
  so `render()` can set `self.flower_drawable.mesh = self.growth_config.
  plant.flower_mesh_name` every frame for near-zero cost — switching
  species at runtime (`set_species`) never needs a new buffer allocated.
  `Plant::bloom_intensity` scales the same drawable to zero size when not
  currently in bloom, so the draw call no longer needs its own separate
  `height >= flowering_height_threshold` gate — that would just duplicate
  a check `bloom_intensity` already encodes.
