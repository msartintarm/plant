# Asset authoring convention

Every `*.svg` file in this folder is one mesh. `build.rs` tessellates it into
a static GPU mesh, keyed by its filename stem (`leaf.svg` → mesh id
`"leaf"`). **To replace a placeholder with real art, overwrite the file at
its existing path, following the rules below — no Rust code changes are
needed.** Rust code only needs to change when adding a brand-new mesh id or
changing which meshes get drawn, never when re-authoring an existing one's
appearance.

## Rules

1. **`viewBox` doesn't matter to the pipeline at all** — pick whatever's
   convenient to draw in (or skip it and just size your canvas). usvg bakes
   the viewBox-to-viewport transform into every path's coordinates, which
   would otherwise silently shift each asset's anchor around depending on
   viewBox bounds — `build.rs` cancels that out automatically (see rule 2),
   so viewBox is purely a drawing-canvas convenience, not part of the
   contract.

2. **The origin `(0, 0)` — in path coordinates, as you drew them — is the
   mesh's anchor/pivot point** — this is the whole convention. Draw your
   shape positioned around `(0, 0)` wherever you want that asset's
   attachment point to be:
   - `stem_segment` — anchor at the bottom-center (segments stack tip-to-
     base by chaining anchors).
   - `leaf` — anchor at the base of the blade (its petiole/attachment
     point), not its center.
   - `pot` — anchor at the top-center of the rim (the stem's base attaches
     there).
   - `sun` / `moon` / `wall` / `window_frame` — anchor at center is simplest,
     but doesn't matter as long as the Rust-side placement code (see
     `render/scene.rs`) agrees with wherever you put it.

3. **Draw normally, y-down, like any SVG** — the pipeline flips Y when
   baking (SVG is y-down; render space is y-up), so don't do anything
   special for that.

4. **No auto-normalization — 1 baked local unit is 1 raw SVG user unit,
   as drawn.** This pipeline doesn't rescale by your viewBox size, so pick
   whatever's convenient (e.g. draw a pot roughly 60 units wide). Each asset
   gets its own Rust-side `scale` at the point it's drawn (see
   `render/scene.rs`), which is what actually fits it into the scene —
   different assets' raw unit systems never need to agree with each other.

5. **Solid fills only, for now** — `fill="#rrggbb"` on `<path>`/`<rect>`/
   `<circle>`/`<polygon>`/etc. (usvg converts basic shapes to paths
   automatically). Gradients, patterns, and strokes aren't tessellated yet;
   an element with an unsupported paint bakes to flagged magenta
   (`#ff00ff`) so it's obvious at a glance in-engine rather than silently
   wrong. A single file can combine multiple flat-colored shapes (e.g. a pot
   body plus a darker rim shape) — they all bake into one combined mesh.

6. **Keep each file flat — no nested `<g transform="...">` groups, no
   per-shape `transform=` attributes.** The anchor correction in rule 2 is
   computed once per file (from whichever shape happens to be first) and
   applied to every shape in it; that's only correct if every top-level
   shape shares the same transform, which a flat file guarantees. Position
   everything directly in path coordinates instead.

## Adding a new asset (not just replacing one)

Drop a new `<name>.svg` file in this folder — `build.rs` picks up any
`*.svg` file automatically, no build script changes needed. You do still
need to reference `"<name>"` from Rust wherever something should actually
draw it (see `MeshRegistry::get` in `render/meshes.rs`), since *where* and
*whether* a mesh appears in the scene is code, not art.
