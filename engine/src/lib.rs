pub mod sim;

// `render`'s wgpu/wasm-bindgen orchestration only exists (and only needs to
// compile) on wasm32 — see the target-gated dependency block in Cargo.toml
// — but `render::scene`/`render::config` are pure math/data with no wasm
// dependency of their own, so the module itself is *not* gated here; the
// gate lives inside `render/mod.rs`, scoped to just the parts that actually
// need wgpu/web-sys. That's what lets `cargo test` exercise the same
// placement geometry (sun/moon position, leaf/branch frames, wall
// coverage) a human would otherwise only be able to check by looking at a
// rendered screenshot.
mod render;

#[cfg(target_arch = "wasm32")]
pub use render::Simulation;
