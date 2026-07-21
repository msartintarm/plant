//! Static GPU meshes baked from `assets/svg/*.svg` by `build.rs` — see
//! `assets/svg/README.md` for the authoring convention. `GENERATED_MESHES`
//! (below, via `include!`) is `&[(name, vertices, indices)]`; `MeshRegistry`
//! turns that into real GPU buffers once at startup.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

include!(concat!(env!("OUT_DIR"), "/meshes_generated.rs"));

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

pub struct MeshRegistry {
    meshes: HashMap<&'static str, Mesh>,
    /// Max |x| and max |y| (independently, not a combined radius) among a
    /// mesh's own baked vertices — computed once here (not per-frame) so
    /// `scene::outline_uniform` can turn a target on-screen pixel margin
    /// into the right amount of extra `scale` for *that* mesh's own native
    /// SVG units, which differ freely from one asset to the next (see that
    /// function's doc comment). Kept per-axis rather than a single combined
    /// radius specifically because of `stem_segment.svg`: it's long and
    /// thin (its local y-extent, running the segment's whole height, is
    /// roughly 10x its local x-extent, just the stem's thickness) — a
    /// shared radius dominated by the long axis made the margin along the
    /// *short* axis (the stem's actual visible sides) come out far too
    /// small, so its outline read thin/faint next to a leaf's (whose local
    /// extent is much closer to square in both axes, so the same bug was
    /// invisible there).
    half_extents: HashMap<&'static str, (f32, f32)>,
}

impl MeshRegistry {
    pub fn load_all(device: &wgpu::Device) -> Self {
        let meshes = GENERATED_MESHES
            .iter()
            .map(|(name, vertices, indices)| {
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let mesh = Mesh {
                    vertex_buffer,
                    index_buffer,
                    index_count: indices.len() as u32,
                };
                (*name, mesh)
            })
            .collect();
        let half_extents = GENERATED_MESHES
            .iter()
            .map(|(name, vertices, _)| {
                let half_extent = vertices.iter().fold((0.0f32, 0.0f32), |(mx, my), v| {
                    (mx.max(v.position[0].abs()), my.max(v.position[1].abs()))
                });
                (*name, half_extent)
            })
            .collect();
        MeshRegistry { meshes, half_extents }
    }

    /// Panics on an unknown name — every name drawn from `scene.rs` should
    /// correspond to a real file in `assets/svg/`; a mismatch is a code bug,
    /// not a recoverable runtime condition.
    pub fn get(&self, name: &str) -> &Mesh {
        self.meshes
            .get(name)
            .unwrap_or_else(|| panic!("no baked mesh named {name:?} — check engine/assets/svg/"))
    }

    /// See the `half_extents` field doc. Panics on an unknown name for the
    /// same reason `get` does.
    pub fn local_half_extent(&self, name: &str) -> (f32, f32) {
        *self
            .half_extents
            .get(name)
            .unwrap_or_else(|| panic!("no baked mesh named {name:?} — check engine/assets/svg/"))
    }
}
