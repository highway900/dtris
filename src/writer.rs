use gltf_json as json;

use std::{fs, mem};

use json::validation::Checked::Valid;
use json::validation::USize64;
use std::borrow::Cow;
use std::io::Write;

use crate::Vertex;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, clap::ValueEnum)]
pub enum Output {
    /// Output standard glTF.
    Standard,

    /// Output binary glTF.
    Binary,
}

/// Calculate bounding coordinates of a list of vertices, used for the clipping distance of the model
fn bounding_coords(points: &[Vertex]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    for point in points {
        let p = point.position;
        for i in 0..3 {
            min[i] = f32::min(min[i], p[i]);
            max[i] = f32::max(max[i], p[i]);
        }
    }
    (min, max)
}

fn align_to_multiple_of_four(n: &mut usize) {
    *n = (*n + 3) & !3;
}

fn to_padded_byte_vector<T: bytemuck::NoUninit>(data: &[T]) -> Vec<u8> {
    let byte_slice: &[u8] = bytemuck::cast_slice(data);
    let mut new_vec: Vec<u8> = byte_slice.to_owned();

    while new_vec.len() % 4 != 0 {
        new_vec.push(0); // pad to multiple of four bytes
    }

    new_vec
}

pub fn write_gltf(
    vertices: &[Vertex],
    indices: &[u32],
    output: Output,
    output_filename_base: &str,
    output_dir: &str,
) {
    if vertices.is_empty() || indices.is_empty() {
        println!(
            "Vertices or indices are empty, skipping export for {}.",
            output_filename_base
        );
        return;
    }

    let (min, max) = bounding_coords(&vertices);

    let mut root = gltf_json::Root::default();

    // --- 1. Buffer Data Construction ---
    // Vertex data comes first, then index data.
    // Both are padded to multiples of 4 bytes.
    let vertex_buffer_data = to_padded_byte_vector(vertices);
    let index_buffer_data = to_padded_byte_vector(indices); // Indices are u32, naturally 4-byte aligned

    let mut combined_buffer_data = Vec::new();
    combined_buffer_data.extend_from_slice(&vertex_buffer_data);
    let indices_buffer_offset = combined_buffer_data.len(); // Offset where index data begins
    combined_buffer_data.extend_from_slice(&index_buffer_data);

    let total_buffer_length = combined_buffer_data.len();

    let buffer_uri = if output == Output::Standard {
        Some(format!("{}.bin", output_filename_base))
    } else {
        None
    };

    // let buffer_length = vertices.len() * mem::size_of::<Vertex>();
    let buffer = root.push(json::Buffer {
        byte_length: USize64::from(total_buffer_length),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: buffer_uri,
    });

    // --- 2. BufferViews ---
    // BufferView for Vertex Data
    let vertex_buffer_view = root.push(json::buffer::View {
        buffer,
        byte_offset: Some(USize64(0)), // Starts at the beginning of the combined buffer
        byte_length: USize64::from(vertex_buffer_data.len()),
        byte_stride: Some(json::buffer::Stride(mem::size_of::<Vertex>())),
        target: Some(Valid(json::buffer::Target::ArrayBuffer)), // For vertex attributes
        name: Some("vertexBufferView".into()),
        extensions: Default::default(),
        extras: Default::default(),
    });

    // BufferView for Index Data
    let index_buffer_view = root.push(json::buffer::View {
        buffer,
        byte_offset: Some(USize64::from(indices_buffer_offset)),
        byte_length: USize64::from(index_buffer_data.len()),
        // byte_stride is not needed for tightly packed indices
        target: Some(Valid(json::buffer::Target::ElementArrayBuffer)), // For indices
        name: Some("indexBufferView".into()),
        byte_stride: None,
        extensions: Default::default(),
        extras: Default::default(),
    });

    // --- 3. Accessors ---
    // Accessor for Vertex Positions
    let positions_accessor = root.push(json::Accessor {
        buffer_view: Some(vertex_buffer_view),
        byte_offset: Some(USize64(0)), // Offset of 'position' within Vertex struct
        count: USize64::from(vertices.len()),
        component_type: Valid(json::accessor::GenericComponentType(
            json::accessor::ComponentType::F32,
        )),
        type_: Valid(json::accessor::Type::Vec3),
        min: Some(json::Value::from(Vec::from(min))),
        max: Some(json::Value::from(Vec::from(max))),
        name: Some("positions".into()),
        extensions: Default::default(),
        extras: Default::default(),
        normalized: true,
        sparse: None,
    });

    // Accessor for Vertex Colors
    // Assumes 'color' is the second field in Vertex struct
    let color_byte_offset = mem::size_of::<[f32; 3]>(); // Offset of 'color' field
    let colors_accessor = root.push(json::Accessor {
        buffer_view: Some(vertex_buffer_view),
        byte_offset: Some(USize64::from(color_byte_offset)),
        count: USize64::from(vertices.len()),
        component_type: Valid(json::accessor::GenericComponentType(
            json::accessor::ComponentType::F32,
        )),
        type_: Valid(json::accessor::Type::Vec3),
        name: Some("colors".into()),
        extensions: Default::default(),
        extras: Default::default(),
        // Min/max for colors are optional
        min: Some(json::Value::from(Vec::from(min))),
        max: Some(json::Value::from(Vec::from(max))),
        normalized: true,
        sparse: None,
    });

    // Accessor for Indices (New)
    let indices_accessor = root.push(json::Accessor {
        buffer_view: Some(index_buffer_view),
        byte_offset: Some(USize64(0)), // Starts at the beginning of its buffer view
        count: USize64::from(indices.len()),
        component_type: Valid(json::accessor::GenericComponentType(
            json::accessor::ComponentType::U32, // Assuming u32 indices
        )),
        type_: Valid(json::accessor::Type::Scalar), // Indices are scalar values
        name: Some("indices".into()),
        extensions: Default::default(),
        extras: Default::default(),
        // Min/max for colors are optional
        min: Some(json::Value::from(Vec::from(min))),
        max: Some(json::Value::from(Vec::from(max))),
        normalized: Default::default(),
        sparse: Default::default(),
    });

    // --- 4. Mesh Primitive ---
    let primitive = json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::BTreeMap::new();
            map.insert(Valid(json::mesh::Semantic::Positions), positions_accessor);
            map.insert(Valid(json::mesh::Semantic::Colors(0)), colors_accessor);
            // Add other attributes like Normals or TexCoords here
            // e.g., map.insert(Valid(json::mesh::Semantic::Normals), normals_accessor);
            map
        },
        indices: Some(indices_accessor), // <-- Link to the index accessor
        mode: Valid(json::mesh::Mode::Triangles),
        extensions: Default::default(),
        extras: Default::default(),
        material: Default::default(),
        targets: Default::default(),
    };

    let mesh = root.push(json::Mesh {
        primitives: vec![primitive],
        name: Some(format!("{}_mesh", output_filename_base).into()),
        extensions: Default::default(),
        extras: Default::default(),
        weights: None,
    });

    let node = root.push(json::Node {
        mesh: Some(mesh),
        name: Some(format!("{}_node", output_filename_base).into()),
        ..Default::default()
    });

    let scene = root.push(json::Scene {
        nodes: vec![node],
        name: Some(format!("{}_scene", output_filename_base).into()),
        extensions: Default::default(),
        extras: Default::default(),
    });
    root.scene = Some(scene); // Set this as the default scene

    match output {
        Output::Standard => {
            let _ = fs::create_dir(format!("{output_dir}"));

            let writer = fs::File::create(format!("{output_dir}/{output_filename_base}.gltf"))
                .expect("I/O error");
            json::serialize::to_writer_pretty(writer, &root).expect("Serialization error");

            // Write the combined buffer data (vertices + indices)
            let mut writer = fs::File::create(format!("{output_dir}/{output_filename_base}.bin"))
                .expect("I/O error");
            writer
                .write_all(&combined_buffer_data)
                .expect("I/O error writing bin file");
        }
        Output::Binary => {
            let json_string = json::serialize::to_string(&root).expect("Serialization error");
            let mut json_offset = json_string.len();
            align_to_multiple_of_four(&mut json_offset);
            let glb = gltf::binary::Glb {
                header: gltf::binary::Header {
                    magic: *b"glTF",
                    version: 2,
                    // N.B., the size of binary glTF file is limited to range of `u32`.
                    length: (json_offset + total_buffer_length)
                        .try_into()
                        .expect("file size exceeds binary glTF limit"),
                },
                // Use the combined buffer data for the binary chunk
                bin: Some(Cow::Borrowed(&combined_buffer_data)),
                json: Cow::Owned(json_string.into_bytes()),
            };
            let writer = std::fs::File::create(format!("{output_dir}/{output_filename_base}.glb"))
                .expect("I/O error");
            glb.to_writer(writer).expect("glTF binary output error");
        }
    }
}
