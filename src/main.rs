mod image_feature_detection;
mod writer;

use clap::Parser;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage, imageops::FilterType};
use rand::Rng;
use spade::{DelaunayTriangulation, HasPosition, Point2, Triangulation};
use std::{collections::HashMap, f32::consts::PI};

use writer::{Output, write_gltf}; // Import necessary spade types

// The Vertex struct as provided
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

// --- Function to Sample Points from Image ---

/// Samples random points from a PNG image.
///
/// Args:
///     img: Image DynamicImage.
///     num_samples: The number of random points to sample.
///
/// Returns:
///     A Result containing a Vec of Vertex structs, where each vertex
///     has a position corresponding to the sampled pixel coordinates (x, y, 0.0)
///     and the color of that pixel, or an error if image loading/processing fails.
fn sample_image_points(img: &DynamicImage, num_samples: u32) -> Result<Vec<Vertex>> {
    // 2. Ensure it's RGB for easy color access
    let rgb_img: RgbImage = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    println!("Image dimensions: {}x{}", width, height);

    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!("Image has zero width or height"));
    }

    // 3. Prepare for sampling
    let mut sampled_vertices = Vec::with_capacity(num_samples as usize);
    let mut rng = rand::rng();

    println!("Sampling {} points...", num_samples);
    // 4. Sample random points
    for _ in 0..num_samples {
        let x = rng.random_range(0..width);
        let y = rng.random_range(0..height);

        // Get pixel color - get_pixel returns Rgb<u8>
        let pixel = rgb_img.get_pixel(x, y);
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        // Create vertex: Position is (x, y, 0), Color is (r, g, b)
        sampled_vertices.push(Vertex {
            // Use pixel coordinates directly for spade, Z=0 for 2D
            position: [x as f32, y as f32, 0.0],
            color: [r, g, b],
        });
    }

    println!(
        "Successfully sampled {} random points.",
        sampled_vertices.len()
    );
    Ok(sampled_vertices)
}

/// Samples points from an image in a grid pattern.
///
/// # Arguments
/// * `img` - The input dynamic image.
/// * `spacing` - The distance between sampled points in pixels.
///
/// # Returns
/// A vector of `Vertex` where each vertex
/// has a position corresponding to the sampled pixel coordinates (x, y, 0.0)
/// and the color of that pixel, or an error if image loading/processing fails.
fn sample_image_grid(img: &DynamicImage, spacing: u32) -> Result<Vec<Vertex>> {
    let rgb_img: RgbImage = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    println!("Image dimensions: {}x{}\n", width, height);

    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!("Image has zero width or height"));
    }

    let mut sampled_vertices = Vec::new(); // No initial capacity needed, we'll push as we go

    println!("Sampling points in a grid with spacing {}...", spacing);

    // Iterate through the grid
    for y in (0..height).step_by(spacing as usize) {
        for x in (0..width).step_by(spacing as usize) {
            // Get pixel color - get_pixel returns Rgb<u8>
            let pixel = rgb_img.get_pixel(x, y);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // Create vertex: Position is (x, y, 0), Color is (r, g, b)
            sampled_vertices.push(Vertex {
                // Use pixel coordinates directly for spade, Z=0 for 2D
                position: [x as f32, y as f32, 0.0],
                color: [r, g, b],
            });
        }
    }

    println!(
        "Successfully sampled {} points in a grid.",
        sampled_vertices.len()
    );
    Ok(sampled_vertices)
}

fn sample_from_image_to_image(
    img: &DynamicImage,
    img_to_sample: &DynamicImage,
    threshold: u8,
) -> Result<Vec<Vertex>> {
    // Convert to grayscale for thresholding
    let gray_img = img.to_luma8();
    let (width, height) = gray_img.dimensions();
    println!("Image dimensions: {}x{}", width, height);

    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!("Image has zero width or height"));
    }

    let mut sampled_vertices = Vec::new();

    println!("Sampling points with grayscale value > {}...", threshold);

    // Iterate through all pixels
    for y in 0..height {
        for x in 0..width {
            let pixel_value = gray_img.get_pixel(x, y)[0];

            // Check if pixel value is above the threshold
            if pixel_value > threshold {
                let pixel = img_to_sample.get_pixel(x, y);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;

                // Create vertex
                sampled_vertices.push(Vertex {
                    position: [x as f32, y as f32, 0.0],
                    color: [r, g, b],
                });
            }
        }
    }

    println!(
        "Successfully sampled {} points above threshold.",
        sampled_vertices.len()
    );
    Ok(sampled_vertices)
}

/// Samples points from an image where pixel grayscale value is above a threshold.
///
/// Args:
///     img: Image DynamicImage.
///     threshold: The grayscale threshold (0-255). Pixels with a value
///                strictly greater than this threshold are sampled.
///
/// Returns:
///     A Result containing a Vec of Vertex structs for pixels above the threshold,
///     where each vertex has a position corresponding to the pixel coordinates (x, y, 0.0)
///     and the color of that pixel (mapped from grayscale to RGB),
///     or an error if image processing fails.
#[allow(dead_code)]
fn sample_image_points_threshold(img: &DynamicImage, threshold: u8) -> Result<Vec<Vertex>> {
    // Convert to grayscale for thresholding
    let gray_img = img.to_luma8();
    let (width, height) = gray_img.dimensions();
    println!("Image dimensions: {}x{}", width, height);

    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!("Image has zero width or height"));
    }

    let mut sampled_vertices = Vec::new();

    println!("Sampling points with grayscale value > {}...", threshold);

    // Iterate through all pixels
    for y in 0..height {
        for x in 0..width {
            let pixel_value = gray_img.get_pixel(x, y)[0];

            // Check if pixel value is above the threshold
            if pixel_value > threshold {
                let pixel = img.get_pixel(x, y);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;

                // Create vertex
                sampled_vertices.push(Vertex {
                    position: [x as f32, y as f32, 0.0],
                    color: [r, g, b],
                });
            }
        }
    }

    println!(
        "Successfully sampled {} points above threshold.",
        sampled_vertices.len()
    );
    Ok(sampled_vertices)
}

// --- Data structure to hold extra info during triangulation ---
// spade's vertices can store associated data. We use this to keep track
// of the original color, as spade might deduplicate points.
#[derive(Clone, Debug, Copy)]
struct PointData {
    color: [f32; 3],
}

#[derive(Debug, Copy, Clone)]
struct PointWithData {
    position: Point2<f32>,
    data: PointData,
}

impl PointWithData {
    const fn new(x: f32, y: f32, data: PointData) -> Self {
        Self {
            position: Point2::new(x, y),
            data,
        }
    }
}

impl HasPosition for PointWithData {
    type Scalar = f32;

    fn position(&self) -> Point2<Self::Scalar> {
        self.position
    }
}

// --- Function to Perform Delaunay Triangulation ---

/// Performs Delaunay triangulation on a set of 2D points (extracted from vertices).
///
/// Args:
///     input_vertices: A slice of Vertex structs containing the points to triangulate.
///                     Only the x and y components of the position are used.
///
/// Returns:
///     A Result containing a tuple:
///     - Vec<Vertex>: The final list of unique vertices used in the triangulation.
///                    Their positions might be slightly adjusted by spade, and
///                    their order will likely differ from the input.
///     - Vec<u32>:    A list of indices into the returned Vertex vector,
///                    defining the triangles (each group of 3 indices is one triangle).
///     Or an error if triangulation fails.
fn create_delaunay_mesh(input_vertices: &[Vertex]) -> Result<(Vec<Vertex>, Vec<u32>)> {
    if input_vertices.is_empty() {
        println!("No input vertices provided for triangulation.");
        return Ok((Vec::new(), Vec::new())); // Return empty mesh
    }

    println!("Preparing points for triangulation...");
    // 1. Convert input Vertices into spade's Point2 format, associating color data.
    let points_with_data: Vec<PointWithData> = input_vertices
        .iter()
        .map(|v| {
            PointWithData::new(
                v.position[0],
                v.position[1],                // Use x, y for 2D point
                PointData { color: v.color }, // Store color
            )
        })
        .collect();

    // 2. Perform the Delaunay triangulation using bulk_load for efficiency
    println!(
        "Performing Delaunay triangulation on {} points...",
        points_with_data.len()
    );
    let delaunay = DelaunayTriangulation::<PointWithData>::bulk_load(points_with_data) // The vertex type is inferred here
        .context("Spade triangulation failed. Are points collinear?")?;

    println!(
        "Triangulation complete. Found {} vertices and {} triangles.",
        delaunay.num_vertices(),
        delaunay.num_inner_faces() // Use inner faces for standard triangles
    );

    // 3. Prepare output buffers
    let num_final_vertices = delaunay.num_vertices();
    let mut final_vertices: Vec<Vertex> = Vec::with_capacity(num_final_vertices);
    // Estimate triangle count (3 indices per triangle)
    let mut indices: Vec<u32> = Vec::with_capacity(delaunay.num_inner_faces() * 3);

    // 4. Map spade's internal vertex handles to indices in our final_vertices buffer
    // This is crucial because spade might reorder or deduplicate vertices.
    let mut vertex_handle_to_index: HashMap<_, u32> = HashMap::with_capacity(num_final_vertices);

    println!("Extracting final vertices...");
    for (i, v_handle) in delaunay.vertices().enumerate() {
        let pos = v_handle.position(); // spade's Point2<f32>
        let data = v_handle.data(); // Our PointData struct

        // Create the final Vertex using spade's position and our stored color
        let final_vertex = Vertex {
            position: [pos.x, pos.y, 0.0], // Keep Z = 0
            color: data.data.color,
        };
        final_vertices.push(final_vertex);

        // Store the mapping from spade's handle to our new index
        vertex_handle_to_index.insert(v_handle, i as u32);
    }

    println!("Extracting triangle indices...");
    // 5. Iterate through the faces (triangles) of the triangulation
    for face in delaunay.inner_faces() {
        // Use inner_faces to avoid outer hull issues
        let [h1, h2, h3] = face.vertices(); // Get the three vertex handles for the triangle

        // Look up the index in our final_vertices buffer for each handle
        if let (Some(i1), Some(i2), Some(i3)) = (
            vertex_handle_to_index.get(&h1),
            vertex_handle_to_index.get(&h2),
            vertex_handle_to_index.get(&h3),
        ) {
            // Add the indices to the index buffer
            indices.push(*i1);
            indices.push(*i2);
            indices.push(*i3);
        } else {
            // This shouldn't happen if the map was built correctly
            eprintln!("Warning: Could not find index for a vertex handle in a triangle.");
        }
    }

    println!(
        "Mesh generation complete. Final vertices: {}, Indices: {} ({} triangles)",
        final_vertices.len(),
        indices.len(),
        indices.len() / 3
    );
    Ok((final_vertices, indices))
}

/// Samples the four corner points of the input image.
///
/// Args:
///     img: Image DynamicImage.
///
/// Returns:
///     A Result containing a Vec of Vertex structs representing the four corners,
///     or an error if image processing fails.
fn sample_image_edges(img: &DynamicImage) -> Result<Vec<Vertex>> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    println!("Image dimensions: {}x{}", width, height);

    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!("Image has zero width or height"));
    }

    let mut sampled_vertices = Vec::with_capacity(4);

    // Define the four corners
    let corners = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ];

    println!("Sampling image corners...");

    for (x, y) in corners.iter() {
        let pixel = rgb_img.get_pixel(*x, *y);
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        sampled_vertices.push(Vertex {
            position: [*x as f32, *y as f32, 0.0],
            color: [r, g, b],
        });
    }

    println!("Successfully sampled {} corners.", sampled_vertices.len());
    Ok(sampled_vertices)
}

fn remap(input_pixel: u8, black_point: u8, white_point: u8) -> u8 {
    // Handle edge cases where black_point >= white_point (invalid input).  Return original pixel in this case to avoid division by zero and unexpected results.
    if black_point >= white_point {
        return input_pixel;
    }

    let input_range = white_point as f32 - black_point as f32;
    let pixel_relative_to_range = (input_pixel as f32 - black_point as f32) / input_range; // Normalize to 0-1 range.

    // Scale and shift to the output range (0-255).
    (pixel_relative_to_range * 255.0) as u8
}

fn adjust_levels(img: &DynamicImage, black_point: u8, white_point: u8) -> DynamicImage {
    let mut new_img = RgbImage::new(img.width(), img.height());

    for y in 0..img.height() {
        for x in 0..img.width() {
            let pixel = img.get_pixel(x, y);
            let px = [
                remap(pixel.0[0], black_point, white_point), // Red channel
                remap(pixel.0[1], black_point, white_point), // Green channel
                remap(pixel.0[2], black_point, white_point), // Blue channel
            ];
            new_img.put_pixel(x, y, Rgb(px));
        }
    }
    DynamicImage::ImageRgb8(new_img)
}

/// Flips the Z-axis of a slice of vertices by negating the Z component.
fn flip_z_axis(vertices: &mut [Vertex]) {
    for vertex in vertices.iter_mut() {
        vertex.position[2] *= -1.0;
    }
}

/// Adds duplicate vertices for points on the left edge of the image,
/// positioned at the right edge, to close the seam in spherical remapping.
///
/// # Arguments
/// * `vertices` - A mutable vector of `Vertex` to process.
/// * `width` - The width of the original 2D rectangle (image).
fn add_horizontal_seam_duplicates(vertices: &mut Vec<Vertex>, width: u32) {
    let mut duplicates = Vec::new();
    for vertex in vertices.iter() {
        // Check if the vertex is on the left edge (x-coordinate is 0)
        if vertex.position[0] == 0.0 {
            let mut duplicate_vertex = vertex.clone();
            // Set the x-coordinate of the duplicate to the right edge (width - 1)
            duplicate_vertex.position[0] = (width - 1) as f32;
            duplicates.push(duplicate_vertex);
        }
    }
    // Add the collected duplicates to the original vector
    vertices.extend(duplicates);
}

/// Remaps a 2D point from a rectangle to a point on a sphere.
///
/// The 2D point is assumed to be within the bounds [0, width-1] for x
/// and [0, height-1] for y. It is mapped to spherical coordinates (theta, phi)
/// and then converted to Cartesian coordinates (x, y, z) on a sphere of the
/// given scale (radius).
///
/// Args:
///     vertex: The input Vertex with a 2D position [x, y, 0.0].
///     width: The width of the original 2D rectangle (image).
///     height: The height of the original 2D rectangle (image).
///     scale: The radius of the target sphere.
///
/// Returns:
///     A new Vertex with its position remapped to the sphere's surface.
fn remap_rectangle_to_sphere(vertex: Vertex, width: f32, height: f32, scale: f32) -> Vertex {
    let x_2d = vertex.position[0];
    let y_2d = vertex.position[1];

    // Normalize coordinates to [0, 1]. Handle potential division by zero if width/height is 1.
    // If width or height is 1, map all points to 0.0 along that axis.
    let u = if width > 1.0 {
        x_2d / (width - 1.0)
    } else {
        0.0
    }; // Ranges from 0 to 1
    let v = if height > 1.0 {
        y_2d / (height - 1.0)
    } else {
        0.0
    }; // Ranges from 0 to 1

    // Map to spherical coordinates:
    // theta (longitude): maps u from [0, 1] to [0, 2*PI]
    let theta = u * 2.0 * PI;
    // phi (colatitude): maps v from [0, 1] to [0, PI]
    // This maps the top edge (v=0) to the North Pole (phi=0) and the bottom edge (v=1) to the South Pole (phi=PI).
    let phi = v * PI;

    // Convert spherical (scale, theta, phi) to Cartesian (x, y, z)
    // Using standard physics convention where phi is angle from positive Z-axis,
    // theta is angle from positive X-axis in XY plane.
    // If we want Y to be the 'up' axis (common in graphics), we adjust:
    // Y = r * cos(phi) (where phi is angle from +Y axis)
    // X = r * sin(phi) * cos(theta)
    // Z = r * sin(phi) * sin(theta)
    let x_3d = scale * phi.sin() * theta.cos();
    let y_3d = scale * phi.cos(); // Y is 'up'
    let z_3d = scale * phi.sin() * theta.sin();

    // Create new vertex with remapped position and original color
    Vertex {
        position: [x_3d, y_3d, z_3d],
        color: vertex.color,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SamplingMethod {
    /// Randomly samples points from the image.
    Random,
    /// Samples points from the image in a grid pattern.
    Grid,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input image file path
    #[arg(short, long)]
    image_file: String,
    #[arg(short, long, default_value = "grid")]
    sample: SamplingMethod,
    /// Spacing for grid sampling (ignored for random sampling)
    #[arg(long, default_value = "50")]
    grid_spacing: u32,
    /// Number of samples for random sampling
    #[arg(long, default_value = "3000")]
    detail_samples: u32,
    /// Write a GLTF or GLB
    #[arg(long, default_value = "standard")]
    output_type: Output,
    /// Scales the output on write
    #[arg(long, default_value = "200.0")]
    output_scale: f32,
    /// Image will be scaled down
    #[arg(long, default_value = "0.5")]
    input_img_scale: f32,
    /// Black level adjustment
    #[arg(long, default_value = "64")]
    black_level: u8,
    /// White level adjustment
    #[arg(long, default_value = "255")]
    white_level: u8,
    /// Edge threshold, all values above this grey scale value will be sampled
    #[arg(long, default_value = "2")]
    grey_threshold: u8,
    /// Output directory
    #[arg(long, default_value = "output")]
    output_dir: String,
    /// Debug info
    #[arg(short, long)]
    debug_info: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let image_file = args.image_file.clone();
    let detail_samples = args.detail_samples.clone();
    let sphere_scale = args.output_scale;
    let input_img_scale = args.input_img_scale;

    println!("Loading image from: {:?}", image_file);
    let img_base = image::open(image_file.clone())
        .with_context(|| format!("Failed to open image: {:?}", image_file))?;
    let img = if input_img_scale != 1.0 {
        img_base.resize(
            (img_base.width() as f32 * input_img_scale) as u32,
            (img_base.height() as f32 * input_img_scale) as u32,
            FilterType::CatmullRom,
        )
    } else {
        img_base
    };
    let img = adjust_levels(&img, args.black_level, args.white_level);

    let kernel_edge_detection = [-1.0_f32, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];
    let edge_img = img.filter3x3(&kernel_edge_detection);
    edge_img
        .save("edge.png")
        .expect("Could not save detail image");
    let mut sampled_vertices = sample_from_image_to_image(&edge_img, &img, args.grey_threshold)?;

    let img_sampled_vertices = match args.sample {
        SamplingMethod::Random => {
            let corner_verts = sample_image_edges(&img)?;
            sampled_vertices.extend(corner_verts);

            sample_image_points(&img, detail_samples)?
        }
        SamplingMethod::Grid => sample_image_grid(&img, args.grid_spacing)?,
    };

    sampled_vertices.extend(img_sampled_vertices);

    // Add duplicate vertices for the horizontal seam to close the sphere for spherical mapping
    let img_width = img.width();
    add_horizontal_seam_duplicates(&mut sampled_vertices, img_width);

    // Handle case where no vertices were sampled (e.g., 0 samples requested or empty image)
    if sampled_vertices.is_empty() {
        println!("No vertices were sampled from the image (including corners). Exiting.");
        return Ok(());
    }

    // 2. Create the Delaunay mesh
    let (mut mesh_vertices, mesh_indices) = create_delaunay_mesh(&sampled_vertices)?;

    flip_z_axis(&mut mesh_vertices);

    println!(
        "Remapping {} mesh vertices to sphere...",
        mesh_vertices.len()
    );
    let remapped_sphere_vertices: Vec<Vertex> = mesh_vertices
        .iter()
        .map(|v| {
            remap_rectangle_to_sphere(*v, img.width() as f32, img.height() as f32, sphere_scale)
        })
        .collect();
    println!("Remapping complete.");

    // --- Output ---
    println!("\n--- Triangulation Results ---");
    println!("Generated {} mesh vertices.", mesh_vertices.len());
    println!(
        "Generated {} mesh indices (representing {} triangles).",
        mesh_indices.len(),
        mesh_indices.len() / 3
    );

    if args.debug_info {
        if !mesh_vertices.is_empty() {
            println!("\nFirst 5 mesh vertices:");
            for v in mesh_vertices.iter().filter(|p| p.color[0] > 0.0).take(5) {
                println!(
                    "  Pos: [{:.2}, {:.2}, {:.2}], Color: [{:.2}, {:.2}, {:.2}]",
                    v.position[0], v.position[1], v.position[2], v.color[0], v.color[1], v.color[2]
                );
            }
        }

        if mesh_indices.len() >= 3 {
            println!("\nFirst 5 triangles (indices):");
            for i in 0..std::cmp::min(5, mesh_indices.len() / 3) {
                let i1 = mesh_indices[i * 3];
                let i2 = mesh_indices[i * 3 + 1];
                let i3 = mesh_indices[i * 3 + 2];
                println!("  Triangle {}: ({}, {}, {})", i, i1, i2, i3);
            }
        }
    }

    fn filenamer(name: &str, meta: Option<&str>) -> String {
        if let Some(meta) = meta {
            return format!("{}.{}", name, meta);
        }
        return format!("{}", name);
    }

    write_gltf(
        &mesh_vertices,
        &mesh_indices,
        args.output_type,
        filenamer(image_file.as_str(), None).as_str(),
        &args.output_dir.clone(),
    );

    write_gltf(
        &remapped_sphere_vertices,
        &mesh_indices,
        args.output_type,
        filenamer(image_file.as_str(), Some("sphere")).as_str(),
        &args.output_dir.clone(),
    );

    Ok(())
}
