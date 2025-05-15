use image::{DynamicImage, GenericImageView, GrayImage, Luma, Pixel, Rgb, RgbImage};
use rand::Rng;

pub fn detect_texture_detail(img: &DynamicImage) -> GrayImage {
    let width = img.width();
    let height = img.height();
    let mut detail = GrayImage::new(width, height);

    // First get local mean using a box filter
    let mean_kernel = [
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
    ];

    let local_mean = img.filter3x3(&mean_kernel);
    let samples = 80000;
    let mut rng = rand::rng();

    let mut sample_count = 0;
    for _ in 0..samples {
        let x = rng.random_range(1..width - 1);
        let y = rng.random_range(1..height - 1);

        let mean = local_mean.get_pixel(x, y)[0] as f32;
        let mut variance = 0.0;

        // Sum squared differences
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = (x as i32 + dx) as u32;
                let ny = (y as i32 + dy) as u32;
                let pixel_val = img.get_pixel(nx, ny)[0] as f32;
                variance += (pixel_val - mean).powi(2);
            }
        }

        variance /= 9.0; // Average

        // Standard deviation (sqrt of variance) is a good measure of detail
        let detail_value = (variance.sqrt() * 2.0).min(255.0) as u8;
        if detail_value > 6_u8 {
            detail.put_pixel(x, y, Luma([255_u8]));
            sample_count += 1;
        }
        // detail.put_pixel(x, y, Luma([detail_value]));
    }

    println!("Sample count: {}", sample_count);
    detail
}

pub fn detect_texture_detail_color(img: &DynamicImage, samples: i32) -> DynamicImage {
    let width = img.width();
    let height = img.height();
    let mut detail_color = RgbImage::new(width, height);

    // First get local mean using a box filter
    let mean_kernel = [
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
    ];

    let local_mean = img.filter3x3(&mean_kernel);
    // let samples = 80000;
    let mut rng = rand::rng();

    let mut sample_count = 0;
    for _ in 0..samples {
        let x = rng.random_range(1..width - 1);
        let y = rng.random_range(1..height - 1);

        let mean = local_mean.get_pixel(x, y)[0] as f32;
        let mut variance = 0.0;

        // Sum squared differences
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = (x as i32 + dx) as u32;
                let ny = (y as i32 + dy) as u32;
                // Use the original image for variance calculation
                let pixel_val = img.get_pixel(nx, ny).to_luma()[0] as f32;
                variance += (pixel_val - mean).powi(2);
            }
        }

        variance /= 9.0; // Average

        // Standard deviation (sqrt of variance) is a good measure of detail
        let detail_value = (variance.sqrt() * 2.0).min(255.0) as u8;

        // Map detail value to a color. Simple grayscale mapping for now.
        let color = Rgb([detail_value, detail_value, detail_value]);
        detail_color.put_pixel(x, y, color);

        sample_count += 1;
    }

    println!("Sample count (color): {}", sample_count);
    DynamicImage::ImageRgb8(detail_color)
}
