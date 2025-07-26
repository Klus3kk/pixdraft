use image::Rgba;
use crate::image_core::PixImage;

/// Simple box blur implementation
pub fn box_blur(image: &PixImage, radius: u32) -> PixImage {
    if radius == 0 {
        return image.clone();
    }
    
    let raw_data = image.get_raw_data();
    let mut result = raw_data.clone();
    let (width, height) = raw_data.dimensions();
    
    let kernel_size = radius * 2 + 1;
    let kernel_area = (kernel_size * kernel_size) as f64;
    
    for y in 0..height {
        for x in 0..width {
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut a_sum = 0u32;
            
            // Apply kernel
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let px = (x as i32 + kx as i32 - radius as i32).clamp(0, width as i32 - 1) as u32;
                    let py = (y as i32 + ky as i32 - radius as i32).clamp(0, height as i32 - 1) as u32;
                    
                    let Rgba([r, g, b, a]) = *raw_data.get_pixel(px, py);
                    r_sum += r as u32;
                    g_sum += g as u32;
                    b_sum += b as u32;
                    a_sum += a as u32;
                }
            }
            
            let new_r = (r_sum as f64 / kernel_area) as u8;
            let new_g = (g_sum as f64 / kernel_area) as u8;
            let new_b = (b_sum as f64 / kernel_area) as u8;
            let new_a = (a_sum as f64 / kernel_area) as u8;
            
            result.put_pixel(x, y, Rgba([new_r, new_g, new_b, new_a]));
        }
    }
    
    PixImage::from_rgba_image(result)
}

/// Gaussian blur (simplified approximation using multiple box blurs)
pub fn gaussian_blur(image: &PixImage, radius: f64) -> PixImage {
    if radius <= 0.0 {
        return image.clone();
    }
    
    // Approximate Gaussian blur with 3 box blurs
    let box_radius = ((radius * 0.57735) + 0.5) as u32;
    
    let mut result = image.clone();
    for _ in 0..3 {
        result = box_blur(&result, box_radius);
    }
    
    result
}