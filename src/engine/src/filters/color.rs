use image::Rgba;
use rayon::prelude::*;
use crate::image_core::PixImage;

/// Adjust saturation of image
pub fn saturation(image: &PixImage, amount: f64) -> PixImage {
    let raw_data = image.get_raw_data();
    let mut result = raw_data.clone();
    
    let saturation_factor = (amount / 100.0 + 1.0).clamp(0.0, 3.0);
    
    result.par_enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        let Rgba([r, g, b, a]) = *pixel;
        
        // Convert to grayscale for desaturation
        let gray = (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) as u8;
        
        // Interpolate between grayscale and original color
        let new_r = (gray as f64 + (r as f64 - gray as f64) * saturation_factor).clamp(0.0, 255.0) as u8;
        let new_g = (gray as f64 + (g as f64 - gray as f64) * saturation_factor).clamp(0.0, 255.0) as u8;
        let new_b = (gray as f64 + (b as f64 - gray as f64) * saturation_factor).clamp(0.0, 255.0) as u8;
        
        *pixel = Rgba([new_r, new_g, new_b, a]);
    });
    
    PixImage::from_rgba_image(result)
}

/// Adjust hue of image (simple implementation)
/// Note: This is a basic implementation and may not handle all edge cases
pub fn hue_shift(image: &PixImage, degrees: f64) -> PixImage {
    let raw_data = image.get_raw_data();
    let mut result = raw_data.clone();
    
    let hue_shift = (degrees % 360.0) / 360.0;
    
    result.par_enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        let Rgba([r, g, b, a]) = *pixel;
        
        // Convert RGB to HSV, shift hue, convert back
        let (h, s, v) = rgb_to_hsv(r, g, b);
        let new_h = (h + hue_shift) % 1.0;
        let (new_r, new_g, new_b) = hsv_to_rgb(new_h, s, v);
        
        *pixel = Rgba([new_r, new_g, new_b, a]);
    });
    
    PixImage::from_rgba_image(result)
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;
    
    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let delta = max - min;
    
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / delta) % 6.0
    } else if max == g {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    } / 6.0;
    
    let s = if max == 0.0 { 0.0 } else { delta / max };
    let v = max;
    
    (h, s, v)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}