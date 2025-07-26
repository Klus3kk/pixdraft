use image::Rgba;
use rayon::prelude::*;
use crate::image_core::PixImage;

/// Apply brightness adjustment to image
pub fn brightness(image: &PixImage, amount: f64) -> PixImage {
    let raw_data = image.get_raw_data();
    let mut result = raw_data.clone();
    
    // Clamp brightness adjustment to reasonable range
    let brightness_factor = (amount / 100.0).clamp(-1.0, 1.0);
    
    // Process pixels in parallel for performance
    result.par_enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        let Rgba([r, g, b, a]) = *pixel;
        
        // Apply brightness adjustment
        let new_r = ((r as f64 + brightness_factor * 255.0).clamp(0.0, 255.0)) as u8;
        let new_g = ((g as f64 + brightness_factor * 255.0).clamp(0.0, 255.0)) as u8;
        let new_b = ((b as f64 + brightness_factor * 255.0).clamp(0.0, 255.0)) as u8;
        
        *pixel = Rgba([new_r, new_g, new_b, a]);
    });
    
    PixImage::from_rgba_image(result)
}

/// Apply contrast adjustment to image
pub fn contrast(image: &PixImage, amount: f64) -> PixImage {
    let raw_data = image.get_raw_data();
    let mut result = raw_data.clone();
    
    // Convert contrast percentage to multiplier
    let contrast_factor = ((amount + 100.0) / 100.0).clamp(0.0, 10.0);
    
    result.par_enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        let Rgba([r, g, b, a]) = *pixel;
        
        // Apply contrast adjustment (subtract 128, multiply, add 128)
        let new_r = (((r as f64 - 128.0) * contrast_factor + 128.0).clamp(0.0, 255.0)) as u8;
        let new_g = (((g as f64 - 128.0) * contrast_factor + 128.0).clamp(0.0, 255.0)) as u8;
        let new_b = (((b as f64 - 128.0) * contrast_factor + 128.0).clamp(0.0, 255.0)) as u8;
        
        *pixel = Rgba([new_r, new_g, new_b, a]);
    });
    
    PixImage::from_rgba_image(result)
}

/// Invert image colors
pub fn invert(image: &PixImage) -> PixImage {
    let raw_data = image.get_raw_data();
    let mut result = raw_data.clone();
    
    result.par_enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        let Rgba([r, g, b, a]) = *pixel;
        *pixel = Rgba([255 - r, 255 - g, 255 - b, a]);
    });
    
    PixImage::from_rgba_image(result)
}

/// Convert to grayscale using perceived lightness (copied from StackOverflow :))
/// This uses the L* lightness channel from CIE L*a*b* color space
/// for better perceptual accuracy compared to simple averaging.
/// It applies gamma correction to convert sRGB to linear RGB before calculating luminance.
/// The result is a grayscale image where the lightness is preserved more accurately.
pub fn grayscale(image: &PixImage) -> PixImage {
    let raw_data = image.get_raw_data();
    let mut result = raw_data.clone();
    
    result.par_enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        let Rgba([r, g, b, a]) = *pixel;
        
        // Convert sRGB to linear values (gamma correction)
        let r_linear = srgb_to_linear(r as f64 / 255.0);
        let g_linear = srgb_to_linear(g as f64 / 255.0);
        let b_linear = srgb_to_linear(b as f64 / 255.0);
        
        // Calculate luminance using standard coefficients
        let luminance = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear;
        
        // Convert back to perceived lightness (L*)
        let lightness = if luminance > 0.008856 {
            116.0 * luminance.powf(1.0/3.0) - 16.0
        } else {
            903.3 * luminance
        };
        
        // Convert L* (0-100) back to 0-255 range
        let gray = (lightness * 255.0 / 100.0).clamp(0.0, 255.0) as u8;
        
        *pixel = Rgba([gray, gray, gray, a]);
    });
    
    PixImage::from_rgba_image(result)
}

/// Convert sRGB to linear RGB (gamma correction)
fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}