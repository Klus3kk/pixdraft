use pyo3::prelude::*;
use image::{ImageBuffer, RgbaImage, Rgba};

/// Core image representation for PixTrick
/// Handles large images with chunked processing
#[pyclass]
#[derive(Clone)]
pub struct PixImage {
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub channels: u8,
    
    // Internal image data - could be chunked for very large images
    data: RgbaImage,
}

#[pymethods]
impl PixImage {
    #[new]
    pub fn new(width: u32, height: u32) -> Self {
        let data = ImageBuffer::new(width, height);
        PixImage {
            width,
            height,
            channels: 4, // RGBA
            data,
        }
    }
    
    /// Load image from file path
    #[staticmethod]
    pub fn from_path(path: &str) -> PyResult<Self> {
        let img = image::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load image: {}", e)))?;
        
        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        
        Ok(PixImage {
            width,
            height,
            channels: 4,
            data: rgba_img,
        })
    }
    
    /// Save image to file path
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.data.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to save image: {}", e)))
    }
    
    /// Get pixel value at coordinates
    pub fn get_pixel(&self, x: u32, y: u32) -> PyResult<(u8, u8, u8, u8)> {
        if x >= self.width || y >= self.height {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Pixel coordinates out of bounds"));
        }
        
        let pixel = self.data.get_pixel(x, y);
        Ok((pixel[0], pixel[1], pixel[2], pixel[3]))
    }
    
    /// Set pixel value at coordinates
    pub fn set_pixel(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) -> PyResult<()> {
        if x >= self.width || y >= self.height {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Pixel coordinates out of bounds"));
        }
        
        self.data.put_pixel(x, y, Rgba([r, g, b, a]));
        Ok(())
    }
    
    /// Clone the image
    pub fn clone(&self) -> Self {
        PixImage {
            width: self.width,
            height: self.height,
            channels: self.channels,
            data: self.data.clone(),
        }
    }
    
    /// Get basic image info
    pub fn info(&self) -> String {
        format!("PixImage {}x{} ({} channels)", self.width, self.height, self.channels)
    }
}

impl PixImage {
    /// Internal method to get raw image data for processing
    pub(crate) fn get_raw_data(&self) -> &RgbaImage {
        &self.data
    }
    
    /// Internal method to get mutable raw image data
    pub(crate) fn get_raw_data_mut(&mut self) -> &mut RgbaImage {
        &mut self.data
    }
    
    /// Create from existing RgbaImage
    pub(crate) fn from_rgba_image(img: RgbaImage) -> Self {
        let (width, height) = img.dimensions();
        PixImage {
            width,
            height,
            channels: 4,
            data: img,
        }
    }
}