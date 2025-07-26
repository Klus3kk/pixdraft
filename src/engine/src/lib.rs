use pyo3::prelude::*;

mod image_core; // Core image handling module
mod node_system; // Node system for image processing (prefered this over sequential processing)
mod filters; // Image filters module

use image_core::PixImage; 
use node_system::{NodeGraph, FilterNode};

/// Image processing engine for PixTrick
#[pymodule]
fn pixtrick_engine(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PixImage>()?;
    m.add_class::<NodeGraph>()?;
    m.add_class::<FilterNode>()?;
    // Filter functions
    m.add_function(wrap_pyfunction!(load_image, m)?)?;
    m.add_function(wrap_pyfunction!(save_image, m)?)?;
    
    Ok(())
}

/// Load an image from file path
#[pyfunction]
fn load_image(path: &str) -> PyResult<PixImage> {
    PixImage::from_path(path)
}

/// Save an image to file path
#[pyfunction]
fn save_image(image: &PixImage, path: &str) -> PyResult<()> {
    image.save(path)
}