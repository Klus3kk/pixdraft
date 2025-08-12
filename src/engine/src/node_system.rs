use pyo3::prelude::*;
use std::collections::HashMap;
use crate::image_core::PixImage;
use crate::filters;

/// Represents a filter node in the processing graph
#[pyclass]
#[derive(Clone)]
pub struct FilterNode {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub filter_type: String,
    #[pyo3(get)]
    pub enabled: bool,
    
    // Parameters as key-value pairs
    parameters: HashMap<String, f64>,
    inputs: Vec<String>,  // IDs of input nodes
}

#[pymethods]
impl FilterNode {
    #[new]
    pub fn new(id: String, filter_type: String) -> Self {
        FilterNode {
            id,
            filter_type,
            enabled: true,
            parameters: HashMap::new(),
            inputs: Vec::new(),
        }
    }
    
    /// Set a parameter value
    pub fn set_parameter(&mut self, key: String, value: f64) {
        self.parameters.insert(key, value);
    }
    
    /// Get a parameter value
    pub fn get_parameter(&self, key: &str) -> Option<f64> {
        self.parameters.get(key).copied()
    }
    
    /// Add an input node connection
    pub fn add_input(&mut self, node_id: String) {
        self.inputs.push(node_id);
    }
    
    /// Remove an input node connection
    pub fn remove_input(&mut self, node_id: &str) {
        self.inputs.retain(|id| id != node_id);
    }
    
    /// Get all input node IDs
    pub fn get_inputs(&self) -> Vec<String> {
        self.inputs.clone()
    }
    
    /// Toggle node enabled/disabled
    pub fn toggle_enabled(&mut self) {
        self.enabled = !self.enabled;
    }

    /// Get all parameter names for this filter type
    pub fn get_parameter_names(&self) -> Vec<String> {
        match self.filter_type.as_str() {
            "brightness" => vec!["amount".to_string()],
            "contrast" => vec!["amount".to_string()],
            "saturation" => vec!["amount".to_string()],
            "hue_shift" => vec!["degrees".to_string()],
            "box_blur" => vec!["radius".to_string()],
            "gaussian_blur" => vec!["radius".to_string()],
            "invert" | "grayscale" => vec![], // No parameters
            _ => vec![],
        }
    }
}

/// Node-based processing graph for non-destructive editing
#[pyclass]
pub struct NodeGraph {
    nodes: HashMap<String, FilterNode>,
    execution_order: Vec<String>,
    source_image: Option<PixImage>,
    cache: HashMap<String, PixImage>, // Cache processed results for performance
}

#[pymethods]
impl NodeGraph {
    #[new]
    pub fn new() -> Self {
        NodeGraph {
            nodes: HashMap::new(),
            execution_order: Vec::new(),
            source_image: None,
            cache: HashMap::new(),
        }
    }
    
    /// Set the source image for processing
    pub fn set_source_image(&mut self, image: PixImage) {
        self.source_image = Some(image);
        self.clear_cache(); // Clear cache when source changes
    }
    
    /// Add a filter node to the graph
    pub fn add_node(&mut self, node: FilterNode) -> PyResult<()> {
        let node_id = node.id.clone();
        
        if self.nodes.contains_key(&node_id) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Node with ID '{}' already exists", node_id)
            ));
        }
        
        self.nodes.insert(node_id.clone(), node);
        self.execution_order.push(node_id);
        self.clear_cache();
        
        Ok(())
    }
    
    /// Remove a node from the graph
    pub fn remove_node(&mut self, node_id: &str) -> PyResult<()> {
        if !self.nodes.contains_key(node_id) {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ));
        }
        
        // Remove from nodes and execution order
        self.nodes.remove(node_id);
        self.execution_order.retain(|id| id != node_id);
        
        // Remove this node from other nodes' inputs
        for node in self.nodes.values_mut() {
            node.remove_input(node_id);
        }

        self.clear_cache();
        
        Ok(())
    }
    
    /// Get a node by ID
    pub fn get_node(&self, node_id: &str) -> PyResult<FilterNode> {
        self.nodes.get(node_id)
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))
    }
    
    /// Update a node in the graph
    pub fn update_node(&mut self, node: FilterNode) -> PyResult<()> {
        let node_id = node.id.clone();
        
        if !self.nodes.contains_key(&node_id) {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ));
        }
        
        self.nodes.insert(node_id, node);
        self.clear_cache(); 
        Ok(())
    }
    
    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }
    
    /// Process the entire graph and return the final image
    pub fn process(&mut self) -> PyResult<PixImage> {
        let source = self.source_image.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("No source image set"))?;
        
        let mut current_image = source.clone();
        
        // Execute nodes in order
        for node_id in &self.execution_order.clone() {
            if let Some(node) = self.nodes.get(node_id) {
                if node.enabled {
                    // Check cache first
                    let cache_key = format!("{}_{}", node_id, self.get_node_hash(node));
                    
                    if let Some(cached_result) = self.cache.get(&cache_key) {
                        current_image = cached_result.clone();
                    } else {
                        current_image = self.apply_filter(&current_image, node)?;
                        // Cache the result
                        self.cache.insert(cache_key, current_image.clone());
                    }
                }
            }
        }
        
        Ok(current_image)
    }
    
    /// Get the current execution order
    pub fn get_execution_order(&self) -> Vec<String> {
        self.execution_order.clone()
    }
    
    /// Reorder nodes for execution
    pub fn set_execution_order(&mut self, order: Vec<String>) -> PyResult<()> {
        // Validate that all node IDs exist
        for node_id in &order {
            if !self.nodes.contains_key(node_id) {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Node '{}' not found", node_id)
                ));
            }
        }
        
        self.execution_order = order;
        self.clear_cache();
        Ok(())
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Filter types available in the system
    #[staticmethod]
    pub fn get_available_filters() -> Vec<String> {
        vec![
            "brightness".to_string(),
            "contrast".to_string(),
            "saturation".to_string(),
            "hue_shift".to_string(),
            "box_blur".to_string(),
            "gaussian_blur".to_string(),
            "invert".to_string(),
            "grayscale".to_string(),
        ]
    }

}

impl NodeGraph {
    /// Apply a single filter to an image 
    fn apply_filter(&self, image: &PixImage, node: &FilterNode) -> PyResult<PixImage> {
        // This is where i'll implement the actual filter logic, for now i just wrote some placeholders
        // In a real implementation, this would apply the filter based on node parameters
        // and return a new PixImage instance.
        match node.filter_type.as_str() {
            "brightness" => {
                let amount = node.get_parameter("amount").unwrap_or(0.0);
                Ok(filters::brightness(image, amount))
            },
            "contrast" => {
                let amount = node.get_parameter("amount").unwrap_or(0.0);
                Ok(filters::contrast(image, amount))
            },
            "saturation" => {
                let amount = node.get_parameter("amount").unwrap_or(100.0);
                Ok(filters::saturation(image, amount))
            },
            "hue_shift" => {
                let degrees = node.get_parameter("degrees").unwrap_or(0.0);
                Ok(filters::hue_shift(image, degrees))
            },
            "box_blur" => {
                let radius = node.get_parameter("radius").unwrap_or(1.0) as u32;
                Ok(filters::box_blur(image, radius))
            },
            "gaussian_blur" => {
                let radius = node.get_parameter("radius").unwrap_or(1.0);
                Ok(filters::gaussian_blur(image, radius))
            },
            "invert" => {
                Ok(filters::invert(image))
            },
            "grayscale" => {
                Ok(filters::grayscale(image))
            },
            _ => {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown filter type: {}", node.filter_type)
                ))
            }
        }
    }

    /// Generate a hash for a node's current state (for caching)
    fn get_node_hash(&self, node: &FilterNode) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        node.filter_type.hash(&mut hasher);
        node.enabled.hash(&mut hasher);

        // Hash parameters in sorted order for consistency
        let mut params: Vec<_> = node.parameters.iter().collect();
        params.sort_by_key(|&(k, _)| k);
        for (key, value) in params {
            key.hash(&mut hasher);
            value.to_bits().hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }
}