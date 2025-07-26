use pyo3::prelude::*;
use std::collections::HashMap;
use crate::image_core::PixImage;

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
}

/// Node-based processing graph for non-destructive editing
#[pyclass]
pub struct NodeGraph {
    nodes: HashMap<String, FilterNode>,
    execution_order: Vec<String>,
    source_image: Option<PixImage>,
}

#[pymethods]
impl NodeGraph {
    #[new]
    pub fn new() -> Self {
        NodeGraph {
            nodes: HashMap::new(),
            execution_order: Vec::new(),
            source_image: None,
        }
    }
    
    /// Set the source image for processing
    pub fn set_source_image(&mut self, image: PixImage) {
        self.source_image = Some(image);
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
        Ok(())
    }
    
    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }
    
    /// Process the entire graph and return the final image
    pub fn process(&self) -> PyResult<PixImage> {
        let source = self.source_image.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("No source image set"))?;
        
        let mut current_image = source.clone();
        
        // Execute nodes in order (for now, we'll implement proper dependency resolution later)
        for node_id in &self.execution_order {
            if let Some(node) = self.nodes.get(node_id) {
                if node.enabled {
                    current_image = self.apply_filter(&current_image, node)?;
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
        Ok(())
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
                // TODO: Implement brightness filter
                Ok(image.clone())
            },
            "contrast" => {
                // TODO: Implement contrast filter
                Ok(image.clone())
            },
            "blur" => {
                // TODO: Implement blur filter
                Ok(image.clone())
            },
            _ => {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown filter type: {}", node.filter_type)
                ))
            }
        }
    }
}