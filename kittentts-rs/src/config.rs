use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Model configuration matching the JSON config.json format.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub model: String,
    pub voices: String,
    pub model_file: String,
    #[serde(default)]
    pub speed_priors: HashMap<String, f32>,
    #[serde(default)]
    pub voice_aliases: HashMap<String, String>,
}

impl ModelConfig {
    /// Load configuration from a config.json file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let config: ModelConfig = serde_json::from_str(&data)?;
        if config.model_type != "ONNX1" && config.model_type != "ONNX2" {
            anyhow::bail!("Unsupported model type: {}", config.model_type);
        }
        Ok(config)
    }
}
