use crate::{Gpt2Config, Gpt2Model};
use anyhow::{anyhow, Result};
use burn::prelude::*;
use burn::record::{BinGzFileRecorder, FullPrecisionSettings};
use std::path::Path;

/// Save model weights in binary format
pub fn save_model<B: Backend>(
    model: &Gpt2Model<B>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let recorder = BinGzFileRecorder::<FullPrecisionSettings>::default();
    model
        .clone()
        .save_file(path.as_ref().to_path_buf(), &recorder)
        .map_err(|e| anyhow!("Failed to save model: {}", e))?;
    Ok(())
}

/// Load model weights from binary format
pub fn load_model<B: Backend>(
    config: Gpt2Config,
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<Gpt2Model<B>> {
    let mut model = Gpt2Model::new(config, device);
    let recorder = BinGzFileRecorder::<FullPrecisionSettings>::default();
    model = model
        .load_file(path.as_ref().to_path_buf(), &recorder, device)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;
    Ok(model)
}