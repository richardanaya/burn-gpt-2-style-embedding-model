use crate::Gpt2Tokenizer;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

/// Batch item for the Burn training system
#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub sentence1: Tensor<B, 2, Int>,
    pub sentence2: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 1>,
}

impl<B: Backend> TrainingBatch<B> {
    pub fn new(
        sentence1: Tensor<B, 2, Int>,
        sentence2: Tensor<B, 2, Int>,
        labels: Tensor<B, 1>,
    ) -> Self {
        Self {
            sentence1,
            sentence2,
            labels,
        }
    }
}

/// Training example from our dataset
#[derive(Clone, Debug)]
pub struct TrainingItem {
    pub sentence1: String,
    pub sentence2: String,
    pub label: f32,
}

impl TrainingItem {
    pub fn new(sentence1: String, sentence2: String, label: f32) -> Self {
        Self {
            sentence1,
            sentence2,
            label,
        }
    }
}

/// Batcher to convert training items to batched tensors
#[derive(Clone)]
pub struct TrainingBatcher {
    tokenizer: Gpt2Tokenizer,
}

impl TrainingBatcher {
    pub fn new(tokenizer: Gpt2Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl<B: Backend> Batcher<B, TrainingItem, TrainingBatch<B>> for TrainingBatcher {
    fn batch(&self, items: Vec<TrainingItem>, device: &B::Device) -> TrainingBatch<B> {
        let mut sentence1_ids = Vec::new();
        let mut sentence2_ids = Vec::new();
        let mut labels = Vec::new();

        // Tokenize all sentences
        for item in items {
            if let (Ok(tokens1), Ok(tokens2)) = (
                self.tokenizer.encode(&item.sentence1, true),
                self.tokenizer.encode(&item.sentence2, true),
            ) {
                sentence1_ids.push(tokens1);
                sentence2_ids.push(tokens2);
                labels.push(item.label);
            }
        }

        if sentence1_ids.is_empty() {
            // Panic if all tokenization failed - returning empty batch would cause learner to panic
            panic!("All items in batch failed tokenization. This indicates a problem with the input data or tokenizer configuration. Check your dataset for malformed entries.");
        }

        // Pad sequences
        let max_len1 = sentence1_ids.iter().map(|s| s.len()).max().unwrap_or(0);
        let max_len2 = sentence2_ids.iter().map(|s| s.len()).max().unwrap_or(0);

        let batch_size = sentence1_ids.len();
        let mut padded_sentence1 = Vec::with_capacity(batch_size * max_len1);
        let mut padded_sentence2 = Vec::with_capacity(batch_size * max_len2);

        // GPT-2 pad token ID is 50256, not 0 (which is a valid BPE token)
        const PAD_TOKEN_ID: u32 = 50256;

        for seq in sentence1_ids.iter() {
            let mut padded = seq.clone();
            padded.resize(max_len1, PAD_TOKEN_ID);
            padded_sentence1.extend(padded.iter().map(|&x| x as i64));
        }

        for seq in sentence2_ids.iter() {
            let mut padded = seq.clone();
            padded.resize(max_len2, PAD_TOKEN_ID);
            padded_sentence2.extend(padded.iter().map(|&x| x as i64));
        }

        let sentence1_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(&padded_sentence1[..]), device)
                .reshape([batch_size, max_len1]);

        let sentence2_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(&padded_sentence2[..]), device)
                .reshape([batch_size, max_len2]);

        let labels_tensor = Tensor::<B, 1>::from_data(TensorData::from(&labels[..]), device);

        TrainingBatch::new(sentence1_tensor, sentence2_tensor, labels_tensor)
    }
}
