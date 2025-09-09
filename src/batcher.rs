use crate::Gpt2Tokenizer;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

/// Batch item for the Burn training system with sequence length tracking for masking
#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub sentence1: Tensor<B, 2, Int>,
    pub sentence2: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 1>,
    /// Original lengths of sentence1 before padding (for masking)
    pub sentence1_lengths: Vec<usize>,
    /// Original lengths of sentence2 before padding (for masking)
    pub sentence2_lengths: Vec<usize>,
}

impl<B: Backend> TrainingBatch<B> {
    pub fn new(
        sentence1: Tensor<B, 2, Int>,
        sentence2: Tensor<B, 2, Int>,
        labels: Tensor<B, 1>,
        sentence1_lengths: Vec<usize>,
        sentence2_lengths: Vec<usize>,
    ) -> Self {
        Self {
            sentence1,
            sentence2,
            labels,
            sentence1_lengths,
            sentence2_lengths,
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
        let mut sentence1_lengths = Vec::new();
        let mut sentence2_lengths = Vec::new();
        let mut labels = Vec::new();

        // Tokenize all sentences and track original lengths
        for item in items {
            if let (Ok((tokens1, len1)), Ok((tokens2, len2))) = (
                self.tokenizer.encode_with_length(&item.sentence1, true),
                self.tokenizer.encode_with_length(&item.sentence2, true),
            ) {
                sentence1_ids.push(tokens1);
                sentence2_ids.push(tokens2);
                sentence1_lengths.push(len1);
                sentence2_lengths.push(len2);
                labels.push(item.label);
            }
        }

        if sentence1_ids.is_empty() {
            // Panic if all tokenization failed - returning empty batch would cause learner to panic
            panic!("All items in batch failed tokenization. This indicates a problem with the input data or tokenizer configuration. Check your dataset for malformed entries.");
        }

        // Note: The tokenizer's encode_with_length already handles padding to max_length,
        // so all sequences should have the same length already
        let max_len1 = sentence1_ids[0].len(); // All should be same length due to padding
        let max_len2 = sentence2_ids[0].len(); // All should be same length due to padding

        let batch_size = sentence1_ids.len();
        let mut padded_sentence1 = Vec::with_capacity(batch_size * max_len1);
        let mut padded_sentence2 = Vec::with_capacity(batch_size * max_len2);

        for seq in sentence1_ids.iter() {
            padded_sentence1.extend(seq.iter().map(|&x| x as i64));
        }

        for seq in sentence2_ids.iter() {
            padded_sentence2.extend(seq.iter().map(|&x| x as i64));
        }

        let sentence1_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(&padded_sentence1[..]), device)
                .reshape([batch_size, max_len1]);

        let sentence2_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(&padded_sentence2[..]), device)
                .reshape([batch_size, max_len2]);

        let labels_tensor = Tensor::<B, 1>::from_data(TensorData::from(&labels[..]), device);

        TrainingBatch::new(
            sentence1_tensor, 
            sentence2_tensor, 
            labels_tensor,
            sentence1_lengths,
            sentence2_lengths,
        )
    }
}
