use anyhow::{anyhow, Result};
use csv::ReaderBuilder;
use serde::Deserialize;
use std::path::Path;

/// A training example with two sentences and a similarity label
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingExample {
    pub id: u32,
    pub sentence1: String,
    pub sentence2: String,
    pub label: u8, // 0 = dissimilar, 1 = similar
}

/// Dataset containing training examples
#[derive(Debug)]
pub struct Dataset {
    pub examples: Vec<TrainingExample>,
}

impl Dataset {
    /// Load dataset from a TSV file
    /// Expected format: id\tsentence1\tsentence2\tlabel
    pub fn from_tsv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .from_path(path_ref)?;

        let mut examples = Vec::new();

        for result in reader.deserialize() {
            match result {
                Ok(example) => {
                    let example: TrainingExample = example;

                    // Validate label
                    if example.label > 1 {
                        return Err(anyhow!("Invalid label: {}. Expected 0 or 1", example.label));
                    }

                    examples.push(example);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to parse row: {}", e);
                    continue;
                }
            }
        }

        if examples.is_empty() {
            return Err(anyhow!("No valid examples found in dataset"));
        }

        println!(
            "Loaded {} examples from {}",
            examples.len(),
            path_ref.display()
        );

        Ok(Dataset { examples })
    }

    /// Get the number of examples in the dataset
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get an iterator over the examples
    pub fn iter(&self) -> impl Iterator<Item = &TrainingExample> {
        self.examples.iter()
    }

    /// Shuffle the examples using the provided random number generator
    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        use rand::seq::SliceRandom;
        self.examples.shuffle(rng);
    }

    /// Split dataset into batches of the given size
    pub fn batches(&self, batch_size: usize) -> Vec<&[TrainingExample]> {
        self.examples.chunks(batch_size).collect()
    }

    /// Limit the dataset to the first N examples (for testing)
    pub fn limit(&mut self, max_examples: usize) {
        if max_examples > 0 && max_examples < self.examples.len() {
            self.examples.truncate(max_examples);
        }
    }

    /// Get statistics about the dataset
    pub fn statistics(&self) -> DatasetStats {
        let total = self.examples.len();
        let similar_count = self.examples.iter().filter(|ex| ex.label == 1).count();
        let dissimilar_count = total - similar_count;

        DatasetStats {
            total_examples: total,
            similar_pairs: similar_count,
            dissimilar_pairs: dissimilar_count,
            similar_ratio: if total > 0 {
                similar_count as f32 / total as f32
            } else {
                0.0
            },
        }
    }
}

/// Statistics about a dataset
#[derive(Debug)]
pub struct DatasetStats {
    pub total_examples: usize,
    pub similar_pairs: usize,
    pub dissimilar_pairs: usize,
    pub similar_ratio: f32,
}

impl DatasetStats {
    /// Print formatted statistics
    pub fn print(&self) {
        println!("Dataset Statistics:");
        println!("  Total examples: {}", self.total_examples);
        println!(
            "  Similar pairs (label=1): {} ({:.1}%)",
            self.similar_pairs,
            self.similar_ratio * 100.0
        );
        println!(
            "  Dissimilar pairs (label=0): {} ({:.1}%)",
            self.dissimilar_pairs,
            (1.0 - self.similar_ratio) * 100.0
        );
    }
}

/// Training batch containing tokenized inputs and labels
#[derive(Debug)]
pub struct TrainingBatch {
    pub sentence1_ids: Vec<Vec<u32>>,
    pub sentence2_ids: Vec<Vec<u32>>,
    pub labels: Vec<u8>,
}

impl TrainingBatch {
    /// Create a training batch from a slice of examples using the provided tokenizer
    pub fn from_examples(
        examples: &[TrainingExample],
        tokenizer: &crate::Gpt2Tokenizer,
    ) -> Result<Self> {
        let mut sentence1_ids = Vec::with_capacity(examples.len());
        let mut sentence2_ids = Vec::with_capacity(examples.len());
        let mut labels = Vec::with_capacity(examples.len());

        for example in examples {
            let s1_tokens = tokenizer.encode(&example.sentence1, true)?;
            let s2_tokens = tokenizer.encode(&example.sentence2, true)?;

            sentence1_ids.push(s1_tokens);
            sentence2_ids.push(s2_tokens);
            labels.push(example.label);
        }

        Ok(TrainingBatch {
            sentence1_ids,
            sentence2_ids,
            labels,
        })
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.labels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_dataset_from_tsv() -> Result<()> {
        // Create a temporary TSV file
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "id\tsentence1\tsentence2\tlabel")?;
        writeln!(temp_file, "1\tHello world\tHi there\t1")?;
        writeln!(temp_file, "2\tGood morning\tBad evening\t0")?;

        temp_file.flush()?;

        // Load dataset
        let dataset = Dataset::from_tsv(temp_file.path())?;

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.examples[0].sentence1, "Hello world");
        assert_eq!(dataset.examples[0].sentence2, "Hi there");
        assert_eq!(dataset.examples[0].label, 1);
        assert_eq!(dataset.examples[1].label, 0);

        Ok(())
    }

    #[test]
    fn test_dataset_statistics() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "id\tsentence1\tsentence2\tlabel")?;
        writeln!(temp_file, "1\tHello\tHi\t1")?;
        writeln!(temp_file, "2\tGood\tBad\t0")?;
        writeln!(temp_file, "3\tYes\tYeah\t1")?;
        temp_file.flush()?;

        let dataset = Dataset::from_tsv(temp_file.path())?;
        let stats = dataset.statistics();

        assert_eq!(stats.total_examples, 3);
        assert_eq!(stats.similar_pairs, 2);
        assert_eq!(stats.dissimilar_pairs, 1);
        assert!((stats.similar_ratio - 2.0 / 3.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_dataset_batches() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "id\tsentence1\tsentence2\tlabel")?;
        for i in 1..=5 {
            writeln!(temp_file, "{}\tsentence{}\tother{}\t{}", i, i, i, i % 2)?;
        }
        temp_file.flush()?;

        let dataset = Dataset::from_tsv(temp_file.path())?;
        let batches = dataset.batches(2);

        assert_eq!(batches.len(), 3); // ceil(5/2) = 3
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
        assert_eq!(batches[2].len(), 1);

        Ok(())
    }
}
