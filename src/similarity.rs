use crate::{Gpt2Model, Gpt2Tokenizer};
use anyhow::Result;
use burn::prelude::*;

/// Calculate cosine similarity between two vectors
/// Returns a value between -1 and 1, where 1 means identical vectors
pub fn cosine_similarity<B: Backend>(
    vec1: Tensor<B, 1>,
    vec2: Tensor<B, 1>,
) -> Tensor<B, 1> {
    // Compute dot product
    let dot_product = vec1.clone() * vec2.clone();
    let dot_product = dot_product.sum();

    // Compute magnitudes
    let magnitude1 = vec1.clone() * vec1;
    let magnitude1 = magnitude1.sum().sqrt();
    
    let magnitude2 = vec2.clone() * vec2;
    let magnitude2 = magnitude2.sum().sqrt();

    // Avoid division by zero
    let denominator = magnitude1 * magnitude2;
    
    // Return cosine similarity
    dot_product / denominator
}

/// Calculate Euclidean distance between two vectors
/// Returns a non-negative value where 0 means identical vectors
pub fn euclidean_distance<B: Backend>(
    vec1: Tensor<B, 1>,
    vec2: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let diff = vec1 - vec2;
    let squared_diff = diff.clone() * diff;
    squared_diff.sum().sqrt()
}

/// Calculate Manhattan distance between two vectors  
/// Returns a non-negative value where 0 means identical vectors
pub fn manhattan_distance<B: Backend>(
    vec1: Tensor<B, 1>,
    vec2: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let diff = vec1 - vec2;
    diff.abs().sum()
}

/// Normalize a vector to unit length
pub fn normalize_vector<B: Backend>(vec: Tensor<B, 1>) -> Tensor<B, 1> {
    let magnitude = vec.clone() * vec.clone();
    let magnitude = magnitude.sum().sqrt();
    vec / magnitude
}

/// Similarity calculator for sentence pairs using different metrics
pub struct SimilarityCalculator<B: Backend> {
    model: Gpt2Model<B>,
    tokenizer: Gpt2Tokenizer,
}

impl<B: Backend<FloatElem = f32>> SimilarityCalculator<B> {
    /// Create a new similarity calculator
    pub fn new(model: Gpt2Model<B>, tokenizer: Gpt2Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Calculate cosine similarity between two sentences (0 to 1 range)
    /// Returns a value between 0 and 1, where 1 means very similar sentences
    pub fn calculate_similarity(&self, sentence1: &str, sentence2: &str) -> Result<f32> {
        // Get embeddings for both sentences
        let embedding1 = self.get_sentence_embedding(sentence1)?;
        let embedding2 = self.get_sentence_embedding(sentence2)?;

        // Calculate cosine similarity
        let similarity = cosine_similarity(embedding1, embedding2);
        
        // Convert to scalar and normalize to 0-1 range
        let similarity_value = similarity.into_scalar();
        let normalized_similarity = (similarity_value + 1.0) / 2.0; // Convert from [-1,1] to [0,1]
        
        Ok(normalized_similarity)
    }

    /// Calculate multiple similarity metrics between two sentences
    pub fn calculate_all_metrics(&self, sentence1: &str, sentence2: &str) -> Result<SimilarityMetrics> {
        let embedding1 = self.get_sentence_embedding(sentence1)?;
        let embedding2 = self.get_sentence_embedding(sentence2)?;

        // Cosine similarity (normalized to 0-1)
        let cosine_sim = cosine_similarity(embedding1.clone(), embedding2.clone());
        let cosine_similarity = (cosine_sim.into_scalar() + 1.0) / 2.0;

        // Euclidean distance
        let euclidean_distance = euclidean_distance(embedding1.clone(), embedding2.clone()).into_scalar();

        // Manhattan distance  
        let manhattan_distance = manhattan_distance(embedding1.clone(), embedding2.clone()).into_scalar();

        // Dot product similarity (with normalized vectors)
        let norm_embed1 = normalize_vector(embedding1);
        let norm_embed2 = normalize_vector(embedding2);
        let dot_product = (norm_embed1 * norm_embed2).sum().into_scalar();

        Ok(SimilarityMetrics {
            cosine_similarity,
            euclidean_distance,
            manhattan_distance,
            dot_product_similarity: dot_product,
        })
    }

    /// Get embedding for a single sentence
    fn get_sentence_embedding(&self, sentence: &str) -> Result<Tensor<B, 1>> {
        // Tokenize the sentence
        let token_ids = self.tokenizer.encode(sentence, true)?;
        
        // Get a device from one of the model's parameters
        let device = &self.model.token_embedding.weight.device();
        
        // Convert to tensor and add batch dimension
        let input_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::from(&token_ids.iter().map(|&x| x as i64).collect::<Vec<_>>()[..]),
            device,
        ).unsqueeze_dim(0);

        // Get sentence embedding (average of token embeddings)
        let sentence_embedding = self.model.get_sentence_embedding(input_tensor);
        let sentence_embedding: Tensor<B, 2> = sentence_embedding.squeeze_dims(&[1]); // Remove seq dimension 
        
        // Remove batch dimension
        let result: Tensor<B, 1> = sentence_embedding.squeeze_dims(&[0]);
        Ok(result)
    }

    /// Calculate similarity matrix for a list of sentences
    pub fn similarity_matrix(&self, sentences: &[&str]) -> Result<Vec<Vec<f32>>> {
        let embeddings: Result<Vec<_>, _> = sentences
            .iter()
            .map(|&sentence| self.get_sentence_embedding(sentence))
            .collect();
        let embeddings = embeddings?;

        let mut matrix = vec![vec![0.0; sentences.len()]; sentences.len()];

        for i in 0..embeddings.len() {
            for j in 0..embeddings.len() {
                if i == j {
                    matrix[i][j] = 1.0; // Perfect similarity with self
                } else {
                    let similarity = cosine_similarity(embeddings[i].clone(), embeddings[j].clone());
                    matrix[i][j] = (similarity.into_scalar() + 1.0) / 2.0; // Normalize to 0-1
                }
            }
        }

        Ok(matrix)
    }

    /// Find the most similar sentence from a list of candidates
    pub fn find_most_similar(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<(usize, f32)> {
        let query_embedding = self.get_sentence_embedding(query)?;
        
        let mut best_similarity = -1.0;
        let mut best_index = 0;

        for (i, &candidate) in candidates.iter().enumerate() {
            let candidate_embedding = self.get_sentence_embedding(candidate)?;
            let similarity = cosine_similarity(query_embedding.clone(), candidate_embedding);
            let normalized_similarity = (similarity.into_scalar() + 1.0) / 2.0;

            if normalized_similarity > best_similarity {
                best_similarity = normalized_similarity;
                best_index = i;
            }
        }

        Ok((best_index, best_similarity))
    }
}

/// Container for different similarity metrics
#[derive(Debug, Clone)]
pub struct SimilarityMetrics {
    pub cosine_similarity: f32,
    pub euclidean_distance: f32,
    pub manhattan_distance: f32,
    pub dot_product_similarity: f32,
}

impl SimilarityMetrics {
    /// Print formatted metrics
    pub fn print_formatted(&self, sentence1: &str, sentence2: &str) {
        println!("Similarity metrics for:");
        println!("  Sentence 1: \"{}\"", sentence1);
        println!("  Sentence 2: \"{}\"", sentence2);
        println!("Results:");
        println!("  Cosine Similarity:    {:.4} (0=different, 1=identical)", self.cosine_similarity);
        println!("  Euclidean Distance:   {:.4} (0=identical, higher=different)", self.euclidean_distance);
        println!("  Manhattan Distance:   {:.4} (0=identical, higher=different)", self.manhattan_distance);
        println!("  Dot Product Sim:      {:.4} (normalized vectors)", self.dot_product_similarity);
    }
}

/// Utility function to print similarity matrix in a readable format
pub fn print_similarity_matrix(sentences: &[&str], matrix: &[Vec<f32>]) {
    println!("\nSimilarity Matrix:");
    println!("Sentences:");
    for (i, sentence) in sentences.iter().enumerate() {
        println!("  {}: \"{}\"", i, sentence);
    }
    
    println!("\nMatrix (cosine similarity, 0.0=different, 1.0=identical):");
    print!("     ");
    for i in 0..sentences.len() {
        print!("{:>6}", i);
    }
    println!();
    
    for (i, row) in matrix.iter().enumerate() {
        print!("{:>3}: ", i);
        for value in row {
            print!("{:>6.3}", value);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_cosine_similarity() {
        let device = Default::default();
        
        // Create two identical vectors
        let vec1 = Tensor::<TestBackend, 1>::from_data(
            TensorData::from(&[1.0, 2.0, 3.0][..]),
            &device,
        );
        let vec2 = vec1.clone();
        
        let similarity = cosine_similarity(vec1, vec2);
        let result = similarity.into_scalar();
        
        // Should be very close to 1.0 (identical vectors)
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let device = Default::default();
        
        // Create two identical vectors
        let vec1 = Tensor::<TestBackend, 1>::from_data(
            TensorData::from(&[1.0, 2.0, 3.0][..]),
            &device,
        );
        let vec2 = vec1.clone();
        
        let distance = euclidean_distance(vec1, vec2);
        let result = distance.into_scalar();
        
        // Should be very close to 0.0 (identical vectors)
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let device = Default::default();
        
        // Create two different vectors
        let vec1 = Tensor::<TestBackend, 1>::from_data(
            TensorData::from(&[1.0, 2.0, 3.0][..]),
            &device,
        );
        let vec2 = Tensor::<TestBackend, 1>::from_data(
            TensorData::from(&[2.0, 3.0, 4.0][..]),
            &device,
        );
        
        let distance = manhattan_distance(vec1, vec2);
        let result = distance.into_scalar();
        
        // Should be 3.0 (sum of absolute differences: |1-2| + |2-3| + |3-4| = 3)
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_vector() {
        let device = Default::default();
        
        let vec = Tensor::<TestBackend, 1>::from_data(
            TensorData::from(&[3.0, 4.0][..]),
            &device,
        );
        
        let normalized = normalize_vector(vec);
        
        // Check that magnitude is 1
        let magnitude_squared = (normalized.clone() * normalized).sum();
        let magnitude = magnitude_squared.sqrt().into_scalar();
        
        assert!((magnitude - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_metrics_creation() {
        let metrics = SimilarityMetrics {
            cosine_similarity: 0.95,
            euclidean_distance: 1.2,
            manhattan_distance: 2.1,
            dot_product_similarity: 0.88,
        };

        assert!((metrics.cosine_similarity - 0.95).abs() < 1e-6);
        assert!((metrics.euclidean_distance - 1.2).abs() < 1e-6);
    }
}