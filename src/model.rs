use anyhow::{anyhow, Result};
use burn::nn::{
    attention::{MultiHeadAttention, MultiHeadAttentionConfig},
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig,
    Linear, LinearConfig,
};
use burn::prelude::*;
use burn::record::{BinGzFileRecorder, CompactRecorder, FullPrecisionSettings};
use burn::tensor::activation;
use std::path::Path;

/// Configuration for the GPT-2 model
/// Based on GPT-2 117M parameters:
/// - 12 transformer blocks
/// - 12 attention heads per block
/// - 768 embedding dimensions
/// - Context window of 1024 tokens
#[derive(Config, Debug)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub dropout: f64,
    /// Margin parameter for contrastive loss
    #[config(default = 1.0)]
    pub margin: f32,
}

impl Default for Gpt2Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocabulary size
            max_seq_len: 1024, // Maximum sequence length
            d_model: 768,      // Embedding dimension
            n_heads: 12,       // Number of attention heads (GPT-2 117M)
            n_layers: 12,      // Number of transformer layers (GPT-2 117M)
            dropout: 0.1,      // Dropout rate
            margin: 1.0,       // Default margin for contrastive loss
        }
    }
}

impl Gpt2Config {
    /// Initialize the model from config
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2Model<B> {
        Gpt2Model::new(self.clone(), device)
    }
}

/// Multi-layer perceptron (feed-forward network) used in transformer blocks
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    /// Creates a new MLP with the given configuration
    pub fn new(d_model: usize, dropout: f64, device: &B::Device) -> Self {
        // GPT-2 uses 4x expansion in the MLP
        let d_ff = d_model * 4;

        let linear_1 = LinearConfig::new(d_model, d_ff)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        let linear_2 = LinearConfig::new(d_ff, d_model)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        let dropout = DropoutConfig::new(dropout).init();

        Self {
            linear_1,
            linear_2,
            dropout,
        }
    }

    /// Forward pass through the MLP
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_1.forward(x);
        let x = activation::gelu(x); // GPT-2 uses GELU activation
        let x = self.linear_2.forward(x);
        self.dropout.forward(x)
    }
}

/// Transformer block containing multi-head attention and feed-forward network
///
/// Each transformer block performs two main operations:
/// 1. **Multi-Head Attention**: Learn relationships between tokens
/// 2. **Feed-Forward Network**: Process and refine the attended information
///
/// ## Progressive Understanding Across Layers
///
/// **What Different Transformer Blocks Learn:**
///
/// **Blocks 1-3 (Early Processing):**
/// - Basic syntactic patterns and part-of-speech information
/// - Local word relationships (adjective-noun, subject-verb)
/// - Simple grammatical structures
///
/// **Blocks 4-8 (Intermediate Processing):**
/// - Complex syntactic relationships across longer distances
/// - Named entity recognition and entity relationships
/// - Coreference resolution (understanding what pronouns refer to)
/// - More abstract grammatical concepts
///
/// **Blocks 9-12 (Final Processing):**
/// - High-level semantic concepts and abstract meanings
/// - Long-range semantic dependencies
/// - Contextual disambiguation of word meanings
/// - Task-specific representations optimal for downstream tasks
///
/// **Why This Progression Matters for Embeddings:**
///
/// By the time we reach the final blocks, the model has built up a sophisticated
/// understanding that captures not just what words are present, but what the
/// sentence *means* in context. This is why we use the final layer for embeddings
/// rather than earlier layers - we want the most semantically rich representation.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    mlp: Mlp<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    /// Creates a new transformer block
    pub fn new(config: &Gpt2Config, device: &B::Device) -> Self {
        // Multi-head attention configuration
        let attention_config = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_dropout(config.dropout)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            });

        let attention = attention_config.init(device);

        // MLP (feed-forward network)
        let mlp = Mlp::new(config.d_model, config.dropout, device);

        // Layer normalization
        let norm_1 = LayerNormConfig::new(config.d_model).init(device);
        let norm_2 = LayerNormConfig::new(config.d_model).init(device);

        Self {
            attention,
            mlp,
            norm_1,
            norm_2,
        }
    }

    /// Forward pass through the transformer block
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-normalization variant used in GPT-2
        // Self-attention with residual connection
        let normed = self.norm_1.forward(x.clone());
        let mha_input = burn::nn::attention::MhaInput::self_attn(normed);
        let attended = self.attention.forward(mha_input);
        let x = x + attended.context;

        // Feed-forward with residual connection
        let normed = self.norm_2.forward(x.clone());
        let fed_forward = self.mlp.forward(normed);
        x + fed_forward
    }
}

/// GPT-2 model for generating embeddings
#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    pub token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    transformer_blocks: Vec<TransformerBlock<B>>,
    ln_f: LayerNorm<B>,
    dropout: Dropout,
    /// Margin parameter for contrastive loss
    pub margin: f32,
}

impl<B: Backend> Gpt2Model<B> {
    /// Creates a new GPT-2 model with the given configuration
    pub fn new(config: Gpt2Config, device: &B::Device) -> Self {
        // Token embeddings
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        // Position embeddings
        let position_embedding = EmbeddingConfig::new(config.max_seq_len, config.d_model)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.01,
            })
            .init(device);

        // Transformer blocks
        let mut transformer_blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            transformer_blocks.push(TransformerBlock::new(&config, device));
        }

        // Final layer normalization
        let ln_f = LayerNormConfig::new(config.d_model).init(device);

        // Dropout
        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            token_embedding,
            position_embedding,
            transformer_blocks,
            ln_f,
            dropout,
            margin: config.margin,
        }
    }

    /// Forward pass through the model to get embeddings from the final transformer layer
    ///
    /// ## Why We Use the Final Layer for Embeddings
    ///
    /// This method processes input through ALL 12 transformer layers and returns embeddings
    /// from the final layer. Here's why this architectural choice is important:
    ///
    /// ### Progressive Representation Learning in Transformers
    ///
    /// Each transformer layer builds increasingly sophisticated representations:
    ///
    /// **Layer 1-3 (Early Layers): Low-level linguistic features**
    /// - Basic syntactic patterns (noun phrases, verb phrases)
    /// - Part-of-speech information
    /// - Simple word relationships
    /// - Local dependencies and basic grammar
    ///
    /// **Layer 4-8 (Middle Layers): Intermediate linguistic structures**  
    /// - Complex syntactic relationships
    /// - Named entity recognition
    /// - Coreference patterns (what "it" refers to)
    /// - Intermediate semantic relationships
    ///
    /// **Layer 9-12 (Final Layers): High-level semantic understanding**
    /// - Abstract semantic concepts and meanings
    /// - Long-range semantic dependencies
    /// - Contextual word meanings and disambiguation
    /// - Task-relevant high-level representations
    ///
    /// ### Why Final Layer is Best for Sentence Embeddings
    ///
    /// **1. Maximum Contextual Understanding**
    ///
    /// Example: "The bank by the river was steep"
    /// - Early layers might represent "bank" with financial concepts mixed in
    /// - Final layer understands from full context that "bank" means riverbank
    ///
    /// **2. Complete Information Integration**
    /// The final layer has processed information from the entire sequence
    /// through multiple attention mechanisms, creating the most informed representation.
    ///
    /// **3. Abstract Semantic Concepts**
    /// Later layers capture higher-level semantic relationships that are crucial
    /// for similarity tasks:
    ///
    /// Example: "The cat is sleeping" vs "A feline is resting"
    /// - Final layer representations will be more similar due to semantic understanding
    /// - Early layer representations might focus more on surface-level differences
    ///
    /// **4. Research Evidence**
    /// Studies show that different layers capture different types of information:
    /// - Syntactic information peaks in middle layers
    /// - Semantic information is strongest in final layers
    /// - For sentence-level tasks, final layers consistently perform best
    ///
    /// ### Alternative Layer Selection Strategies We Could Use
    ///
    /// **1. Concatenating Multiple Layers**
    /// ```rust
    /// // Combine layers 9-12 for richer representation
    /// // let combined = concat([layer9, layer10, layer11, layer12], dim=2)
    /// ```
    /// - **Pros**: Captures both intermediate and final representations
    /// - **Cons**: 4x larger embeddings, potential redundancy
    ///
    /// **2. Weighted Layer Combination**
    /// ```rust  
    /// // Learn weights: final_embedding = w1*layer9 + w2*layer10 + w3*layer11 + w4*layer12
    /// ```
    /// - **Pros**: Optimal layer combination for specific tasks
    /// - **Cons**: Requires additional training and parameters
    ///
    /// **3. Task-Specific Layer Selection**
    /// ```rust
    /// // Use layer 6-8 for syntactic tasks, layer 10-12 for semantic tasks
    /// ```
    /// - **Pros**: Tailored to specific linguistic aspects
    /// - **Cons**: Requires task-specific knowledge and experimentation
    ///
    /// **4. Earlier Layer Selection (NOT recommended for semantic embeddings)**
    /// ```rust
    /// // Using layer 3-6 embeddings
    /// ```
    /// - **Why not?**: Misses crucial semantic processing
    /// - **Result**: Embeddings focus more on surface form than meaning
    /// - **Example**: "happy" and "joyful" might seem different despite similar meaning
    ///
    /// ### Empirical Evidence from Research
    ///
    /// **Layer Analysis Studies Show:**
    /// - **Layer 1-3**: Best for part-of-speech tagging, basic syntax
    /// - **Layer 4-8**: Best for syntactic parsing, named entity recognition  
    /// - **Layer 9-12**: Best for semantic similarity, sentiment analysis, textual entailment
    ///
    /// **For Embedding Tasks Specifically:**
    /// - Sentence similarity tasks: Final layers (10-12) perform best
    /// - Semantic textual similarity: Layer 12 consistently wins
    /// - Paraphrase detection: Later layers show highest correlation with human judgments
    ///
    /// ### Why Not Average Across All Layers?
    ///
    /// While we could average embeddings from all layers, this would:
    /// - Dilute the sophisticated semantic understanding from final layers
    /// - Include lower-level syntactic noise that's less relevant for semantic similarity
    /// - Lose the benefit of the model's progressive abstraction process
    ///
    /// ### Conclusion
    ///
    /// By using the final layer, we get embeddings that represent the model's best
    /// understanding of semantic content, making them ideal for similarity tasks.
    /// This is why most successful embedding models (BERT, RoBERTa, etc.) use
    /// final or near-final layer representations for downstream tasks.
    ///
    /// Returns embeddings for each token in the sequence from the final transformer layer.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Token embeddings: Convert token IDs to initial dense representations
        let token_embeds = self.token_embedding.forward(input_ids);

        // Position embeddings: Add positional information so model knows word order
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);
        let position_embeds = self.position_embedding.forward(positions);

        // Combine token and position embeddings to create input representations
        let mut x = token_embeds + position_embeds;
        x = self.dropout.forward(x);

        // Pass through ALL 12 transformer blocks sequentially
        // Each block adds progressively more sophisticated understanding:
        // - Early blocks: syntax, basic word relationships
        // - Middle blocks: complex grammar, entity recognition
        // - Final blocks: semantic meaning, contextual understanding
        for block in &self.transformer_blocks {
            x = block.forward(x);
        }

        // Final layer normalization - prepare the sophisticated final layer representations
        // These embeddings now contain the model's best semantic understanding
        self.ln_f.forward(x)
    }

    /// Get sentence embeddings using mean pooling strategy
    ///
    /// ## Mean Pooling Explained
    ///
    /// Mean pooling is a simple but effective method for converting variable-length sequences
    /// of token embeddings into fixed-size sentence representations. Here's how it works:
    ///
    /// 1. **Input**: Token embeddings from the final transformer layer [batch_size, seq_len, d_model]
    /// 2. **Process**: Average all token embeddings along the sequence dimension
    /// 3. **Output**: Single sentence embedding [batch_size, d_model]
    ///
    /// ### Example:
    ///
    /// Input sentence: "The cat sleeps"
    /// Token embeddings after final transformer layer:
    /// - "The":    [0.1, -0.2, 0.3, ...]  (768 dimensions)
    /// - "cat":    [0.5, 0.1, -0.1, ...]  (768 dimensions)  
    /// - "sleeps": [-0.2, 0.4, 0.2, ...]  (768 dimensions)
    ///
    /// Mean pooled sentence embedding:
    /// Average of all three: [(0.1+0.5-0.2)/3, (-0.2+0.1+0.4)/3, (0.3-0.1+0.2)/3, ...]
    /// Result: [0.133, 0.1, 0.133, ...] (768 dimensions)
    ///
    /// ## Why Mean Pooling?
    ///
    /// **Advantages:**
    /// - **Simplicity**: Easy to implement and understand
    /// - **Robustness**: Works well across different sentence lengths
    /// - **Semantic Averaging**: Captures overall semantic content of the sentence
    /// - **No Additional Parameters**: Doesn't require training additional components
    ///
    /// **Considerations:**
    /// - All tokens are weighted equally (including less important function words)
    /// - May dilute important information from key tokens
    /// - Doesn't account for varying importance of different words
    ///
    /// ## Alternative Pooling Strategies We Could Use
    ///
    /// ### 1. Max Pooling
    /// Take the maximum value across all tokens for each dimension:
    /// ```rust
    /// // embeddings.max_dim(1) // Takes element-wise maximum
    /// ```
    /// - **Pros**: Captures the most salient features, preserves strong signals
    /// - **Cons**: May be dominated by outlier tokens, loses averaging effect
    ///
    /// ### 2. CLS Token (BERT-style)
    /// Add a special [CLS] token at the beginning, use its embedding as sentence representation:
    /// ```rust
    /// // embeddings.narrow(1, 0, 1) // Take first token ([CLS]) embedding
    /// ```
    /// - **Pros**: Designed specifically for sentence-level tasks, trainable
    /// - **Cons**: Requires training the model to use CLS effectively
    ///
    /// ### 3. Attention-Based Pooling
    /// Learn attention weights to weight tokens by importance:
    /// ```rust
    /// // attention_weights = softmax(W * embeddings)
    /// // sentence_embedding = sum(attention_weights * embeddings)
    /// ```
    /// - **Pros**: Learns which tokens are most important, adaptive weighting
    /// - **Cons**: Requires additional parameters and training complexity
    ///
    /// ### 4. Last Token Pooling (GPT-style)
    /// Use the embedding of the last token (like GPT models often do):
    /// ```rust
    /// // embeddings.narrow(1, seq_len-1, 1) // Take last token embedding
    /// ```
    /// - **Pros**: Captures information that has "seen" the entire sequence
    /// - **Cons**: May not capture information from earlier parts of the sentence
    ///
    /// ## Why We Chose Mean Pooling for This Educational Implementation
    ///
    /// For this educational model, mean pooling is ideal because:
    /// 1. **Conceptual Clarity**: Easy to understand and explain
    /// 2. **No Training Required**: Works immediately with any pre-trained transformer
    /// 3. **Good Baseline**: Provides reasonable performance across many tasks
    /// 4. **Stable Results**: Less sensitive to individual token anomalies
    /// 5. **Research Foundation**: Widely used in research, making it a good starting point
    ///
    /// In production systems, you might consider more sophisticated approaches like
    /// attention-based pooling or fine-tuning with task-specific pooling strategies.
    pub fn get_sentence_embedding(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let embeddings = self.forward(input_ids); // [batch_size, seq_len, d_model]

        // Apply mean pooling: average over sequence length dimension to get sentence embeddings
        // This transforms [batch_size, seq_len, d_model] -> [batch_size, d_model]
        let pooled = embeddings.mean_dim(1); // This gives [batch_size, 1, d_model]
        pooled.squeeze_dims(&[1]) // Remove the singleton seq dimension to get [batch_size, d_model]
    }

    /// Get embeddings for multiple sentences
    pub fn encode_sentences(&self, input_ids_batch: Vec<Tensor<B, 1, Int>>) -> Vec<Tensor<B, 1>> {
        let mut embeddings = Vec::new();

        for input_ids in input_ids_batch {
            let input_batch = input_ids.unsqueeze_dim(0); // Add batch dimension
            let sentence_embedding = self.get_sentence_embedding(input_batch);
            let embedding = sentence_embedding.squeeze_dims(&[0]); // Remove batch dimension
            embeddings.push(embedding);
        }

        embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_gpt_n_model_creation() {
        let device = Default::default();
        let config = Gpt2Config::default();
        let _model = Gpt2Model::<TestBackend>::new(config, &device);

        // Test model creation doesn't panic
        // Model is created successfully if we reach here
    }

    #[test]
    fn test_forward_pass() {
        let device = Default::default();
        let config = Gpt2Config::default();
        let model = Gpt2Model::<TestBackend>::new(config, &device);

        // Create dummy input
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 10], &device);

        // Forward pass should not panic
        let output = model.forward(input_ids);
        let [batch_size, seq_len, d_model] = output.dims();

        assert_eq!(batch_size, 1);
        assert_eq!(seq_len, 10);
        assert_eq!(d_model, 768);
    }

    #[test]
    fn test_sentence_embedding() {
        let device = Default::default();
        let config = Gpt2Config::default();
        let model = Gpt2Model::<TestBackend>::new(config, &device);

        // Create dummy input
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 10], &device);

        // Get sentence embeddings
        let embeddings = model.get_sentence_embedding(input_ids);
        let [batch_size, d_model] = embeddings.dims();

        assert_eq!(batch_size, 2);
        assert_eq!(d_model, 768);
    }
}

/// Save model weights in binary format
///
/// This function serializes the model's trained parameters to disk using Burn's
/// binary format with gzip compression. The saved model can be loaded later
/// using `load_model` function.
///
/// ## Parameters
/// - `model`: Reference to the trained model to save
/// - `path`: File path where to save the model (typically with .mpk extension)
///
/// ## File Format
/// The model is saved in Burn's binary format with full precision settings,
/// which preserves all model weights and parameters exactly as trained.
pub fn save_model<B: Backend>(model: &Gpt2Model<B>, path: impl AsRef<Path>) -> Result<()> {
    let recorder = BinGzFileRecorder::<FullPrecisionSettings>::default();
    model
        .clone()
        .save_file(path.as_ref().to_path_buf(), &recorder)
        .map_err(|e| anyhow!("Failed to save model: {}", e))?;
    Ok(())
}

/// Load model weights from binary format
///
/// This function creates a new model with the given configuration and loads
/// previously saved weights from disk. This is used to restore a trained model
/// for inference or to continue training from a checkpoint.
///
/// ## Parameters
/// - `config`: Model configuration (must match the configuration used when saving)
/// - `path`: File path to the saved model file
/// - `device`: Backend device where to load the model
///
/// ## Returns
/// A `Gpt2Model` instance with loaded weights ready for inference
///
/// ## Example
/// ```rust,no_run
/// use burn_gpt_n_embedding_model::{Gpt2Config, load_model};
/// use burn::backend::wgpu::{Wgpu, WgpuDevice};
/// # use anyhow::Result;
/// # fn example() -> Result<()> {
/// let config = Gpt2Config::default();
/// let device = WgpuDevice::default();
/// let model = load_model::<Wgpu>(config, "trained_model.mpk", &device)?;
/// # Ok(())
/// # }
/// ```
pub fn load_model<B: Backend>(
    config: Gpt2Config,
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<Gpt2Model<B>> {
    let mut model = Gpt2Model::new(config, device);
    let recorder = CompactRecorder::new();
    model = model
        .load_file(path.as_ref().to_path_buf(), &recorder, device)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;
    Ok(model)
}

// Complex Burn training traits removed - using simpler manual training approach
