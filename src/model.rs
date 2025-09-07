use burn::prelude::*;
use burn::nn::{
    Linear, LinearConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig,
    LayerNorm, LayerNormConfig, Initializer, 
    attention::{MultiHeadAttention, MultiHeadAttentionConfig},
};
use burn::tensor::activation;

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
}

impl Default for Gpt2Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocabulary size
            max_seq_len: 1024, // Maximum sequence length
            d_model: 768,      // Embedding dimension
            n_heads: 12,       // Number of attention heads
            n_layers: 12,      // Number of transformer layers
            dropout: 0.1,      // Dropout rate
        }
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
        }
    }

    /// Forward pass through the model to get embeddings
    /// Returns embeddings for each token in the sequence
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Token embeddings
        let token_embeds = self.token_embedding.forward(input_ids);

        // Position embeddings  
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);
        let position_embeds = self.position_embedding.forward(positions);

        // Combine token and position embeddings
        let mut x = token_embeds + position_embeds;
        x = self.dropout.forward(x);

        // Pass through transformer blocks
        for block in &self.transformer_blocks {
            x = block.forward(x);
        }

        // Final layer normalization
        self.ln_f.forward(x)
    }

    /// Get sentence embeddings by averaging token embeddings
    /// This is a simple approach - more sophisticated methods could be used
    pub fn get_sentence_embedding(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let embeddings = self.forward(input_ids); // [batch_size, seq_len, d_model]
        
        // Average over sequence length dimension to get sentence embeddings
        embeddings.mean_dim(1) // [batch_size, d_model]
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
    fn test_gpt2_model_creation() {
        let device = Default::default();
        let config = Gpt2Config::default();
        let model = Gpt2Model::<TestBackend>::new(config, &device);
        
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
        let [batch_size, seq_len, d_model] = embeddings.dims();
        
        assert_eq!(batch_size, 2);
        assert_eq!(seq_len, 1);
        assert_eq!(d_model, 768);
    }
}