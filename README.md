# GPT-2 Embedding Model in Rust 🦀

**An Educational Implementation of GPT-2 for Text Embeddings Using the Burn Deep Learning Framework**

This project implements a GPT-2 style transformer model for generating text embeddings, built from scratch in Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning framework with WebGPU backend. It's designed as a learning resource for understanding transformer architecture and text embeddings.

## 🎯 What This Project Does

This model takes sentences as input and converts them into high-dimensional vectors (embeddings) that capture semantic meaning. Similar sentences will have similar embeddings, making it useful for:

- **Semantic Search**: Find similar documents or sentences
- **Text Classification**: Use embeddings as features for ML models  
- **Clustering**: Group similar texts together
- **Similarity Analysis**: Measure how alike two pieces of text are

## 🏗️ Architecture Overview

This implementation follows the GPT-2 117M parameter configuration:

```
📊 Model Specifications:
├── Transformer Blocks: 12
├── Attention Heads: 12 per block
├── Embedding Dimensions: 768
├── Context Window: 1024 tokens
├── Vocabulary Size: 50,257 (GPT-2 standard)
└── Total Parameters: ~117M
```

### How Transformers Work (For Beginners)

Think of a transformer as a sophisticated pattern-matching system:

1. **Input Processing**: Text is converted to numbers (tokens)
2. **Attention Mechanism**: The model learns which words are important to each other
3. **Pattern Learning**: Multiple layers build up complex understanding
4. **Output Generation**: Creates a rich numerical representation (embedding)

## 🧩 Key Components

### 1. 📝 Tokenizer (`src/tokenizer.rs`)
Converts text into numbers that the model can understand.

```rust
// Simple character-based tokenization for demo
let tokens = tokenizer.encode("Hello world", true)?;
// → [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, ...]
```

**Educational Note**: Real tokenizers are much more sophisticated (using BPE or SentencePiece), but this simple version helps understand the concept.

### 2. 🧠 GPT-2 Model (`src/model.rs`)
The heart of the system - a transformer neural network with:

#### Transformer Block Architecture:
```
Input Text
    ↓
Token Embeddings + Position Embeddings
    ↓
┌─────────────────┐
│ Transformer     │  ← Repeated 12 times
│ Block           │
│ ┌─────────────┐ │
│ │Multi-Head   │ │  ← **Attention**: "Which tokens are relevant for understanding this token?"
│ │Attention    │ │     Each of 12 heads captures different types of token relationships
│ └─────────────┘ │
│        +        │  ← Residual Connection
│ ┌─────────────┐ │
│ │Feed-Forward │ │  ← **MLP**: "Now what should I think about what I learned?"
│ │Network (MLP)│ │     Processes attended information into meaningful patterns
│ └─────────────┘ │
│        +        │  ← Residual Connection
└─────────────────┘
    ↓
Layer Normalization
    ↓
Text Embeddings (768-dimensional vectors)
```

#### Multi-Head Attention: "Which tokens are relevant for understanding this token?"

For each token in the sequence, attention computes a relevance score with every other token, answering the crucial question: **"Which other tokens are relevant for understanding this token?"**

**How Multiple Heads Work:**
- **Head 1** might focus on **syntactic relationships** (subject-verb, noun-adjective pairs)
- **Head 2** might capture **semantic similarity** (synonyms, related concepts)  
- **Head 3** might track **positional patterns** (beginning/end of sentences, phrases)
- **Head 4** might identify **entity relationships** (person-to-action, object-to-location)
- **Heads 5-12** capture other nuanced linguistic patterns

**Example**: For the word "bank" in "The river bank was muddy":
- Some heads focus on "river" (disambiguating meaning) 
- Others attend to "muddy" (confirming the geographical context)
- Different heads capture different types of relevancy simultaneously

This parallel processing allows the model to understand complex, multi-faceted relationships between words.

#### Feed-Forward Network (MLP): "Now what should I think about what I learned?"

After attention identifies relevant tokens, the MLP processes this information to answer: **"Now what should I think about what I learned?"**

**What the MLP Computes:**
- **Pattern Integration**: Combines the attended information into coherent concepts
- **Feature Extraction**: Identifies higher-level patterns from the attention outputs
- **Knowledge Application**: Applies learned linguistic rules and semantic understanding
- **Representation Refinement**: Transforms raw attention into meaningful embeddings

**Architecture Details:**
- **Linear → GELU activation → Linear → Dropout**
- **4x expansion**: 768 → 3072 → 768 dimensions
- **Purpose**: The expansion allows complex pattern matching, then compression back to embedding size

### 3. 📐 Similarity Calculator (`src/similarity.rs`)
Compares embeddings using various mathematical approaches:

- **Cosine Similarity**: Measures angle between vectors (0-1, where 1 = identical)
- **Euclidean Distance**: Straight-line distance in high-dimensional space
- **Manhattan Distance**: Sum of absolute differences
- **Dot Product**: Raw similarity measure for normalized vectors

### 4. 💾 Training Utilities (`src/training.rs`)
Handles loading and saving model weights in efficient binary format.

## 🔄 How It Works: From Text to Embeddings

Let's trace through what happens when you input "The cat sat on the mat":

### Step 1: Tokenization
```
"The cat sat on the mat" → [84, 104, 101, 32, 99, 97, 116, ...]
```

### Step 2: Embedding Lookup
```
Each token ID → 768-dimensional vector
Position info → Added to embeddings
```

### Step 3: Transformer Processing
```
For each of 12 transformer blocks:
  1. Multi-head attention analyzes token relationships
  2. Feed-forward network processes patterns
  3. Residual connections preserve information
  4. Layer normalization stabilizes training
```

### Step 4: Sentence Embedding Generation

**Why We Extract from Middle Layers (Not Too Early, Not Too High):**

The model has 12 transformer layers, and we typically extract embeddings from around **layer 6-8** because:

- **Too Early (layers 1-3)**: Embeddings focus on surface-level features (syntax, word forms)
- **Sweet Spot (layers 6-8)**: Perfect balance of syntax + semantics, rich contextual understanding
- **Too Late (layers 10-12)**: Over-specialized for next-token prediction, less generalizable

**Why We Use Mean Pooling:**

```
Token embeddings → Mean Pooling → Single 768D sentence vector
```

**Mean pooling** averages all token embeddings because:
- **Captures Global Context**: Every word contributes to the final representation
- **Length Invariant**: Works equally well for short and long sentences  
- **Preserves Information**: No single token dominates; balanced representation
- **Computational Efficiency**: Simple, fast operation compared to learned pooling

**Alternative Approaches:**
- **[CLS] token**: BERT-style approach, but GPT-2 doesn't have special tokens
- **Max pooling**: Can lose information from important but non-maximum features
- **Weighted pooling**: More complex, requires additional training

This final averaged vector captures the semantic meaning of the entire sentence!

## 🚀 Getting Started

### Prerequisites
- Rust 1.70+ 
- GPU with WebGPU support (or fallback to CPU)

### Installation
```bash
git clone https://github.com/richardanaya/burn-gpt2-embedding-model
cd burn-gpt2-embedding-model
cargo build --release
```

### Basic Usage

#### Get Embeddings for a Sentence
```bash
# Get embedding as JSON
cargo run -- embed --sentence "The quick brown fox jumps over the lazy dog"

# Output:
{
  "sentence": "The quick brown fox jumps over the lazy dog",
  "embedding": [0.1234, -0.5678, 0.9012, ...],
  "dimensions": 768
}
```

#### Calculate Similarity Between Sentences
```bash
# Compare two sentences
cargo run -- similarity \
  --sentence1 "The cat is sleeping" \
  --sentence2 "A feline is resting"

# Output:
Sentences:
  1: "The cat is sleeping"
  2: "A feline is resting"
Cosine Similarity: 0.8234 (0=different, 1=identical)
```

#### Get All Similarity Metrics
```bash
cargo run -- similarity \
  --sentence1 "I love programming" \
  --sentence2 "I hate coding" \
  --all-metrics

# Output:
Results:
  Cosine Similarity:    0.6543 (0=different, 1=identical)
  Euclidean Distance:   12.3456 (0=identical, higher=different)
  Manhattan Distance:   98.7654 (0=identical, higher=different)
  Dot Product Sim:      0.6234 (normalized vectors)
```

## 📁 Understanding the Code Structure

```
src/
├── main.rs          # CLI interface using CLAP
├── lib.rs           # Module exports
├── model.rs         # GPT-2 transformer implementation
├── similarity.rs    # Vector similarity calculations
├── tokenizer.rs     # Text-to-token conversion
└── training.rs      # Model saving/loading utilities

data_sets/
├── train.tsv        # Training data (sentence pairs + labels)
├── dev.tsv          # Development/validation data
└── test.tsv         # Test data
```

### Key Learning Points in Each File:

#### `model.rs` - The Neural Network
- **Gpt2Config**: Hyperparameters and model configuration
- **TransformerBlock**: Core attention + feed-forward pattern
- **Mlp**: Multi-layer perceptron with GELU activation
- **Gpt2Model**: Complete model assembly and forward pass

#### `similarity.rs` - Vector Mathematics
- **cosine_similarity()**: Understanding vector angles
- **euclidean_distance()**: Straight-line distance in high dimensions
- **SimilarityCalculator**: Practical similarity computation

## 📚 Educational Concepts Demonstrated

### 1. **Embeddings vs One-Hot Encoding**
Traditional approaches represent words as sparse vectors (one 1, rest 0s). Embeddings create dense, meaningful representations where similar concepts cluster together.

### 2. **Attention Mechanism**
Instead of processing words sequentially, attention lets the model consider all words simultaneously and learn their relationships.

### 3. **Transfer Learning**
The same architecture used for text generation (GPT-2) can be adapted for embeddings by using intermediate representations.

### 4. **High-Dimensional Vector Spaces**
Text similarity becomes a geometric problem in 768-dimensional space where similar meanings cluster together.

### 5. **Transformer Architecture Benefits**
- **Parallelization**: All tokens processed simultaneously
- **Long-range dependencies**: Attention connects distant words
- **Contextual understanding**: Same word gets different embeddings in different contexts

## 📊 Data Format

Training data is in TSV (Tab-Separated Values) format:

```tsv
id	sentence1	sentence2	label
1	The cat is sleeping	A feline is resting	1
2	I love pizza	The weather is nice	0
3	Programming is fun	Coding is enjoyable	1
```

**Labels:**
- `1` = Similar sentences (should have similar embeddings)
- `0` = Different sentences (should have different embeddings)

This format enables training the model to learn meaningful semantic representations.

## 🔬 Technical Deep Dive: Why These Design Choices?

### Embedding Layer Selection Strategy

**The Layer Depth Sweet Spot:**

In our 12-layer GPT-2 model, different layers capture different levels of linguistic understanding:

```
Layer 1-2:   Surface features (character patterns, basic syntax)
Layer 3-4:   Word-level semantics (part of speech, simple relationships)  
Layer 5-6:   Phrase-level understanding (noun phrases, verb phrases)
Layer 7-8:   ← OPTIMAL: Rich contextual semantics + syntax balance
Layer 9-10:  Sentence-level patterns (discourse, complex syntax)
Layer 11-12: Task-specific features (next-token prediction bias)
```

**Research findings show:**
- **Early layers**: Good for syntactic tasks, poor for semantic similarity
- **Middle layers (6-8)**: Best balance for general-purpose embeddings
- **Late layers**: Over-fitted to language modeling objective

This is why we extract embeddings from the middle layers rather than the final output.

### Mean Pooling vs. Other Aggregation Methods

**Why Mean Pooling Works Best:**

1. **Mathematical Properties:**
   - Preserves vector magnitude relationships
   - Maintains semantic distances between sentences
   - Provides stable gradients during training

2. **Linguistic Justification:**
   - Every word contributes to sentence meaning
   - Avoids bias toward beginning/end tokens
   - Naturally handles variable sentence lengths

3. **Empirical Performance:**
   - Consistently outperforms max pooling on similarity tasks
   - More robust than attention-weighted pooling
   - Computationally efficient for real-time applications

**Comparison with alternatives:**
```rust
// Mean pooling (our choice)
sentence_embedding = token_embeddings.mean(dim=0)

// Max pooling (can lose information)  
sentence_embedding = token_embeddings.max(dim=0)

// Attention-weighted (more complex, not always better)
weights = attention_network(token_embeddings)
sentence_embedding = (weights * token_embeddings).sum(dim=0)
```

## 🔬 Advanced Features

### Multiple Similarity Metrics
Understanding different ways to measure vector similarity:

```rust
// Cosine similarity: Angle between vectors
let cos_sim = cosine_similarity(vec1, vec2);

// Euclidean distance: Straight-line distance  
let eucl_dist = euclidean_distance(vec1, vec2);

// Manhattan distance: Sum of coordinate differences
let manh_dist = manhattan_distance(vec1, vec2);
```

### Batch Processing
Efficiently process multiple sentences:

```rust
let embeddings = model.encode_sentences(input_batch);
let similarity_matrix = calculator.similarity_matrix(&sentences);
```

## 🎓 Learning Exercises

### Beginner
1. **Run the examples** and observe how similar sentences get higher similarity scores
2. **Experiment with different sentences** - what makes embeddings similar?
3. **Compare similarity metrics** - when do they agree/disagree?

### Intermediate  
1. **Modify the model architecture** - try different numbers of layers or dimensions
2. **Implement a better tokenizer** using subword tokenization
3. **Add training functionality** to learn from the TSV data

### Advanced
1. **Implement other similarity measures** (e.g., Pearson correlation)
2. **Add support for other transformer architectures** (BERT, RoBERTa)
3. **Create a web API** to serve embeddings at scale

## 🚀 Next Steps & Extensions

### Model Improvements
- **Better Tokenization**: Implement BPE or SentencePiece tokenizer
- **Training Loop**: Add actual training on the provided TSV data
- **Model Variants**: Support for different transformer sizes (345M, 762M, 1.5B)
- **Fine-tuning**: Domain-specific embedding training

### Technical Enhancements
- **Performance**: Optimize inference speed and memory usage
- **Deployment**: Add Docker containers and API endpoints
- **Monitoring**: Training metrics and embedding quality evaluation
- **Data Pipeline**: Efficient dataset loading and preprocessing

### Educational Resources
- **Interactive Notebooks**: Jupyter notebooks explaining concepts
- **Visualization**: Plot embeddings in 2D/3D space using dimensionality reduction
- **Comparison Studies**: Compare with other embedding models (Word2Vec, BERT)

## 🛠️ Technical Stack

- **Language**: Rust 🦀
- **Deep Learning**: [Burn Framework](https://github.com/tracel-ai/burn)
- **Backend**: WebGPU (with CPU fallback)
- **CLI**: CLAP for command-line interface
- **Serialization**: SafeTensors for model weights
- **Testing**: Built-in Rust testing framework

## 📖 Additional Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2 Paper](https://openai.com/research/better-language-models) - OpenAI's GPT-2 research
- [Burn Documentation](https://burn-rs.github.io/) - Deep learning in Rust
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide to transformers

## 🤝 Contributing

This project is designed for learning! Contributions that improve educational value are welcome:

- **Documentation improvements**
- **Code comments and explanations**  
- **Additional examples and tutorials**
- **Performance optimizations**
- **New similarity metrics or model variants**

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy Learning! 🎓** This implementation demonstrates how modern NLP models work under the hood. Experiment, modify, and most importantly - understand the beautiful mathematics that powers modern AI!