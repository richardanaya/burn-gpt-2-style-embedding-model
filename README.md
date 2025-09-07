# A GPT-2-like Embedding Model in Rust to Play with Training 🦀

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

#### Understanding Tokenization: From Text to Numbers

**Why Tokenization Matters:**
Neural networks can only work with numbers, not text. Tokenization is the crucial bridge that converts human-readable text into numerical sequences that transformers can process.

#### Real-World Tokenization vs Our Demo

**Our Demo Tokenizer (Simplified):**
- Maps each character directly to a number
- "Hello" → [72, 101, 108, 108, 111] (ASCII values)
- Good for learning, but inefficient for real applications

**Real-World Tokenizers (BPE/SentencePiece):**
Modern tokenizers like GPT-2's use **Byte Pair Encoding (BPE)** which creates subword tokens:

```
"unhappiness" might become:
├── "un" → Token ID 1234
├── "happy" → Token ID 5678  
└── "ness" → Token ID 9012

Instead of 11 character tokens, we get just 3 meaningful subword tokens!
```

**Why Subword Tokenization is Powerful:**
- **Vocabulary Efficiency**: 50,257 tokens can represent millions of words
- **Unknown Word Handling**: Can tokenize words never seen in training
- **Semantic Preservation**: "happy" and "unhappy" share the "happy" token
- **Compression**: Reduces sequence length compared to character-level

#### Fragment Word Lists in Real Tokenizers

Real tokenizers maintain extensive vocabularies of word fragments:

```
Common BPE tokens include:
├── Whole words: "the" (464), "and" (290), "of" (286)
├── Prefixes: "re" (260), "un" (403), "pre" (661)  
├── Suffixes: "ing" (278), "ed" (276), "ly" (306)
├── Subwords: "tion" (357), "ness" (408), "able" (489)
└── Characters: "a" (64), "e" (68), "!" (0)
```

**Token ID Assignment Process:**
1. **Text Input**: "The unhappy cat"
2. **BPE Segmentation**: ["The", "un", "happy", "cat"] 
3. **Lookup in Vocabulary**: [464, 403, 2995, 3797]
4. **Add Special Tokens**: [50256] + [464, 403, 2995, 3797] + [50256] (start/end)

#### From Token IDs to Transformer Input

Once we have token IDs, the transformer needs to convert them into rich numerical representations:

**Step 1: Token Embedding Lookup**
```rust
// Each token ID → 768-dimensional vector
Token ID 464 ("The") → [0.1, -0.3, 0.8, 0.2, ...] // 768 numbers
Token ID 403 ("un")  → [-0.2, 0.5, 0.1, -0.7, ...] // 768 numbers  
Token ID 2995 ("happy") → [0.4, 0.1, -0.2, 0.9, ...] // 768 numbers
```

**Step 2: Position Encoding**
```rust
// Add positional information so model knows word order
Position 0 → [0.0, 0.1, 0.0, 0.3, ...] // 768 numbers
Position 1 → [0.1, 0.0, 0.2, 0.1, ...] // 768 numbers
Position 2 → [0.2, 0.3, 0.1, 0.0, ...] // 768 numbers

// Final input = Token Embedding + Position Embedding
"The" input = [0.1, -0.2, 0.8, 0.5, ...] // Ready for transformer!
```

**Step 3: What These Numbers Mean**
Each of the 768 dimensions potentially captures different linguistic features:
- Dimensions 1-100: Semantic meaning (animal, emotion, action)
- Dimensions 101-200: Syntactic role (noun, verb, adjective)  
- Dimensions 201-300: Grammatical features (tense, number, gender)
- Dimensions 301-768: Complex interactions and contextual nuances

This rich 768-dimensional representation gives the transformer everything it needs to understand and process the text!

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
│ │Multi-Head   │ │  ← Attention: "What words relate to what?"
│ │Attention    │ │
│ └─────────────┘ │
│        +        │  ← Residual Connection
│ ┌─────────────┐ │
│ │Feed-Forward │ │  ← Processing: "What patterns do I see?"
│ │Network (MLP)│ │
│ └─────────────┘ │
│        +        │  ← Residual Connection
└─────────────────┘
    ↓
Layer Normalization
    ↓
Text Embeddings (768-dimensional vectors)
```

#### Multi-Head Attention: The Heart of Understanding

**What Multi-Head Attention Does:**
Multi-head attention allows the model to focus on different parts of the input simultaneously. Think of it as having multiple "attention spotlights" that can each focus on different aspects of language understanding.

**Why Multiple Heads?**
Each head can specialize in different types of relationships and linguistic patterns:

**🎯 Head Specialization Examples:**
- **Head 1**: Focuses on subject-verb relationships ("cat" ↔ "sleeps")
- **Head 2**: Focuses on adjective-noun relationships ("big red" ↔ "car")  
- **Head 3**: Focuses on local word dependencies ("very" ↔ "quickly")
- **Head 4**: Focuses on long-range dependencies ("The" ↔ "car" across sentence)
- **Head 5**: Focuses on syntactic patterns (detecting phrases and clauses)
- **Head 6**: Focuses on semantic similarity (synonyms and related concepts)
- **Head 7**: Focuses on positional patterns (beginning/end of sentences)
- **Head 8**: Focuses on rare/specific patterns (idioms, technical terms)
- **Head 9**: Focuses on temporal relationships ("before", "after", "while")
- **Head 10**: Focuses on causal relationships ("because", "therefore", "since")
- **Head 11**: Focuses on comparative relationships ("more", "less", "than")
- **Head 12**: Focuses on contextual disambiguation (resolving word meanings)

#### 🔍 Concrete Example: "The big red car drove quickly"

Let's see how different attention heads might analyze this sentence:

**Head 1 (Subject-Verb Focus):**
```
"car" ──────────────► "drove" 
 ↑                      ↑
Strong attention: 0.9   │
                       │
The model learns: "car" is the subject performing "drove"
```

**Head 2 (Adjective-Noun Focus):**
```
"big" ────► "car" ◄──── "red"
 ↑           ↑           ↑
Attention:  0.8        0.8
           
The model learns: "big" and "red" both modify "car"
```

**Head 3 (Verb-Adverb Focus):**
```
"drove" ──────────────► "quickly"
  ↑                       ↑
Attention: 0.9            │
                         │
The model learns: "quickly" modifies how "drove" happened
```

**Head 4 (Long-Range Dependencies):**
```
"The" ────────────────────────────► "car"
 ↑                                   ↑
Attention: 0.6                      │
                                   │
The model learns: "The" determines the specific "car" being discussed
```

**Head 5 (Syntactic Patterns):**
```
Noun Phrase Detection:
["The" + "big" + "red" + "car"] → Attention cluster: 0.7-0.9
                                                      
Verb Phrase Detection:  
["drove" + "quickly"] → Attention cluster: 0.8
                                          
The model learns: Grammatical phrase boundaries
```

**Head 6 (Semantic Relationships):**
```
"big" ◄──► "red"     (both are descriptive attributes)
  ↑         ↑
Attention: 0.4       

"drove" ◄──► "quickly" (action and manner are semantically linked)  
    ↑         ↑
Attention: 0.7

The model learns: Words with similar semantic roles
```

#### 💡 How Multiple Heads Work Together

**Parallel Processing:**
All 12 heads process the sentence simultaneously, each creating their own attention patterns.

**Information Integration:**
```
Input: "The big red car drove quickly"
        ↓
┌─────────────────────────────────────┐
│ Head 1: Subject-verb patterns       │ ──┐
│ Head 2: Adjective-noun patterns     │   │
│ Head 3: Verb-adverb patterns        │   │  
│ Head 4: Long-range dependencies     │   │ → Combined Understanding
│ Head 5: Syntactic structure         │   │
│ Head 6: Semantic relationships      │   │
│ ... (6 more heads)                  │   │
└─────────────────────────────────────┘ ──┘
        ↓
Rich 768-dimensional representation capturing:
- Grammatical roles and relationships  
- Semantic meaning and context
- Syntactic structure and patterns
- Long and short-range dependencies
```

**Why 12 Heads Specifically?**
- **Computational Efficiency**: 768 dimensions ÷ 12 heads = 64 dimensions per head
- **Linguistic Coverage**: Enough heads to capture diverse language patterns  
- **Empirical Optimization**: Research shows 12 heads work well for this model size
- **Specialization vs Redundancy**: Balance between having specialized heads and backup capacity

**The Magic of Attention:**
Each head essentially asks: "For each word, which other words in the sentence are most important for understanding its meaning in this context?" The answers from all 12 heads combine to create a rich, contextual understanding that goes far beyond simple word-by-word processing.

#### 🔧 How Head Results Get Merged Together

After all 12 heads have processed the input in parallel, their outputs need to be combined before feeding into the rest of the transformer. Here's exactly how this works:

**Step 1: Individual Head Outputs**
Each head produces attention-weighted representations for each token:
```
Head 1 output: [token1: 64-dim vector, token2: 64-dim vector, ...]
Head 2 output: [token1: 64-dim vector, token2: 64-dim vector, ...]
...
Head 12 output: [token1: 64-dim vector, token2: 64-dim vector, ...]
```

**Step 2: Concatenation**
The outputs from all heads are concatenated (joined together) for each token:
```
For each token position:
├── Head 1 output: [a₁, a₂, ..., a₆₄]    (64 dimensions)
├── Head 2 output: [b₁, b₂, ..., b₆₄]    (64 dimensions)  
├── Head 3 output: [c₁, c₂, ..., c₆₄]    (64 dimensions)
│   ...
└── Head 12 output: [l₁, l₂, ..., l₆₄]   (64 dimensions)
                    ↓
Concatenated: [a₁, a₂, ..., a₆₄, b₁, b₂, ..., b₆₄, c₁, ..., l₆₄]
              └─────────────── 768 dimensions total ──────────────┘
```

**Step 3: Linear Projection**
The concatenated 768-dimensional vector goes through a learned linear transformation:
```rust
// Pseudo-code representation
multi_head_output = concatenate([head1, head2, ..., head12])  // Shape: [768]
projected_output = linear_projection(multi_head_output)       // Shape: [768]
```

This linear projection (also called the "output projection") allows the model to:
- **Blend Information**: Learn optimal combinations of different head outputs
- **Reduce Redundancy**: Filter out redundant information between heads
- **Maintain Dimensionality**: Keep 768 dimensions for the residual connection

**Step 4: Residual Connection & Layer Norm**
```
Original input (768-dim)
       +                    ← Residual connection (preserves original info)
Projected multi-head output (768-dim)
       ↓
Layer Normalization         ← Stabilizes and normalizes the combined result
       ↓
Fed into Feed-Forward Network
```

**Visual Summary:**
```
Input: "The big red car"
         ↓
┌─────────────────────────────────────┐
│    Multi-Head Attention Block       │
│ ┌─────┐ ┌─────┐       ┌─────┐      │ Each head: 768→64 dims
│ │Head1│ │Head2│  ...  │Head12│      │
│ │64dim│ │64dim│       │64dim │      │
│ └─────┘ └─────┘       └─────┘      │
│     │       │           │          │
│     └───────┼───────────┘          │ Concatenate: 12×64→768
│             ↓                      │
│    [768-dimensional vector]        │
│             ↓                      │
│    Linear Projection (768→768)     │ Learn optimal blending
│             ↓                      │
│    [768-dimensional output]        │
└─────────────────────────────────────┘
         ↓
    + Residual + LayerNorm
         ↓
    Feed-Forward Network
```

**Why This Design Works:**
- **Parallel Specialization**: Each head can focus on different patterns independently
- **Information Preservation**: Concatenation keeps all specialized information
- **Adaptive Combination**: Linear projection learns how to best combine head outputs
- **Stable Training**: Residual connections prevent information loss during deep processing

This merging process ensures that the rich, specialized understanding from all 12 attention heads is effectively combined and passed forward through the transformer, creating the powerful contextual representations that make transformers so effective at understanding language!

#### Feed-Forward Network (MLP)
- **Purpose**: Processes the attended information to extract patterns
- **Architecture**: Linear → GELU activation → Linear → Dropout
- **Size**: 4x expansion (768 → 3072 → 768 dimensions)

### 3. 📐 Similarity Calculator (`src/similarity.rs`)
Compares embeddings using various mathematical approaches:

- **Cosine Similarity**: Measures angle between vectors (0-1, where 1 = identical)
- **Euclidean Distance**: Straight-line distance in high-dimensional space
- **Manhattan Distance**: Sum of absolute differences
- **Dot Product**: Raw similarity measure for normalized vectors

### 4. 📚 Embedding Strategy: Mean Pooling from Final Layer

This implementation uses a specific strategy for creating sentence embeddings that's worth understanding in detail:

#### 🎯 Mean Pooling Explained

**What is Mean Pooling?**
Mean pooling converts variable-length sequences of token embeddings into fixed-size sentence representations by averaging:

```
Input sentence: "The cat sleeps"
Token embeddings from final layer:
├── "The":    [0.1, -0.2, 0.3, ...] (768 dims)
├── "cat":    [0.5, 0.1, -0.1, ...] (768 dims)  
└── "sleeps": [-0.2, 0.4, 0.2, ...] (768 dims)

Mean pooled result:
[(0.1+0.5-0.2)/3, (-0.2+0.1+0.4)/3, (0.3-0.1+0.2)/3, ...]
= [0.133, 0.1, 0.133, ...] (768 dims)
```

**Why Mean Pooling?**
- ✅ **Simplicity**: Easy to understand and implement
- ✅ **Robustness**: Works across different sentence lengths
- ✅ **No Training**: Requires no additional parameters
- ✅ **Stable**: Less sensitive to outlier tokens
- ⚠️ **Equal Weighting**: Treats all words equally (pro and con)

#### 🏗️ Why Final Layer vs Earlier Layers?

**Transformer Layers Build Progressive Understanding:**

```
Layer 1-3:   Basic syntax, part-of-speech, local word relationships
    ↓
Layer 4-8:   Complex grammar, named entities, coreference  
    ↓  
Layer 9-12:  Semantic meaning, context, abstract concepts
    ↓
Final Output: Rich semantic embeddings perfect for similarity
```

**Example: "The bank by the river was steep"**

- **Early Layers (1-3)**: Might mix financial and geographical meanings of "bank"
- **Middle Layers (4-8)**: Start to resolve meaning from nearby context words
- **Final Layers (9-12)**: Fully understand "bank" means riverbank from complete context

**Research Evidence:**
- Studies show semantic information peaks in final layers
- Sentence similarity tasks perform best with layers 10-12
- Earlier layers capture syntax; later layers capture meaning

#### 🔄 Alternative Approaches We Could Use

**Different Pooling Strategies:**
```rust
// Max Pooling: Take maximum value for each dimension
embeddings.max_dim(1)  // Captures strongest signals

// CLS Token (BERT-style): Use special classification token  
embeddings.narrow(1, 0, 1)  // Take first token embedding

// Attention Pooling: Learn which tokens are most important
attention_weights * embeddings  // Requires training
```

**Different Layer Strategies:**
```rust
// Multi-layer combination: Use multiple layers
concat([layer9, layer10, layer11, layer12])

// Weighted layers: Learn optimal layer combination  
w1*layer9 + w2*layer10 + w3*layer11 + w4*layer12
```

**Why We Chose Final Layer + Mean Pooling:**
1. **Educational Clarity**: Easy to understand and explain
2. **Strong Baseline**: Proven effective across many tasks
3. **No Extra Training**: Works with any transformer model
4. **Research Foundation**: Widely used in academic work

This combination gives us embeddings that capture semantic meaning while being simple enough to understand and modify for learning purposes!

### 5. 💾 Training Utilities (`src/training.rs`)
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

### Step 4: Sentence Embedding
```
Token embeddings → Averaged → Single 768D sentence vector
```

This final vector captures the semantic meaning of the entire sentence!

## 🚀 Getting Started

### Prerequisites
- Rust 1.70+ 
- GPU with WebGPU support (or fallback to CPU)

### Installation
```bash
git clone https://github.com/richardanaya/burn-gpt-n-embedding-model
cd burn-gpt-n-embedding-model
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
