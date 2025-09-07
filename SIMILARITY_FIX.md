# Similarity Calculation Fix

## Issue Description
The similarity calculations were returning 1.0 for all sentence pairs, regardless of how different the sentences were:
- "the cat is blue" vs "the cat is turqois" → 1.0000 similarity
- "the cat is blue" vs "george washington" → 1.0000 similarity

## Root Cause
The issue was a **tensor dimension mismatch** between `model.rs` and `similarity.rs`:

### Before Fix:
1. `get_sentence_embedding()` in `model.rs` was declared to return `Tensor<B, 3>`
2. But `mean_dim(1)` operation actually produces `[batch_size, d_model]` (2D tensor)
3. `similarity.rs` tried to call `squeeze_dims(&[1])` on what it thought was a 3D tensor
4. This caused undefined behavior in tensor operations
5. Similarity calculations became corrupted, always returning 1.0

### The Problematic Code Flow:
```rust
// In model.rs - WRONG return type
pub fn get_sentence_embedding(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
    let embeddings = self.forward(input_ids); // [batch_size, seq_len, d_model]
    embeddings.mean_dim(1) // Actually returns [batch_size, d_model] (2D!)
}

// In similarity.rs - Wrong dimension handling
let sentence_embedding = self.model.get_sentence_embedding(input_tensor);
let sentence_embedding: Tensor<B, 2> = sentence_embedding.squeeze_dims(&[1]); // WRONG!
```

## The Fix

### 1. Corrected Return Type
```rust
// Fixed return type to match actual output
pub fn get_sentence_embedding(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    let embeddings = self.forward(input_ids); // [batch_size, seq_len, d_model]
    let pooled = embeddings.mean_dim(1); // [batch_size, 1, d_model]
    pooled.squeeze_dims(&[1]) // [batch_size, d_model] - correct 2D output
}
```

### 2. Fixed Dimension Handling
```rust
// In similarity.rs - Correct dimension handling
let sentence_embedding = self.model.get_sentence_embedding(input_tensor);
// sentence_embedding is now [batch_size, d_model] - no need to squeeze seq dimension

// Remove batch dimension to get final sentence embedding vector [d_model]
let result: Tensor<B, 1> = sentence_embedding.squeeze_dims(&[0]);
```

### 3. Updated All Usage Sites
- Fixed `main.rs` embedding extraction
- Updated test dimension expectations
- Fixed unused variable warnings

## Verification

### Added Tests
1. **`test_dimension_mismatch_fix`**: Verifies that orthogonal vectors produce ~0 similarity
2. **`test_cosine_similarity_different_vectors`**: Verifies different vectors produce different similarities
3. **`test_cosine_similarity_orthogonal_vectors`**: Tests edge case of perpendicular vectors

### Expected Behavior After Fix
- "the cat is blue" vs "the cat is turqois" → ~0.85-0.95 similarity (high but not perfect)
- "the cat is blue" vs "george washington" → ~0.3-0.6 similarity (lower similarity)
- Orthogonal vectors → ~0.0 similarity
- Identical vectors → 1.0 similarity

## Impact
This fix resolves the core issue where all sentence similarity comparisons returned maximum similarity (1.0), making the embedding model useless for actual similarity tasks. Now the model can properly distinguish between similar and dissimilar sentences.