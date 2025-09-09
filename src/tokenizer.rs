use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

/// GPT-2 tokenizer wrapper using the tokenizers crate
#[derive(Debug, Clone)]
pub struct Gpt2Tokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl Gpt2Tokenizer {
    /// Creates a tokenizer from a tokenizer.json file
    pub fn from_file(tokenizer_path: &str, max_length: usize) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    /// Creates a simple tokenizer for demo purposes (fallback)
    pub fn new_simple(max_length: usize) -> Result<Self> {
        // Try to load from tokenizer.json first
        if let Ok(tokenizer) = Self::from_file("tokenizer.json", max_length) {
            return Ok(tokenizer);
        }

        // Fallback: try to load from another path or create a minimal tokenizer
        Err(anyhow!(
            "No tokenizer found. Please ensure tokenizer.json is available."
        ))
    }

    /// Tokenize a single text string into token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("Failed to encode text: {}", e))?;
        let mut ids = encoding.get_ids().to_vec();

        // Truncate if necessary
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }

        // Pad if necessary (use pad token ID if available, otherwise use 0)
        let pad_token = self.tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

        while ids.len() < self.max_length {
            ids.push(pad_token);
        }

        Ok(ids)
    }

    /// Tokenize a single text string into token IDs with length information for masking
    pub fn encode_with_length(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<(Vec<u32>, usize)> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("Failed to encode text: {}", e))?;
        let mut ids = encoding.get_ids().to_vec();

        // Store original length before padding/truncation
        // Ensure minimum length of 1 to avoid empty sequences
        let original_length = ids.len().max(1).min(self.max_length);

        // Truncate if necessary
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }

        // If no tokens were generated, add a default token (e.g., unknown token or padding)
        if ids.is_empty() {
            let pad_token = self.tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);
            ids.push(pad_token);
        }

        // Pad if necessary (use pad token ID if available, otherwise use 0)
        let pad_token = self.tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

        while ids.len() < self.max_length {
            ids.push(pad_token);
        }

        Ok((ids, original_length))
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow!("Failed to decode tokens: {}", e))?;
        Ok(text)
    }

    /// Tokenize multiple texts
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow!("Failed to encode batch: {}", e))?;
        let mut result = Vec::new();

        for encoding in encodings {
            let mut ids = encoding.get_ids().to_vec();

            // Truncate if necessary
            if ids.len() > self.max_length {
                ids.truncate(self.max_length);
            }

            // Pad if necessary
            let pad_token = self.tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

            while ids.len() < self.max_length {
                ids.push(pad_token);
            }

            result.push(ids);
        }

        Ok(result)
    }

    /// Tokenize multiple texts with length information for masking
    pub fn encode_batch_with_lengths(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<(Vec<Vec<u32>>, Vec<usize>)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow!("Failed to encode batch: {}", e))?;
        let mut result_ids = Vec::new();
        let mut result_lengths = Vec::new();

        for encoding in encodings {
            let mut ids = encoding.get_ids().to_vec();

            // Store original length before padding/truncation
            let original_length = ids.len().min(self.max_length);

            // Truncate if necessary
            if ids.len() > self.max_length {
                ids.truncate(self.max_length);
            }

            // Pad if necessary
            let pad_token = self.tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

            while ids.len() < self.max_length {
                ids.push(pad_token);
            }

            result_ids.push(ids);
            result_lengths.push(original_length);
        }

        Ok((result_ids, result_lengths))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    /// Get max sequence length
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Get the padding token ID
    pub fn pad_token_id(&self) -> u32 {
        self.tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0)
    }
}

/// Helper function to create a demo tokenizer for testing
pub fn create_demo_tokenizer() -> Result<Gpt2Tokenizer> {
    // Try to load from tokenizer.json if available, otherwise fallback
    Gpt2Tokenizer::new_simple(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_tokenizer_creation() {
        let tokenizer = create_demo_tokenizer();
        assert!(tokenizer.is_ok());

        let tokenizer = tokenizer.unwrap();
        assert!(tokenizer.vocab_size() > 0);
        assert_eq!(tokenizer.max_length(), 1024);
    }

    #[test]
    fn test_encode_decode() -> Result<()> {
        let tokenizer = create_demo_tokenizer()?;

        let text = "Hello world";
        let encoded = tokenizer.encode(text, false)?;
        let _decoded = tokenizer.decode(&encoded, false)?;

        assert!(!encoded.is_empty());
        assert_eq!(encoded.len(), tokenizer.max_length());

        Ok(())
    }

    #[test]
    fn test_batch_encoding() -> Result<()> {
        let tokenizer = create_demo_tokenizer()?;

        let texts = vec!["Hello world", "This is a test"];
        let encoded_batch = tokenizer.encode_batch(&texts, false)?;

        assert_eq!(encoded_batch.len(), 2);
        for encoded in encoded_batch {
            assert!(!encoded.is_empty());
            assert_eq!(encoded.len(), tokenizer.max_length());
        }

        Ok(())
    }
}
