use anyhow::Result;

/// Simple tokenizer wrapper for demo purposes
#[derive(Debug)]
pub struct Gpt2Tokenizer {
    vocab_size: usize,
    max_length: usize,
}

impl Gpt2Tokenizer {
    /// Creates a simple tokenizer for demo purposes
    pub fn new_simple(max_length: usize) -> Result<Self> {
        Ok(Self {
            vocab_size: 50257, // GPT-2 standard vocab size
            max_length,
        })
    }

    /// Tokenize a single text string into token IDs
    /// This is a very simple demo tokenizer - real tokenization would be much more complex
    pub fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        // Simple character-based tokenization for demo
        let mut ids: Vec<u32> = text
            .chars()
            .map(|c| (c as u32) % self.vocab_size as u32)
            .collect();
        
        // Truncate if necessary
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }
        
        // Pad if necessary
        while ids.len() < self.max_length {
            ids.push(50256); // End of text token ID
        }
        
        Ok(ids)
    }

    /// Decode token IDs back to text (simplified)
    pub fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        let text: String = ids
            .iter()
            .filter(|&&id| id != 50256) // Skip padding tokens
            .map(|&id| char::from_u32(id).unwrap_or('?'))
            .collect();
        Ok(text)
    }

    /// Tokenize multiple texts
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        texts.iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get max sequence length
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

/// Helper function to create a demo tokenizer for testing
pub fn create_demo_tokenizer() -> Result<Gpt2Tokenizer> {
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