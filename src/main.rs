use anyhow::Result;
use burn_gpt_n_embedding_model::{
      train_model, Dataset,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tokenizers::{models::bpe::{BpeTrainerBuilder, BPE}, pre_tokenizers::byte_level::ByteLevel, normalizers::{strip::Strip, unicode::NFC, utils::Sequence}, processors::template::TemplateProcessing, decoders::byte_level::ByteLevel as ByteLevelDecoder, AddedToken, TokenizerBuilder, Tokenizer};
use std::collections::HashSet;

/// GPT-2 Embedding Model CLI
///
/// A command-line interface for training and using a GPT-2 style embedding model
/// built with the Burn deep learning framework and WebGPU backend.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create tokenizer from datasets
    Tokens {
        #[command(subcommand)]
        tokens_command: TokensCommands,
    },
    /// Train the model on TSV data
    Train {
        /// Path to training TSV file
        #[arg(short, long)]
        train_data: PathBuf,

        /// Path to validation TSV file
        #[arg(short, long)]
        validation_data: PathBuf,

        /// Path to tokenizer.json file (default: tokenizer.json)
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer_path: PathBuf,

        /// Path to save model checkpoints (default: training_output/)
        #[arg(short, long, default_value = "training_output")]
        output_dir: PathBuf,

        /// Number of training epochs
        #[arg(short, long, default_value = "10")]
        epochs: usize,

        /// Batch size for training
        #[arg(short, long, default_value = "8")]
        batch_size: usize,

        /// Initial learning rate
        #[arg(long, default_value = "1e-5")]
        initial_lr: f64,

        /// Number of attention heads (default: 12)
        #[arg(long, default_value = "4")]
        n_heads: usize,

        /// Number of transformer layers (default: 12)
        #[arg(long, default_value = "4")]
        n_layers: usize,

        /// Embedding dimension size (default: 256)
        #[arg(long, default_value = "256")]
        d_model: usize,

        /// Maximum sequence length / context size (default: 1024)
        #[arg(long, default_value = "256")]
        context_size: usize,

        /// Limit training examples for testing (0 = no limit)
        #[arg(long, default_value = "0")]
        limit_train: usize,

        /// Limit validation examples for testing (0 = no limit)
        #[arg(long, default_value = "0")]
        limit_validation: usize,

        /// Disable TUI and use simple console output for training
        #[arg(long)]
        no_tui: bool,
    },
}

#[derive(Subcommand)]
enum TokensCommands {
    /// Create a tokenizer.json file from all datasets
    Create {
        /// Directory containing TSV datasets (default: data_sets/)
        #[arg(short, long, default_value = "data_sets")]
        data_dir: PathBuf,
        
        /// Output path for tokenizer.json (default: tokenizer.json)
        #[arg(short, long, default_value = "tokenizer.json")]
        output: PathBuf,
        
        /// Vocabulary size (default: auto-detect from data)
        #[arg(short, long)]
        vocab_size: Option<usize>,
    },
    /// Count tokens in an existing tokenizer
    Count {
        /// Path to tokenizer.json file (default: tokenizer.json)
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer_path: PathBuf,
    },
}

fn load_datasets(
    train_data_path: &PathBuf,
    validation_data_path: &PathBuf,
    limit_train: usize,
    limit_validation: usize,
) -> Result<(Dataset, Dataset)> {
    println!("Loading training data from: {}", train_data_path.display());
    let mut train_dataset = Dataset::from_tsv(train_data_path)?;

    if limit_train > 0 {
        println!(
            "🔬 Limiting training data to {} examples for testing",
            limit_train
        );
        train_dataset.limit(limit_train);
    }

    train_dataset.statistics().print();
    println!();

    println!("Loading validation data from: {}", validation_data_path.display());
    let mut validation_dataset = Dataset::from_tsv(validation_data_path)?;

    if limit_validation > 0 {
        println!(
            "🔬 Limiting validation data to {} examples for testing (before: {})",
            limit_validation,
            validation_dataset.examples.len()
        );
        validation_dataset.limit(limit_validation);
        println!("🔬 After limiting: {} examples", validation_dataset.examples.len());
    } else {
        println!(
            "🔬 No validation limit specified (limit_validation = {})",
            limit_validation
        );
    }

    validation_dataset.statistics().print();
    println!();

    Ok((train_dataset, validation_dataset))
}

fn create_tokenizer(data_dir: &PathBuf, output_path: &PathBuf, vocab_size: Option<usize>) -> Result<()> {
    println!("🔤 Creating tokenizer from datasets in: {}", data_dir.display());
    
    // Collect all text from TSV files
    let mut training_data = Vec::new();
    
    // Find all TSV files in the directory
    let tsv_files = std::fs::read_dir(data_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map_or(false, |ext| ext == "tsv")
        })
        .collect::<Vec<_>>();
    
    if tsv_files.is_empty() {
        anyhow::bail!("No TSV files found in directory: {}", data_dir.display());
    }
    
    println!("📁 Found {} TSV file(s):", tsv_files.len());
    for file in &tsv_files {
        println!("  - {}", file.file_name().to_string_lossy());
    }
    
    // Read all TSV files and extract text content
    let mut unique_words = HashSet::new();
    for file_entry in tsv_files {
        let file_path = file_entry.path();
        println!("📖 Processing: {}", file_path.display());
        
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(&file_path)?;
        
        let headers = reader.headers()?.clone();
        println!("   Headers: {:?}", headers);
        
        for result in reader.records() {
            let record = result?;
            // Extract all text fields (assuming sentence1, sentence2 columns exist)
            if let (Some(sentence1), Some(sentence2)) = (record.get(1), record.get(2)) {
                training_data.push(sentence1.to_string());
                training_data.push(sentence2.to_string());
                
                // Count unique words for vocab size estimation
                for word in sentence1.split_whitespace() {
                    unique_words.insert(word.to_lowercase());
                }
                for word in sentence2.split_whitespace() {
                    unique_words.insert(word.to_lowercase());
                }
            }
        }
    }
    
    println!("📊 Collected {} sentences for tokenizer training", training_data.len());
    
    // Determine vocabulary size
    let vocab_size = match vocab_size {
        Some(size) => size,
        None => {
            // Use a heuristic: start with unique words count and add some buffer for subword tokens
            let base_vocab = unique_words.len();
            let estimated_vocab = std::cmp::max(5000, std::cmp::min(50000, base_vocab * 3));
            println!("📈 Auto-detected vocab size: {} (based on {} unique words)", estimated_vocab, base_vocab);
            estimated_vocab
        }
    };
    
    // Create trainer
    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(vec![
            AddedToken::from("[UNK]", true),
            AddedToken::from("[CLS]", true),
            AddedToken::from("[SEP]", true),
            AddedToken::from("[PAD]", true),
            AddedToken::from("[MASK]", true),
        ])
        .build();

    // Create BPE tokenizer with post-processor for special tokens
    let post_processor = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .map_err(|e| anyhow::anyhow!("Failed to create template: {}", e))?
        .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build template: {}", e))?;

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(post_processor))
        .with_decoder(Some(ByteLevelDecoder::default()))
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build tokenizer: {}", e))?;
    
    // Write training data to temporary files (tokenizers API requires files)
    let temp_dir = std::env::temp_dir();
    let temp_files = training_data
        .chunks(training_data.len() / 3 + 1) // Split into multiple files to avoid huge single file
        .enumerate()
        .map(|(i, chunk)| {
            let temp_file = temp_dir.join(format!("tokenizer_training_{}.txt", i));
            std::fs::write(&temp_file, chunk.join("\n")).map(|_| temp_file)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let temp_file_paths: Vec<String> = temp_files
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    
    // Train the tokenizer
    println!("🏋️ Training BPE tokenizer with vocab size: {} on {} temporary files", vocab_size, temp_file_paths.len());
    
    tokenizer.train_from_files(&mut trainer, temp_file_paths)
        .map_err(|e| anyhow::anyhow!("Training failed: {}", e))?;
    
    // Clean up temporary files
    for temp_file in temp_files {
        let _ = std::fs::remove_file(temp_file);
    }
    
    // Save tokenizer
    println!("💾 Saving tokenizer to: {}", output_path.display());
    tokenizer.save(output_path, false).map_err(|e| anyhow::anyhow!("Save failed: {}", e))?;
    
    println!("✅ Tokenizer created successfully!");
    
    Ok(())
}

fn count_tokens(tokenizer_path: &PathBuf) -> Result<()> {
    println!("📊 Counting tokens in: {}", tokenizer_path.display());
    
    // Check if file exists
    if !tokenizer_path.exists() {
        anyhow::bail!("Tokenizer file not found: {}", tokenizer_path.display());
    }
    
    // Load the tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    // Get vocabulary size
    let vocab_size = tokenizer.get_vocab_size(true); // true includes added tokens
    let vocab_size_without_added = tokenizer.get_vocab_size(false); // false excludes added tokens
    
    println!("🔢 Total vocabulary size: {}", vocab_size);
    println!("📝 Base vocabulary size (excluding special tokens): {}", vocab_size_without_added);
    println!("🏷️  Special tokens: {}", vocab_size - vocab_size_without_added);
    
    // Get some example tokens
    let vocab = tokenizer.get_vocab(true);
    println!("\n🔤 Sample tokens:");
    
    // Show first 10 tokens (typically special tokens)
    let mut sorted_tokens: Vec<_> = vocab.iter().collect();
    sorted_tokens.sort_by_key(|&(_, &id)| id);
    
    for (token, &id) in sorted_tokens.iter().take(10) {
        println!("  {}: {}", id, token);
    }
    
    if sorted_tokens.len() > 10 {
        println!("  ... and {} more", sorted_tokens.len() - 10);
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize WebGPU device
    let device = burn::backend::wgpu::WgpuDevice::default();

    match &cli.command {
        Commands::Tokens { tokens_command } => {
            match tokens_command {
                TokensCommands::Create { data_dir, output, vocab_size } => {
                    create_tokenizer(data_dir, output, *vocab_size)
                }
                TokensCommands::Count { tokenizer_path } => {
                    count_tokens(tokenizer_path)
                }
            }
        },
        Commands::Train {
            train_data,
            validation_data,
            tokenizer_path,
            output_dir,
            epochs,
            batch_size,
            initial_lr,
            n_heads,
            n_layers,
            d_model,
            context_size,
            limit_train,
            limit_validation,
            no_tui,
        } => {
            // Load datasets
            let (train_dataset, validation_dataset) = load_datasets(
                train_data,
                validation_data,
                *limit_train,
                *limit_validation,
            )?;

            // Display training configuration
            println!("\n🔥 Starting Training with Burn Framework");
            println!("==========================================");
            println!("📊 Training Examples: {}", train_dataset.len());
            println!("📊 Validation Examples: {}", validation_dataset.len());
            println!("🔄 Epochs: {}", epochs);
            println!("📦 Batch Size: {}", batch_size);
            println!("🎯 Learning Rate: {:.2e}", initial_lr);
            println!("💾 Output dir: {}", output_dir.display());
            println!();

            train_model(
                train_dataset,
                Some(validation_dataset),
                tokenizer_path,
                output_dir,
                *epochs,
                *batch_size,
                *initial_lr,
                *n_heads,
                *n_layers,
                *d_model,
                *context_size,
                *no_tui,
                device,
            )
            .await
        }
    }
}
