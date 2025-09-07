I want to make a GPT-2 style embedding model

Model GPT-2 117M
Transformer Blocks  12
Attention Blocks 12
Heads per Block 12
Dimensions 768

The application should be able to be run as a CLI.

* train safe tensors from the .tsv
* training should be able to be stopped at any point gracefully and have the safe tensors saved at that point
* take a sentence and give me its vector
* take two sentences and give me their similarity 0-1

Technologies
* it should be written in Rust
* weights should be stored in safetensor files
* it should have tests
* it should use the burn library with webgpu backend
* it should use CLAP as the CLI frontend

Style
* The goal of this codebase is to be learned from, so try to keep comments in to give the high level picture clear to a reader
