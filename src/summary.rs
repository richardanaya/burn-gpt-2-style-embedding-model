use burn::train::LearnerSummary;

/// Print educational explanations of training metrics based on actual values
pub fn print_educational_metrics_explanation(summary: &LearnerSummary) {
    println!("\n📊 Training Results Explanation:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Find training and validation loss metrics
    let train_loss = summary
        .metrics
        .train
        .iter()
        .find(|m| m.name == "Loss")
        .and_then(|m| m.entries.last())
        .map(|e| e.value);

    let valid_loss = summary
        .metrics
        .valid
        .iter()
        .find(|m| m.name == "Loss")
        .and_then(|m| m.entries.last())
        .map(|e| e.value);

    println!("📈 Loss Values:");
    if let Some(train_val) = train_loss {
        println!(
            "   • Training Loss: {:.3} - Measures how well the model fits training data",
            train_val
        );
    }
    if let Some(valid_val) = valid_loss {
        println!(
            "   • Validation Loss: {:.3} - Measures performance on unseen data",
            valid_val
        );
    }

    println!();
    println!("🔍 Key Observations:");

    // Provide conditional analysis based on actual values
    match (train_loss, valid_loss) {
        (Some(train_val), Some(valid_val)) => {
            if valid_val < train_val {
                let diff = train_val - valid_val;
                println!("   • ✅ Good Generalization: Validation loss ({:.3}) is {:.3} lower than training loss", 
                         valid_val, diff);
                println!("     This suggests the model isn't overfitting and may generalize well");
            } else if valid_val > train_val {
                let diff = valid_val - train_val;
                if diff > 0.1 {
                    println!("   • ⚠️  Possible Overfitting: Validation loss ({:.3}) is {:.3} higher than training loss", 
                             valid_val, diff);
                    println!(
                        "     Consider reducing model complexity or increasing regularization"
                    );
                } else {
                    println!("   • ✅ Normal Pattern: Validation loss ({:.3}) slightly higher than training loss ({:.3})", 
                             valid_val, train_val);
                    println!("     This is typical and suggests healthy learning");
                }
            } else {
                println!(
                    "   • ✅ Perfect Balance: Training and validation losses are nearly equal"
                );
                println!("     This indicates good generalization without overfitting");
            }

            // Loss magnitude analysis
            if train_val > 1.0 || valid_val > 1.0 {
                println!("   • 📈 High Loss Values: Consider running more epochs or adjusting learning rate");
            } else if train_val < 0.1 && valid_val < 0.1 {
                println!("   • 🎯 Very Low Loss: Model has learned the patterns well");
            } else {
                println!("   • 📊 Moderate Loss: Values in reasonable range for early training");
            }
        }
        (Some(train_val), None) => {
            println!(
                "   • Training Loss: {:.3} (validation data not available)",
                train_val
            );
        }
        (None, Some(valid_val)) => {
            println!(
                "   • Validation Loss: {:.3} (training data not available)",
                valid_val
            );
        }
        (None, None) => {
            println!("   • No loss metrics available for analysis");
        }
    }

    println!();
    println!("🚀 Next Steps:");
    if summary.epochs <= 2 {
        println!(
            "   • Consider running more epochs - {} epoch(s) is typically insufficient",
            summary.epochs
        );
        println!("     for a 117M parameter GPT-2 model to learn meaningful patterns");
    } else if summary.epochs < 10 {
        println!(
            "   • Good start with {} epochs - monitor loss trends over more epochs",
            summary.epochs
        );
    } else {
        println!(
            "   • With {} epochs completed, evaluate if further training improves results",
            summary.epochs
        );
    }

    if let (Some(train_val), Some(valid_val)) = (train_loss, valid_loss) {
        if (train_val - valid_val).abs() > 0.2 {
            println!("   • Large gap between train/validation suggests reviewing data split or regularization");
        }
    }

    println!("   • Monitor that validation loss follows training loss without diverging");
    println!("   • Use embeddings for downstream tasks to evaluate real-world performance");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
