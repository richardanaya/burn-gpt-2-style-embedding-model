use burn::prelude::*;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{Adaptor, Metric, MetricEntry, MetricMetadata};
use burn::train::RegressionOutput;

/// Input type for similarity accuracy metric
pub struct SimilarityAccuracyInput<B: Backend> {
    pub predictions: Tensor<B, 1>, // Model predictions (similarity scores)
    pub targets: Tensor<B, 1>,     // True labels (0 or 1)
}

/// Custom accuracy metric for similarity prediction following official Burn pattern
pub struct SimilarityAccuracyMetric<B: Backend> {
    state: NumericMetricState,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for SimilarityAccuracyMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> SimilarityAccuracyMetric<B> {
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            _b: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Metric for SimilarityAccuracyMetric<B> {
    type Input = SimilarityAccuracyInput<B>;

    fn name(&self) -> String {
        "Similarity Accuracy".to_string()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size] = item.predictions.dims();

        // Convert predictions to binary (threshold at 0.5)
        let predicted_labels = item.predictions.clone().greater_elem(0.5);
        let true_labels = item.targets.clone().greater_elem(0.5);

        // Count correct predictions
        let correct = predicted_labels.equal(true_labels).int().sum();
        let accuracy_value = correct.clone().into_scalar().elem::<f32>() as f64 / batch_size as f64;

        self.state.update(
            accuracy_value * 100.0, // Convert to percentage
            batch_size,
            FormatOptions::new(self.name()).precision(2).unit("%"),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> burn::train::metric::Numeric for SimilarityAccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Adaptor to convert from RegressionOutput to SimilarityAccuracyInput
impl<B: Backend> Adaptor<SimilarityAccuracyInput<B>> for RegressionOutput<B> {
    fn adapt(&self) -> SimilarityAccuracyInput<B> {
        // Extract predictions and targets from the regression output
        // For similarity task: output contains predictions, targets contain true labels
        let predictions = self.output.clone().flatten::<1>(0, 1); // Flatten to 1D
        let targets = self.targets.clone().flatten::<1>(0, 1); // Flatten to 1D

        SimilarityAccuracyInput {
            predictions,
            targets,
        }
    }
}