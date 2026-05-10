use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub success: bool,
    pub calibrated_params: BTreeMap<String, f64>,
    pub objective_value: f64,
    pub errors: Vec<f64>,
    pub market_values: Vec<f64>,
    pub model_values: Vec<f64>,
    pub rmse: f64,
    pub mape: f64,
    pub max_error: f64,
    pub iterations: usize,
    pub message: String,
}

impl CalibrationResult {
    pub fn summary(&self) -> String {
        let mut output = format!(
            "Calibration Results\n============================================================\nSuccess: {}\nMessage: {}\nIterations: {}\n\nObjective Value: {:e}\nRMSE: {:e}\nMAPE: {:.2}%\nMax Error: {:e}\n\nCalibrated Parameters:\n------------------------------------------------------------\n",
            self.success,
            self.message,
            self.iterations,
            self.objective_value,
            self.rmse,
            self.mape * 100.0,
            self.max_error,
        );

        for (name, value) in &self.calibrated_params {
            output.push_str(&format!("  {name:<15} = {value:>12.6}\n"));
        }

        output
    }

    pub fn get_errors_by_maturity(&self, maturities: &[f64]) -> Vec<(f64, f64)> {
        let mut grouped: BTreeMap<u64, (f64, f64, usize)> = BTreeMap::new();

        for (&maturity, &error) in maturities.iter().zip(self.errors.iter()) {
            let key = maturity.to_bits();
            let entry = grouped.entry(key).or_insert((maturity, 0.0, 0));
            entry.1 += error.abs();
            entry.2 += 1;
        }

        grouped
            .into_values()
            .map(|(maturity, sum_abs_error, count)| (maturity, sum_abs_error / count as f64))
            .collect()
    }
}
