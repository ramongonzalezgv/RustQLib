use quantlib_models::{BlackScholesModel, HestonModel};
use rayon::prelude::*;

use super::core::{MonteCarloConfig, NormalRng};

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlackScholesSimulator {
    config: MonteCarloConfig,
}

impl BlackScholesSimulator {
    pub(crate) fn new(config: MonteCarloConfig) -> Self {
        Self { config }
    }

    pub(crate) fn simulate_terminal(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        if self.should_parallelize_terminal() {
            self.simulate_terminal_parallel(spot, model, ttm, rng)
        } else {
            self.simulate_terminal_serial(spot, model, ttm, rng)
        }
    }

    pub(crate) fn simulate_paths(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        if self.config.parallel {
            self.simulate_paths_parallel(spot, model, ttm, rng)
        } else {
            self.simulate_paths_serial(spot, model, ttm, rng)
        }
    }

    fn should_parallelize_terminal(&self) -> bool {
        if !self.config.parallel {
            return false;
        }

        // Exact BS terminal pricing does little work per path. For typical batch sizes,
        // Rayon scheduling overhead dominates and serial execution is faster.
        let threads = rayon::current_num_threads().max(1);
        let min_work = self
            .config
            .chunk_size
            .saturating_mul(threads)
            .saturating_mul(256);
        self.config.n_paths >= min_work
    }

    fn simulate_terminal_serial(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        let drift = (model.r - model.q - 0.5 * model.sigma * model.sigma) * ttm;
        let diffusion = model.sigma * ttm.sqrt();

        let mut spots = Vec::with_capacity(self.config.n_paths);
        for _ in 0..self.config.n_paths {
            let z = rng.standard_normal();
            spots.push(spot * (drift + diffusion * z).exp());
        }
        spots
    }

    fn simulate_terminal_parallel(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        let drift = (model.r - model.q - 0.5 * model.sigma * model.sigma) * ttm;
        let diffusion = model.sigma * ttm.sqrt();
        let chunk_size = self.config.chunk_size.min(self.config.n_paths).max(1);
        let mut spots = vec![0.0; self.config.n_paths];

        for chunk in spots.chunks_mut(chunk_size) {
            let chunk_len = chunk.len();
            let mut normals = Vec::with_capacity(chunk_len);
            for _ in 0..chunk_len {
                normals.push(rng.standard_normal());
            }

            chunk
                .par_iter_mut()
                .zip(normals.par_iter().copied())
                .for_each(|(s, z)| *s = spot * (drift + diffusion * z).exp());
        }
        spots
    }

    fn simulate_paths_serial(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        let dt = ttm / self.config.n_steps as f64;
        let drift_step = (model.r - model.q - 0.5 * model.sigma * model.sigma) * dt;
        let diffusion_step = model.sigma * dt.sqrt();

        let mut paths = vec![vec![0.0; self.config.n_steps + 1]; self.config.n_paths];
        for path in &mut paths {
            path[0] = spot;
            for step in 0..self.config.n_steps {
                let z = rng.standard_normal();
                path[step + 1] = path[step] * (drift_step + diffusion_step * z).exp();
            }
        }

        paths
    }

    fn simulate_paths_parallel(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        let dt = ttm / self.config.n_steps as f64;
        let drift_step = (model.r - model.q - 0.5 * model.sigma * model.sigma) * dt;
        let diffusion_step = model.sigma * dt.sqrt();
        let chunk_size = self.config.chunk_size.min(self.config.n_paths).max(1);

        let mut paths = vec![vec![0.0; self.config.n_steps + 1]; self.config.n_paths];
        for path_chunk in paths.chunks_mut(chunk_size) {
            let chunk_len = path_chunk.len();
            let mut normals = Vec::with_capacity(chunk_len * self.config.n_steps);
            for _ in 0..(chunk_len * self.config.n_steps) {
                normals.push(rng.standard_normal());
            }

            path_chunk
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_idx, path)| {
                    let base = local_idx * self.config.n_steps;
                    path[0] = spot;
                    for step in 0..self.config.n_steps {
                        let z = normals[base + step];
                        path[step + 1] = path[step] * (drift_step + diffusion_step * z).exp();
                    }
                });
        }

        paths
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HestonSimulator {
    config: MonteCarloConfig,
}

impl HestonSimulator {
    pub(crate) fn new(config: MonteCarloConfig) -> Self {
        Self { config }
    }

    pub(crate) fn simulate_terminal(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        if self.config.parallel {
            self.simulate_terminal_parallel(spot, model, ttm, rng)
        } else {
            self.simulate_terminal_serial(spot, model, ttm, rng)
        }
    }

    pub(crate) fn simulate_paths(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        if self.config.parallel {
            self.simulate_paths_parallel(spot, model, ttm, rng)
        } else {
            self.simulate_paths_serial(spot, model, ttm, rng)
        }
    }

    fn simulate_terminal_serial(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        let dt = ttm / self.config.n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let rho = model.rho;
        let rho_orth = (1.0 - rho * rho).max(0.0).sqrt();

        let mut terminals = Vec::with_capacity(self.config.n_paths);
        for _ in 0..self.config.n_paths {
            let mut s = spot;
            let mut v = model.v0;
            for _ in 0..self.config.n_steps {
                let z1 = rng.standard_normal();
                let z2 = rng.standard_normal();
                let d_w1 = sqrt_dt * z1;
                let d_w2 = sqrt_dt * (rho * z1 + rho_orth * z2);

                let v_pos = v.max(0.0);
                s *= ((model.r - model.q - 0.5 * v_pos) * dt + v_pos.sqrt() * d_w1).exp();

                let v_next =
                    v + model.kappa * (model.theta - v) * dt + model.sigma * v_pos.sqrt() * d_w2;
                v = v_next.max(0.0);
            }
            terminals.push(s);
        }
        terminals
    }

    fn simulate_terminal_parallel(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        let dt = ttm / self.config.n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let rho = model.rho;
        let rho_orth = (1.0 - rho * rho).max(0.0).sqrt();
        let chunk_size = self.config.chunk_size.min(self.config.n_paths).max(1);

        let mut terminals = vec![0.0; self.config.n_paths];
        for terminals_chunk in terminals.chunks_mut(chunk_size) {
            let chunk_len = terminals_chunk.len();
            let mut normals = Vec::with_capacity(chunk_len * self.config.n_steps * 2);
            for _ in 0..(chunk_len * self.config.n_steps * 2) {
                normals.push(rng.standard_normal());
            }

            terminals_chunk
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_idx, terminal)| {
                    let mut s = spot;
                    let mut v = model.v0;
                    let base = local_idx * self.config.n_steps * 2;

                    for step in 0..self.config.n_steps {
                        let z_idx = base + (step * 2);
                        let z1 = normals[z_idx];
                        let z2 = normals[z_idx + 1];
                        let d_w1 = sqrt_dt * z1;
                        let d_w2 = sqrt_dt * (rho * z1 + rho_orth * z2);

                        let v_pos = v.max(0.0);
                        s *= ((model.r - model.q - 0.5 * v_pos) * dt + v_pos.sqrt() * d_w1).exp();

                        let v_next = v
                            + model.kappa * (model.theta - v) * dt
                            + model.sigma * v_pos.sqrt() * d_w2;
                        v = v_next.max(0.0);
                    }
                    *terminal = s;
                });
        }

        terminals
    }

    fn simulate_paths_serial(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        let dt = ttm / self.config.n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let rho = model.rho;
        let rho_orth = (1.0 - rho * rho).max(0.0).sqrt();

        let mut paths = vec![vec![0.0; self.config.n_steps + 1]; self.config.n_paths];
        for path in &mut paths {
            path[0] = spot;
            let mut v = model.v0;

            for step in 0..self.config.n_steps {
                let z1 = rng.standard_normal();
                let z2 = rng.standard_normal();
                let d_w1 = sqrt_dt * z1;
                let d_w2 = sqrt_dt * (rho * z1 + rho_orth * z2);

                let v_pos = v.max(0.0);
                path[step + 1] = path[step]
                    * ((model.r - model.q - 0.5 * v_pos) * dt + v_pos.sqrt() * d_w1).exp();

                let v_next =
                    v + model.kappa * (model.theta - v) * dt + model.sigma * v_pos.sqrt() * d_w2;
                v = v_next.max(0.0);
            }
        }

        paths
    }

    fn simulate_paths_parallel(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        let dt = ttm / self.config.n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let rho = model.rho;
        let rho_orth = (1.0 - rho * rho).max(0.0).sqrt();
        let chunk_size = self.config.chunk_size.min(self.config.n_paths).max(1);

        let mut paths = vec![vec![0.0; self.config.n_steps + 1]; self.config.n_paths];
        for path_chunk in paths.chunks_mut(chunk_size) {
            let chunk_len = path_chunk.len();
            let mut normals = Vec::with_capacity(chunk_len * self.config.n_steps * 2);
            for _ in 0..(chunk_len * self.config.n_steps * 2) {
                normals.push(rng.standard_normal());
            }

            path_chunk
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_idx, path)| {
                    path[0] = spot;
                    let mut v = model.v0;
                    let base = local_idx * self.config.n_steps * 2;

                    for step in 0..self.config.n_steps {
                        let z_idx = base + (step * 2);
                        let z1 = normals[z_idx];
                        let z2 = normals[z_idx + 1];
                        let d_w1 = sqrt_dt * z1;
                        let d_w2 = sqrt_dt * (rho * z1 + rho_orth * z2);

                        let v_pos = v.max(0.0);
                        path[step + 1] = path[step]
                            * ((model.r - model.q - 0.5 * v_pos) * dt + v_pos.sqrt() * d_w1).exp();

                        let v_next = v
                            + model.kappa * (model.theta - v) * dt
                            + model.sigma * v_pos.sqrt() * d_w2;
                        v = v_next.max(0.0);
                    }
                });
        }

        paths
    }
}
