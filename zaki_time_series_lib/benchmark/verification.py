from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, integrate

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger

logger = get_logger(__name__)


class Lorenz96:
    def __init__(self, n_vars: int = 40, F: float = 8.0, dt: float = 0.01):
        self.n_vars = n_vars
        self.F = F
        self.dt = dt
        self.dlog = DetailedLogger(f"data.Lorenz96_N{n_vars}F{F}")

    def _rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        n = self.n_vars
        dx = np.zeros(n)
        dx[0] = (x[1] - x[n - 2]) * x[n - 1] - x[0] + self.F
        dx[1] = (x[2] - x[n - 1]) * x[0] - x[1] + self.F
        for i in range(2, n - 1):
            dx[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + self.F
        dx[n - 1] = (x[0] - x[n - 3]) * x[n - 2] - x[n - 1] + self.F
        return dx

    def generate(self, n_steps: int = 10000, transient: int = 1000,
                 seed: int = 42) -> np.ndarray:
        self.dlog.log_section(f"Generating Lorenz96 N={self.n_vars}, F={self.F}")
        rng = np.random.RandomState(seed)
        x0 = rng.uniform(-0.1, 0.1, self.n_vars)
        x0[0] = x0[0] + 1.0
        self.dlog.get_logger().info(f"Integrating {n_steps + transient} steps, dt={self.dt}...")
        t_span = np.arange(0, (n_steps + transient) * self.dt, self.dt)
        sol = integrate.solve_ivp(
            self._rhs, (t_span[0], t_span[-1]), x0,
            method='RK45', t_eval=t_span, rtol=1e-6, atol=1e-8
        )
        data = sol.y[:, transient:].T
        self.dlog.log_data_shape(data, "Lorenz96")
        self.dlog.get_logger().info(
            f"Lorenz96 generated: shape={data.shape}, "
            f"min={data.min():.4f}, max={data.max():.4f}, "
            f"mean={data.mean():.4f}, std={data.std():.4f}"
        )
        return data

    def generate_single(self, n_steps: int = 10000, transient: int = 1000,
                        seed: int = 42) -> np.ndarray:
        data = self.generate(n_steps, transient, seed)
        return data[:, 0]

    def lyapunov_estimate(self, data: np.ndarray, sample_every: int = 10) -> float:
        self.dlog.get_logger().info("Estimating Lyapunov exponent from data...")
        diffs = np.diff(data[::sample_every], axis=0)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        mask = dists > 1e-10
        if not mask.any():
            return 0.0
        log_div = np.log(dists[mask])
        t = np.arange(len(log_div)) * self.dt * sample_every
        lyap = np.polyfit(t, log_div, 1)[0]
        self.dlog.get_logger().info(f"Estimated Lyapunov exponent: {lyap:.4f}")
        return float(lyap)

    @staticmethod
    def split_train_test(data: np.ndarray, train_ratio: float = 0.7
                         ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(data)
        split = int(n * train_ratio)
        return data[:split], data[split:]

    @staticmethod
    def compute_prediction_horizon(y_true: np.ndarray, y_pred: np.ndarray,
                                   threshold: float = 0.5) -> int:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        errors = np.abs(y_true - y_pred) / (np.std(y_true) + 1e-10)
        for i in range(len(errors)):
            if errors[i] > threshold:
                return i
        return len(errors)


class Lorenz63:
    r"""
    Lorenz 63 chaotic system:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

    Classic parameters (sigma=10, rho=28, beta=8/3) produce chaotic attractor.
    """
    def __init__(self, sigma: float = 10.0, rho: float = 28.0,
                 beta: float = 8.0 / 3.0, dt: float = 0.01):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.dlog = DetailedLogger(f"data.Lorenz63_s{sigma}r{rho}b{beta:.3f}")

    def _rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        dx = np.zeros(3)
        dx[0] = self.sigma * (x[1] - x[0])
        dx[1] = x[0] * (self.rho - x[2]) - x[1]
        dx[2] = x[0] * x[1] - self.beta * x[2]
        return dx

    def generate(self, n_steps: int = 10000, transient: int = 1000,
                 seed: int = 42) -> np.ndarray:
        self.dlog.log_section(
            f"Generating Lorenz63 sigma={self.sigma}, rho={self.rho}, "
            f"beta={self.beta:.4f}"
        )
        rng = np.random.RandomState(seed)
        x0 = rng.uniform(-5, 5, 3)
        self.dlog.get_logger().info(
            f"Integrating {n_steps + transient} steps, dt={self.dt}..."
        )
        t_span = np.arange(0, (n_steps + transient) * self.dt, self.dt)
        sol = integrate.solve_ivp(
            self._rhs, (t_span[0], t_span[-1]), x0,
            method='RK45', t_eval=t_span, rtol=1e-6, atol=1e-8
        )
        data = sol.y[:, transient:].T
        self.dlog.log_data_shape(data, "Lorenz63")
        self.dlog.get_logger().info(
            f"Lorenz63 generated: shape={data.shape}, "
            f"min={data.min():.4f}, max={data.max():.4f}, "
            f"mean={data.mean():.4f}, std={data.std():.4f}"
        )
        return data

    def generate_single(self, n_steps: int = 10000, transient: int = 1000,
                        seed: int = 42, component: int = 0) -> np.ndarray:
        data = self.generate(n_steps, transient, seed)
        return data[:, component]

    def lyapunov_estimate(self, data: np.ndarray,
                          sample_every: int = 5) -> float:
        self.dlog.get_logger().info("Estimating Lyapunov exponent from data...")
        diffs = np.diff(data[::sample_every], axis=0)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        mask = dists > 1e-10
        if not mask.any():
            return 0.0
        log_div = np.log(dists[mask])
        t = np.arange(len(log_div)) * self.dt * sample_every
        lyap = float(np.polyfit(t, log_div, 1)[0])
        self.dlog.get_logger().info(f"Estimated Lyapunov exponent: {lyap:.4f}")
        return lyap

    @staticmethod
    def compute_prediction_horizon(y_true: np.ndarray, y_pred: np.ndarray,
                                   threshold: float = 0.5) -> int:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        errors = np.abs(y_true - y_pred) / (np.std(y_true) + 1e-10)
        for i in range(len(errors)):
            if errors[i] > threshold:
                return i
        return len(errors)

    @staticmethod
    def split_train_test(data: np.ndarray, train_ratio: float = 0.7
                         ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(data)
        split = int(n * train_ratio)
        return data[:split], data[split:]


class ESPVerification:
    def __init__(self):
        self.dlog = DetailedLogger("verification.ESP")
        self.results: Dict[str, Any] = {}

    def check_spectral_radius(self, W_res: np.ndarray) -> float:
        self.dlog.get_logger().info("Computing spectral radius of reservoir...")
        eigvals = linalg.eigvals(W_res)
        sr = float(max(abs(eigvals)))
        self.results["spectral_radius"] = sr
        self.results["spectral_radius_pass"] = sr < 1.0
        self.dlog.get_logger().info(
            f"Spectral radius: {sr:.6f} - {'PASS' if sr < 1.0 else 'FAIL'} (threshold < 1)"
        )
        return sr

    def check_state_forgetting(self, W_in: np.ndarray, W_res: np.ndarray,
                                n_inputs: int = 1000, input_scaling: float = 0.5,
                                leaky_rate: float = 0.3, seed: int = 42) -> Dict[str, Any]:
        self.dlog.log_section("State Forgetting Test (2 trajectories)")
        rng = np.random.RandomState(seed)
        n = W_res.shape[0]
        d = W_in.shape[1]

        inputs = rng.uniform(-input_scaling, input_scaling, (n_inputs, d))

        x1 = rng.uniform(-1, 1, n)
        x2 = rng.uniform(-1, 1, n) * 10

        distances = []
        for t in range(n_inputs):
            u = inputs[t]
            x1 = (1 - leaky_rate) * x1 + leaky_rate * np.tanh(W_in @ u + W_res @ x1)
            x2 = (1 - leaky_rate) * x2 + leaky_rate * np.tanh(W_in @ u + W_res @ x2)
            distances.append(float(np.linalg.norm(x1 - x2)))

        distances = np.array(distances)
        converged = distances[-1] < 1e-6
        half_life = int(np.argmax(distances < distances[0] / 2)) if np.any(distances < distances[0] / 2) else -1

        result = {
            "final_distance": float(distances[-1]),
            "initial_distance": float(distances[0]),
            "distance_ratio": float(distances[-1] / max(distances[0], 1e-15)),
            "converged": bool(converged),
            "half_life_steps": half_life,
        }
        self.results["state_forgetting"] = result
        self.dlog.get_logger().info(
            f"Initial dist: {distances[0]:.6f}, Final dist: {distances[-1]:.6f} - "
            f"{'CONVERGED' if converged else 'NOT CONVERGED'}"
        )
        self.dlog.get_logger().info(f"Convergence half-life: {half_life} steps")
        return result

    def check_multi_trajectory_convergence(self, W_in: np.ndarray, W_res: np.ndarray,
                                            n_traj: int = 20, n_steps: int = 100,
                                            input_scaling: float = 0.5,
                                            leaky_rate: float = 0.3,
                                            seed: int = 42) -> Dict[str, Any]:
        r"""
        Multi-trajectory ESP test (notebook-style).
        Runs n_traj random initial states through same input sequence,
        measures mean pairwise L2 distance at each step.
        Convergence indicates ESP holds.
        """
        self.dlog.log_section(f"Multi-Trajectory ESP Test ({n_traj} trajectories, {n_steps} steps)")
        rng = np.random.RandomState(seed)
        n = W_res.shape[0]
        d = W_in.shape[1]

        inputs = rng.uniform(-input_scaling, input_scaling, (n_steps, d))

        states = rng.uniform(-1, 1, (n_traj, n))
        trajectories = np.zeros((n_traj, n_steps, n))

        for t in range(n_steps):
            u = inputs[t]
            for k in range(n_traj):
                states[k] = (1 - leaky_rate) * states[k] + leaky_rate * np.tanh(
                    W_in @ u + W_res @ states[k]
                )
            trajectories[:, t] = states

        distances = np.zeros(n_steps)
        n_pairs = 0
        for i in range(n_traj):
            for j in range(i + 1, n_traj):
                diff = trajectories[i] - trajectories[j]
                distances += np.linalg.norm(diff, axis=1)
                n_pairs += 1
        distances /= max(n_pairs, 1)

        converged = distances[-1] < 1e-6
        saturation_ratio = float(distances[-1] / max(distances[0], 1e-15))

        result = {
            "n_trajectories": n_traj,
            "n_steps": n_steps,
            "initial_mean_distance": float(distances[0]),
            "final_mean_distance": float(distances[-1]),
            "saturation_ratio": saturation_ratio,
            "converged": bool(converged),
            "distance_trace": distances.tolist(),
        }
        self.results["multi_trajectory"] = result
        self.dlog.get_logger().info(
            f"Mean pairwise dist: {distances[0]:.6f} -> {distances[-1]:.6f} - "
            f"{'CONVERGED' if converged else 'BOUNDED' if saturation_ratio < 10 else 'DIVERGING'}"
        )
        return result

    def estimate_conditional_lyapunov(self, W_in: np.ndarray, W_res: np.ndarray,
                                       n_steps: int = 5000, input_scaling: float = 0.5,
                                       leaky_rate: float = 0.3, seed: int = 42) -> float:
        self.dlog.log_section("Conditional Lyapunov Exponent")
        rng = np.random.RandomState(seed)
        n = W_res.shape[0]
        d = W_in.shape[1]
        inputs = rng.uniform(-input_scaling, input_scaling, (n_steps, d))

        x = np.zeros(n)
        delta = rng.uniform(-1e-6, 1e-6, n)
        delta = delta / np.linalg.norm(delta) * 1e-8

        lyap_sum = 0.0
        count = 0
        for t in range(n_steps):
            u = inputs[t]
            x_new = (1 - leaky_rate) * x + leaky_rate * np.tanh(W_in @ u + W_res @ x)
            delta_new = (1 - leaky_rate) * delta + leaky_rate * (
                W_res @ delta * (1 - np.tanh(W_in @ u + W_res @ x) ** 2)
            )
            norm_delta = np.linalg.norm(delta_new)
            if norm_delta > 0 and t > 100:
                lyap_sum += np.log(norm_delta / max(np.linalg.norm(delta), 1e-15))
                count += 1
                delta_new = delta_new / norm_delta * 1e-8
            x = x_new
            delta = delta_new

        lyap = float(lyap_sum / max(count, 1))
        self.results["conditional_lyapunov"] = lyap
        self.results["conditional_lyapunov_pass"] = lyap < 0
        self.dlog.get_logger().info(
            f"Conditional Lyapunov: {lyap:.6f} - {'STABLE' if lyap < 0 else 'UNSTABLE'}"
        )
        return lyap

    def verify(self, W_in: np.ndarray, W_res: np.ndarray,
                n_inputs: int = 1000, input_scaling: float = 0.5,
                leaky_rate: float = 0.3, seed: int = 42,
                n_traj: int = 20, n_steps: int = 100) -> Dict[str, Any]:
        self.dlog.log_section("Full ESP Verification")
        sr = self.check_spectral_radius(W_res)
        forgetting = self.check_state_forgetting(W_in, W_res, n_inputs, input_scaling, leaky_rate, seed)
        multi = self.check_multi_trajectory_convergence(W_in, W_res, n_traj, n_steps, input_scaling, leaky_rate, seed)
        lyap = self.estimate_conditional_lyapunov(W_in, W_res, n_inputs, input_scaling, leaky_rate, seed + 1)

        sr_ok = self.results.get("spectral_radius_pass", False)
        lyap_ok = self.results.get("conditional_lyapunov_pass", False)
        forgetting_ok = self.results.get("state_forgetting", {}).get("converged", False)
        multi_ok = self.results.get("multi_trajectory", {}).get("converged", False)

        passed = (sr_ok or lyap_ok) and (forgetting_ok or multi_ok)

        self.results["esp_passed"] = passed
        self.dlog.log_section(f"ESP VERDICT: {'PASS' if passed else 'FAIL'}")
        return self.results


class FSDHCalculator:
    def __init__(self):
        self.dlog = DetailedLogger("verification.FSDH")

    def compute_from_hidden(self, hidden_func: Callable[[np.ndarray], np.ndarray],
                            inputs: np.ndarray, epsilon: float = 1e-5,
                            n_perturbations: int = 10) -> Dict[str, Any]:
        self.dlog.log_section("FSDH Computation (Hidden Space)")
        n_samples = len(inputs)
        sensitivities = []

        for i in range(n_perturbations):
            noise = np.random.RandomState(i).uniform(-epsilon, epsilon, inputs.shape)
            h_orig = hidden_func(inputs)
            h_pert = hidden_func(inputs + noise)
            numerator = np.linalg.norm(h_pert - h_orig, axis=1)
            denominator = np.linalg.norm(noise, axis=1) + 1e-15
            s = np.mean(numerator / denominator)
            sensitivities.append(s)

        result = {
            "fsdh_mean": float(np.mean(sensitivities)),
            "fsdh_std": float(np.std(sensitivities)),
            "fsdh_min": float(np.min(sensitivities)),
            "fsdh_max": float(np.max(sensitivities)),
            "epsilon": epsilon,
            "n_perturbations": n_perturbations,
        }
        self.dlog.get_logger().info(
            f"FSDH: mean={result['fsdh_mean']:.6f} +/- {result['fsdh_std']:.6f}"
        )
        return result

    def compute_from_predictions(self, model, inputs: np.ndarray,
                                  epsilon: float = 1e-5,
                                  n_perturbations: int = 10) -> Dict[str, Any]:
        self.dlog.log_section("FSDH Computation (Prediction Space)")
        if not model.is_fitted:
            raise RuntimeError("Model must be fitted first")

        base_pred = model.predict(len(inputs), inputs)
        sensitivities = []

        for i in range(n_perturbations):
            rng = np.random.RandomState(i)
            noise = rng.uniform(-epsilon, epsilon, inputs.shape)
            pert_pred = model.predict(len(inputs), inputs + noise)
            numerator = np.linalg.norm(pert_pred - base_pred)
            denominator = np.linalg.norm(noise) + 1e-15
            sensitivities.append(float(numerator / denominator))

        result = {
            "fsdh_pred_mean": float(np.mean(sensitivities)),
            "fsdh_pred_std": float(np.std(sensitivities)),
            "epsilon": epsilon,
            "n_perturbations": n_perturbations,
        }
        self.dlog.get_logger().info(
            f"FSDH (pred): mean={result['fsdh_pred_mean']:.6f} +/- {result['fsdh_pred_std']:.6f}"
        )
        return result
