# Contributing

Thanks for considering contributing to Zaki Time Series Library.

## Development Setup

```bash
git clone https://github.com/ZakLr/Zaki-time-series-lib-AQC
cd Zaki-time-series-lib-AQC
pip install -e ".[all]"
```

## Guidelines

- Keep the unified interface: all models share `fit(y)` / `predict(horizon)`.
- Add experiments under `zaki_time_series_lib/experiments/`.
- Run benchmarks before submitting to verify no regression.
- Follow existing code style (no docstrings, minimal comments).

## Pull Request Process

1. Open an issue describing the change.
2. Fork the repo and create a feature branch.
3. Submit a PR linking the issue.
4. Ensure all existing benchmarks still pass.
