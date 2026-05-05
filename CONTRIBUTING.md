# Contributing to CDIE

Thank you for your interest in contributing to CDIE (Causal Decision Intelligence Engine)! We welcome bug reports, feature requests, and pull requests.

## Development Setup

1. Fork the repository and clone it locally.
2. Follow the [Installation Guide](docs/installation.md) to set up your environment.
3. Install development dependencies (e.g., using a virtual environment and `requirements.txt`).

## Testing and CI

We rely on GitHub Actions for Continuous Integration. Before submitting a PR, ensure that:
- **Linting:** Your code passes `ruff` checks.
- **Type Checking:** `mypy` finds no errors.
- **Tests:** The `pytest` test suite passes. 

See `.github/workflows/main.yml` for exact commands.

For details on adding a new causal method or testing against causal ground truths, please see our [Development Guide](docs/development.md).

## Submitting a Pull Request

1. Create a feature branch (`git checkout -b feature/your-feature-name`).
2. Commit your changes with clear, descriptive messages.
3. Push to your fork and submit a Pull Request against the `main` branch.
4. Ensure all CI checks pass.
