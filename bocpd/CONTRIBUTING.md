# Contributing to BOCPD

Thank you for considering contributing to BOCPD! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by its [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue on our GitHub repository with the following information:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any relevant logs or screenshots
- Your environment (Python version, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear, descriptive title
- Provide a detailed description of the suggested enhancement
- Explain why this enhancement would be useful
- Include any relevant examples or references

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Run the tests to ensure they pass
5. Submit a pull request

Please follow these guidelines when submitting a pull request:

- Follow the Python code style (PEP 8)
- Include tests for new features or bug fixes
- Update documentation as needed
- Add an entry to the changelog

## Development Setup

1. Clone your fork of the repository
   ```bash
   git clone https://github.com/yourusername/bocpd.git
   cd bocpd
   ```

2. Create a virtual environment and install the development dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Run the tests to ensure everything is working
   ```bash
   pytest
   ```

## Code Style

This project follows PEP 8 style guidelines. We use `ruff` for linting and `mypy` for type checking.

To check your code:
```bash
ruff check .
mypy .
```

## Testing

We use `pytest` for testing. Please write tests for any new functionality and ensure all tests pass before submitting a pull request.

To run the tests:
```bash
pytest
```

To check test coverage:
```bash
pytest --cov=bocpd
```

## Documentation

Documentation is written in Markdown. Please update the documentation when you add or modify features.

## Implementing New Likelihood Models

If you want to implement a new likelihood model:

1. Create a new class that implements the `ConjugateLikelihood` interface in `likelihoods.py`
2. Implement all required methods: `init_stats`, `predictive_prob`, `update_cp`, `update_growth`, and `predictive_mean`
3. Add tests for your new likelihood model
4. Update the documentation to include your new model

## Implementing New Hazard Functions

To implement a new hazard function:

1. Create a new class that inherits from the `Hazard` class
2. Implement the `prob(r: int, t: int) -> float` method
3. Add tests for your new hazard function
4. Update the documentation

## Release Process

1. Update version in `pyproject.toml`
2. Update the changelog
3. Create a new tag with the version number
4. Push the tag to GitHub
5. Create a new release on GitHub
6. Publish to PyPI

## Questions?

If you have any questions or need help, please create an issue on GitHub or contact the maintainers.

Thank you for your contribution!
