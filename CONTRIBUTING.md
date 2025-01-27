# Contributing to There's Waldo

Thank you for your interest in contributing to There's Waldo! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/theres-waldo.git
   cd theres-waldo
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards:
   - Use type hints
   - Follow PEP 8 style guide
   - Add docstrings for new functions/classes
   - Include unit tests for new functionality

3. Format your code:
   ```bash
   # Format code
   black src/

   # Sort imports
   isort src/

   # Run type checking
   mypy src/
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```
   We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request

## Pull Request Guidelines

- Fill out the PR template completely
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow the coding standards
- Keep PRs focused and atomic

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage report
pytest --cov=src/waldo_finder tests/
```

## Development Environment

We recommend using Visual Studio Code with the following extensions:
- Python
- Pylance
- Black Formatter
- isort
- mypy

## Project Structure

```
theres-waldo/
├── config/                 # Hydra configuration files
├── src/
│   └── waldo_finder/      # Main package
│       ├── model.py       # Model architecture
│       ├── data.py        # Data loading and processing
│       ├── train.py       # Training script
│       └── inference.py   # Inference script
├── tests/                 # Test files
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

## Documentation

- Add docstrings to all public functions/classes
- Update README.md for significant changes
- Include examples in docstrings
- Follow Google Python Style Guide for docstrings

## Reporting Issues

- Use the issue tracker
- Include reproducible examples
- Provide system information
- Follow the issue template

## Feature Requests

- Use the issue tracker with the "enhancement" label
- Clearly describe the feature and its use case
- Provide examples if possible

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open an issue
- Start a discussion
- Reach out to maintainers

Thank you for contributing to There's Waldo! 🎉
