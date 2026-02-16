# Contributing to Per-Harmonic ADSR Synthesis

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Ways to Contribute

- **Bug Reports**: Open an issue describing the bug, steps to reproduce, and your environment
- **Feature Requests**: Open an issue describing the feature and use case
- **Code Contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve README, docstrings, or add examples
- **Experiments**: Share results from testing with new audio samples

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/per-harmonic-adsr-synthesis.git
   cd per-harmonic-adsr-synthesis
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Follow [PEP 8](https://pep8.org/) for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Comment complex algorithms

## Testing

Before submitting a pull request:

1. Test with multiple audio samples
2. Run all experiments to ensure no regressions:
   ```bash
   python run.py exp1
   python run.py exp2
   python run.py exp3
   ```

3. Verify the GUI works:
   ```bash
   python run.py
   ```

## Pull Request Process

1. Update documentation if needed
2. Add any new dependencies to `requirements.txt`
3. Describe your changes in the pull request
4. Link any related issues

## Research Contributions

If you use this work in academic research:

- Cite the original paper (see README.md)
- Consider contributing your findings back to the project
- Share any novel applications or improvements

## Questions?

Open an issue for questions about:
- Implementation details
- DSP theory behind the algorithms
- How to extend the system
- Interpretation of experimental results

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
