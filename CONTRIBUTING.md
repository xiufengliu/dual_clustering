# Contributing to Neutrosophic Dual Clustering Random Forest

We welcome contributions to this project! This document provides guidelines for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of machine learning and time series forecasting

### Development Setup

1. **Fork the repository**
   ```bash
   git fork https://github.com/xiufengliu/dual_clustering.git
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/dual_clustering.git
   cd dual_clustering/renewable_energy_forecasting
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🛠️ Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

### 2. Make Changes

- Follow the existing code style and structure
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_clustering.py

# Run with coverage
pytest --cov=src tests/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new clustering algorithm"
```

**Commit Message Format:**
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring
- `style:` for formatting changes

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📝 Code Style Guidelines

### Python Code Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md if adding new features

### Example Function Documentation

```python
def calculate_neutrosophic_components(
    cluster_labels: np.ndarray,
    membership_matrix: np.ndarray,
    entropy_epsilon: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate neutrosophic components from clustering results.
    
    Args:
        cluster_labels: Hard cluster assignments from K-means
        membership_matrix: Fuzzy membership matrix from FCM
        entropy_epsilon: Small value to prevent log(0)
        
    Returns:
        Tuple containing truth, indeterminacy, and falsity components
        
    Raises:
        ValueError: If input arrays have incompatible shapes
    """
```

## 🧪 Testing Guidelines

### Test Structure

- Place tests in the `tests/` directory
- Mirror the source code structure
- Use descriptive test names

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

### Example Test

```python
import pytest
import numpy as np
from src.clustering.dual_clusterer import DualClusterer

class TestDualClusterer:
    def test_fit_predict_basic(self):
        """Test basic fit_predict functionality."""
        clusterer = DualClusterer(n_clusters=3)
        X = np.random.rand(100, 5)
        
        labels, memberships = clusterer.fit_predict(X)
        
        assert labels.shape == (100,)
        assert memberships.shape == (100, 3)
        assert np.all(labels >= 0) and np.all(labels < 3)
```

## 📊 Adding New Features

### New Clustering Algorithms

1. Inherit from `BaseClusterer`
2. Implement required methods
3. Add comprehensive tests
4. Update documentation

### New Forecasting Models

1. Inherit from `BaseForecaster`
2. Implement `fit()` and `predict()` methods
3. Add to baseline model registry
4. Include in experimental evaluation

### New Evaluation Metrics

1. Add to `ForecastingMetrics` class
2. Include statistical significance tests
3. Update visualization components

## 🐛 Reporting Issues

### Bug Reports

Include:
- Python version and OS
- Complete error traceback
- Minimal code to reproduce
- Expected vs actual behavior

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach

## 📋 Pull Request Checklist

Before submitting a pull request:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] Pre-commit hooks pass

## 🔍 Code Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Peer Review**: At least one maintainer reviews
3. **Discussion**: Address feedback and suggestions
4. **Approval**: Maintainer approves and merges

## 📚 Resources

- [Project Documentation](docs/)
- [Issue Tracker](https://github.com/xiufengliu/dual_clustering/issues)
- [Discussions](https://github.com/xiufengliu/dual_clustering/discussions)

## 🤝 Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the code of conduct

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private inquiries

Thank you for contributing to the Neutrosophic Dual Clustering Random Forest project!
