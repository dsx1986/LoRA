# LoRA Tests

This directory contains unit tests for the LoRA library.

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_linear.py
```

To run a specific test:

```bash
pytest tests/test_linear.py::TestLinear::test_forward
```

## Test Coverage

To run tests with coverage:

```bash
pytest --cov=loralib
```

To generate a coverage report:

```bash
pytest --cov=loralib --cov-report=html
```

This will create an HTML report in the `htmlcov` directory.

## Test Structure

- `test_lora_layer.py`: Tests for the base LoRALayer class
- `test_linear.py`: Tests for the Linear layer with LoRA
- `test_embedding.py`: Tests for the Embedding layer with LoRA
- `test_merged_linear.py`: Tests for the MergedLinear layer with LoRA
- `test_conv2d.py`: Tests for the Conv2d layer with LoRA
- `test_utils.py`: Tests for utility functions
