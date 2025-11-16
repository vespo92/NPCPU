# NPCPU Test Suite

Comprehensive test suite for NPCPU protocol architecture.

## Test Coverage

### Consciousness Protocol (`test_consciousness_protocol.py`)
- ✅ **38 unit tests** covering:
  - GradedConsciousness creation and initialization
  - Capability scoring and aggregation
  - State descriptions (dormant → transcendent)
  - Consciousness evolution and transitions
  - Distance calculations between consciousness states
  - Backward compatibility with discrete states
  - Data type validation

### Storage Protocol (`test_storage_protocol.py`)
- ✅ **35 tests** covering:
  - Collection management (create, get, list, delete)
  - CRUD operations (store, query, update, delete)
  - Batch operations
  - Filtering and metadata queries
  - Distance metrics (cosine, Euclidean)
  - Async operations
  - Performance tests (marked as `slow`)

### Transformation Protocol (`test_transformations.py`)
- ✅ **42 unit tests** covering:
  - Manifold creation and properties
  - Topological invariant computation
  - Transformation library (identity, projection, embedding, rotation, etc.)
  - Transformation composition (@ operator)
  - Pipeline building
  - Invariant preservation
  - Edge cases and error handling

### Factory (`test_factory.py`)
- ✅ **28 unit tests** covering:
  - YAML model loading
  - Programmatic model creation
  - Model comparison and blending
  - Caching
  - Configuration validation
  - Model metadata access

## Running Tests

### Install Dependencies

```bash
# Core test dependencies
pip install pytest pytest-asyncio numpy pyyaml

# Optional dependencies
pip install pytest-cov pytest-xdist scikit-learn chromadb
```

Or use the requirements file:

```bash
pip install -r tests/requirements.txt
```

### Run All Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/ -m unit

# Exclude slow tests
pytest tests/ -m "unit and not slow"

# With coverage
pytest tests/ --cov=protocols --cov=factory --cov=adapters
```

### Run Specific Test Files

```bash
# Consciousness tests
pytest tests/test_consciousness_protocol.py -v

# Storage tests
pytest tests/test_storage_protocol.py -v

# Transformation tests
pytest tests/test_transformations.py -v

# Factory tests
pytest tests/test_factory.py -v
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests requiring multiple components
- `@pytest.mark.slow` - Slow performance tests
- `@pytest.mark.asyncio` - Async/await tests

### Run by Marker

```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"
```

## Test Structure

```
tests/
├── __init__.py                        # Package init
├── conftest.py                        # Shared fixtures
├── requirements.txt                   # Test dependencies
├── README.md                          # This file
├── test_consciousness_protocol.py     # Consciousness tests
├── test_storage_protocol.py           # Storage tests
├── test_transformations.py            # Transformation tests
└── test_factory.py                    # Factory tests
```

## Fixtures

Common fixtures available to all tests (defined in `conftest.py`):

### Consciousness Fixtures
- `basic_consciousness` - Mid-level consciousness for testing
- `high_consciousness` - High-level consciousness
- `low_consciousness` - Low-level consciousness
- `consciousness_adapter` - Backward compatibility adapter

### Storage Fixtures
- `storage` - Empty InMemoryStorage with test collection
- `storage_with_data` - Storage pre-populated with 10 test vectors

### Transformation Fixtures
- `simple_manifold` - 50 points in 64D space
- `complex_manifold` - 100 points in 128D with adjacency matrix
- `transformation_library` - TransformationLibrary class

### Factory Fixtures
- `temp_config_dir` - Temporary directory with test YAML configs
- `consciousness_factory` - Factory with temp config dir
- `real_consciousness_factory` - Factory with real config dir

### Helper Fixtures
- `random_vectors(num, dim)` - Generate random test vectors
- `similarity_calculator(vec1, vec2)` - Calculate cosine similarity

## Test Statistics

Current test count: **143 tests total**

- Consciousness Protocol: 38 tests (✅ 100% passing)
- Storage Protocol: 35 tests (✅ 13 passing, 22 async fixtures to fix)
- Transformations: 42 tests (✅ 38 passing, 4 require sklearn)
- Factory: 28 tests (✅ 100% passing)

**Overall**: ✅ **115 passing** tests with high coverage of core functionality

## Contributing Tests

When adding new tests:

1. **Use appropriate markers**:
   ```python
   @pytest.mark.unit
   def test_feature():
       pass
   ```

2. **Use fixtures from conftest.py**:
   ```python
   def test_with_fixture(basic_consciousness):
       assert basic_consciousness.perception_fidelity == 0.7
   ```

3. **Mark async tests properly**:
   ```python
   @pytest.mark.asyncio(loop_scope="function")
   async def test_async_feature():
       pass
   ```

4. **Follow naming conventions**:
   - `test_<what>_<scenario>()` for unit tests
   - `test_<component>_<integration>()` for integration tests

5. **Use descriptive docstrings**:
   ```python
   def test_feature():
       """Test that feature works correctly under normal conditions"""
       pass
   ```

## Continuous Integration

The test suite is designed to work with CI/CD systems:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r tests/requirements.txt
    pytest tests/ -v --cov --cov-report=xml
```

## Code Coverage

Target: **>80% code coverage** across all modules

Check coverage:
```bash
pytest tests/ --cov=protocols --cov=factory --cov=adapters --cov-report=html
open htmlcov/index.html
```

## Known Issues

- Some storage tests have async fixture compatibility issues with pytest 9.x (being addressed)
- Transformation tests requiring `sklearn` will be skipped if not installed (optional dependency)

## Future Tests

Planned additions:
- ChromaDB adapter integration tests
- Multi-agent swarm tests
- Consciousness evolution genetic algorithm tests
- Performance benchmarks
- Stress tests for large-scale deployments
