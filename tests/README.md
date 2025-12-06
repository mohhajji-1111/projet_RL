# Test Suite

Unit tests for the robot navigation RL framework.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_environment.py

# Run with coverage
pytest --cov=src tests/
```

## Test Structure

- `test_environment.py` - Environment functionality
- `test_agent.py` - Agent implementations
- `test_training.py` - Training loop and utilities

## Writing Tests

Example test:
```python
def test_environment_reset():
    env = NavigationEnv()
    state, info = env.reset()
    assert state.shape == (8,)
    assert 'distance_to_goal' in info
```
