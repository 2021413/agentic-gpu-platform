"""Application layer.

Use cases orchestrate domain objects through ports. This layer knows *what* the
platform does, never *how* the outside world is reached: it may import
``domain`` and nothing else. The rule is enforced by
``tests/architecture/test_dependency_rule.py``.
"""
