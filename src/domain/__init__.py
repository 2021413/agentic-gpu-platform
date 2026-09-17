"""Domain layer.

Pure business model: entities, value objects, domain services, events and the
*ports* (abstract interfaces) through which the outside world is reached.

Hard rule, enforced by ``tests/architecture/test_dependency_rule.py``:
this package must never import ``application``, ``infrastructure`` or
``interfaces``, and must never import a web/db/cache framework.
"""
