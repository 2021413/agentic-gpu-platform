"""Data transfer objects of the application layer.

Plain frozen dataclasses on purpose: pydantic belongs to the API boundary,
where untrusted input is validated. Inside the application, types are already
trusted and validation lives in the domain.
"""
