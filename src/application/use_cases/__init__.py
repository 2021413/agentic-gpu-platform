"""Use cases: one class per intent the platform exposes.

Each is a thin, explicit script over domain objects and ports. Business rules
live in the domain; what lives here is sequencing, transactions and the
translation between commands and aggregates.
"""
