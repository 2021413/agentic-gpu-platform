"""The lightweight process that lives next to a GPU.

It owns no orchestration logic. Its whole job is to make one inference server
discoverable and trustworthy: announce it, prove it is alive, stop taking work
on request, and leave cleanly. Everything else is the control plane's business.
"""
