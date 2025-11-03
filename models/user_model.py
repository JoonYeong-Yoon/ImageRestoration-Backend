"""Compatibility alias module.

This file keeps the old import path `models.user_model` working by re-exporting
the model from `db_models.user_model`. Consumer code that imports
`from models.user_model import User` will continue to work.
"""
from db_models.user_model import *  # noqa: F401,F403
