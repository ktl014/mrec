"""Main module."""

from collections import namedtuple
import io
import logging
import logging.config
import json
import os

# Setup default config
with io.open(os.path.join(os.path.dirname(__file__), 'logger.json')) as f:
    logging.config.dictConfig(json.load(f))

logging.root.setLevel(logging.DEBUG)
