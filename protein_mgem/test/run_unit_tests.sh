#!/usr/bin/env bash

pytest scripts/general/analysis.py
pytest scripts/general/attribution.py
pytest scripts/general/data.py
pytest scripts/general/preprocess.py
pytest scripts/general/util.py
