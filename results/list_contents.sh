#!/usr/bin/env bash

RES_DIR=$(dirname "${BASH_SOURCE[0]}")
paste <(find ${RES_DIR} -name 'README.md') <(find ${RES_DIR} -name 'README.md' | xargs head -n 1 | awk 'NR%3 == 2') | sort