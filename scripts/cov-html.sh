#!/bin/bash
coverage run --branch --source=app -m pytest -ssvv tests && coverage html --fail-under=90
