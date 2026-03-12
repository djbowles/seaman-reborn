#!/bin/bash
# NGC Docker login helper — run from WSL2
NGC_KEY=$(awk -F'= ' '/apikey/{print $2}' "$HOME/.ngc/config" | tr -d '\r')
docker login nvcr.io -u '$oauthtoken' -p "$NGC_KEY"
