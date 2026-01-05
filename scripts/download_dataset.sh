#!/bin/bash

# Download map2seq dataset
# Dataset: https://map2seq-assets.schumann.pub/map2seq_v1-0.zip

set -e

DATA_DIR="./data/map2seq"
ZIP_URL="https://map2seq-assets.schumann.pub/map2seq_v1-0.zip"
ZIP_NAME="map2seq_v1-0.zip"

echo "======================================"
echo "Downloading map2seq dataset (ZIP)"
echo "======================================"

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo ""
echo "Downloading ZIP file..."
wget -O "$ZIP_NAME" "$ZIP_URL"

echo ""
echo "Extracting ZIP..."
unzip -o "$ZIP_NAME"

# Optional: clean up zip file
rm "$ZIP_NAME"

echo ""
echo "======================================"
echo "Dataset download complete!"
echo "======================================"
echo ""
echo "Dataset location: $DATA_DIR"
echo ""
echo "Contents:"
echo "  - osm/graph/nodes.txt"
echo "  - osm/graph/links.txt"
echo "  - osm/graph/pois.txt"
echo "  - osm/graph/poi_links.txt"
echo "  - splits/*.json"
echo ""
