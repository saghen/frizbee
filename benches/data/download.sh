#!/usr/bin/env bash
set -xeuo pipefail

# chromium
echo "Downloading chromium benchmark data..."
curl -L -o benches/data/chromium.txt https://gist.github.com/ii14/637689ef8d071824e881a78044670310/raw/dc1dbc859daa38b62f4b9a69dec1fc599e4735e7/data.txt

# unicode
echo "Downloading unicode benchmark data..."
curl -L -o benches/data/unicode.tar.gz https://github.com/lemire/unicode_lipsum/archive/refs/heads/main.tar.gz
echo "Extracting unicode benchmark data..."
rm -r benches/data/unicode
mkdir benches/data/unicode
tar -xzf benches/data/unicode.tar.gz --strip-components=1 -C benches/data/unicode
rm benches/data/unicode.tar.gz
echo "Done"
