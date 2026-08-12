#!/bin/sh

echo "Downloading chromium.txt benchmark data..."
curl -L -o benches/data/chromium.txt https://gist.github.com/ii14/637689ef8d071824e881a78044670310/raw/dc1dbc859daa38b62f4b9a69dec1fc599e4735e7/data.txt
echo
echo "Downloading arabic_unicode.txt benchmark data..."
curl -L -o benches/data/arabic_unicode.txt https://gist.github.com/saghen/d6d582b681bebecc2d2ad4a6ec9534e5/raw/3ffad717bbee2f77262c5d2d9f2e92b032f18760/arabic_unicode.txt
echo
echo "Downloading korean_unicode.txt benchmark data..."
curl -L -o benches/data/korean_unicode.txt https://gist.github.com/saghen/f9b4c6e9870ee08913275520c178bafa/raw/8fd9355ee26ddc99a493eccd2475da382e6b7f8a/korean_unicode.txt
echo
echo "All benchmarks downloaded"
