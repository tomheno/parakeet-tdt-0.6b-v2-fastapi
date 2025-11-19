#!/bin/bash
# Install FFmpeg for torchcodec

echo "Installing FFmpeg and required libraries..."
echo ""

# Update package lists
apt-get update

# Install FFmpeg and all development libraries
apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev

echo ""
echo "Verifying FFmpeg installation..."
ffmpeg -version

echo ""
echo "Checking for required libraries..."
ldconfig -p | grep libavutil
ldconfig -p | grep libavcodec
ldconfig -p | grep libavformat

echo ""
echo "FFmpeg installation complete!"
echo ""