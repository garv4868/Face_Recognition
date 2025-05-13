# Face Recognition Program

This program implements a basic face recognition system using Principal Component Analysis (PCA) for dimensionality reduction and a Multi-Layer Perceptron (MLP) classifier for face identification.

## Features:
- Loads face images from a structured dataset directory
- Preprocesses images by converting to grayscale and resizing to 100x100 pixels
- Normalises the data by subtracting the mean and dividing by the standard deviation
- Reduces dimensionality using PCA (100 components)
- Trains a neural network classifier with two hidden layers (100 and 50 neurons)
- Provides a prediction function for new face images

## Usage:
The program expects a dataset directory structure where subdirectories represent different individuals (labels) containing their face images. It demonstrates prediction on a sample image.

## License:
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
