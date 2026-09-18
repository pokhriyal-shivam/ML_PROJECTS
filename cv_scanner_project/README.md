# Real-Time Document Scanner using OpenCV

A real-time document scanner built using **Python and OpenCV**. The system uses a webcam to detect a document, identify its four corners, correct its perspective, straighten and crop the document, and convert it into a clean grayscale scan.

## Features
* Real-time document detection using a webcam
* Automatic edge detection
* Contour detection
* Automatic detection of four document corners
* Corner ordering for accurate perspective transformation
* Perspective correction
* Automatic document cropping and straightening
* Grayscale document conversion
* Save the scanned document as an image

## Technologies Used

* **Python**
* **OpenCV**
* **NumPy**

## How It Works

The project processes the webcam frame through the following computer vision pipeline:

```text
Webcam
   ↓
Grayscale Conversion
   ↓
Gaussian Blur
   ↓
Canny Edge Detection
   ↓
Contour Detection
   ↓
Four-Corner Detection
   ↓
Corner Ordering
   ↓
Width & Height Calculation
   ↓
Perspective Transformation
   ↓
Document Cropping & Straightening
   ↓
Grayscale Scan
   ↓
Save
```

## Project Structure

```text
real-time-document-scanner/
│
├── document_scanner.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/real-time-document-scanner.git
```

### 2. Open the project folder

```bash
cd real-time-document-scanner
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

## Usage

Run the program using:

```bash
python document_scanner.py
```

Make sure your webcam is connected and accessible.

Place a document in front of the webcam. The program will detect the document outline and display the corrected scan.

### Controls

| Key | Action                    |
| --- | ------------------------- |
| `S` | Save the scanned document |
| `Q` | Quit the application      |

The scanned document will be saved as:

```text
scanned_document.jpg
```

## Computer Vision Concepts Used

### Grayscale Conversion

Converts the webcam image from a color image into grayscale to simplify image processing.

### Gaussian Blur

Reduces image noise before edge detection.

### Canny Edge Detection

Detects strong edges in the image, helping identify the boundaries of the document.

### Contour Detection

Finds continuous boundaries around objects in the image.

### Contour Approximation

Approximates detected contours into simpler geometric shapes. A contour with four points is considered a potential document.

### Perspective Transformation

Corrects the perspective of the document so that a tilted document appears straight.

### Grayscale Scanning

The corrected document is converted into grayscale to produce a simple scanned-document appearance.

## Future Improvements

* Automatic document enhancement
* Adaptive thresholding for a cleaner scan
* Shadow and background removal
* Automatic brightness and contrast adjustment
* Support for multiple documents
* PDF generation
* Mobile or web-based interface

## Author

**Shivam Pokhriyal**

This project was created as a practical **Computer Vision project using OpenCV and Python**.
