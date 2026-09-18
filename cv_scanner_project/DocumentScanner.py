import cv2
import numpy as np

# Function to get a frame from the webcam

def get_frame(cap):
    ret, frame = cap.read()
    return ret, frame

# Start webcam

cap = cv2.VideoCapture(0)

# Request higher webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Function to arrange the four document corners

def order_points(points):
    sums = points.sum(axis=1)                         # Calculate x + y for every corner☻
    differences = points[:, 0] - points[:, 1]         # Calculate x - y for every corner

    top_left = points[sums.argmin()]                  # Smallest x + y = Top-Left
    bottom_right = points[sums.argmax()]              # Largest x + y = Bottom-Right
    top_right = points[differences.argmax()]          # Largest x - y = Top-Right
    bottom_left = points[differences.argmin()]        # Smallest x - y = Bottom-Left

    return [
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ]


# Main webcam loop

while True:

    # Get frame from webcam
    ret, frame = get_frame(cap)

    # Check whether frame was captured successfully
    if not ret:
        print("Failed to capture frame")
        break

   
    original_frame = frame.copy()


    # 1. Convert frame to grayscale

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # 2. Reduce noise

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # 3. Detect edges

    edges = cv2.Canny(
        blur,
        50,
        150
    )


    # 4. Find contours

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # Sort contours from largest to smallest
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    # No document detected initially
    document_contour = None


    # 5. Find a four-sided contour

    for contour in contours:

        # Calculate contour perimeter
        perimeter = cv2.arcLength(
            contour,
            True
        )

        # Approximate contour shape
        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # Document should have four corners
        if len(approximation) == 4:
            document_contour = approximation
            break


    # Process document if detected

    if document_contour is not None:

        # Draw green outline around document
        cv2.drawContours(
            frame,
            [document_contour],
            -1,
            (0, 255, 0),
            3
        )

        # Convert contour into 4 x 2 array
        points = document_contour.reshape(4, 2)

        # Arrange corners in correct order
        top_left, top_right, bottom_right, bottom_left = order_points(points)


        # 6. Calculate document width

        width_top = cv2.norm(
            top_right - top_left
        )

        width_bottom = cv2.norm(
            bottom_right - bottom_left
        )

        # Use the larger width
        max_width = int(
            max(width_top, width_bottom)
        )


        # 7. Calculate document height

        height_left = cv2.norm(
            bottom_left - top_left
        )

        height_right = cv2.norm(
            bottom_right - top_right
        )

        # Use the larger height
        max_height = int(
            max(height_left, height_right)
        )


        # 8. Create destination points

        destination_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)


        # 9. Create perspective transformation matrix

        matrix = cv2.getPerspectiveTransform(
            np.array([
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ], dtype=np.float32),
            destination_points
        )


        # 10. Straighten and crop the document

        # Use original_frame instead of frame
        scanned_document = cv2.warpPerspective(
            original_frame,
            matrix,
            (max_width, max_height)
        )


        # Show straightened document
        cv2.imshow(
            "Scanned Document",
            scanned_document
        )


        # 11. Convert scanned document to grayscale

        scanned_gray = cv2.cvtColor(
            scanned_document,
            cv2.COLOR_BGR2GRAY
        )

        # Show grayscale scan
        cv2.imshow(
            "Clean Scan",
            scanned_gray
        )


    # Show original webcam
    cv2.imshow(
        "Webcam",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # Press S to save the scan
    if key == ord('s'):

        if document_contour is not None:
            cv2.imwrite(
                "scanned_document.jpg",
                scanned_gray
            )
            print("Scanned document saved!")


    # Press Q to quit
    if key == ord('q'):
        break


# Release webcam and close windows

cap.release()
cv2.destroyAllWindows()


# Our pipeline

'''
# Webcam
#     ↓
# Grayscale
#     ↓
# Gaussian Blur
#     ↓
# Canny Edges
#     ↓
# Contours
#     ↓
# 4-Corner Detection
#     ↓
# Corner Ordering
#     ↓
# Width / Height
#     ↓
# Perspective Transform
#     ↓
# Crop / Straighten
#     ↓
# Grayscale Scan
#     ↓
# Save

'''
