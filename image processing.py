import cv2

# Load image
img = cv2.imread('D:/python-pip/image processing..webp')  # Replace with your image path
img = cv2.resize(img, (800, 800))

# 1. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Apply median blur
gray_blur = cv2.medianBlur(gray, 5)

# 3. Detect edges using adaptive thresholding
edges = cv2.adaptiveThreshold(gray_blur, 205,
                              cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 9, 9)

# 4. Apply bilateral filter to smooth color
color = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=300)

# 5. Combine edges and color image
cartoon = cv2.bitwise_and(color, color, mask=edges)

# Show results
cv2.imshow("Original Image", img)
cv2.imshow("Cartoon Image", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

