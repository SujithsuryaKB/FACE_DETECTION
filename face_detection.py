import cv2

def detect_faces(image_path):
    """
    Detect faces in an image using OpenCV Haar Cascade.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None
    """
    # Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # Convert the image to grayscale (required for Haar Cascade)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Face Detection", image)
    print(f"Detected {len(faces)} face(s) in the image.")

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = "face_image.jpg"  # Replace with the path to your image
    detect_faces(image_path)
