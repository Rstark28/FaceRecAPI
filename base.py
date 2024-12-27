from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import logging
from pymongo import MongoClient
import os
import io
from PIL import Image

app = Flask(__name__)

# Set up MongoDB connection
client = MongoClient("mongodb+srv://djangoproj210:123@cluster0.ghk5p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['face_recognition']
collection = db['images']

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Purpose: Set up logging configuration.
# Output: None
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Purpose: Check if the file extension is allowed.
# Input: filename (str) - The filename of the uploaded image.
# Output: True if the extension is allowed, False otherwise.
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Purpose: Detect and align a face in an image.
# Input: image (numpy.ndarray) - Image containing a face.
# Output: face (numpy.ndarray) - Aligned face region.
# Raises: IOError - If the face cascade classifier is not loaded, ValueError - If no face is detected.
def detect_and_align_face(image):
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise IOError("Failed to load face cascade classifier.")

    # Detect faces in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100))
    if len(faces) == 0:
        raise ValueError("No face detected. Ensure the image contains a clear, forward-facing face.")

    # Select the largest face for alignment
    faces = sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)
    x, y, w, h = faces[0]
    return gray[y:y+h, x:x+w]

# Purpose: Partition a face image into a grid of smaller regions.
# Input: image (numpy.ndarray) - Aligned face image.
#        grid_size (tuple) - Number of rows and columns to divide the image into.
# Output: partitions (list) - List of smaller regions of the face image.
def partition_face(image, grid_size=(8, 8)):
    h, w = image.shape
    block_h, block_w = h // grid_size[0], w // grid_size[1]
    return [image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] for i in range(grid_size[0]) for j in range(grid_size[1])]

# Purpose: Extract Local Binary Pattern (LBP) features from a list of image partitions.
# Input: partitions (list) - List of image partitions.
#        radius (int) - Radius for LBP calculation.
#        points (int) - Number of points to sample on the circle.
# Output: features (numpy.ndarray) - Concatenated LBP histograms of all partitions.
def extract_lbp_features(partitions, radius=1, points=8):
    # Calculate LBP histograms for each partition
    histograms = []
    for partition in partitions:
        lbp = local_binary_pattern(partition, points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
        histograms.append(hist / hist.sum())
    return np.concatenate(histograms)

# Purpose: Compare two sets of LBP features using the Chi-Squared kernel.
# Input: features1 (numpy.ndarray) - LBP features of the first face.
#        features2 (numpy.ndarray) - LBP features of the second face.
#        weights (numpy.ndarray) - Weights for each feature dimension.
# Output: score (float) - Similarity score between the two faces.
def compare_faces(features1, features2, weights=None):
    # Apply weights to the features if provided
    if weights is not None:
        features1 *= weights
        features2 *= weights

    # Calculate similarity score using Chi-squared kernel
    similarity = -chi2_kernel(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]

    # Normalize and convert to percentage
    similarity_percentage = max(0, 100 - abs(similarity * 100))
    return similarity_percentage

# Route to upload image and store LBP features
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    image_file = request.files['image']
    name = request.form.get('name')  # Get the name from the form data

    if not image_file or not name:
        return jsonify({"error": "Missing image or name"}), 400

    # Check if the file is allowed
    if not allowed_file(image_file.filename):
        return jsonify({"error": "Invalid file format. Only PNG, JPG, and JPEG are allowed."}), 400

    try:
        # Read image from the request (no need to save it to the file system)
        image = Image.open(io.BytesIO(image_file.read()))
        image = np.array(image)

        # Detect and align the face
        face = detect_and_align_face(image)

        # Partition the face into a grid
        partitions = partition_face(face)

        # Extract LBP features from partitions
        lbp_features = extract_lbp_features(partitions)

        # Compare with existing images in the database
        matches = []
        for entry in collection.find():
            for stored_features in entry["lbp_features"]:
                similarity_score = compare_faces(
                    np.array(stored_features),
                    lbp_features
                )
                matches.append({
                    "name": entry["name"],
                    "similarity_score": similarity_score,
                    "id": str(entry["_id"])  # Include MongoDB document ID for reference
                })

        # Sort matches by similarity score in descending order
        matches = sorted(matches, key=lambda x: x["similarity_score"], reverse=True)[:3]

        # Log the matches
        for match in matches:
            logging.info(f"Match found: {match['name']} with similarity score {match['similarity_score']}")

        # Store the new image's LBP features in the database
        existing_entry = collection.find_one({"name": name})
        if existing_entry:
            # If the person already exists, append the LBP features
            collection.update_one(
                {"name": name},
                {
                    "$push": {
                        "lbp_features": lbp_features.tolist()
                    }
                }
            )
            logging.info(f"LBP features updated for {name}.")
        else:
            # If it's a new person, create a new entry
            collection.insert_one({
                "name": name,
                "lbp_features": [lbp_features.tolist()]
            })
            logging.info(f"New entry created for {name}.")

        return jsonify({
            "message": "Image uploaded and processed successfully.",
            "top_matches": matches
        }), 200

    except Exception as e:
        logging.error(f"Error processing image for {name}: {e}")
        return jsonify({"error": f"Error processing image: {e}"}), 500