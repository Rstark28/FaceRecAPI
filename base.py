from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from PIL import Image
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import chi2_kernel
import logging
import io
from dotenv import load_dotenv
import os
import faiss

app = Flask(__name__)

# Load environment variables from .env
load_dotenv()

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
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

# Purpose: Preprocess an image by resizing it to a target size.
# Input: image (numpy.ndarray) - Image to be preprocessed.
#        target_size (tuple) - Target size for resizing.
# Output: preprocessed_image (numpy.ndarray) - Resized image.
def preprocess_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size)

# Purpose: Detect and align a face in an image.
# Input: image (numpy.ndarray) - Image containing a face.
# Output: face (numpy.ndarray) - Aligned face region.
# Raises: IOError - If the face cascade classifier is not loaded, ValueError - If no face is detected.
def detect_and_align_face(image):
    # Load the DNN model for face detection (deploy and weights from OpenCV or other sources)
    model_path = "deploy.prototxt"  # Path to the deploy prototxt file
    weights_path = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the weights file
    net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

    if net.empty():
        raise IOError("Failed to load DNN face detection model.")

    # Prepare the image for the DNN model
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104.0, 177.0, 123.0], swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Parse detections and find the largest face
    max_confidence = 0
    best_face = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            face_area = (x2 - x1) * (y2 - y1)
            if confidence > max_confidence:
                max_confidence = confidence
                best_face = (x1, y1, x2, y2)

    if best_face is None:
        raise ValueError("No face detected. Ensure the image contains a clear, forward-facing face.")

    # Extract and align the face
    x1, y1, x2, y2 = best_face
    face = image[y1:y2, x1:x2]

    return face


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

# Purpose: Create a Faiss index for fast similarity search.
# Input: features (numpy.ndarray) - LBP features of all images in the database.
# Output: index (faiss.IndexFlatL2) - Faiss index for similarity search.
def create_faiss_index(features):
    d = features.shape[1]  # Dimensionality of features
    index = faiss.IndexFlatL2(d)
    index.add(features)
    return index

# Purpose: Find the top k matches for a query image in the Faiss index.
# Input: index (faiss.IndexFlatL2) - Faiss index for similarity search.
#        query_features (numpy.ndarray) - LBP features of the query image.
#        k (int) - Number of top matches to retrieve.
# Output: distances (numpy.ndarray) - Distances to the top k matches.
#         indices (numpy.ndarray) - Indices of the top k matches.
def find_top_matches(index, query_features, k=3):
    distances, indices = index.search(query_features.reshape(1, -1), k)
    return distances[0], indices[0]


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

        # Preprocess and detect face
        image = preprocess_image(image)
        face = detect_and_align_face(image)

        # Partition the face into a grid
        h, w = face.shape
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError("Face dimensions are not evenly divisible by the grid size.")

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

    except ValueError as ve:
        logging.error(f"Value error while processing image for {name}: {ve}")
        return jsonify({"error": f"Value error: {ve}"}), 400

    except Exception as e:
        logging.error(f"Error processing image for {name}: {e}", exc_info=True)
        return jsonify({"error": f"Error processing image: {e}"}), 500
