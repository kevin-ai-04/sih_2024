import cv2
import numpy as np
from matplotlib import pyplot as plt
from pythreejs import *
from IPython.display import display
import trimesh
from PIL import Image

# Load the image
img = cv2.imread('floor1.png', 0)


# Apply thresholding to remove text
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.erode(thresh, kernel, iterations=2)
thresh = cv2.dilate(thresh, kernel, iterations=2)

# Apply edge detection
edges = cv2.Canny(thresh, threshold1=50, threshold2=150)

# Find contours with cv2.RETR_TREE to get internal contours as well
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create lists for OBJ export
vertices = []
faces = []

# Function to create a simple wall object and store vertices
def create_wall(contour, height=3):
    # Create a 3D mesh from the contour
    for i in range(len(contour)):
        v1 = [contour[i][0][0], 0, contour[i][0][1]]
        v2 = [contour[(i+1)%len(contour)][0][0], 0, contour[(i+1)%len(contour)][0][1]]
        v3 = [contour[(i+1)%len(contour)][0][0], height, contour[(i+1)%len(contour)][0][1]]
        v4 = [contour[i][0][0], height, contour[i][0][1]]

        # Add vertices in clockwise order to create faces (for the OBJ file)
        vertices.extend([v1, v2, v3, v4])

        # Add face indices (4 vertices per wall), assuming consecutive numbering
        base_idx = len(vertices) - 4
        faces.append([base_idx + 1, base_idx + 2, base_idx + 3, base_idx + 4])  # Define faces in correct order

# Add walls to the scene
for contour in contours:
    create_wall(contour)

# Save OBJ file after generating all walls
def save_obj(vertices, faces, filename="floor_plan_3d.obj"):
    with open(filename, "w") as file:
        # Write vertices
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write faces
        for face in faces:
            file.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")

save_obj(vertices, faces)


