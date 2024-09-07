import cv2
import numpy as np
from matplotlib import pyplot as plt
from pythreejs import *
from IPython.display import display
import trimesh

# Load the image
img = cv2.imread('floor2.png', 0)

# Apply edge detection
edges = cv2.Canny(img, threshold1=50, threshold2=150)

# Find contours with cv2.RETR_TREE to get internal contours as well
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Approximate contours to polygons and filter
polygons = [cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True) for contour in contours]

# Extracting straight lines (for walls) using Hough Line Transformation
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Create scene and camera
camera = PerspectiveCamera(position=[10, 10, 10], fov=75)
scene = Scene(children=[
    AmbientLight(color='#777777'),
    DirectionalLight(color='white', position=[3, 5, 1], intensity=0.5)
])

# Lists for OBJ export
vertices = []
faces = []

# Function to create a simple wall object and store vertices
def create_wall(x1, y1, x2, y2, height=3, thickness=0.1):
    # Use the thickness parameter for the depth of the wall (Z-axis in 3D space)
    geometry = BoxGeometry(width=abs(x2 - x1), height=height, depth=thickness)
    material = MeshBasicMaterial(color='gray')
    wall = Mesh(geometry=geometry, material=material)

    # Adjust wall position to place it properly in the scene
    wall.position = [(x1 + x2) / 2, height / 2, (y1 + y2) / 2]

    # Add vertices for OBJ export (top and bottom vertices of the box)
    v1 = [x1, 0, y1]
    v2 = [x2, 0, y2]
    v3 = [x2, height, y2]
    v4 = [x1, height, y1]

    # Add vertices in clockwise order to create faces (for the OBJ file)
    vertices.extend([v1, v2, v3, v4])

    # Add face indices (4 vertices per wall), assuming consecutive numbering
    base_idx = len(vertices) - 4
    faces.append([base_idx + 1, base_idx + 2, base_idx + 3, base_idx + 4])  # Define faces in correct order

    return wall


# Add walls to the scene
for line in lines:
    x1, y1, x2, y2 = line[0]
    scene.add(create_wall(x1 / 100, y1 / 100, x2 / 100, y2 / 100))

# Create the renderer
renderer = Renderer(camera=camera, scene=scene, controls=[OrbitControls(controlling=camera)], width=600, height=400)

# Display the 3D scene
display(renderer)

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
