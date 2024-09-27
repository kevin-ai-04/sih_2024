import cv2
import numpy as np
from pythreejs import *
from IPython.display import display
import flet as ft
import os
import subprocess
import trimesh
from PIL import Image

def main(page: ft.Page):
    # Set the title and window size
    page.title = "FPC"
    page.window.width = 600
    page.window.height = 400
    page.horizontal_alignment = "center"
    page.vertical_alignment = "center"

    # Create a Text control to display the selected image path
    image_path_text = ft.Text(value="", text_align="center", size=16)
    
    selected_file_path = None  # Variable to store the selected file path

    # Define a function to handle the file selection result
    def dialog_picker(e: ft.FilePickerResultEvent):
        nonlocal selected_file_path
        if e.files:
            selected_file_path = e.files[0].path
            image_path_text.value = f"Selected image path: {selected_file_path}"
            print(selected_file_path)
            page.update()  # Refresh the page to display the selected path

    def running(e):
        if selected_file_path:
            print('running')
            # Load the image
            img = cv2.imread(selected_file_path, 0)  # Use the selected image path

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
            def create_wall(contour, height=100):
                # Create a 3D mesh from the contour
                for i in range(len(contour)):
                    x1, y1 = contour[i][0]
                    x2, y2 = contour[(i+1) % len(contour)][0]
                    
                    # Define the wall vertices
                    v1 = [x1, 0, y1]
                    v2 = [x2, 0, y2]
                    v3 = [x2, height, y2]
                    v4 = [x1, height, y1]

                    # Add vertices in clockwise order to create faces (for the OBJ file)
                    vertices.extend([v1, v2, v3, v4])

                    # Add face indices (4 vertices per wall), assuming consecutive numbering
                    base_idx = len(vertices) - 4
                    faces.append([base_idx + 1, base_idx + 2, base_idx + 3, base_idx + 4])  # Define faces in correct order

            # Add walls to the scene
            for contour in contours:
                create_wall(contour)

            # Debugging output
            print(f"Total vertices: {len(vertices)}")
            print(f"Total faces: {len(faces)}")
            # Update face validation to account for 1-based indexing in OBJ
            for face in faces:
                if any(idx - 1 >= len(vertices) for idx in face):  # Subtract 1 for 0-based indexing
                    print(f"Face with invalid index: {face}")

            # Save OBJ file after generating all walls
            def save_obj(vertices, faces, filename="floor_plan_3d.obj"):
                with open(filename, "w") as file:
                    # Write vertices
                    for vertex in vertices:
                        file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                    
                    # Write faces, note OBJ is 1-based indexing
                    for face in faces:
                        file.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")

                # Automatically open the 3D file using the system's default application
                if os.name == 'posix':  # macOS or Linux
                    #subprocess.run(['open', filename])
                    print("OBJ file created successfully!!")
                elif os.name == 'nt':  # Windows
                    os.startfile(filename)

            save_obj(vertices, faces)

            # # Optional: Display the 3D model using trimesh
            # try:
            #     # Convert faces to 0-based indexing for trimesh
            #     trimesh_faces = np.array(faces) - 1

            #     mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=trimesh_faces)
            #     mesh.show()
            # except Exception as e:
            #     print(f"Error displaying 3D model: {e}")

    # Create a FilePicker control
    file_picker = ft.FilePicker(on_result=dialog_picker)

    # Add the FilePicker control to the page (hidden by default)
    page.add(file_picker)

    # Add controls to the page
    page.add(
        ft.Column(
            controls=[
                ft.Text("2D to 3D floor plan", text_align="center", size=24),
                ft.ElevatedButton(text="Select Image", on_click=lambda _: file_picker.pick_files()),
                image_path_text,
                ft.ElevatedButton(text="Run", on_click=lambda e: running(e)),  # Pass the event to the running function
            ],
            horizontal_alignment="center",  # Center-align content horizontally
            alignment="center",  # Center-align content vertically
            spacing=10  # Optional: add space between items
        )
    )

# Run the Flet app
ft.app(target=main)
