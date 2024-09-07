import cv2
import numpy as np
from pythreejs import *
from IPython.display import display
import flet as ft
import os
import subprocess

def main(page: ft.Page):
    # Set the title and window size
    page.title = "FPC"
    page.window.width = 600
    page.window.height = 400
    page.horizontal_alignment="center"
    page.vertical_alignment="center"

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

            if img is None:
                print("Error: Could not load image. Please check the file path.")
            else:
                # Edge detection with adjusted thresholds for better wall and door/window detection
                edges = cv2.Canny(img, threshold1=40, threshold2=130)

                # Detect straight lines using Hough Line Transform with adjusted parameters
                lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=60, minLineLength=40, maxLineGap=15)

                if lines is not None:
                    # Create the scene and camera
                    camera = PerspectiveCamera(position=[10, 10, 10], fov=75)
                    scene = Scene(children=[
                        AmbientLight(color='#777777'),
                        DirectionalLight(color='white', position=[3, 5, 1], intensity=0.5)
                    ])

                    vertices = []
                    faces = []

                    def create_scaled_wall(x1, y1, x2, y2, thickness=0.135 * 1.12, height=1.5 * 0.75, scale_factor=1.5, is_window=False, is_door=False):
                        """
                        Creates a 3D wall with optional window or door representation.
                        """
                        dx = x2 - x1
                        dy = y2 - y1
                        length = np.sqrt(dx ** 2 + dy ** 2)

                        # Scale the length and height
                        length *= scale_factor
                        height *= scale_factor

                        # Normalize direction vector to get perpendicular for thickness
                        norm_x = -dy / length
                        norm_y = dx / length

                        # Calculate inner and outer points
                        x1_inner = x1 + norm_x * thickness / 2
                        y1_inner = y1 + norm_y * thickness / 2
                        x2_inner = x2 + norm_x * thickness / 2
                        y2_inner = y2 + norm_y * thickness / 2

                        x1_outer = x1 - norm_x * thickness / 2
                        y1_outer = y1 - norm_y * thickness / 2
                        x2_outer = x2 - norm_x * thickness / 2
                        y2_outer = y2 - norm_y * thickness / 2

                        if is_window:
                            # Window as hollow rectangle in the middle of the wall
                            geometry = BoxGeometry(width=length * 0.8, height=height * 0.5, depth=thickness * 0.8)
                            material = MeshBasicMaterial(color='lightblue', opacity=0.2, transparent=True)
                        elif is_door:
                            # Leave space blank for doors
                            geometry = BoxGeometry(width=length, height=height / 2, depth=thickness)
                            material = MeshBasicMaterial(color='gray', opacity=0)
                        else:
                            # Normal wall material
                            geometry = BoxGeometry(width=length, height=height, depth=thickness)
                            material = MeshBasicMaterial(color='gray')

                        wall = Mesh(geometry=geometry, material=material)
                        wall.position = [(x1 + x2) / 2, height / 2, (y1 + y2) / 2]

                        # Append vertices for OBJ export
                        v1 = [x1_inner, 0, y1_inner]
                        v2 = [x2_inner, 0, y2_inner]
                        v3 = [x2_outer, 0, y2_outer]
                        v4 = [x1_outer, 0, y1_outer]
                        v5 = [x1_inner, height, y1_inner]
                        v6 = [x2_inner, height, y2_inner]
                        v7 = [x2_outer, height, y2_outer]
                        v8 = [x1_outer, height, y1_outer]

                        vertices.extend([v1, v2, v3, v4, v5, v6, v7, v8])

                        base_idx = len(vertices) - 8
                        faces.extend([
                            [base_idx + 1, base_idx + 2, base_idx + 6, base_idx + 5],  # Front face
                            [base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6],  # Right face
                            [base_idx + 3, base_idx + 4, base_idx + 8, base_idx + 7],  # Back face
                            [base_idx + 4, base_idx + 1, base_idx + 5, base_idx + 8],  # Left face
                            [base_idx + 5, base_idx + 6, base_idx + 7, base_idx + 8],  # Top face
                            [base_idx + 1, base_idx + 4, base_idx + 3, base_idx + 2]   # Bottom face
                        ])

                        return wall

                    def detect_windows_v4(lines, img, window_aspect_ratio_threshold=0.6, brightness_threshold=150):
                        """
                        Detects windows as non-filled rectangles based on aspect ratio and brightness threshold.
                        """
                        windows = []
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            width = abs(x2 - x1)
                            height = abs(y2 - y1)
                            aspect_ratio = width / height if height > 0 else float('inf')

                            if aspect_ratio > window_aspect_ratio_threshold:
                                region = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                                if region.size == 0:  # Check if the region is empty
                                    continue
                                intensity = np.mean(region)
                                if intensity > brightness_threshold:
                                    windows.append(tuple(line[0]))  # Use tuple to make comparison easier
                        return windows

                    def detect_doors_v4(edges, img, contour_threshold=0.1):
                        """
                        Detect doors by identifying thin lines and semicircle shapes using contours.
                        """
                        doors = []
                        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            perimeter = cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

                            # Check for potential door contours based on shape (thin lines or semicircles)
                            if len(approx) >= 5:
                                bounding_rect = cv2.boundingRect(approx)
                                aspect_ratio = bounding_rect[2] / bounding_rect[3]
                                
                                if 0.9 < aspect_ratio < 1.1:
                                    doors.append(tuple(bounding_rect))  # Use tuple for easier comparison

                        return doors

                    def merge_lines(lines, distance_threshold=15):
                        merged_lines = []
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            if len(merged_lines) == 0:
                                merged_lines.append(line)
                            else:
                                prev_x1, prev_y1, prev_x2, prev_y2 = merged_lines[-1][0]
                                distance = np.sqrt((x1 - prev_x2) ** 2 + (y1 - prev_y2) ** 2)
                                if distance < distance_threshold:
                                    merged_lines[-1] = [[min(prev_x1, x1), min(prev_y1, y1), max(prev_x2, x2), max(prev_y2, y2)]]
                                else:
                                    merged_lines.append(line)
                        return merged_lines

                    merged_lines = merge_lines(lines)
                    windows = detect_windows_v4(merged_lines, img)
                    doors = detect_doors_v4(edges, img)

                    for line in merged_lines:
                        x1, y1, x2, y2 = line[0]
                        is_window = tuple(line[0]) in windows
                        is_door = any(cv2.pointPolygonTest(np.array([door]), (x1, y1), False) >= 0 for door in doors)

                        scene.add(create_scaled_wall(x1 / 100, y1 / 100, x2 / 100, y2 / 100, thickness=0.135 * 1.12, scale_factor=1.5, is_window=is_window, is_door=is_door))

                    renderer = Renderer(camera=camera, scene=scene, controls=[OrbitControls(controlling=camera)], width=600, height=400)
                    display(renderer)

                    def save_obj(vertices, faces, filename="floor_plan_3d.obj"):
                        with open(filename, "w") as file:
                            for vertex in vertices:
                                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                            for face in faces:
                                file.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")

                        # Open the file after saving
                        if os.name == 'nt':  # For Windows
                            os.startfile(filename)
                        elif os.name == 'posix':  # For macOS and Linux
                            if 'darwin' in os.uname().sysname.lower():  # macOS
                                subprocess.call(['open', filename])
                            else:  # Linux
                                subprocess.call(['xdg-open', filename])

                    save_obj(vertices, faces)
                else:
                    print("No lines were detected. Please check the edge detection.")
        else:
            print("No file selected. Please select an image file.")

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