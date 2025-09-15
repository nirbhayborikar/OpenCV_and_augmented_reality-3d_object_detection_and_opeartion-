import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# Function to read an image
def read_image(image):
    load_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)
    return load_image



# Function to display an image
def show_image(load_image):
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('image', load_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Function to detect markers
def detect_markers(load_image):
    """
    Returns the first detected marker's corner coordinates.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)#Dictionary fetching
    parameters = cv2.aruco.DetectorParameters()# Initialize detector parameters
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(load_image, dictionary, parameters=parameters)
    # Detect markers in the image

    return marker_corners[0][0].astype(int)

# Function to grab corners of an image
def grab_corners(load_image):
    points = np.array([
        [0, 0],
        [(load_image.shape[1]) - 1, 0],
        [(load_image.shape[1]) - 1, (load_image.shape[0]) - 1],
        [0, (load_image.shape[0]) - 1]
    ])
    return points # Returns corner points of the image.


# Function to remove black pixels in an image with small non-zero values to avoid issue during processing.
def remove_black_pixels(load_image):
    for i in range(load_image.shape[0]):
        for j in range(load_image.shape[1]):
            if load_image[i, j].any() == 0:
                load_image[i, j] = [1,1,1]
    return load_image

# Function to calculate the polygon width and height
def get_polygon_width_height(marker_corners):
    polygon_width = int(
        (abs(marker_corners[0, 0] - marker_corners[1, 0]) + abs(marker_corners[2, 0] - marker_corners[3, 0])) / 2
    )
    polygon_height = int(
        (abs(marker_corners[0, 1] - marker_corners[3, 1]) + abs(marker_corners[1, 1] - marker_corners[2, 1])) / 2
    )
    return polygon_width, polygon_height # return height and width of polygon from the marker corner.

# Function to create a polygon in the frame
def create_polygon_in_frame(polygon_width, polygon_height, frame_center):
    
    polygon_in_frame = np.array([
        [frame_center[0] - 0.5 * polygon_width, frame_center[1] - 0.5 * polygon_height],
        [frame_center[0] + 0.5 * polygon_width, frame_center[1] - 0.5 * polygon_height],
        [frame_center[0] + 0.5 * polygon_width, frame_center[1] + 0.5 * polygon_height],
        [frame_center[0] - 0.5 * polygon_width, frame_center[1] + 0.5 * polygon_height]
    ])
    return polygon_in_frame #Creates a polygon which centered at the frame's center

# Function to augments the frame onto the detected marker in the image_with_marker.
def augment_image(frame, image_with_marker):
    """
    
    """
    frame = remove_black_pixels(frame)
    frame_corners = grab_corners(frame)
    frame_center = np.median(frame_corners, axis=0)
    marker_corners = detect_markers(image_with_marker)
    
    # Define the polygon for augmentation (e.g., fixed width and height)
    polygon_in_frame = create_polygon_in_frame(polygon_width=250, polygon_height=250, frame_center=frame_center)

    # Calculate the transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(polygon_in_frame.astype(np.float32), marker_corners.astype(np.float32))

    # Warp the image
    warped_image = cv2.warpPerspective(frame, transformation_matrix, dsize=(image_with_marker.shape[1], image_with_marker.shape[0]))

    # Create a mask and overlay the warped image on the marker
    mask = np.zeros_like(image_with_marker)
    for i in range(warped_image.shape[0]):
        for j in range(warped_image.shape[1]):
            if warped_image[i, j].any() == 0:
                mask[i, j] = 255

    image_with_marker_black = cv2.bitwise_and(mask, image_with_marker)
    augmented_frame = cv2.bitwise_or(warped_image, image_with_marker_black)
    return augmented_frame

# Main execution
temple = read_image('/home/nirbhay-borikar/Pictures/Photos/cats 2.jpg')
classroom = read_image('/home/nirbhay-borikar/opencv_env/documeents of opencv/Task1_aruco_marker/Room with ArUco Markers-20241110/20221115_113437.jpg')

# Perform augmentation
augmented_frame = augment_image(temple,classroom)

# Display the result using matplotlib
preview = cv2.cvtColor(augmented_frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(200, 200))
plt.imshow(preview)

# Save the final image to a file
output_path = '/home/nirbhay-borikar/Pictures/Photos/output_image_6.jpg'
cv2.imwrite(output_path, cv2.cvtColor(augmented_frame, cv2.COLOR_BGR2RGB))
print(f"Final image saved at: {output_path}")

