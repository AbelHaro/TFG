import cv2

def split_image_with_overlap(original_image, new_width, new_height, overlap):
    """
    Splits an image into smaller sub-images with a specified overlap.

    Args:
        original_image (numpy.ndarray): The original image to split.
        new_width (int): Width of the new sub-images.
        new_height (int): Height of the new sub-images.
        overlap (int): Overlap in pixels between sub-images.

    Returns:
        list: A list of sub-images.
    """
    # Get the dimensions of the original image
    original_height, original_width = original_image.shape[:2]

    # Check if the new dimensions are valid
    if new_width > original_width or new_height > original_height:
        raise ValueError("New dimensions cannot be larger than the original image dimensions.")

    # Calculate the step size for x and y (accounting for overlap)
    step_x = new_width - overlap
    step_y = new_height - overlap
    
    horizontal_steps = (original_width - new_width) // step_x + 1
    vertical_steps = (original_height - new_height) // step_y + 1
    
    print(f"Splitting the image into {horizontal_steps}x{vertical_steps} sub-images.")

    # Initialize a list to store the sub-images
    sub_images = []

    # Loop through the image and extract sub-images
    for y in range(0, original_height - new_height + 1, step_y):
        for x in range(0, original_width - new_width + 1, step_x):
            # Extract the sub-image
            sub_image = original_image[y:y + new_height, x:x + new_width]
            sub_images.append(sub_image)
            
    print(f"Split the image into {len(sub_images)} sub-images.")

    return sub_images


# Example usage
if __name__ == "__main__":
    # Load the original image
    original_image = cv2.imread('./prueba.jpg')

    # Check if the image was loaded successfully
    if original_image is None:
        print("Error: Could not load image.")
        exit()

    # Define the new size and overlap
    new_width = 1200
    new_height = 1200
    overlap = 300

    # Split the image
    sub_images = split_image_with_overlap(original_image, new_width, new_height, overlap)

    # Save the sub-images
    for i, sub_image in enumerate(sub_images):
        output_filename = f'sub_image_{i + 1}.jpg'
        cv2.imwrite(output_filename, sub_image)
        print(f"Saved {output_filename}")

    print("Image splitting completed.")
    
    