import os
import cv2
from logger import Logger
import numpy as np

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

def correct_perspective(mask_img_path, real_img_path, logger, aspect_ratio=2.35, padding=50):

    logger.info(f'Correcting perspective for image: {real_img_path}\n')

    # Load the mask image
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_COLOR)

    # Convert the mask image to grayscale
    mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # Threshold the mask image to get a binary image
    _, mask_img_bin = cv2.threshold(mask_img_gray, 127, 255, cv2.THRESH_BINARY)

    # Define the kernel size
    kernel_size = 5

    # Define the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply morphological opening to the binary mask image
    mask_img_opened = cv2.morphologyEx(mask_img_bin, cv2.MORPH_OPEN, kernel)

    # Find all connected components in the binary mask image
    num_labels, labels_im = cv2.connectedComponents(mask_img_opened)

    #log the number of components
    logger.info(f'Number of components: {num_labels}')
    
    #print each component
    for i in range(num_labels):
        logger.info(f'Component {i}: {len(labels_im[labels_im == i])}')

    # Compute the area of each component
    area = [len(labels_im[labels_im == i]) for i in range(num_labels)]

    #log the area of each component
    logger.info(f'\nArea of each component: {area}')

    # Find the index of the largest component (ignore the background, i.e., index 0)
    largest_comp_idx = area[1:].index(max(area[1:])) + 1  # +1 due to background

    #log the largest component
    logger.info(f'Largest component: {largest_comp_idx}\n')

    # Keep only the largest component in the mask image
    largest_comp = (labels_im == largest_comp_idx).astype('uint8') * 255

    # Find the contours in the binary mask image
    contours, _ = cv2.findContours(largest_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #log the number of contours
    logger.info(f'\nNumber of contours: {len(contours)}')

    #print each contour
    for i in range(len(contours)):
        logger.info(f'Contour {i}: {len(contours[i])} points, area: {cv2.contourArea(contours[i])}')

    # Among all contours, find the one with the maximum area
    cnt = max(contours, key=cv2.contourArea)

    #log the largest contour
    logger.info(f'Largest contour: {len(cnt)} points, area: {cv2.contourArea(cnt)}\n')


    # Find the minimum area rectangle that encloses the contour
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = order_points(box)  # ensure the points are in the correct order

    #log the rectangle
    logger.info(f'\nRectangle: {rect}\n')

    #log the bounding box
    logger.info(f'\nBounding box: \n{box}\n')

    # Define the dimensions of the output image
    output_width = 500 + 2 * padding  # add padding to the width
    output_height = round(output_width * aspect_ratio)  # adjust the height according to the aspect ratio
    output_size = (output_width, output_height)

    #log the output width, height and size
    logger.info(f'Output width: {output_width}\n Output height: {output_height}\n Output size: {output_size}\n')

    # The corners of the rectangle in the output image (adjusted for padding)
    dst_pts = np.array([[padding, padding],  # top-left corner (subtracted padding)
                        [output_size[0]-padding-1, padding],  # top-right corner (subtracted padding)
                        [output_size[0]-padding-1, output_size[1]-padding-1],  # bottom-right corner (added padding)
                        [padding, output_size[1]-padding-1]], dtype='float32')  # bottom-left corner (added padding)

    #log the destination points
    logger.info(f'Destination points: \n{dst_pts}\n')

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(box, dst_pts)

    #log the perspective transformation matrix
    logger.info(f'Perspective transformation matrix: \n{M}\n')

    # Load the real image
    real_img = cv2.imread(real_img_path, cv2.IMREAD_COLOR)

    # Apply the perspective transformation to the real image
    img_warped = cv2.warpPerspective(real_img, M, output_size)

    #log the real image
    logger.info(f'Real image: {real_img.shape}')

    #log the warped image
    logger.info(f'Warped image: {img_warped.shape}')


    return img_warped



def main():
    # Step 1: Prompt user if okay to delete existing output directory
    if os.path.exists('output'):
        response = input("Output directory already exists. Do you want to delete it? (y/n): ")
        if response != 'n':
            os.system('rm -rf output')
        else:
            print("You decided not to delete the output directory. Continue at your own risk.")

            #wait for user to press enter
            input("Press Enter to continue...")

    # Step 2: Create output directories
    if not os.path.exists('output'):
        os.makedirs('output')
        os.makedirs('output/photos')
        os.makedirs('output/logs')

    # Step 3: Initialize the logger
    logger = Logger('output.log', 'output').get_logger()

    # Step 4: Get the list of the mask image paths
    mask_image_paths = os.listdir('segmentations')
    logger.info("Mask image paths: " + str(mask_image_paths))


    #Print to log
    logger.info("Processing " + str(len(mask_image_paths)) + " images")

    for mask_img_name in mask_image_paths:

        logger.info("Processing mask: " + mask_img_name)

        try:
            
            # Construct paths
            mask_img_path = os.path.join('segmentations', mask_img_name)
            real_img_name = mask_img_name.split('.')[0] + '.jpg'
            real_img_path = os.path.join('photos', real_img_name)

            # Initialize the logger for the current image
            individual_logger = Logger(f'logs/{real_img_name.split(".")[0]}.log', 'output').get_logger()
            logger.info(f'Correcting image: {real_img_name}')
            
            # Step 5: Call correct_perspective() and save the result
            img_warped = correct_perspective(mask_img_path, real_img_path, individual_logger, aspect_ratio=2.35, padding=50)

            cv2.imwrite(os.path.join('output', 'photos', real_img_name), img_warped)

            
            individual_logger.info(f'Successfully processed and saved image: {real_img_name}')

        except Exception as e:
            # Log the error if an exception is thrown
            individual_logger = Logger(f'logs/{real_img_name.split(".")[0]}.log', 'output').get_logger()
            individual_logger.error(f'Error processing image: {real_img_name}, error: {str(e)}')

    # Step 6: Log the completion of all operations
    logger.info('All images processed.')

if __name__ == "__main__":
    main()

