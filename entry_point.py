import cv2
import numpy as np
import os
from utils import show_process_steps, full_annotation, show_summary
from improc import get_white_connected_components, filter_center_masses, classify_white_components


def parse_image(file_path):
    # Load image
    im_rgb = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)

    # Processing log, containing the steps of the process
    processing_log = [
        ('Original Image', im_rgb)
    ]

    # Get white components and their center masses
    white_components, y_cms, x_cms, components_matrix = get_white_connected_components(im_gray, processing_log)
    filter_center_masses(white_components, x_cms, y_cms, components_matrix, im_gray, processing_log)

    # Classify the white components as musical notes
    white_components, y_cms, x_cms, white_component_notes = classify_white_components(
        white_components, np.squeeze(x_cms), y_cms,
        im_gray, components_matrix, processing_log)
    full_annotation(im_rgb.copy(), y_cms, x_cms, white_component_notes, white_components, components_matrix, processing_log)

    # Show the process steps and the final result
    show_process_steps(processing_log)
    show_summary(processing_log)


if __name__ == '__main__':
    images_path = 'images'
    for file in os.listdir(images_path):
        parse_image(os.path.join(images_path, file))
