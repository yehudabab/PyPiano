import cv2
import numpy as np
from sklearn import linear_model
from utils import expand2rgb, draw_connected_components, draw_center_masses, draw_f_keys, outline_black_key_patch


def get_thick_edges(im_gray, steps_log):
    # Canny for edges detection
    thick_edges = np.zeros_like(im_gray)
    edges = cv2.Canny(im_gray, 100, 200)

    # Wiggle the image a bit to get thicker edges, closing holes
    for i in range(-4, 4):
        thick_edges += (np.roll(edges, i, 0) + np.roll(edges, i, 1))
    thick_edges = 1 - (thick_edges > 0).astype(np.uint8)

    # Log
    steps_log.append(('Edges Detection + Thickening', expand2rgb(255 * thick_edges)))
    return thick_edges


def get_white_connected_components(im_gray, steps_log):
    # Get edges and connected components
    edges = get_thick_edges(im_gray, steps_log)
    num_components, components_matrix = cv2.connectedComponents(edges, connectivity=4)

    # Log
    draw_connected_components(range(num_components),
                              components_matrix,
                              [im_gray.shape[0], im_gray.shape[1], 3],
                              'Connected Components',
                              steps_log)

    # Filter out non white components
    white_components = []
    for c_idx in range(num_components):
        gray_patch = im_gray[np.where(components_matrix == c_idx)]
        patch_size = np.prod(gray_patch.shape)
        if np.sum(gray_patch) / patch_size > 200 and patch_size > 100:
            white_components.append((c_idx, patch_size))
    size_avg = np.average([c for _, c in white_components])

    white_components = [idx for idx, s in white_components if 0.5 * size_avg < s < 2 * size_avg]

    # Log
    draw_connected_components(white_components,
                              components_matrix,
                              [im_gray.shape[0], im_gray.shape[1], 3],
                              'White Connected Components',
                              steps_log)

    # Calculate center masses
    y_cms, x_cms = [], []
    for c_idx in white_components:
        z = np.zeros_like(im_gray)
        z[np.where(components_matrix == c_idx)] = 1
        nz_indices = np.nonzero(z)
        y_cm, x_cm = int(np.mean(nz_indices[0])), int(np.mean(nz_indices[1]))
        y_cms.append(y_cm)
        x_cms.append(x_cm)

    # Log
    draw_center_masses(white_components, components_matrix, im_gray, 'Center Masses', steps_log)
    return white_components, np.array(y_cms), np.array(x_cms)[..., None], components_matrix


def filter_center_masses(white_components, x_cms, y_cms, components_matrix, im_gray, processing_log):
    try:
        # Get a linear fit of the center masses by throwing out outliers,
        # then filter out center masses that don't conform with the consensus
        model = linear_model.RANSACRegressor()
        model.fit(x_cms, y_cms)
        dists = np.abs(np.array(model.predict(x_cms) - y_cms))
        avg = np.average(dists)
        for component_idx in range(len(white_components)):
            if dists[component_idx] > 5. * avg:
                del white_components[component_idx]
                del y_cms[component_idx]
                del x_cms[component_idx]
    except Exception:
        pass

    # Log
    draw_center_masses(white_components, components_matrix, im_gray, 'Filtered Center Masses', processing_log)


def classify_white_components(white_components, x_cms, y_cms, im_gray, components_matrix, processing_log):
    # Sort white components by horizontal position
    x_cms, y_cms, white_components = zip(*sorted(zip(x_cms, y_cms, white_components), key=lambda xyw: xyw[0]))

    # Build log image
    log_image = expand2rgb(im_gray)

    # Determine which white component (i.e. piano key) has a black key to its right
    black_key_exists = []
    for idx in range(len(white_components) - 1):

        # Calculate the border between the white key and its right neighbor
        wc_a_patch = np.where(components_matrix == white_components[idx])
        wc_b_patch = np.where(components_matrix == white_components[idx+1])
        border_points = []
        for y in range(max(wc_a_patch[0][0], wc_b_patch[0][0]),
                       min(wc_a_patch[0][-1], wc_b_patch[0][-1])):
            max_x_in_a = np.max(wc_a_patch[1][np.where(wc_a_patch[0] == y)])
            min_x_in_b = np.min(wc_b_patch[1][np.where(wc_b_patch[0] == y)])
            border_points.append((int((min_x_in_b + max_x_in_a)/2.), y))

        # Log - draw borders
        for x, y in border_points:
            log_image[(y-2):(y+2), (x-2):(x+2), :] = np.array([0, 0, 255])

        # Determine a patch along the upper border to inspect
        inspection_middle_points = sorted(border_points, key=lambda p: p[1])
        inspection_middle_points = inspection_middle_points[
                                   int(0.1 * len(inspection_middle_points)):
                                   int(0.3 * len(inspection_middle_points))]
        inspection_x, inspection_y = int(np.average([p[0] for p in inspection_middle_points])),\
                                     int(np.average([p[1] for p in inspection_middle_points]))
        patch_value = np.sum(im_gray[(inspection_y-15):(inspection_y+15), (inspection_x-15):(inspection_x+15)])

        # If the patch is relatively black, there's a black key there
        black_threshold = 30*30*255*0.6
        black_key_exists.append(1 if patch_value < black_threshold else 0)

        # Log - outline inspection patches that saw a black key
        if patch_value < black_threshold:
            outline_black_key_patch(log_image, inspection_y, inspection_x)

    # Log
    processing_log.append(('Borders & Black Keys', log_image))

    # Find a sequence of 3 white keys that have a black key to their right
    # The first of those is an F key
    f_keys = [k - 1 for k in list(np.where(np.convolve(np.array(black_key_exists), [1, 1, 1], mode='same') == 3)[0])]
    f_keys_wc_indices = [white_components[k] for k in f_keys]

    # Log
    draw_f_keys(f_keys_wc_indices, components_matrix, im_gray, 'F Key Detection', processing_log)

    # Assign a note to each white component,
    # according to its position in relation to the F notes
    notes = []
    keys_order = ['F', 'G', 'A', 'B', 'C', 'D', 'E']
    for wc_idx in range(len(white_components)):
        offset = (wc_idx - f_keys[0]) % 7
        notes.append(keys_order[offset])
    return white_components, y_cms, x_cms, notes
