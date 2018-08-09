import matplotlib.pyplot as plt
import numpy as np
import cv2


def expand2rgb(im):
    return np.transpose(np.stack([im]*3), axes=(1, 2, 0))


def imshow(im):
    plt.imshow(im)
    plt.show()


def outline_black_key_patch(log_image, inspection_y, inspection_x):
    outline_color = np.array([255, 0, 0])
    log_image[(inspection_y - 15):(inspection_y - 12), (inspection_x - 15):(inspection_x + 15), :] = outline_color
    log_image[(inspection_y + 12):(inspection_y + 15), (inspection_x - 15):(inspection_x + 15), :] = outline_color
    log_image[(inspection_y - 15):(inspection_y + 15), (inspection_x - 15):(inspection_x - 12), :] = outline_color
    log_image[(inspection_y - 15):(inspection_y + 15), (inspection_x + 12):(inspection_x + 15), :] = outline_color


def draw_connected_components(components_indices, components_matrix, im_shape, title, steps_log):
    cc_rgb = np.zeros(im_shape, dtype=np.int)
    for c_idx in components_indices:
        y_locations, x_locations = np.where(components_matrix == c_idx)
        cc_rgb[y_locations, x_locations, :] = np.random.randint(0, 255, size=(3, ), dtype=np.int)
    steps_log.append((title, cc_rgb))


def draw_center_masses(white_components, components_matrix, im_gray, title, steps_log):
    radius = 7
    im_rgb = expand2rgb(im_gray)
    for c_idx in white_components:
        z = np.zeros_like(im_gray)
        z[np.where(components_matrix == c_idx)] = 1
        indices = np.nonzero(z)
        y_cm, x_cm = int(np.mean(indices[0])), int(np.mean(indices[1]))
        im_rgb[(y_cm - radius):(y_cm + radius), (x_cm - radius):(x_cm + radius), :] = np.array([255, 0, 0])
    steps_log.append((title, im_rgb))


def show_summary(processing_log):
    fig = plt.figure()

    fig.suptitle('Processing Summary', fontsize=16)
    for step_idx, step_data in enumerate([processing_log[0], processing_log[-1]]):
        ax = plt.subplot('12%i' % (step_idx + 1))
        ax.tick_params(
            bottom=False,
            right=False,
            left=False,
            top=False,
            labelbottom=False,
            labelleft=False)
        ax.set_title('%s' % step_data[0])
        ax.imshow(step_data[1])
    plt.show()


def show_process_steps(processing_log):
    fig = plt.figure()
    fig.suptitle('Processing Steps', fontsize=16)
    for step_idx, step_data in enumerate(processing_log):
        ax = plt.subplot('33%i' % (step_idx + 1))
        ax.tick_params(
            bottom=False,
            right=False,
            left=False,
            top=False,
            labelbottom=False,
            labelleft=False)
        ax.set_title('Step %i: %s' % (step_idx + 1, step_data[0]))
        ax.imshow(step_data[1])
    plt.show()


def draw_f_keys(f_keys_indices, components_matrix, gray_im, title, processing_log):
    rgb_im = expand2rgb(gray_im)
    for f_key in f_keys_indices:
        rgb_im[np.where(components_matrix == f_key)] = np.array([0, 0, 255])
    processing_log.append((title, rgb_im))


def highlight_note(rgb_im, components_matrix, note_cc_id, color_delta):
    rgb_im[np.where(components_matrix == note_cc_id)] -= color_delta
    rgb_im[np.where(components_matrix == note_cc_id)] = \
        np.clip(rgb_im[np.where(components_matrix == note_cc_id)], 0, 255)


def put_text(rgb_im, y, x, text):
    rgb_im[(y - 12):(y + 12), (x - 12):(x + 12), :] = np.array([0, 0, 0])
    cv2.putText(rgb_im, text, (x - 8, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=255, thickness=2)


def full_annotation(rgb_im, y_cms, x_cms, white_component_notes, white_components, components_matrix, processing_log):
    for idx, wc_idx in enumerate(white_components):
        color_delta = (np.array([100, 100, 0] if idx % 2 == 0 else [100, 0, 100], dtype=np.uint8))
        highlight_note(rgb_im, components_matrix, wc_idx, color_delta)
        put_text(rgb_im, y_cms[idx], x_cms[idx], white_component_notes[idx])
    processing_log.append(('Full Annotation', rgb_im))
