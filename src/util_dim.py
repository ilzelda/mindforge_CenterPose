import numpy as np

def resize_into_padded_image(kps):
    cam_width = 640
    cam_height = 480
    
    points = np.array(kps, dtype=np.int32).reshape(4, 2)

    points[:, 0] = points[:, 0] * 512 / cam_width
    
    normalized_height = (512 * cam_height / cam_width)
    padding = (512 - normalized_height) / 2
    points[:, 1] = points[:, 1] / cam_height * normalized_height + padding

    return points

def calculate_object_dimensions(marker_result, dets):
    marker_corners = np.array(marker_result)
    resized_marker_corners = resize_into_padded_image(marker_corners)
    top_length = ((resized_marker_corners[0][0] - resized_marker_corners[1][0]) ** 2 + (resized_marker_corners[0][1] - resized_marker_corners[1][1]) ** 2) ** 0.5
    left_length = ((resized_marker_corners[0][0] - resized_marker_corners[3][0]) ** 2 + (resized_marker_corners[0][1] - resized_marker_corners[3][1]) ** 2) ** 0.5

    points = dets['kps']
    x_length = None
    z_length = None
    
    ori = 6
    x_dir = 2
    z_dir = 5
    y_dir = 8

    box_size_pixel = {}
    for j, e in enumerate([[ori, x_dir], [ori, z_dir]]): # [x, z]
        temp = [e[0] - 1, e[1] - 1]

        if points[temp[1], 0] <= -10000 or points[temp[1], 1] <= -10000 or points[temp[0], 0] <= -10000 or \
                points[temp[0], 1] <= -10000:
            pass
        else:
            start_point = (int(points[temp[0], 0]), int(points[temp[0], 1]))
            end_point = (int(points[temp[1], 0]), int(points[temp[1], 1]))
            pixel_length = ((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) ** 0.5

            if j == 0:
                box_size_pixel['x'] = pixel_length
                x_length = pixel_length / top_length * 5
            else:
                box_size_pixel['z'] = pixel_length
                z_length = pixel_length / left_length * 5
    
    y_length = x_length / dets['obj_scale'][0]
    box_size_pixel['y'] = ((points[3][0] - points[2][0]) ** 2 + (points[3][1] - points[2][1]) ** 2) ** 0.5

    return {'dimension' : {
                'x': (x_length.item(), x_dir, (0, 255, 0)), 
                'y': (y_length.item(), y_dir, (128, 0, 128)), 
                'z': (z_length.item(), z_dir, (0, 128, 255))
            },
            'info' : {
                'marker_corners': resized_marker_corners,
                'marker_top_length': top_length,
                'marker_left_length': left_length,
                'box_size_pixel' : box_size_pixel
            }
    }