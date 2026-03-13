import numpy as np
import random


def summarize_tensor(x):
    return f"\033[34m{str(tuple(x.shape)).ljust(24)}\033[0m (\033[31mmin {x.min().item():+.4f}\033[0m / \033[32mmean {x.mean().item():+.4f}\033[0m / \033[33mmax {x.max().item():+.4f}\033[0m)"


def calculate_mouth_open_similarity(landmarks_list, select_idx, top_k=50, ascending=True):
    num_landmarks = len(landmarks_list)
    mouth_open_ratios = np.zeros(num_landmarks)
    print(np.shape(landmarks_list))
    for i, landmarks in enumerate(landmarks_list):
        mouth_top = np.array(landmarks[62])
        mouth_bottom = np.array(landmarks[66])
        mouth_open_ratio = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_open_ratios[i] = mouth_open_ratio

    differences = np.abs(mouth_open_ratios - mouth_open_ratios[select_idx])
    signed_differences = (mouth_open_ratios - mouth_open_ratios[select_idx]).tolist()
    print(differences.shape)
    if ascending:
        top_indices = np.argsort(differences)[:top_k]
    else:
        top_indices = np.argsort(-differences)[:top_k]
    return top_indices.tolist(), signed_differences


#############################################################################################
def get_closed_mouth(landmarks_list, ascending=True, top_k=50):
    num_landmarks = len(landmarks_list)

    mouth_open_ratios = np.zeros(num_landmarks)
    for i, landmarks in enumerate(landmarks_list):
        mouth_top = np.array(landmarks[62])
        mouth_bottom = np.array(landmarks[66])
        mouth_open_ratio = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_open_ratios[i] = mouth_open_ratio

    if ascending:
        top_indices = np.argsort(mouth_open_ratios)[:top_k]
    else:
        top_indices = np.argsort(-mouth_open_ratios)[:top_k]
    return top_indices


def calculate_landmarks_similarity(selected_idx, landmarks_list, image_shapes, start_index, end_index, top_k=50, ascending=True):
    """
    Calculate the similarity between sets of facial landmarks and return the indices of the most similar faces.

    Parameters:
    landmarks_list (list): A list containing sets of facial landmarks, each element is a set of landmarks.
    image_shapes (list): A list containing the shape of each image, each element is a (width, height) tuple.
    start_index (int): The starting index of the facial landmarks.
    end_index (int): The ending index of the facial landmarks.
    top_k (int): The number of most similar landmark sets to return. Default is 50.
    ascending (bool): Controls the sorting order. If True, sort in ascending order; If False, sort in descending order. Default is True.

    Returns:
    similar_landmarks_indices (list): A list containing the indices of the most similar facial landmarks for each face.
    resized_landmarks (list): A list containing the resized facial landmarks.
    """
    num_landmarks = len(landmarks_list)
    resized_landmarks = []

    for i in range(num_landmarks):
        landmark_array = np.array(landmarks_list[i])
        selected_landmarks = landmark_array[start_index:end_index]
        resized_landmark = resize_landmark(selected_landmarks, w=image_shapes[i][0], h=image_shapes[i][1], new_w=256, new_h=256)
        resized_landmarks.append(resized_landmark)

    resized_landmarks_array = np.array(resized_landmarks)
    distances = np.linalg.norm(resized_landmarks_array - resized_landmarks_array[selected_idx][np.newaxis, :], axis=2)
    overall_distances = np.mean(distances, axis=1)

    if ascending:
        sorted_indices = np.argsort(overall_distances)
        similar_landmarks_indices = sorted_indices[1:top_k + 1].tolist()
    else:
        sorted_indices = np.argsort(-overall_distances)
        similar_landmarks_indices = sorted_indices[0:top_k].tolist()

    return similar_landmarks_indices


def process_bbox_musetalk(face_array, landmark_array):
    x_min_face, y_min_face, x_max_face, y_max_face = map(int, face_array)
    x_min_lm = min([int(x) for x, y in landmark_array])
    y_min_lm = min([int(y) for x, y in landmark_array])
    x_max_lm = max([int(x) for x, y in landmark_array])
    y_max_lm = max([int(y) for x, y in landmark_array])
    x_min = min(x_min_face, x_min_lm)
    y_min = min(y_min_face, y_min_lm)
    x_max = max(x_max_face, x_max_lm)
    y_max = max(y_max_face, y_max_lm)

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)

    return [x_min, y_min, x_max, y_max]


def shift_landmarks_to_face_coordinates(landmark_list, face_list):
    """
        Translates the data in landmark_list to the coordinates of the cropped larger face.

        Parameters:
        landmark_list (list): A list containing multiple sets of facial landmarks.
        face_list (list): A list containing multiple facial images.

        Returns:
        landmark_list_shift (list): The list of translated landmarks.
        bbox_union (list): The list of union bounding boxes.
        face_shapes (list): The list of facial shapes.
    """
    landmark_list_shift = []
    bbox_union = []
    face_shapes = []

    for i in range(len(face_list)):
        landmark_array = np.array(landmark_list[i])
        face_array = face_list[i]
        f_landmark_bbox = process_bbox_musetalk(face_array, landmark_array)
        x_min, y_min, x_max, y_max = f_landmark_bbox
        landmark_array[:, 0] = landmark_array[:, 0] - f_landmark_bbox[0]
        landmark_array[:, 1] = landmark_array[:, 1] - f_landmark_bbox[1]
        landmark_list_shift.append(landmark_array)
        bbox_union.append(f_landmark_bbox)
        face_shapes.append((x_max - x_min, y_max - y_min))

    return landmark_list_shift, bbox_union, face_shapes


def resize_landmark(landmark, w, h, new_w, new_h):
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]

    return landmark_resized


def get_src_idx(drive_idx, T, sample_method, landmarks_list, image_shapes, top_k_ratio):
    """
        Calculate the source index (src_idx) based on the given drive index, T, s, e, and sampling method.

        Parameters:
        - drive_idx (int): The current drive index.
        - T (int): Total number of frames or a specific range limit.
        - sample_method (str): Sampling method, which can be "random" or other methods.
        - landmarks_list (list): List of facial landmarks.
        - image_shapes (list): List of image shapes.
        - top_k_ratio (float): Ratio for selecting top k similar frames.

        Returns:
        - src_idx (int): The calculated source index.
    """
    if sample_method == "random":
        src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)
    elif sample_method == "pose_similarity":
        top_k = int(top_k_ratio * len(landmarks_list))
        try:
            top_k = int(top_k_ratio * len(landmarks_list))
            landmark_start_idx = 0
            landmark_end_idx = 16
            pose_similarity_list = calculate_landmarks_similarity(drive_idx, landmarks_list, image_shapes, landmark_start_idx, landmark_end_idx, top_k=top_k, ascending=True)
            src_idx = random.choice(pose_similarity_list)
            while abs(src_idx - drive_idx) < 5:
                src_idx = random.choice(pose_similarity_list)
        except Exception as e:
            print(e)
            return None
    elif sample_method == "pose_similarity_and_closed_mouth":
        landmark_start_idx = 0
        landmark_end_idx = 16
        try:
            top_k = int(top_k_ratio * len(landmarks_list))
            closed_mouth_list = get_closed_mouth(landmarks_list, ascending=True, top_k=top_k)
            pose_similarity_list = calculate_landmarks_similarity(drive_idx, landmarks_list, image_shapes, landmark_start_idx, landmark_end_idx, top_k=top_k, ascending=True)
            common_list = list(set(closed_mouth_list).intersection(set(pose_similarity_list)))
            if len(common_list) == 0:
                src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)
            else:
                src_idx = random.choice(common_list)

            while abs(src_idx - drive_idx) < 5:
                src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)

        except Exception as e:
            print(e)
            return None

    elif sample_method == "pose_similarity_and_mouth_dissimilarity":
        top_k = int(top_k_ratio * len(landmarks_list))
        try:
            top_k = int(top_k_ratio * len(landmarks_list))

            landmark_start_idx = 0
            landmark_end_idx = 16

            pose_similarity_list = calculate_landmarks_similarity(drive_idx, landmarks_list, image_shapes, landmark_start_idx, landmark_end_idx, top_k=top_k, ascending=True)

            landmark_start_idx = 60
            landmark_end_idx = 67

            mouth_dissimilarity_list = calculate_landmarks_similarity(drive_idx, landmarks_list, image_shapes, landmark_start_idx, landmark_end_idx, top_k=top_k, ascending=False)

            common_list = list(set(pose_similarity_list).intersection(set(mouth_dissimilarity_list)))
            if len(common_list) == 0:
                src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)
            else:
                src_idx = random.choice(common_list)

            while abs(src_idx - drive_idx) < 5:
                src_idx = random.randint(drive_idx - 5 * T, drive_idx + 5 * T)

        except Exception as e:
            print(e)
            return None

    else:
        raise ValueError(f"Unknown sample_method: {sample_method}")
    return src_idx
