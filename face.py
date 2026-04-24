'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    img_np = img.permute(1, 2, 0).contiguous().numpy()
    face_locations = face_recognition.face_locations(img_np, number_of_times_to_upsample=1, model="hog")
    for loc in face_locations:
        top, right, bottom, left = loc
        detection_results.append([float(left), float(top), float(right - left), float(bottom - top)])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    img_names = list(imgs.keys())
    encodings_list = []
    valid_names = []
    for name in img_names:
        img = imgs[name]
        img_np = img.permute(1, 2, 0).contiguous().numpy()
        face_locs = face_recognition.face_locations(img_np)
        if len(face_locs) == 0:
            h, w = img_np.shape[0], img_np.shape[1]
            face_locs = [(0, w, h, 0)]
        face_encs = face_recognition.face_encodings(img_np, known_face_locations=face_locs, num_jitters=1)
        if len(face_encs) > 0:
            encodings_list.append(torch.tensor(face_encs[0], dtype=torch.float32))
            valid_names.append(name)

    if len(encodings_list) == 0:
        return cluster_results

    encodings = torch.stack(encodings_list)
    N = encodings.shape[0]

    best_labels = None
    best_inertia = float('inf')
    for seed in range(20):
        torch.manual_seed(seed)
        first_idx = torch.randint(0, N, (1,)).item()
        centroid_indices = [first_idx]
        for _ in range(1, K):
            chosen = encodings[torch.tensor(centroid_indices)]
            dists = torch.cdist(encodings, chosen).min(dim=1).values
            dists_sq = dists ** 2
            total = dists_sq.sum()
            if total.item() == 0:
                next_idx = torch.randint(0, N, (1,)).item()
            else:
                probs = dists_sq / total
                next_idx = torch.multinomial(probs, 1).item()
            centroid_indices.append(next_idx)
        centroids = encodings[torch.tensor(centroid_indices)].clone()
        labels = torch.zeros(N, dtype=torch.long)
        for iteration in range(300):
            distances = torch.cdist(encodings, centroids)
            new_labels = distances.argmin(dim=1)
            if iteration > 0 and torch.equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(K):
                mask = labels == k
                if mask.any():
                    centroids[k] = encodings[mask].mean(dim=0)
        final_dists = torch.cdist(encodings, centroids)
        inertia = final_dists.gather(1, labels.unsqueeze(1)).sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()

    for i, name in enumerate(valid_names):
        cluster_idx = best_labels[i].item()
        cluster_results[cluster_idx].append(name)

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)