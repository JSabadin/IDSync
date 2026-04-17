import sys
from contextlib import contextmanager
import torch
import kornia
from torch.nn import functional as F

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

@contextmanager
def temporary_sys_path(path):
    """
    A context manager to temporarily add a directory to sys.path.
    """
    original_sys_path = sys.path.copy()  # Make a copy of the original sys.path
    sys.path.insert(0, path)             # Insert the desired path at the beginning
    try:
        yield
    finally:
        sys.path = original_sys_path     # Restore the original sys.path

def load_retinaface(pytorch_retinaface_library_path):
    """
    Function to load the RetinaFace model from Pytorch_Retinaface.
    """
    with temporary_sys_path(pytorch_retinaface_library_path):
        try:
            from models.retinaface import RetinaFace 
            from data import cfg_re50
            from utils.box_utils import decode, decode_landm
            from layers.functions.prior_box import PriorBox

            return cfg_re50, RetinaFace, decode, decode_landm, PriorBox
        except ImportError as e:
            print(f"Error importing RetinaFace model: {e}")
            sys.exit(1)

def torch_nms(dets, thresh):
    """
    Non-Maximum Suppression (NMS) using PyTorch.
    
    Parameters:
    - dets (torch.Tensor): Tensor of shape (N, 5), where each row is [x1, y1, x2, y2, score].
    - thresh (float): IoU threshold for suppression.

    Returns:
    - keep (list): Indices of boxes to keep.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]  # Index of the current box with highest score
        keep.append(i.item())

        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1 + 1, min=0.0)
        h = torch.clamp(yy2 - yy1 + 1, min=0.0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def detect_faces(images, model, cfg, decode, decode_landm, PriorBox, confidence_threshold=0.6, nms_threshold=0.4, top_k=5000, keep_top_k=750, resize=1):
    """
    Detects faces in the input tensor using a PyTorch model.

    Parameters:
    - images (torch.Tensor): Input images with batch dimension normalized with mean=0.5, std=0.5.
    - model (torch.nn.Module): Loaded PyTorch model for inference.
    - cfg (dict): Configuration dictionary for the model.
    - decode, decode_landm, PriorBox functions!

    Returns:
    - List[dict]: A list of dictionaries containing:
        - 'boxes' (torch.Tensor): Bounding boxes of detected faces.
        - 'scores' (torch.Tensor): Confidence scores of detected faces.
        - 'landmarks' (torch.Tensor): Keypoints for each detected face.
    """
    images = images.float()

    device = images.device
    bs, _, h, w = images.shape

    # Denormalize and apply RetinaFace-specific normalization
    images = (images * 0.5 + 0.5) * 255  # Denormalize to [0, 255]
    mean = torch.tensor([104, 117, 123], dtype=images.dtype, device=device).view(1, 3, 1, 1)
    images = images - mean

    # Scaling tensors for box and landmark decoding
    scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
    scale1 = torch.tensor([w, h] * 5, dtype=torch.float32, device=device)

    # Forward pass through the model
    loc, conf, landms = model(images)

    # Generate prior boxes
    priorbox = PriorBox(cfg, image_size=(h, w))
    priors = priorbox.forward().to(device)
    prior_data = priors.data

    all_detections = []
    for b in range(bs):
        # Decode boxes, scores, and landmarks
        boxes = decode(loc[b], prior_data, cfg['variance'])
        boxes = boxes * scale / resize

        scores = conf[b][:, 1]  # Confidence for class 1 (faces)
        landmarks = decode_landm(landms[b], prior_data, cfg['variance'])
        landmarks = landmarks * scale1 / resize

        # Filter boxes by confidence
        inds = scores > confidence_threshold
        boxes = boxes[inds]
        scores = scores[inds]
        landmarks = landmarks[inds]

        if boxes.numel() == 0:
                # No detections
                all_detections.append({
                    'boxes': torch.empty((0, 4), device=device),
                    'scores': torch.empty((0,), device=device),
                    'landmarks': torch.empty((0, 10), device=device),
                })
                continue

        # Keep top-k before NMS
        order = scores.argsort(descending=True)[:top_k]
        boxes = boxes[order]
        scores = scores[order]
        landmarks = landmarks[order]

        # Perform NMS
        dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1)  # (N, 5)
        keep = torch_nms(dets, nms_threshold)
        dets = dets[keep]
        landmarks = landmarks[keep]

        # Keep top-k after NMS
        dets = dets[:keep_top_k]
        landmarks = landmarks[:keep_top_k]

        # If detections exist, keep only the largest face
        if len(dets) > 0:
            largest_idx = torch.argmax((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))
            dets = dets[largest_idx:largest_idx + 1]
            landmarks = landmarks[largest_idx:largest_idx + 1]

        # Store results
        all_detections.append({
            'boxes': dets[:, :4],  # Bounding boxes
            'landmarks': landmarks  # Keypoints
        })

    return all_detections


def align_and_crop_faces(tensor, detections, output_size=112, transform_size=256, padding=True):
    """
    Aligns and crops detected faces using facial landmarks.

    Parameters:
    - tensor (torch.Tensor): Input tensor with batch dimension normalized with mean=0.5, std=0.5.
    - detections (List[dict]): Output from `detect_faces`, containing bounding boxes and landmarks.
    - output_size (int): Desired output image size (output_size x output_size).
    - transform_size (int): Intermediate size for transformations.
    - padding (bool): Whether to add padding for areas outside the bounding box.

    Returns:
    - List[torch.Tensor]: Aligned and cropped face images.
    """
    output_faces = []

    for b, det in enumerate(detections):
        boxes, landmarks = det['boxes'], det['landmarks']
        if len(boxes) == 0:
            # If no bounding boxes, resize the original image
            resized_tensor = F.interpolate(
                tensor[b].unsqueeze(0), size=(output_size, output_size), mode='bilinear', align_corners=False
            ).squeeze(0)
            output_faces.append(resized_tensor)
            continue

        largest_idx = torch.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        keypoints = landmarks[largest_idx].reshape(-1, 2)


        left_eye, right_eye, nose, mouth_left, mouth_right = keypoints
        aligned_face = align_and_crop(
            tensor[b], left_eye, right_eye, mouth_left, mouth_right,
            output_size, transform_size, padding
        )
        output_faces.append(aligned_face)

    output_faces_tensor = torch.stack(output_faces)

    return output_faces_tensor


def align_and_crop(img, left_eye, right_eye, mouth_left, mouth_right, output_size=112, transform_size=256, padding=True):
    """
    Align the face by computing a perspective transformation based on facial landmarks,
    then crop and resize to the desired output size.

    Parameters:
    - img (torch.Tensor): Input image (H, W, C).
    - left_eye (tuple): Coordinates of the left eye (x, y).
    - right_eye (tuple): Coordinates of the right eye (x, y).
    - mouth_left (tuple): Coordinates of the left mouth corner (x, y).
    - mouth_right (tuple): Coordinates of the right mouth corner (x, y).
    - output_size (int): Desired output image size (output_size x output_size).
    - transform_size (int): Intermediate size for transformations.
    - padding (bool): Whether to add padding for areas outside the bounding box.

    Returns:
    - torch.Tensor: Aligned and cropped face of size (3, output_size, output_size).
    """
    device = img.device
    
    original_dtype = img.dtype
    if original_dtype == torch.float16 or original_dtype == torch.bfloat16:
        img = img.to(torch.float32)

    C, H, W = img.shape

    eye_center = torch.tensor([(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2], device=device, dtype=img.dtype)
    mouth_center = torch.tensor([(mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2], device=device, dtype=img.dtype)

    eye_to_eye = torch.tensor([right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]], device=device, dtype=img.dtype)
    eye_to_mouth = mouth_center - eye_center

    x_axis = eye_to_eye / torch.norm(eye_to_eye)
    y_axis = torch.tensor([-x_axis[1], x_axis[0]], device=device, dtype=img.dtype)  # Perpendicular vector
    scale = max(torch.norm(eye_to_eye) * 2.0, torch.norm(eye_to_mouth) * 1.8) * 0.7

    center = eye_center + eye_to_mouth * 0.1
    quad = torch.stack([
        center - x_axis * scale - y_axis * scale,
        center - x_axis * scale + y_axis * scale,
        center + x_axis * scale + y_axis * scale,
        center + x_axis * scale - y_axis * scale
    ])

    dst_quad = torch.tensor([
        [0, 0],
        [0, transform_size - 1],
        [transform_size - 1, transform_size - 1],
        [transform_size - 1, 0]
    ], dtype=torch.float32, device=device)

    matrix = kornia.geometry.transform.get_perspective_transform(quad.unsqueeze(0), dst_quad.unsqueeze(0))

    img_tensor = img.unsqueeze(0)
    transformed = kornia.geometry.transform.warp_perspective(img_tensor, matrix, dsize=(transform_size, transform_size))

    transformed_cropped = torch.nn.functional.interpolate(
        transformed, size=(output_size, output_size), mode='bilinear', align_corners=False
    ).squeeze(0)

    if padding:
        pad = int(scale * 0.1)
        max_pad = output_size // 4
        pad = min(pad, max_pad)
        transformed_cropped = torch.nn.functional.pad(
            transformed_cropped, (pad, pad, pad, pad), mode='reflect'
        )
        h, w = transformed_cropped.shape[1:3]
        transformed_cropped = transformed_cropped[:, pad:h - pad, pad:w - pad]

    return transformed_cropped.to(original_dtype)


def atribute_inference(atribute_model, images: torch.Tensor, task: str) -> torch.Tensor:
    """
    images: list of images (np.ndarray) with shape (H, W, C)
    fr_model: ONNX model for face recognition. Expects input shape (N, C, H, W). Images are normalized with mean=0.5, std=0.5 (with /255). In range [-1, 1].
    """
    atribute_model.eval()
    return atribute_model(images, task=task)