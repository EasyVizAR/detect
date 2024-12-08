import operator
import os
import sys
import io
import time
from http import HTTPStatus
import requests
import torch
import imageio
import numpy as np
from torchvision.ops import box_convert
from matplotlib import patches
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from GroundingDINO.groundingdino.util.inference import Model, predict
import groundingdino.datasets.transforms as T
import traceback
from segment_anything import SamPredictor, sam_model_registry
import scipy.ndimage

# Global configurations
QUEUE_NAME = os.environ.get("QUEUE_NAME", "detection") # The queue name to process
WAIT_TIMEOUT = os.environ.get("WAIT_TIMEOUT", 30) # The time to wait for new items
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "easyvizar.wings.cs.wisc.edu") # The VIZAR server to connect
DATA_PATH = os.environ.get("DATA_PATH", "./") # The path to save the annotated images locally
MIN_RETRY_INTERVAL = 5 # The minimum time to wait before retrying
MARK_ALL_OBJECTS = True # Whether to mark all objects in the image
DINO_CONFIG = os.environ.get("CONFIG_PATH", "detect/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_WEIGHTS = os.environ.get("WEIGHTS_PATH", "detect/GroundingDINO/groundingdino/weights/groundingdino_swint_ogc.pth")
SAM_PTH = os.environ.get("SAM_PTH", "detect/sam_vit_b_01ec64.pth")
TEXT_PROMPT = os.environ.get("TEXT_PROMPT", "doors . chairs . tables . ladders . desks") # The text prompt for DINO
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

def upload_results_to_server(url, result, annotated_png):
    """
    Uploads the result information and annotated image to the server.

    Parameters:
    - url (str): The URL to send the result information.
    - result_info (dict): The JSON payload with status and other info.
    - annotated_png (bytes): The annotated image in PNG format.
    """
    # Send result information
    response = requests.patch(url, json=result)
    if not response.ok:
        print(f"Failed to update status for {url}: {response.status_code}")
        print(response.text)

    # Upload the annotated image
    headers = {"Content-Type": "image/png"}
    annotated_url = f"{url}/annotated.png"
    response = requests.put(annotated_url, data=annotated_png, headers=headers)
    if response.ok:
        print(f"Annotated image uploaded to {annotated_url}")
        # print(response.text)
    else:
        print(f"Failed to upload annotated image: {response.status_code}")

def choose_source(item):
    """
    Choose the image source to process.

    Parameters:
    - item (dict): The item from the VIZAR server.

    Returns:
    - source (str): The source of the image to process.
    """
    path = item.get("imagePath")
    url = item.get("imageUrl")

    if path not in [None, ""]:
        full_path = os.path.join(DATA_PATH, path)
        if os.path.isfile(full_path):
            return full_path

    if url.startswith("http"):
        return url

    if url.startswith("/"):
        return f"http://{VIZAR_SERVER}{url}"

    raise Exception(f"Cannot load image path ({path}) or URL ({url})")

def transform_image(image):
    """
    Transform the image to the desired format.

    Parameters:
    - image (np.ndarray): The source image in RGB format.

    Returns:
    - image (np.ndarray): The source image in RGB format.
    - image_transformed (torch.Tensor): The transformed image for the model.
    """
    image_pil = Image.fromarray(image).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_pil, None)
    return image, image_transformed

def annotate_with_matplotlib(image_source: np.ndarray
                             ,boxes: torch.Tensor
                             ,masks
                             ,logits: torch.Tensor
                             ,phrases: List[str]
                             ,image_filename: str):
    """
    Annotate the image using Matplotlib and save it.

    Parameters:
    - image_source (np.ndarray): The source image in RGB format.
    - boxes (torch.Tensor): Bounding box coordinates.
    - masks (List[np.ndarray]): The segmentation masks in a list of (C, H, W) format.
    - logits (torch.Tensor): Confidence scores for each bounding box.
    - phrases (List[str]): Labels for each bounding box.
    - image_filename (str): The path to save the annotated image.

    Returns:
    - annotated_png (bytes): The annotated image in PNG format.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.tensor([w, h, w, h])
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    fig, ax = plt.subplots(1)
    fig.set_size_inches(w / 100, h / 100)
    ax.imshow(image_source)

    for box, mask, phrase, score in zip(boxes, masks, phrases, logits):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            width, height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            xmin, ymin - 5,
            f"{phrase} {score:.2f}",
            fontsize=10,
            color='red',
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
        )
        mask = np.transpose(mask, (1, 2, 0))  # CHW to HWC
        colored_mask = np.zeros((*mask.shape[:2], 4), dtype=np.float32)  # Initialize RGBA array
        colored_mask[..., :3] = [0, 0, 1] # Blue mask color
        colored_mask[..., 3] = mask[..., 0] * 0.35
        ax.imshow(colored_mask)

    ax.axis('off')

    # Save to buffer and return as PNG binary
    buffer = io.BytesIO()
    plt.savefig(image_filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    buffer.seek(0)  # Rewind the buffer for reading
    plt.close(fig)
    return buffer.getvalue()

def get_queue_names():
    '''
    Get the list of supported queue names from the server.
    
    Returns:
    - supported_queue_names (set): The set of supported queue names.
    '''
    url = f"http://{VIZAR_SERVER}/photos/queues"
    response = requests.get(url)
    if response.ok and response.status_code == HTTPStatus.OK:
        items = response.json()
        return set(x['name'] for x in items)
    else:
        return set([QUEUE_NAME, "done"])


def get_next_queue(item, supported_queue_names):
    '''
    Get the next queue name based on the item annotations.

    Parameters:
    - item (dict): The item from the VIZAR server.
    - supported_queue_names (set): The set of supported queue names.

    Returns:
    - queue_name (str): The next queue name to process.
    '''
    annotations = item.get('annotations', [])
    has_person = any(x['label'] == "person" for x in annotations)

    if has_person and "identification" in supported_queue_names:
        return "identification"
    elif len(annotations) > 0 and "detection-3d" in supported_queue_names:
        return "detection-3d"
    else:
        return "done"

def compute_contour(mask, image_height, image_width):
    """
    Computes normalized contour points from a binary mask.

    Parameters:
    - mask (np.ndarray): Binary mask (H, W) with values 0 or 1.
    - image_height (int): Height of the image.
    - image_width (int): Width of the image.

    Returns:
    - normalized_contour (list): List of [x, y] points normalized to [0, 1].
    """
    # Binary dilation to find edges
    dilated = scipy.ndimage.binary_dilation(mask)
    contour = dilated ^ mask  # XOR operation to get the edge

    # Extract contour points
    contour_points = np.argwhere(contour)  # (row, col) format

    # Normalize coordinates
    normalized_contour = [
        [col / image_width, row / image_height] for row, col in contour_points
    ]
    return normalized_contour


def main():
    model = Model(
        model_config_path=DINO_CONFIG,
        model_checkpoint_path=DINO_WEIGHTS,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    sam = sam_model_registry["vit_b"](checkpoint=SAM_PTH)
    predictor = SamPredictor(sam)
    output_dir = "./images" # The directory to save annotated images locally
    os.makedirs(output_dir, exist_ok=True)
    
    while True:
        sys.stdout.flush()
        supported_queue_names = get_queue_names()
        location_url = f"https://{VIZAR_SERVER}/locations/acf9cc39-a7a8-4ea4-bc10-8959cae35582"
        query_url = f"https://{VIZAR_SERVER}/photos?camera_location_id=acf9cc39-a7a8-4ea4-bc10-8959cae35582"
        start_time = time.time()
        items = []
        try:
            location_response = requests.get(location_url)
            if location_response.ok and location_response.status_code == HTTPStatus.OK:
                location_info = location_response.json()
            text_prompt = location_info['description']
            text_prompt = " . ".join(item.strip() for item in text_prompt.split(","))
            print(f"Using text prompt: {text_prompt}")
            response = requests.get(query_url)
            if response.ok and response.status_code == HTTPStatus.OK:
                items = response.json()
        except requests.exceptions.RequestException as error:
            print(error)

        if len(items) == 0:
            elapsed = time.time() - start_time
            if elapsed < MIN_RETRY_INTERVAL:
                time.sleep(MIN_RETRY_INTERVAL - elapsed)
            continue

        for item in items:
            item['priority_tuple'] = (-1 * item.get("priority", 0), item.get("created"))

        items.sort(key=operator.itemgetter("priority_tuple"))
        for item in items:
            try:
                source = choose_source(item)
                print(f"Processing image from {source}...")
                np_image = imageio.v3.imread(source)
                # Process the image with DINO
                image_source, image = transform_image(np_image)
                print(f"np_image type: {type(np_image)}, shape: {np_image.shape if isinstance(np_image, np.ndarray) else 'N/A'}")
                boxes, logits, phrases = predict(
                    image=image,
                    caption=text_prompt,
                    model=model.model,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device="cuda" if model.device == "cuda" else "cpu" # assume CUDA is available
                )
                image_filename = os.path.join(output_dir, f"cs_arc_lab{item['id']}.png")
                h, w, _ = image_source.shape
                boxes_xyxy = box_convert(boxes=boxes * torch.tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy()
                if boxes_xyxy.shape[0] == 0:
                    print("No objects detected.")
                else:
                    print(boxes_xyxy.shape)
                    predictor.set_image(np_image)
                    masks = []
                    contours = []
                    for box in boxes_xyxy:
                        print(f"BOX: {box}")
                        mask, _, _ = predictor.predict(box=box, multimask_output=False)
                        print(mask.shape)
                        masks.append(mask) # mask is in (1, H, W) format
                        # contour = compute_contour(mask[0], np_image.shape[0], np_image.shape[1])
                        # print(f"CONTOUR: {contour}")
                        # contours.append(contour)  # Add to the list of contours
                    print(f"MASKS: generated {len(masks)} masks")
                
                annotated_png = annotate_with_matplotlib(
                    image_source=image_source,
                    boxes=boxes,
                    masks=masks,
                    logits=logits,
                    phrases=phrases,
                    image_filename=image_filename
                )

                result = {
                    "status": get_next_queue(item, supported_queue_names),
                    "annotations": [
                        {
                            "boundary": {
                                "height": float(box[3] - box[1])/h,
                                "left": float(box[0])/w,
                                "top": float(box[1])/h,
                                "width": float(box[2] - box[0])/w
                            },
                            "confidence": float(score),
                            "contour": [],
                            "label": phrase
                        }
                        for box, phrase, score in zip(boxes_xyxy, phrases, logits)
                    ]
                }
                upload_url = f"https://{VIZAR_SERVER}/photos/{item['id']}" # IMPORTANT: upload url is different (with port 5000)
                # upload_results_to_server(upload_url, result, annotated_png)
            except Exception as error:
                print(f"Error processing item {item['id']}: {error}")
                traceback.print_exc()

if __name__ == "__main__":
    main()

# /locations/<location_id>