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
from PIL import Image
from GroundingDINO.groundingdino.util.inference import Model, predict
import groundingdino.datasets.transforms as T
import traceback
from segment_anything import SamPredictor, sam_model_registry

# Environment variables
QUEUE_NAME = os.environ.get("QUEUE_NAME", "detection")
WAIT_TIMEOUT = os.environ.get("WAIT_TIMEOUT", 30)
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "easyvizar.wings.cs.wisc.edu") # The VIZAR server to connect
DATA_PATH = os.environ.get("DATA_PATH", "./") # The path to save the annotated images locally
MIN_RETRY_INTERVAL = 5 # The minimum time to wait before retrying
# MARK_ALL_OBJECTS = True
DINO_CONFIG = os.environ.get("CONFIG_PATH", "detect/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_WEIGHTS = os.environ.get("WEIGHTS_PATH", "detect/GroundingDINO/groundingdino/weights/groundingdino_swint_ogc.pth")
SAM_PTH = os.environ.get("SAM_PTH", "detect/sam_vit_b_01ec64.pth")
BOX_THRESHOLD = 0.45
TEXT_THRESHOLD = 0.25
COLOR_MAP = {
    "door": [119/255, 170/255, 221/255],  # #77AADD
    "chair": [238/255, 136/255, 102/255], # #EE8866
    "table": [238/255, 221/255, 136/255], # #EEDD88
    "ladder": [255/255, 170/255, 187/255] # #FFAABB
} # https://cran.r-project.org/web/packages/khroma/vignettes/tol.html palette 2.7


def upload_results_to_server(url, result, annotated_png, mask_png):
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

    # Check if this frame has any detections
    if annotated_png is None or mask_png is None:
        return
    
    # Upload the annotated image
    headers = {"Content-Type": "image/png"}
    annotated_url = f"{url}/annotated.png"
    response = requests.put(annotated_url, data=annotated_png, headers=headers)
    headers = {"Content-Type": "image/png"}
    mask_url = f"{url}/mask.png"
    response = requests.put(mask_url, data=mask_png, headers=headers)
    if response.ok:
        print(f"Annotated image uploaded to {annotated_url}")
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
        return f"https://{VIZAR_SERVER}{url}"

    raise Exception(f"Cannot load image path ({path}) or URL ({url})")


def transform_image(image):
    """
    Transform the image before grounding dino processing.

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


def annotate_with_numpy(image_source: np.ndarray, 
                        boxes: torch.Tensor, 
                        masks: list, 
                        phrases: list):
    """
    Annotate the image using NumPy arrays and save both the annotated image and mask.

    Parameters:
    - image_source (np.ndarray): The source image in RGB format.
    - boxes (torch.Tensor): Bounding box coordinates.
    - masks (List[np.ndarray]): The segmentation masks in a list of (C, H, W) format.
    - phrases (List[str]): Labels for each bounding box.

    Returns:
    - annotated_png (bytes): The annotated image in PNG format.
    - mask_png (bytes): The colored mask image in PNG format.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.tensor([w, h, w, h])
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    overlay = np.zeros((h, w, 4), dtype=np.float32)  # RGBA
    mask_overlay = np.zeros((h, w, 4), dtype=np.float32)  # Mask-specific overlay (colored mask)

    for mask, phrase in zip(masks, phrases):
        color = COLOR_MAP.get(phrase, None)
        if color is None: 
            continue

        mask = np.transpose(mask, (1, 2, 0))  # CHW to HWC
        # First channel contains the most confident mask
        mask = mask[..., 0] if mask.shape[-1] > 0 else np.zeros((h, w), dtype=np.float32)
        
        for i in range(3):  # Apply color to both overlays
            overlay[..., i] = np.where(mask > 0, color[i], overlay[..., i])
            mask_overlay[..., i] = np.where(mask > 0, color[i], mask_overlay[..., i])

        overlay[..., 3] = np.where(mask > 0, 1, overlay[..., 3])  # Set alpha where mask is present
        mask_overlay[..., 3] = np.where(mask > 0, 1, mask_overlay[..., 3])  # Same for mask

    # Blend the overlay with the original image
    annotated_image = image_source.astype(np.float32) / 255
    alpha = overlay[..., 3:4]  # Ensure alpha shape is (h, w, 1) for broadcasting
    annotated_image = annotated_image * (1 - alpha) + overlay[..., :3] * alpha
    annotated_image = (annotated_image * 255).astype(np.uint8)

    # Convert mask overlay to an image
    mask_image = (mask_overlay * 255).astype(np.uint8)  # Convert float to uint8

    # Save the annotated image as PNG
    annotated_buffer = io.BytesIO()
    Image.fromarray(annotated_image).save(annotated_buffer, format="PNG")
    annotated_buffer.seek(0)

    # Save the mask as PNG
    mask_buffer = io.BytesIO()
    Image.fromarray(mask_image).save(mask_buffer, format="PNG")  # Keep colors and transparency
    mask_buffer.seek(0)

    return annotated_buffer.getvalue(), mask_buffer.getvalue()


def get_queue_names():
    '''
    Get the list of supported queue names from the server.
    
    Returns:
    - supported_queue_names (set): The set of supported queue names.
    '''
    url = f"https://{VIZAR_SERVER}/photos/queues"
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

def main():
    # Initialize the gdino model and sam predictor
    gdino_model = Model(
        model_config_path=DINO_CONFIG,
        model_checkpoint_path=DINO_WEIGHTS,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    sam = sam_model_registry["vit_b"](checkpoint=SAM_PTH)
    sam_predictor = SamPredictor(sam)
    
    while True:
        sys.stdout.flush()

        # Set of photo queues supported by the server
        supported_queue_names = get_queue_names()

        query_url = "http://{}/photos?queue_name={}&wait={}".format(VIZAR_SERVER, QUEUE_NAME, WAIT_TIMEOUT)
        start_time = time.time()

        items = []

        try:
            response = requests.get(query_url)
            if response.ok and response.status_code == HTTPStatus.OK:
                items = response.json()
        except requests.exceptions.RequestException as error:
            print(error)

        # Check if the empty/error response from the server was sooner than
        # expected.  If so, add an extra delay to avoid spamming the server.
        # We need this in case long-polling is not working as expected.
        if len(items) == 0:
            elapsed = time.time() - start_time
            if elapsed < MIN_RETRY_INTERVAL:
                time.sleep(MIN_RETRY_INTERVAL - elapsed)
            continue

        for item in items:
            # Sort by priority level (descending), then creation time (ascending)
            item['priority_tuple'] = (-1 * item.get("priority", 0), item.get("created"))

        items.sort(key=operator.itemgetter("priority_tuple"))

        # After sorting, generate a text prompt list for each item since 
        # they can be from different locations
        text_prompt = []
        for item in items:
            if 'location' in item:
                formatted_text_prompt = " . ".join(part.strip() 
                                                   for part in item['location']['description']
                                                   .split(","))
                text_prompt.append(formatted_text_prompt) # e.g. "door . chair . table"
            else:
                raise Exception("Item does not have a location \
                                or location description does not have anything.")
        for i, item in enumerate(items):
            try:
                source = choose_source(item)
                np_image = imageio.v3.imread(source)

                # Process the image with DINO
                image_source, image = transform_image(np_image)
                boxes, logits, phrases = predict(
                    image=image,
                    caption=text_prompt[i],
                    model=gdino_model.model,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device="cuda" if gdino_model.device == "cuda" else "cpu"
                )
                if boxes.shape[0] == 0:
                    print("No objects detected.")
                    result = {
                        "status": get_next_queue(item, supported_queue_names),
                        "annotations": []
                    }
                    annotated_png = None
                    mask_png = None
                else:
                    h, w, _ = image_source.shape
                    boxes_xyxy = box_convert(boxes=boxes * torch.tensor([w, h, w, h])
                                            ,in_fmt="cxcywh"
                                            ,out_fmt="xyxy").numpy()
                    # Process the image with SAM
                    sam_predictor.set_image(np_image)
                    masks = []
                    for box in boxes_xyxy:
                        mask, _, _ = sam_predictor.predict(box=box, multimask_output=False)
                        masks.append(mask) # mask here is in (1, H, W) format
                    
                    annotated_png, mask_png = annotate_with_numpy(
                        image_source=image_source,
                        boxes=boxes,
                        masks=masks,
                        phrases=phrases
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
                                "label": phrase
                            }
                            for box, phrase, score in zip(boxes_xyxy, phrases, logits)
                        ]
                    }
                upload_url = f"https://{VIZAR_SERVER}/photos/{item['id']}"
                upload_results_to_server(upload_url, result, annotated_png, mask_png)
            except Exception as error:
                print(f"Error processing item {item['id']}: {error}")
                traceback.print_exc()

if __name__ == "__main__":
    main()