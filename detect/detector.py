import ast
import collections
import io
import math
import os
import time

from distutils.version import LooseVersion

import imageio
import numpy
import onnxruntime

from scipy.spatial.transform import Rotation
from skimage.color import hsv2rgb


DATA_PATH = os.environ.get("DATA_PATH", "./")
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "localhost:5000")

PROVIDER_PRIORITY_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Newer versions support writting PNG directly to a memory buffer.
SUPPORT_IO_BUFFER = LooseVersion(imageio.__version__) >= "2.31"


def encode_png(data):
    if SUPPORT_IO_BUFFER:
        buffer = io.BytesIO()
        imageio.v3.imwrite(buffer, data, extension=".png")
        return buffer.getvalue()
    else:
        imageio.v3.imwrite("/tmp/image.png", data, extension=".png")
        return open("/tmp/image.png", "rb").read()


def generate_palette():
    num_classes = 80
    values = numpy.linspace(0, 1, num_classes)

    palette = numpy.zeros((num_classes, 3), dtype=numpy.uint8)
    for i in range(num_classes):
        # Older version of skimage requires argument to hsv2rgb to look like an
        # image, hence the extra brackets.
        rgb = hsv2rgb([[(values[i], 1, 1)]]).squeeze()
        palette[i, :] = 255.0 * rgb

    return palette


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


class DetectionResult:
    palette = generate_palette()

    def __init__(self, info, image, outputs, remote_info):
        self.info = info
        self.image = image
        self.outputs = outputs
        self.remote_info = remote_info

    def apply_masks(self):
        boxes = self.outputs[0]
        masks = self.outputs[1]
        nms = self.outputs[2]

        num_classes = 80

        num_masks = masks.shape[1]
        #height = masks.shape[2]
        #width = masks.shape[3]
        height = self.image.shape[0] // 4
        width = self.image.shape[1] // 4

        mask_image = numpy.zeros((height, width, 3), dtype=numpy.uint8)
        alpha = numpy.zeros((height, width))
        for i in range(nms.shape[0]):
            cid = nms[i, 1].item()
            bi = nms[i, 2].item()

            weights = boxes[0, 4+num_classes:, bi].squeeze()

            cx = boxes[0, 0, bi].item()
            cy = boxes[0, 1, bi].item()
            w = boxes[0, 2, bi].item()
            h = boxes[0, 3, bi].item()

            minx = int(max(0, cx-w/2) / 4)
            maxx = int(min(self.image.shape[1], cx+w/2) / 4)
            miny = int(max(0, cy-h/2) / 4)
            maxy = int(min(self.image.shape[0], cy+h/2) / 4)

            for y in range(miny, maxy):
                for x in range(minx, maxx):
                    score = sigmoid(numpy.dot(weights, masks[0, :, y, x]))
                    if score > alpha[y, x]:
                        mask_image[y, x, 0:3] = self.palette[cid]
                        alpha[y, x] = score

        # This is just the mask image with alpha channel added so that it is
        # transparent where no objects were detected.
        alpha_uint8 = (alpha * 255).astype(numpy.uint8)
        mask = numpy.dstack((mask_image, alpha_uint8))
        mask_png = encode_png(mask)

        # Use repeat to grow mask back to original image size
        mask_image = mask_image.repeat(4, axis=0).repeat(4, axis=1)
        alpha = alpha.repeat(4, axis=0).repeat(4, axis=1)
        inv_alpha = 1.0 - alpha

        combined = alpha[:,:,None] * mask_image + inv_alpha[:,:,None] * self.image[:,:,0:3]
        combined = combined.astype(numpy.uint8)

        annotated_png = encode_png(combined)

        return annotated_png, mask_png

    def try_localize_objects(self, source):
        # Geometry image is originally 16-bit RGBA, with values in millimeters.
        # The loader truncates to 8-bit, which is equivalent to dividing by 256.
        # Multiply by 256 and divide by 1000 to convert to meters.
        # Zero values in the geometry image represent invalid readings and
        # are replaced with NaN to avoid including in averages.
        try:
            geometry_image = imageio.v3.imread(source)
            geometry = (geometry_image.astype(float) - 128.0) * 256.0 / 1000.0
            geometry[geometry_image == 0] = numpy.nan
        except:
            return False

        height, width, channels = geometry.shape

        boxes = self.outputs[0]
        masks = self.outputs[1]
        nms = self.outputs[2]

        num_classes = 80

        camera_position = self.remote_info.get("camera_position")
        camera_orientation = self.remote_info.get("camera_orientation")
        if camera_position is None or camera_orientation is None:
            return False

        cam_pos = [
            camera_position.get("x", 0),
            camera_position.get("y", 0),
            camera_position.get("z", 0)
        ]
        cam_quat = [
            camera_orientation.get("x", 0),
            camera_orientation.get("y", 0),
            camera_orientation.get("z", 0),
            camera_orientation.get("w", 1)
        ]
        cam_rot = Rotation.from_quat(cam_quat)

        num_masks = masks.shape[1]
        #height = masks.shape[2]
        #width = masks.shape[3]
        height = self.image.shape[0] // 4
        width = self.image.shape[1] // 4

        for i in range(nms.shape[0]):
            cid = nms[i, 1].item()
            bi = nms[i, 2].item()

            weights = boxes[0, 4+num_classes:, bi].squeeze()

            cx = boxes[0, 0, bi].item()
            cy = boxes[0, 1, bi].item()
            w = boxes[0, 2, bi].item()
            h = boxes[0, 3, bi].item()

            minx = int(max(0, cx-w/2) / 4)
            maxx = int(min(self.image.shape[1], cx+w/2) / 4)
            miny = int(max(0, cy-h/2) / 4)
            maxy = int(min(self.image.shape[0], cy+h/2) / 4)

            points = []
            pos_weights = []
            for y in range(miny, maxy):
                for x in range(minx, maxx):
                    score = sigmoid(numpy.dot(weights, masks[0, :, y, x]))

                    xyz = numpy.nanmean(geometry[4*y:4*y+4, 4*x:4*x+4, 0:3], axis=(0,1))
                    if any(numpy.isnan(xyz)):
                        continue

                    pos_weights.append(score)
                    points.append(xyz)

            if len(points) == 0 or sum(pos_weights) < 0.5:
                continue

            position = numpy.average(points, axis=0, weights=pos_weights)

            squared_distances = numpy.sum((points - position) ** 2, axis=1)
            spread = math.sqrt(numpy.average(squared_distances, weights=pos_weights))

            # Apply camera rotation and add to camera position
            # to find object position in world coordinates.
            pos = cam_pos + cam_rot.apply(position)

            print("Detected {} at position {} spread {}".format(self.info['annotations'][i]['label'], pos, spread))

            self.info['annotations'][i]['position'] = {
                "x": pos[0].item(),
                "y": pos[1].item(),
                "z": pos[2].item(),
            }
            self.info['annotations'][i]['position_error'] = spread

        return True


class Detector:
    def __init__(self, model_repo, model_name):
        self.model_repo = model_repo
        self.model_name = model_name
        self.model = None

        self.session = None
        self.names = None
        self.input_shape = None

    def choose_source(self, item):
        path = item.get("imagePath")
        url = item.get("imageUrl")

        if path not in [None, ""]:
            full_path = os.path.join(DATA_PATH, path)
            if os.path.isfile(full_path):
                return full_path

        if url.startswith("http"):
            return url

        if url.startswith("/"):
            return "http://" + VIZAR_SERVER + url

        raise Exception("Cannot load image path ({}) or URL ({})".format(path, url))

    def initialize_model(self):
        model_file = "{}.onnx".format(self.model_name)
        model_path = os.path.join(MODEL_DIR, model_file)

        self.session = onnxruntime.InferenceSession(model_path, providers=PROVIDER_PRIORITY_LIST)

        meta = self.session.get_modelmeta()
        names = meta.custom_metadata_map.get("names", "{}")
        self.names = ast.literal_eval(names)

        self.input_shape = self.session.get_inputs()[0].shape

    def preprocess(self, image):
        height, width, channels = image.shape
        _, exp_channels, exp_height, exp_width = self.input_shape

        # Drop the alpha channel if present
        if channels > exp_channels:
            image = image[:, :, 0:exp_channels]

        # If the image is smaller than expected by the model, we can pad zeroes
        # at the bottom and right of the image.
        pad_bottom = 0
        pad_right = 0
        if height < exp_height:
            pad_bottom = exp_height - height
        if width < exp_width:
            pad_right = exp_width - width
        if pad_bottom > 0 or pad_right > 0:
            pad_width = ((0, pad_bottom), (0, pad_right), (0, 0))
            image = numpy.pad(image, pad_width, "constant")

        image = image.astype(numpy.float32) / 255.0
        image = numpy.transpose(image, axes=[2, 0, 1])
        image = numpy.expand_dims(image, axis=0)

        return image

    def postprocess(self, output, image_shape):
        boxes = output[0]
        nms = output[2].astype(int)

        num_candidates = boxes.shape[2]
        num_classes = 80
        num_masks = 32

        image_height = image_shape[0]
        image_width = image_shape[1]

        annotations = []
        for i in range(nms.shape[0]):
            cid = nms[i, 1].item()
            bi = nms[i, 2].item()

            cx = boxes[0, 0, bi].item()
            cy = boxes[0, 1, bi].item()
            w = boxes[0, 2, bi].item()
            h = boxes[0, 3, bi].item()

            score = boxes[0, 4+cid, bi].item()

            obj = {
                "boundary": {
                    "left": (cx - w/2) / image_width,
                    "top": (cy - h/2) / image_height,
                    "width": w/image_width,
                    "height": h/image_height
                },
                "confidence": score,
                "label": self.names.get(cid, "unknown-{}".format(cid))
            }
            annotations.append(obj)

        return annotations

    def run(self, item):
        detector_info = {
            "model_repo": self.model_repo,
            "model_name": self.model_name,
        }

        try:
            if self.session is None:
                self.initialize_model()

            source = self.choose_source(item)
            print("Processing image from {}...".format(source))

            image = imageio.v3.imread(source)
            image_shape = image.shape

            preprocess_start = time.time()
            processed = self.preprocess(image)

            inference_start = time.time()
            output = self.session.run(None, {"images": processed})

            postprocess_start = time.time()
            annotations = self.postprocess(output, image_shape)
            postprocess_end = time.time()

            detector_info = {
                "model_repo": self.model_repo,
                "model_name": self.model_name,
                "engine_name": onnxruntime.__name__,
                "engine_version": onnxruntime.__version__,
                "torchvision_version": "",
                "torch_version": "",
                "cuda_enabled": self.session.get_providers()[0].startswith("CUDA"),
                "preprocess_duration": inference_start - preprocess_start,
                "inference_duration": postprocess_start - inference_start,
                "nms_duration": 0,
                "postprocess_duration": postprocess_end - postprocess_start,
            }

            image_info = {
                "status": "done",
                "annotations": annotations,
                "detector": detector_info
            }

            return DetectionResult(image_info, image, output, item)

        except Exception as error:
            print(error)
