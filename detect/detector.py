import os

import torch
import torchvision


DATA_PATH = os.environ.get("DATA_PATH", "./")
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "localhost:5000")


class Detector:
    def __init__(self, model_repo, model_name):
        self.model_repo = model_repo
        self.model_name = model_name
        self.model = None

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
        self.model = torch.hub.load(self.model_repo, self.model_name)

    def run(self, item):
        detector_info = {
            "model_repo": self.model_repo,
            "model_name": self.model_name,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_enabled": torch.cuda.is_available(),
        }

        try:
            source = self.choose_source(item)

            print("Processing image from {}...".format(source))
            results = self.model(source)
            results.print()

            output = results.xywhn[0]
            rows, cols = output.shape

            annotations = []
            for i in range(rows):
                # xywhn: x_center, y_center, width, height, confidence, class
                obj = {
                    "boundary": {
                        "left": output[i, 0].item() - 0.5 * output[i, 2].item(),
                        "top": output[i, 1].item() - 0.5 * output[i, 3].item(),
                        "width": output[i, 2].item(),
                        "height": output[i, 3].item()
                    },
                    "confidence": output[i, 4].item(),
                    "label": results.names[int(output[i, 5].item())]
                }
                annotations.append(obj)

            # yolov5 library changed from representing timing information as a
            # list of four times to currently a list of three Profile objects
            # with an attribute storing the elapsed time
            if len(results.times) == 3:
                detector_info['preprocess_duration'] = results.times[0].t
                detector_info['inference_duration'] = results.times[1].t
                detector_info['nms_duration'] = results.times[2].t
            elif len(results.times) == 4:
                detector_info['preprocess_duration'] = results.times[1] - results.times[0]
                detector_info['inference_duration'] = results.times[2] - results.times[1]
                detector_info['nms_duration'] = results.times[3] - results.times[2]

            image_info = {
                "status": "done",
                "annotations": annotations,
                "detector": detector_info
            }
            return image_info

        except Exception as error:
            print(error)

