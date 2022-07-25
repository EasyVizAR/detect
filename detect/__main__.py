import operator
import os
import sys

import requests
import torch


DATA_PATH = os.environ.get("DATA_PATH", "./")
WAIT_TIMEOUT = os.environ.get("WAIT_TIMEOUT", 30)
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "localhost:5000")


class Detector:
    def __init__(self, model):
        self.model = model

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

    def run(self, item):
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

            shape = results.imgs[0].shape
            image_info = {
                "status": "done",
                "width": shape[0],
                "height": shape[1],
                "annotations": annotations
            }
            return image_info

        except Exception as error:
            print(error)


def repair_images():
    """
    Repair images on the server that have an invalid status.
    """
    query_url = "http://{}/photos?status=unknown".format(VIZAR_SERVER)
    response = requests.get(query_url)
    data = response.json()

    for item in data:
        status = "created"
        if item.get("ready", False):
            status = "ready"

        annotations = item.get("annotations")
        if annotations is not None and len(annotations) > 0:
            status = "done"

        change = {"status": status}
        url = "http://{}/photos/{}".format(VIZAR_SERVER, item['id'])
        requests.patch(url, json=change)


def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    detector = Detector(model)

    repair_images()

    while True:
        sys.stdout.flush()

        query_url = "http://{}/photos?status=ready&wait={}".format(VIZAR_SERVER, WAIT_TIMEOUT)
        response = requests.get(query_url)
        if not response.ok or response.status_code == 204:
            continue

        items = response.json()
        for item in items:
            # Sort by priority level (descending), then creation time (ascending)
            item['priority_tuple'] = (-1 * item.get("priority", 0), item.get("created"))

        items.sort(key=operator.itemgetter("priority_tuple"))
        for item in items:
            info = detector.run(item)
            if info is not None:
                url = "http://{}/photos/{}".format(VIZAR_SERVER, item['id'])
                requests.patch(url, json=info)


if __name__ == "__main__":
    main()
