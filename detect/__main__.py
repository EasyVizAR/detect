import os

import requests
import torch


DATA_PATH = os.environ.get("DATA_PATH", "./")
WAIT_TIMEOUT = os.environ.get("WAIT_TIMEOUT", 30)
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "localhost:5000")

TIME_DELTA = 0.001 # add to the most recent timestamp to avoid re-fetching


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
                "width": shape[0],
                "height": shape[1],
                "annotations": annotations
            }
            return image_info

        except Exception as error:
            print(error)


def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    detector = Detector(model)

    url = "http://{}/photos?ready=1".format(VIZAR_SERVER)

    response = requests.get(url)
    data = response.json()

    last_timestamp = 0

    # Keep track of the most recently completed item IDs so that we do not
    # repeat work.
    recently_completed = set()

    for item in data:
        if item['updated'] >= last_timestamp:
            last_timestamp = item['updated']

        if item.get("annotations") in [None, []]:
            info = detector.run(item)

            if info is not None:
                url = "http://{}/photos/{}".format(VIZAR_SERVER, item['id'])
                requests.patch(url, json=info)

        recently_completed.add(item['id'])

    while True:
        url = "http://{}/photos?ready=1&since={}&wait={}".format(VIZAR_SERVER, last_timestamp, WAIT_TIMEOUT)
        print(url)
        response = requests.get(url)
        if not response.ok or response.status_code == 204:
            continue

        new_recently_completed = set()

        items = response.json()
        for item in items:
            if item['updated'] >= last_timestamp:
                last_timestamp = item['updated']

            if item['id'] in recently_completed:
                continue

            info = detector.run(item)
            if info is not None:
                url = "http://{}/photos/{}".format(VIZAR_SERVER, item['id'])
                requests.patch(url, json=info)

            recently_completed.add(item['id'])

        recently_completed = new_recently_completed


if __name__ == "__main__":
    main()
