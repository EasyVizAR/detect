import operator
import os
import sys

import requests

from .detector import Detector


DATA_PATH = os.environ.get("DATA_PATH", "./")
WAIT_TIMEOUT = os.environ.get("WAIT_TIMEOUT", 30)
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "localhost:5000")

# Pretrained models from smallest (fast) to largest (accurate):
# yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
# Consider detecting the CPU/GPU available and choose based on that.
MODEL_REPO = "ultralytics/yolov5"
MODEL_NAME = "yolov5n"


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
    detector = Detector(MODEL_REPO, MODEL_NAME)
    detector.initialize_model()

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

            url = "http://{}/photos/{}".format(VIZAR_SERVER, item['id'])
            if info is not None:
                requests.patch(url, json=info)
            else:
                info = {"status": "error"}
                requests.patch(url, json=info)


if __name__ == "__main__":
    main()
