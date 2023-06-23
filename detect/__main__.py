import collections
import math
import operator
import os
import sys

import requests

from .detector import Detector


WAIT_TIMEOUT = os.environ.get("WAIT_TIMEOUT", 30)
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "localhost:5000")

MODEL_REPO = "custom"
MODEL_NAME = "yolov8n-seg-c04-nms"

MARK_LABELS = set(["door", "dining table"])
MIN_DISTANCE = 1


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


def try_create_features(location_id, item, info):
    # Get current list of features for the location
    features_url = "http://{}/locations/{}/features".format(VIZAR_SERVER, location_id)
    response = requests.get(features_url)
    if not response.ok:
        return

    features = response.json()
    features_by_label = collections.defaultdict(list)

    # Organize the existing features by name.
    # We are only interested in objects of the configured types.
    for feature in features:
        if feature.get("type") == "object" and feature.get("name") in MARK_LABELS:
            features_by_label[feature['name']].append(feature)

    for obj in info.get("annotations", []):
        if obj['label'] not in MARK_LABELS:
            continue

        pos = obj.get("position")
        if pos is None:
            continue

        # Check if there is any already existing feature within configured minimum radius.
        # If so, we will not create another feature.
        duplicate = False
        for other in features_by_label[obj['label']]:
            sq_dist = sum( (pos[d] - other['position'][d])**2 for d in ["x", "y", "z"] )
            if math.sqrt(sq_dist) < MIN_DISTANCE:
                duplicate = True
                break

        if duplicate:
            continue

        # Create a new feature on the map
        new_feature = {
            "name": obj['label'],
            "position": pos,
            "style": {
                "placement": "point"
            },
            "type": "object"
        }
        response = requests.post(features_url, json=new_feature)
        if response.ok:
            new_feature = response.json()
            features_by_label[obj['label']].append(new_feature)


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
            result = detector.run(item)

            url = "http://{}/photos/{}".format(VIZAR_SERVER, item['id'])
            if result is not None:
                requests.patch(url, json=result.info)

                data = result.apply_masks()
                headers = {
                    "Content-Type": "image/png"
                }
                annotated_url = "{}/annotated.png".format(url)
                req = requests.put(annotated_url, data=data, headers=headers)

                geom_url = "{}/geometry.png".format(url)
                if result.try_localize_objects(geom_url):
                    requests.patch(url, json=result.info)

                    camera_location_id = item.get("camera_location_id")
                    if camera_location_id is not None:
                        try_create_features(camera_location_id, item, result.info)

            else:
                info = {"status": "error"}
                requests.patch(url, json=info)


if __name__ == "__main__":
    main()
