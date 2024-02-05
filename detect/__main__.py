import collections
import math
import operator
import os
import sys
import time

from http import HTTPStatus

import requests

from .detector import Detector


QUEUE_NAME = os.environ.get("QUEUE_NAME", "detection")
WAIT_TIMEOUT = os.environ.get("WAIT_TIMEOUT", 30)
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "localhost:5000")
MIN_RETRY_INTERVAL = 5

MODEL_REPO = "yolov8"
MODEL_NAME = "yolov8m-seg-nms"

MARK_ALL_OBJECTS = True
MARK_LABELS = set(["door", "dining table", "desk", "table"])

# Rename some of the labels from the detector before marking them as map features.
LABELS_TO_FEATURE_NAMES = {
#    "dining table": "table",
#    "desk": "table"
}


def repair_images():
    """
    Repair images on the server that have an invalid status.
    """
    query_url = "http://{}/photos?status=error".format(VIZAR_SERVER)
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

        time.sleep(0.1)


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
        label = obj['label']
        if not MARK_ALL_OBJECTS and label not in MARK_LABELS:
            continue
        if label in LABELS_TO_FEATURE_NAMES:
            label = LABELS_TO_FEATURE_NAMES[label]

        pos = obj.get("position")
        if pos is None:
            continue

        pos_error = obj.get("position_error", 10.0)

        # Check if there is any already existing feature within a certain
        # radius.  We use the combined position_error values for the two
        # objects, which give a rough estimate of how wide they are. If they
        # are too close, avoid creating another feature.
        duplicate = False
        for other in features_by_label[label]:
            sq_dist = sum( (pos[d] - other['position'][d])**2 for d in ["x", "y", "z"] )
            dist = math.sqrt(sq_dist)

            # Other point's radius may not be set, which means we do not
            # know the position_error value for that other feature.
            # Just use the new object's position_error twice, then.
            other_radius = other.get("radius")
            if other_radius is None:
                other_radius = pos_error

            threshold = pos_error + other_radius

            if dist < threshold:
                duplicate = True
                break

        if duplicate:
            continue

        # Create a new feature on the map.  We are abusing the radius field
        # here to store the position error / spread.  The radius attribute was
        # meant to control when the feature should be displayed in AR, only
        # when the user is within a certain radius of the feature position.
        # However, it is not really used.
        new_feature = {
            "name": label,
            "position": pos,
            "style": {
                "placement": "point",
                "radius": pos_error
            },
            "type": "object"
        }
        response = requests.post(features_url, json=new_feature)
        if response.ok:
            new_feature = response.json()
            features_by_label[label].append(new_feature)


def main():
    detector = Detector(MODEL_REPO, MODEL_NAME)
    detector.initialize_model()

    repair_images()

    while True:
        sys.stdout.flush()

        query_url = "http://{}/photos?queue_name={}&wait={}".format(VIZAR_SERVER, QUEUE_NAME, WAIT_TIMEOUT)
        start_time = time.time()
        response = requests.get(query_url)

        items = []
        if response.ok and response.status_code == HTTPStatus.OK:
            items = response.json()

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
