import operator
import os
import sys
import time
from http import HTTPStatus

import requests

from .ocr_engine import OCREngine


QUEUE_NAME = os.environ.get("QUEUE_NAME", "ocr")
WAIT_TIMEOUT = int(os.environ.get("WAIT_TIMEOUT", 30))
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "easyvizar.wings.cs.wisc.edu:5001")
MIN_RETRY_INTERVAL = 5

API_TOKEN = os.environ.get("VIZAR_API_TOKEN", "")
API_KEY = os.environ.get("VIZAR_API_KEY", "")


def build_headers(extra=None):
    headers = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    if extra:
        headers.update(extra)

    return headers


def get_queue_names():
    url = f"http://{VIZAR_SERVER}/photos/queues"
    try:
        response = requests.get(url, headers=build_headers())
        if response.ok and response.status_code == HTTPStatus.OK:
            items = response.json()
            queues = set(x["name"] for x in items)
            print("Available queues:", queues)
            return queues
        else:
            print("Queue request failed with status:", response.status_code)
            print("Queue response body:", response.text[:300])
    except requests.exceptions.RequestException as error:
        print("Queue fetch error:", error)

    return {QUEUE_NAME, "done"}


def get_next_queue(result, supported_queue_names):
    return "done"


def main():
    engine = OCREngine()
    engine.initialize_model()
    print("OCR worker started. Waiting for queue items...")
    print("Using server:", VIZAR_SERVER)
    print("Listening on queue:", QUEUE_NAME)

    while True:
        sys.stdout.flush()

        supported_queue_names = get_queue_names()

        query_url = f"http://{VIZAR_SERVER}/photos?queue_name={QUEUE_NAME}&wait={WAIT_TIMEOUT}"
        start_time = time.time()
        items = []

        try:
            response = requests.get(query_url, headers=build_headers())

            if response.status_code == HTTPStatus.NO_CONTENT:
                print("No items currently in queue", QUEUE_NAME)
                items = []

            elif response.ok and response.status_code == HTTPStatus.OK:
                items = response.json()
                print("Polled", len(items), "items from queue", QUEUE_NAME)

            else:
                print("Poll request failed with status:", response.status_code)
                print("Poll response body:", response.text[:300])

        except requests.exceptions.RequestException as error:
            print("Polling error:", error)

        if len(items) == 0:
            elapsed = time.time() - start_time
            if elapsed < MIN_RETRY_INTERVAL:
                time.sleep(MIN_RETRY_INTERVAL - elapsed)
            continue

        for item in items:
            item["priority_tuple"] = (-1 * item.get("priority", 0), item.get("created"))

        items.sort(key=operator.itemgetter("priority_tuple"))

        for item in items:
            print("Got item:", item.get("id"))

            try:
                result = engine.run(item)
            except Exception as error:
                print("Run error:", error)
                result = None

            url = f"http://{VIZAR_SERVER}/photos/{item['id']}"

            if result is not None:
                result.info["status"] = get_next_queue(result, supported_queue_names)

                try:
                    print("PATCH payload:", result.info)

                    patch_response = requests.patch(url, json=result.info, headers=build_headers())
                    print("Patched photo info with status:", patch_response.status_code)

                    if patch_response.status_code >= 400:
                        print("PATCH response body:", patch_response.text)
                        print("PATCH failed, stopping worker so item does not loop forever.")
                        return

                except requests.exceptions.RequestException as error:
                    print("Patch error:", error)
                    return

                try:
                    annotated_png, mask_png = result.apply_masks()
                    content_headers = build_headers({"Content-Type": "image/png"})

                    annotated_url = f"{url}/annotated.png"
                    annotated_response = requests.put(
                        annotated_url,
                        data=annotated_png,
                        headers=content_headers,
                    )
                    print("Uploaded annotated image with status:", annotated_response.status_code)

                    mask_url = f"{url}/mask.png"
                    mask_response = requests.put(
                        mask_url,
                        data=mask_png,
                        headers=content_headers,
                    )
                    print("Uploaded mask image with status:", mask_response.status_code)

                except requests.exceptions.RequestException as error:
                    print("Upload error:", error)
                except Exception as error:
                    print("Mask/apply error:", error)
            else:
                try:
                    error_response = requests.patch(
                        url,
                        json={"status": "error"},
                        headers=build_headers(),
                    )
                    print("Marked item as error with status:", error_response.status_code)
                except requests.exceptions.RequestException as error:
                    print("Error status patch failed:", error)


if __name__ == "__main__":
    main()