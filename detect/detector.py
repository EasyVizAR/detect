import os


DATA_PATH = os.environ.get("DATA_PATH", "./")
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

