from unittest.mock import MagicMock

from detect.detector import Detector


def test_detector_choose_source():
    model = MagicMock()
    detector = Detector(model)

    item = {
        "imagePath": __file__
    }
    source = detector.choose_source(item)
    assert source == item['imagePath']

    item = {
        "imageUrl": "http://example.org/example.jpg"
    }
    source = detector.choose_source(item)
    assert source == item['imageUrl']

    item = {
        "imageUrl": "/example.jpg"
    }
    source = detector.choose_source(item)
    assert source.startswith("http")


def test_detector_run():
    model = MagicMock()
    detector = Detector(model)

    results = MagicMock()
    results.xywhn = (MagicMock(), )
    results.xywhn[0].shape = (0, 0)
    model.return_value = results

    item = {
        "imageUrl": "http://example.org/example.jpg"
    }

    info = detector.run(item)
    assert info['status'] == "done"
