from unittest.mock import MagicMock

from detect.detector import Detector


def test_detector_choose_source():
    detector = Detector("test", "test")
    detector.model = MagicMock()

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
    detector = Detector("test", "test")
    detector.model = MagicMock()

    results = MagicMock()
    results.xywhn = (MagicMock(), )
    results.xywhn[0].shape = (0, 0)
    detector.model.return_value = results

    item = {
        "imageUrl": "http://example.org/example.jpg"
    }

    info = detector.run(item)
    assert info['status'] == "done"
