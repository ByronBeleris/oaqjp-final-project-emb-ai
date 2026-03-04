import json
import unittest
from unittest.mock import patch

from EmotionDetection import emotion_detector


MOCK_EMOTION_SCORES = {
    "I am glad this happened": {
        "anger": 0.01,
        "disgust": 0.01,
        "fear": 0.02,
        "joy": 0.92,
        "sadness": 0.04,
    },
    "I am really mad about this": {
        "anger": 0.93,
        "disgust": 0.02,
        "fear": 0.02,
        "joy": 0.01,
        "sadness": 0.02,
    },
    "I feel disgusted just hearing about this": {
        "anger": 0.04,
        "disgust": 0.9,
        "fear": 0.02,
        "joy": 0.01,
        "sadness": 0.03,
    },
    "I am so sad about this": {
        "anger": 0.02,
        "disgust": 0.02,
        "fear": 0.03,
        "joy": 0.01,
        "sadness": 0.92,
    },
    "I am really afraid that this will happen": {
        "anger": 0.03,
        "disgust": 0.02,
        "fear": 0.9,
        "joy": 0.01,
        "sadness": 0.04,
    },
}


class MockResponse:
    def __init__(self, text):
        self.text = text


def mock_post(*args, **kwargs):
    statement = kwargs["json"]["raw_document"]["text"]
    emotions = MOCK_EMOTION_SCORES[statement]
    response_payload = {"emotionPredictions": [{"emotion": emotions}]}
    return MockResponse(json.dumps(response_payload))


class TestEmotionDetection(unittest.TestCase):
    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_emotions(self, mocked_post):
        mocked_post.side_effect = mock_post
        expected = {
            "I am glad this happened": "joy",
            "I am really mad about this": "anger",
            "I feel disgusted just hearing about this": "disgust",
            "I am so sad about this": "sadness",
            "I am really afraid that this will happen": "fear",
        }

        for statement, dominant_emotion in expected.items():
            with self.subTest(statement=statement):
                response = emotion_detector(statement)
                self.assertEqual(response["dominant_emotion"], dominant_emotion)


if __name__ == "__main__":
    unittest.main()
