import pytest
from pydantic import ValidationError
from src.model_service.main import KafkaMessage, PredictionRequest, MAX_TEXT_LENGTH


def test_valid_kafka_message():
    """Test that a valid message passes validation."""
    msg = KafkaMessage(text="Hello world", timestamp=1234567890.0)
    assert msg.text == "Hello world"
    assert msg.timestamp == 1234567890.0


def test_missing_text_field():
    """Test that missing 'text' field raises ValidationError."""
    with pytest.raises(ValidationError):
        KafkaMessage(timestamp=1234567890.0)


def test_missing_timestamp_field():
    """Test that missing 'timestamp' field raises ValidationError."""
    with pytest.raises(ValidationError):
        KafkaMessage(text="Hello world")


def test_empty_text():
    """Test that empty text is accepted (Pydantic allows empty strings)."""
    msg = KafkaMessage(text="", timestamp=1234567890.0)
    assert msg.text == ""


def test_invalid_timestamp_type():
    """Test that non-numeric timestamp raises ValidationError."""
    with pytest.raises(ValidationError):
        KafkaMessage(text="Hello", timestamp="not-a-number")


def test_extra_fields_ignored():
    """Test that extra fields in the message don't cause errors."""
    msg = KafkaMessage(text="Hello", timestamp=1234567890.0, extra_field="ignored")
    assert msg.text == "Hello"


# --- PredictionRequest validation tests ---

def test_prediction_request_valid():
    """Test valid prediction request."""
    req = PredictionRequest(text="I love this product!")
    assert req.text == "I love this product!"


def test_prediction_request_empty_text():
    """Test that empty/whitespace text is rejected."""
    with pytest.raises(ValidationError):
        PredictionRequest(text="   ")


def test_prediction_request_too_long():
    """Test that text exceeding max length is rejected."""
    with pytest.raises(ValidationError):
        PredictionRequest(text="a" * (MAX_TEXT_LENGTH + 1))


def test_prediction_request_at_max_length():
    """Test that text at exactly max length is accepted."""
    req = PredictionRequest(text="a" * MAX_TEXT_LENGTH)
    assert len(req.text) == MAX_TEXT_LENGTH
