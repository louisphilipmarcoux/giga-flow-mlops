import pytest
from pydantic import ValidationError
from src.model_service.main import KafkaMessage


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
