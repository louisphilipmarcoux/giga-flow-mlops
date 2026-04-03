import sys
from unittest.mock import MagicMock, patch


def test_message_format():
    """Verify that produced messages have the expected schema."""
    # Patch KafkaProducer before importing the module (it connects at import time)
    with patch.dict("sys.modules", {}), patch("kafka.KafkaProducer") as MockProducer:
        mock_instance = MagicMock()
        MockProducer.return_value = mock_instance

        # Remove cached module to force re-import with patch
        sys.modules.pop("src.producer.producer", None)

        # Replace the module-level producer with our mock
        import src.producer.producer as producer_mod
        from src.producer.producer import send_message

        producer_mod.producer = mock_instance

        send_message()

        mock_instance.send.assert_called_once()
        call_args = mock_instance.send.call_args
        topic = call_args[0][0]
        message = call_args[1]["value"]

        assert topic == "giga-flow-messages"
        assert "text" in message
        assert "timestamp" in message
        assert isinstance(message["text"], str)
        assert isinstance(message["timestamp"], float)
        assert len(message["text"]) > 0


def test_dummy_data_variety():
    """Ensure we have enough variety in dummy data."""
    with patch("kafka.KafkaProducer"):
        sys.modules.pop("src.producer.producer", None)
        from src.producer.producer import dummy_data

    assert len(dummy_data) >= 10
    texts_lower = [t.lower() for t in dummy_data]
    has_positive = any("love" in t or "great" in t or "wonderful" in t or "fantastic" in t for t in texts_lower)
    has_negative = any("worst" in t or "frustrated" in t or "terrible" in t or "regret" in t for t in texts_lower)
    assert has_positive, "Dummy data should contain positive examples"
    assert has_negative, "Dummy data should contain negative examples"
