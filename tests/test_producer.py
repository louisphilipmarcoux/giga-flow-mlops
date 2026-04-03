import json
from unittest.mock import MagicMock, patch


def test_message_format():
    """Verify that produced messages have the expected schema."""
    with patch("src.producer.producer.KafkaProducer") as MockProducer:
        mock_instance = MagicMock()
        MockProducer.return_value = mock_instance

        # Import after patching to avoid real Kafka connection
        import importlib
        import src.producer.producer as producer_mod
        producer_mod.producer = mock_instance

        producer_mod.send_message()

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
    from src.producer.producer import dummy_data

    assert len(dummy_data) >= 10
    # Check we have both positive and negative examples
    texts_lower = [t.lower() for t in dummy_data]
    has_positive = any("love" in t or "great" in t or "wonderful" in t or "fantastic" in t for t in texts_lower)
    has_negative = any("worst" in t or "frustrated" in t or "terrible" in t or "regret" in t for t in texts_lower)
    assert has_positive, "Dummy data should contain positive examples"
    assert has_negative, "Dummy data should contain negative examples"
