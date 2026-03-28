import pytest
import httpx
import re
from datetime import datetime, timezone, timedelta
from main import app  # Assuming 'app' is in 'main.py' in the backend root

@pytest.fixture(scope="module")
def client():
    with httpx.Client(app=app, base_url="http://test") as client:
        yield client

def test_health_check_checked_at_is_iso_format_utc(client: httpx.Client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] == "ok"
    assert "checked_at" in data

    checked_at_value = data["checked_at"]

    # Assert that the value ends with 'Z' for UTC
    assert checked_at_value.endswith('Z'), f"checked_at value '{checked_at_value}' does not end with 'Z'"

    # ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ
    # This regex is a basic check for the format, not a full validation of date/time components.
    iso_8601_regex = r"^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3,6}Z$"
    assert re.match(iso_8601_regex, checked_at_value) is not None, \
        f"checked_at value '{checked_at_value}' is not in valid ISO 8601 format"

    # Further validate by attempting to parse and checking timezone
    try:
        # Replace 'Z' with '+00:00' for datetime.fromisoformat to parse it as UTC
        parsed_time = datetime.fromisoformat(checked_at_value.replace('Z', '+00:00'))
        assert parsed_time.tzinfo == timezone.utc, \
            f"checked_at value '{checked_at_value}' is not recognized as UTC"
    except ValueError:
        pytest.fail(f"checked_at value '{checked_at_value}' is not a valid ISO 8601 UTC datetime.")

def test_health_endpoint_checked_at_is_recent(client: httpx.Client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    checked_at_value = data["checked_at"]

    # Parse the checked_at timestamp
    # Replace 'Z' with '+00:00' for datetime.fromisoformat to parse it as UTC
    parsed_time = datetime.fromisoformat(checked_at_value.replace('Z', '+00:00'))

    # Get current UTC time
    now_utc = datetime.now(timezone.utc)

    # Define a small tolerance for recency (e.g., 5 seconds)
    time_difference = now_utc - parsed_time
    assert timedelta(seconds=-5) < time_difference < timedelta(seconds=5), \
        f"checked_at timestamp '{checked_at_value}' is not recent. Difference: {time_difference}"
