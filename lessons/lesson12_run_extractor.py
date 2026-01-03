from __future__ import annotations

import json
from lesson12_extractor import extract_ticket

if __name__ == "__main__":
  text = """
    Hi, I'm Alice Johnson (alice@example.com). The app crashes when I click 'Export'.
    This is blocking our finance report due today. Please help ASAP.
    """

  data, telemetry = extract_ticket(text)
  print(json.dumps(data, indent=2))
  print("\nTelemetry:", json.dumps(telemetry, indent=2))
