import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv(override=True)
API_KEY = os.getenv("GRAPH_API_KEY")
SUBGRAPH_ID = "6yuf1C49aWEscgk5n9D1DekeG1BCk5Z9imJYJT3sVmAT"
API_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
OUTPUT_FILE = "aave_liquidated_users.jsonl"

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
SAVE_EVERY_N_ITERATIONS = 5
SLEEP_BETWEEN_PAGES_SECONDS = 0.5
RETRY_BACKOFF_BASE_SECONDS = 1


def build_query(last_id):
    return f"""
    {{
      liquidates(first: 100, orderBy: id, orderDirection: asc, where: {{id_gt: "{last_id}"}}) {{
        id
        liquidatee {{
          id
          positions {{
            side
            balance
            market {{
              inputToken {{
                symbol
                decimals
              }}
            }}
          }}
          liquidates(first: 1) {{
            timestamp
          }}
        }}
      }}
    }}
    """


def append_jsonl(users, output_path):
    if not users:
        return
    with open(output_path, "a", encoding="utf-8") as file:
        for user in users:
            file.write(json.dumps(user, ensure_ascii=True) + "\n")


def fetch_page_with_retry(query):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            print(f"[Retry {attempt}/{MAX_RETRIES}] Network error: {error}")
            if attempt == MAX_RETRIES:
                return None, "network_error"
            time.sleep(RETRY_BACKOFF_BASE_SECONDS * attempt)
            continue

        if response.status_code != 200:
            print(
                f"[Retry {attempt}/{MAX_RETRIES}] HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
            if attempt == MAX_RETRIES:
                return None, "http_error"
            time.sleep(RETRY_BACKOFF_BASE_SECONDS * attempt)
            continue

        try:
            payload = response.json()
        except ValueError:
            print(f"[Retry {attempt}/{MAX_RETRIES}] Invalid JSON in response.")
            if attempt == MAX_RETRIES:
                return None, "invalid_json"
            time.sleep(RETRY_BACKOFF_BASE_SECONDS * attempt)
            continue

        if payload.get("errors"):
            print(f"[Retry {attempt}/{MAX_RETRIES}] GraphQL errors: {payload['errors']}")
            if attempt == MAX_RETRIES:
                return None, "graphql_error"
            time.sleep(RETRY_BACKOFF_BASE_SECONDS * attempt)
            continue

        # Retrieve liquidation events from this page.
        events = payload.get("data", {}).get("liquidates", [])
        # Extract liquidatee account profiles from liquidation events.
        users = [event["liquidatee"] for event in events if event.get("liquidatee")]
        return users, None

    return None, "unknown_error"


def fetch_all_aave_users():
    if not API_KEY:
        raise RuntimeError("Missing GRAPH_API_KEY in .env")

    last_id = ""
    iteration = 0
    total_users = 0
    pending_users = []

    print("Starting Aave liquidatee extraction...")
    print(f"Incremental output file: {OUTPUT_FILE}")

    while True:
        query = build_query(last_id)
        page_data, error_type = fetch_page_with_retry(query)

        if error_type:
            print(f"Stopping extraction due to repeated API failure: {error_type}")
            break

        if not page_data:
            print("No more data returned by API. Extraction complete.")
            break

        iteration += 1
        total_users += len(page_data)
        pending_users.extend(page_data)
        new_last_id = page_data[-1]["id"]
        if new_last_id == last_id:
            print("Detected non-advancing pagination cursor. Stopping to avoid infinite loop.")
            break
        last_id = new_last_id

        print(
            f"Iteration {iteration}: fetched {len(page_data)} users "
            f"(total: {total_users}, last_id: {last_id})"
        )

        if iteration % SAVE_EVERY_N_ITERATIONS == 0:
            append_jsonl(pending_users, OUTPUT_FILE)
            print(f"Saved {len(pending_users)} users to {OUTPUT_FILE}")
            pending_users = []

        time.sleep(SLEEP_BETWEEN_PAGES_SECONDS)

    if pending_users:
        append_jsonl(pending_users, OUTPUT_FILE)
        print(f"Saved final {len(pending_users)} users to {OUTPUT_FILE}")

    print(f"Extraction finished. Total users fetched: {total_users}")


if __name__ == "__main__":
    fetch_all_aave_users()