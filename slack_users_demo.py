import datetime
import os
import sys
import time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

EXCLUDE_BOT = True
EXCLUDE_GUEST = True


def get_slack_client() -> WebClient:
    token = os.environ.get("SLACK_USER_TOKEN") or os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print(
            "Error: SLACK_USER_TOKEN or SLACK_BOT_TOKEN environment variable not set.",
            file=sys.stderr,
        )
        print(
            "Note: Searching workspace messages requires a User Token ('xoxp-...') with 'search:read' and 'users:read' scopes.",
            file=sys.stderr,
        )
        sys.exit(1)
    return WebClient(token=token)


def get_all_users(client: WebClient):
    users = []
    cursor = None
    while True:
        try:
            response = client.users_list(cursor=cursor, limit=200)
        except SlackApiError as e:
            print(f"Error fetching users: {e.response['error']}", file=sys.stderr)
            sys.exit(1)

        for member in response["members"]:
            if EXCLUDE_BOT and (
                member.get("is_bot", False) or member.get("id") == "USLACKBOT"
            ):
                continue

            if EXCLUDE_GUEST and (
                member.get("is_restricted", False)
                or member.get("is_ultra_restricted", False)
            ):
                continue

            real_name = (
                member.get("real_name")
                or member.get("profile", {}).get("real_name")
                or member.get("name")
            )
            users.append(
                {
                    "id": member["id"],
                    "name": member.get("name", ""),
                    "real_name": real_name,
                    "deleted": member.get("deleted", False),
                }
            )

        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    return users


def get_message_timestamp(client: WebClient, query: str, sort_dir: str):
    try:
        response = client.search_messages(
            query=query,
            sort="timestamp",
            sort_dir=sort_dir,
            count=1,
        )
        matches = response.get("messages", {}).get("matches", [])
        if matches:
            return float(matches[0]["ts"])
    except SlackApiError as e:
        error_code = e.response.get("error", "")
        if error_code == "ratelimited":
            retry_after = int(e.response.headers.get("Retry-After", 2))
            time.sleep(retry_after)
            return get_message_timestamp(client, query, sort_dir)
        elif error_code == "not_allowed_token_type":
            print(
                "Error: 'search.messages' requires a Slack User Token ('xoxp-...'). Bot tokens ('xoxb-...') cannot search.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(f"Slack search error: {error_code}", file=sys.stderr)
    return None


def get_user_extremes(client: WebClient, user_id: str):
    # Slack standard search does not support boolean OR, so query messages sent and mentions separately
    queries = [f"from:<@{user_id}>", f"<@{user_id}>"]

    timestamps = []
    for q in queries:
        earliest = get_message_timestamp(client, q, sort_dir="asc")
        time.sleep(0.3)
        latest = get_message_timestamp(client, q, sort_dir="desc")
        time.sleep(0.3)

        if earliest is not None:
            timestamps.append(earliest)
        if latest is not None:
            timestamps.append(latest)

    if not timestamps:
        return None, None

    return min(timestamps), max(timestamps)


def format_date(ts):
    if ts is None:
        return "N/A"
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
        "%Y-%m-%d"
    )


def main():
    client = get_slack_client()
    users = get_all_users(client)
    users.sort(key=lambda u: u["real_name"].lower())

    print("name,earliest_date,latest_date")
    for user in users:
        earliest_ts, latest_ts = get_user_extremes(client, user["id"])
        earliest_str = format_date(earliest_ts)
        latest_str = format_date(latest_ts)

        display_name = user["real_name"].replace('"', '""')
        if "," in display_name:
            display_name = f'"{display_name}"'

        print(f"{display_name},{earliest_str},{latest_str}")


if __name__ == "__main__":
    main()
