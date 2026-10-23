from pathlib import Path
import os
import sys
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def get_slack_client() -> WebClient:
    token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_USER_TOKEN")
    if not token:
        print(
            "Error: SLACK_BOT_TOKEN or SLACK_USER_TOKEN environment variable not set.",
            file=sys.stderr,
        )
        print(
            "Please export SLACK_BOT_TOKEN='xoxb-...' before running.",
            file=sys.stderr,
        )
        sys.exit(1)
    return WebClient(token=token)


def list_users(client: WebClient):
    active_users = []
    inactive_users = []

    cursor = None
    while True:
        try:
            # users.list requires 'users:read' scope
            response = client.users_list(cursor=cursor, limit=200)
        except SlackApiError as e:
            print(f"Slack API error: {e.response['error']}", file=sys.stderr)
            sys.exit(1)

        for member in response["members"]:
            # Skip bots and Slackbot if desired, but retain user classification
            is_deleted = member.get("deleted", False)
            user_id = member.get("id")
            name = member.get("name")
            real_name = (
                member.get("real_name")
                or member.get("profile", {}).get("real_name")
                or name
            )
            is_bot = member.get("is_bot", False)

            info = {
                "id": user_id,
                "name": name,
                "real_name": real_name,
                "is_bot": is_bot,
            }

            if is_deleted:
                inactive_users.append(info)
            else:
                active_users.append(info)

        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    return active_users, inactive_users


def main():
    client = get_slack_client()
    print("Fetching users from Slack workspace...")
    active_users, inactive_users = list_users(client)

    print(f"\n--- Active Users ({len(active_users)}) ---")
    for user in active_users:
        bot_suffix = " [BOT]" if user["is_bot"] else ""
        print(f" - {user['real_name']} (@{user['name']}, ID: {user['id']}){bot_suffix}")

    print(f"\n--- Inactive / Deactivated Users ({len(inactive_users)}) ---")
    for user in inactive_users:
        bot_suffix = " [BOT]" if user["is_bot"] else ""
        print(f" - {user['real_name']} (@{user['name']}, ID: {user['id']}){bot_suffix}")


if __name__ == "__main__":
    main()
