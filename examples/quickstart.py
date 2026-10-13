"""Rewrite a message with irulan-swe.

    export IRULAN_BASE_URL=...   # an implementation of docs/API.md (no hosted endpoint yet)
    export IRULAN_API_KEY=...
    python3 examples/quickstart.py "I just wanted to quickly check in on the status of the deploy."
"""

import sys

import irulan


def main():
    text = " ".join(sys.argv[1:]) or (
        "I hope you're doing well! I wanted to follow up and ask whether you might have a moment "
        "to review the pull request I opened yesterday. Thank you so much!")
    client = irulan.Client()
    try:
        print(client.rewrite(text))
    except irulan.NotAvailableYet as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
