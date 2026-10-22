"""Post the report on a pull request, one comment per commit, updated on reruns.

    GITHUB_TOKEN=... python qa/pr_comment.py --repo OWNER/REPO --pr N --sha SHA --body pr-comment.md

With --expect-head, refuses unless SHA is still the PR's head commit: the
workflow_run commenter uses it so a report can only land on the PR it was built for.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


def call(method, path, body=None):
    req = urllib.request.Request(
        f"{API}{path}", method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                 "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read() or b"null")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", required=True, type=int)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--expect-head", action="store_true")
    args = ap.parse_args()

    if args.expect_head:
        head = call("GET", f"/repos/{args.repo}/pulls/{args.pr}")["head"]["sha"]
        if head != args.sha:
            print(f"PR #{args.pr} head is {head}, not {args.sha}; not commenting.")
            return
    marker = f"<!-- kittentts-python-platform-report:{args.sha} -->"
    with open(args.body, encoding="utf-8") as f:
        body = f"{marker}\n{f.read()}"

    existing, page = None, 1
    while existing is None:
        batch = call("GET", f"/repos/{args.repo}/issues/{args.pr}/comments?per_page=100&page={page}")
        existing = next((c for c in batch if c["user"]["type"] == "Bot" and marker in c["body"]), None)
        if len(batch) < 100:
            break
        page += 1
    try:
        if existing:
            call("PATCH", f"/repos/{args.repo}/issues/comments/{existing['id']}", {"body": body})
            print(f"Updated the report comment for {args.sha[:7]}: {existing['html_url']}")
        else:
            c = call("POST", f"/repos/{args.repo}/issues/{args.pr}/comments", {"body": body})
            print(f"Posted the report comment for {args.sha[:7]}: {c['html_url']}")
    except urllib.error.HTTPError as e:
        sys.exit(f"GitHub refused the comment ({e.code}): {e.read().decode()[:300]}")


if __name__ == "__main__":
    main()
