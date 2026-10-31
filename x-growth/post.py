#!/usr/bin/env python3
"""Post to X (API v2) as @aleromlan using OAuth 1.0a user context. Stdlib only.

Usage:
  post.py [--dry-run] [--reply-to TWEET_ID] FILE
  post.py --next [--dry-run]     # post the first file in queue/, move it to posted/

A PNG/JPG in media/ with the same name as the queued .txt is attached automatically.

Env: X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
"""
import base64, hashlib, hmac, json, os, secrets, shutil, sys, time, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://api.x.com/2/tweets"
UPLOAD_URL = "https://api.x.com/2/media/upload"
HERE = Path(__file__).resolve().parent


def q(s):
    return urllib.parse.quote(str(s), safe="~-._")


def sign(method, url, params, consumer_secret, token_secret):
    # JSON bodies are not part of the signature base string; only oauth_* and query/form params are.
    norm = "&".join(f"{k}={v}" for k, v in sorted((q(k), q(v)) for k, v in params.items()))
    base = "&".join([method.upper(), q(url), q(norm)])
    key = f"{q(consumer_secret)}&{q(token_secret)}"
    return base64.b64encode(hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()).decode()


def auth_header(method, url):
    oauth = {
        "oauth_consumer_key": os.environ["X_API_KEY"],
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": os.environ["X_ACCESS_TOKEN"],
        "oauth_version": "1.0",
    }
    oauth["oauth_signature"] = sign(method, url, oauth, os.environ["X_API_SECRET"], os.environ["X_ACCESS_TOKEN_SECRET"])
    return "OAuth " + ", ".join(f'{q(k)}="{q(v)}"' for k, v in sorted(oauth.items()))


def upload(image):
    # Multipart bodies are not part of the OAuth 1.0a signature, same as JSON.
    boundary = secrets.token_hex(16)
    data = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"media_category\"\r\n\r\ntweet_image\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"media\"; filename=\"{image.name}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode(),
        image.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(UPLOAD_URL, data=data, method="POST", headers={
        "Authorization": auth_header("POST", UPLOAD_URL), "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)["data"]["id"]
    except urllib.error.HTTPError as e:
        sys.exit(f"X media upload error {e.code}: {e.read().decode()}")


def find_media(path):
    return next((p for ext in ("png", "jpg", "jpeg") if (p := HERE / "media" / f"{path.stem}.{ext}").exists()), None)


def post(text, reply_to=None, dry_run=False, image=None):
    body = {"text": text}
    if image:
        print(f"attaching {image.name}", file=sys.stderr)
        body["media"] = {"media_ids": ["<uploaded>" if dry_run else upload(image)]}
    if reply_to:
        body["reply"] = {"in_reply_to_tweet_id": reply_to}
    if len(text) > 280:
        print(f"note: {len(text)} chars, needs X Premium for long posts", file=sys.stderr)
    if dry_run:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return None
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": auth_header("POST", URL), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"X API error {e.code}: {e.read().decode()}")
    tweet_id = data["data"]["id"]
    print(f"posted https://x.com/aleromlan/status/{tweet_id}")
    return tweet_id


def main(argv):
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    reply_to = None
    if "--reply-to" in argv:
        i = argv.index("--reply-to")
        reply_to = argv[i + 1]
        del argv[i:i + 2]
    if argv == ["--next"]:
        queue = sorted((HERE / "queue").glob("*.txt"))
        if not queue:
            sys.exit("queue is empty")
        path = queue[0]
    elif len(argv) == 1:
        path = Path(argv[0])
    else:
        sys.exit(__doc__)
    tweet_id = post(path.read_text().strip(), reply_to, dry_run, find_media(path))
    if tweet_id and path.parent == HERE / "queue":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        shutil.move(path, HERE / "posted" / f"{stamp}-{tweet_id}-{path.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
