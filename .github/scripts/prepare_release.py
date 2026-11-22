"""Select the next tested commit or resume the current release."""

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_TAG = "v0.2.0"
CI_WORKFLOW = "main_checks.yml"
STABLE_TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
RESERVATION = re.compile(r"mlx-graphs release artifact: ([1-9]\d*) ([1-9]\d*)")


@dataclass
class Release:
    sha: str
    previous_tag: str
    tag: str = ""
    artifact_run_id: str = ""
    artifact_id: str = ""


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def github(path, *, missing_ok=False):
    api = os.environ["GITHUB_API_URL"]
    repo = os.environ["GITHUB_REPOSITORY"]
    request = Request(
        f"{api}/repos/{repo}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if missing_ok and error.code == 404:
            return None
        raise


def main_releases(history):
    positions = set(history)
    releases = []
    refs = git(
        "for-each-ref",
        "--sort=-version:refname",
        "--format=%(refname:short) %(objectname) %(*objectname)",
        "refs/tags/v*",
    )
    for ref in refs.splitlines():
        tag, *objects = ref.split()
        sha = objects[-1]  # An annotated tag includes its peeled commit last.
        if STABLE_TAG.fullmatch(tag) and sha in positions:
            releases.append((tag, sha))
    return releases


def select_release(completed_run=None):
    history = git("rev-list", "--first-parent", "--reverse", "HEAD").splitlines()
    baseline = git("rev-parse", f"{BASE_TAG}^{{commit}}")
    if baseline not in history:
        raise RuntimeError(f"{BASE_TAG} must be on the first-parent history of main")
    history = history[history.index(baseline) :]
    releases = main_releases(history)
    tag, released_sha = releases[0]

    if tag != BASE_TAG:
        release = github(f"releases/tags/{tag}", missing_ok=True)
        if release is None or release["draft"]:
            subject = git(
                "for-each-ref", "--format=%(contents:subject)", f"refs/tags/{tag}"
            )
            reservation = RESERVATION.fullmatch(subject)
            if reservation is None:
                raise RuntimeError(f"{tag} has no GitHub release or build artifact")
            run_id, artifact_id = reservation.groups()
            return Release(released_sha, releases[1][0], tag, run_id, artifact_id)

    commits = history[history.index(released_sha) + 1 :]
    for sha in commits:
        if not git("ls-tree", sha, f".github/workflows/{CI_WORKFLOW}"):
            continue  # main had no post-merge checks before the release rollout.
        query = urlencode({"branch": "main", "event": "push", "head_sha": sha})
        runs = github(f"actions/workflows/{CI_WORKFLOW}/runs?{query}")["workflow_runs"]
        if completed_run and completed_run["head_sha"] == sha:
            # The completion event can arrive before the list API reflects it.
            runs.append(completed_run)
        run = max(
            runs,
            key=lambda item: (
                item["run_number"],
                item["run_attempt"],
                item["status"] == "completed",
            ),
            default=None,
        )
        if run is None or run["status"] != "completed":
            print(f"Waiting for main checks on {sha}")
            return None
        if run["conclusion"] == "success":
            return Release(sha, tag)
        print(f"Skipping {sha}: main checks concluded {run['conclusion']}")
    return None


if __name__ == "__main__":
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event:
        completed_run = json.load(event).get("workflow_run")
    release = select_release(completed_run)
    if release is not None:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            for key, value in asdict(release).items():
                print(f"{key}={value}", file=output)
