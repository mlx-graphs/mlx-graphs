import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from check_commit_title import valid_title
from prepare_release import BASE_TAG, Release, select_release


class ReleaseSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        self.git("init", "--quiet", "--initial-branch=main")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release-test@example.com")
        self.commit("Existing release")
        self.git("tag", BASE_TAG)
        self.runs = {}
        self.released_tags = set()
        git_patch = patch("prepare_release.git", self.git)
        api_patch = patch("prepare_release.github", self.github)
        git_patch.start()
        api_patch.start()
        self.addCleanup(git_patch.stop)
        self.addCleanup(api_patch.stop)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def github(self, path, *, missing_ok=False):
        if path.startswith("releases/tags/"):
            tag = path.rsplit("/", 1)[-1]
            return {"draft": False} if tag in self.released_tags else None
        sha = parse_qs(urlsplit(path).query)["head_sha"][0]
        return {"workflow_runs": self.runs[sha]}

    def commit(self, title):
        self.git("commit", "--quiet", "--allow-empty", "-m", title)
        return self.git("rev-parse", "HEAD")

    def activate(self):
        workflow = self.repo / ".github/workflows/main_checks.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("name: Main checks\n")
        self.git("add", str(workflow))
        return self.commit("ci: automate releases")

    def check(self, sha, conclusion, *, status="completed", attempt=1):
        self.runs[sha] = [
            {
                "status": status,
                "conclusion": conclusion,
                "run_number": 1,
                "run_attempt": attempt,
                "head_sha": sha,
            }
        ]

    def test_bundles_everything_up_to_the_newest_green_commit(self):
        self.commit("Pre-rollout commit")
        earlier = self.activate()
        later = self.commit("feat: add a loader")
        self.check(earlier, "success")
        self.check(later, "success")
        self.assertEqual(select_release(), Release(later, BASE_TAG))

    def test_leaves_unfinished_and_failed_commits_for_the_next_run(self):
        green = self.activate()
        running = self.commit("feat: add a loader")
        failed = self.commit("fix: an unfinished fix")
        self.check(green, "success")
        self.check(running, None, status="in_progress")
        self.check(failed, "failure")
        self.assertEqual(select_release(), Release(green, BASE_TAG))

    def test_uses_the_latest_attempt_of_the_checks(self):
        sha = self.activate()
        self.check(sha, "failure")
        self.runs[sha].append(
            dict(self.runs[sha][0], conclusion="success", run_attempt=2)
        )
        self.assertEqual(select_release(), Release(sha, BASE_TAG))

    def test_nothing_to_release_without_a_new_green_commit(self):
        first = self.activate()
        self.git("tag", "v0.2.1", first)
        self.released_tags.add("v0.2.1")
        self.assertIsNone(select_release())
        failed = self.commit("fix: an unfinished fix")
        self.check(failed, "failure")
        self.assertIsNone(select_release())

    def test_ignores_commits_from_before_the_release_rollout(self):
        self.commit("feat: before the post-merge checks existed")
        self.assertIsNone(select_release())

    def test_resumes_with_the_original_artifact_before_moving_on(self):
        first = self.activate()
        self.git("tag", "-am", "mlx-graphs release artifact: 123 456", "v0.2.1", first)
        self.commit("docs: update the quickstart")
        expected = Release(first, BASE_TAG, "v0.2.1", "123", "456")
        self.assertEqual(select_release(), expected)


class CommitTitleTests(unittest.TestCase):
    def test_accepts_scope_and_breaking_marker(self):
        self.assertTrue(valid_title("feat(loaders)!: change defaults", ["feat"]))

    def test_rejects_typos_empty_descriptions_and_workflow_skips(self):
        for title in ("feature: add dataset", "fix:  ", "fix: handle None [SKIP CI]"):
            with self.subTest(title=title):
                self.assertFalse(valid_title(title, ["feat", "fix"]))
