import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evch.utils.wandb import DummyRun, init_wandb, log_artifact


class _FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class _FakeRun:
    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] | None = None
        self.artifacts: list[tuple[object, list[str]]] = []

    def log_artifact(self, artifact: object, aliases: list[str] | None = None) -> None:
        self.artifacts.append((artifact, aliases or []))


class WandbUtilsTest(unittest.TestCase):
    def test_init_returns_dummy_run_when_disabled(self) -> None:
        run = init_wandb(config={"logging": {"wandb": {"enabled": False}}}, job_type="train", run_name="demo")
        self.assertIsInstance(run, DummyRun)

    def test_online_mode_uses_saved_or_external_credentials(self) -> None:
        fake_run = _FakeRun()
        fake_wandb = SimpleNamespace(init=lambda **kwargs: setattr(fake_run, "init_kwargs", kwargs) or fake_run)

        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            with mock.patch("evch.utils.wandb.find_spec", return_value=object()):
                with mock.patch.dict(os.environ, {}, clear=True):
                    run = init_wandb(
                        config={"logging": {"wandb": {"enabled": True, "mode": "online", "project": "demo"}}},
                        job_type="train",
                        run_name="demo-run",
                    )

        self.assertIs(run, fake_run)
        self.assertIsNotNone(fake_run.init_kwargs)
        self.assertEqual(fake_run.init_kwargs["mode"], "online")
        self.assertEqual(fake_run.init_kwargs["project"], "demo")

    def test_log_artifact_adds_file_to_run(self) -> None:
        fake_run = _FakeRun()
        fake_wandb = SimpleNamespace(Artifact=_FakeArtifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "metrics.json"
            artifact_path.write_text('{"ok": true}', encoding="utf-8")

            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                with mock.patch("evch.utils.wandb.find_spec", return_value=object()):
                    log_artifact(
                        run=fake_run,
                        path=artifact_path,
                        artifact_name="demo-metrics",
                        artifact_type="metrics",
                        aliases=["latest"],
                    )

        self.assertEqual(len(fake_run.artifacts), 1)
        artifact, aliases = fake_run.artifacts[0]
        self.assertEqual(artifact.name, "demo-metrics")
        self.assertEqual(artifact.type, "metrics")
        self.assertEqual(artifact.files, [str(artifact_path)])
        self.assertEqual(aliases, ["latest"])


if __name__ == "__main__":
    unittest.main()
