import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "installer", Path(__file__).resolve().parents[1] / "install.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def test_uninstall_removes_only_the_managed_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "codex").write_text("mod")
            original = root / "system-codex"
            original.write_text("system")
            launcher = root / "codex"
            launcher.symlink_to(data / "codex")
            installer.uninstall(launcher, data)
            self.assertFalse(launcher.is_symlink())
            self.assertEqual(original.read_text(), "system")
            launcher.symlink_to(original)
            with self.assertRaises(ValueError):
                installer.uninstall(launcher, data)
            self.assertEqual(launcher.resolve(), original)

    def test_patch_can_be_applied_twice_without_changing_it_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            target = source / "example.txt"
            target.write_text("before\n")
            patch = root / "example.patch"
            patch.write_text(
                "diff --git a/example.txt b/example.txt\n"
                "--- a/example.txt\n+++ b/example.txt\n"
                "@@ -1 +1 @@\n-before\n+after\n"
            )
            installer.apply_patch(source, patch)
            installer.apply_patch(source, patch)
            self.assertEqual(target.read_text(), "after\n")

    def test_unrelated_launcher_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "codex"
            launcher.write_text("another launcher")
            with self.assertRaises(ValueError):
                installer.check_launcher(launcher, root / "data")
            self.assertEqual(launcher.read_text(), "another launcher")

    def test_runtime_is_found_beside_a_standalone_codex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex = root / "codex"
            helper = root / "codex-code-mode-host"
            codex.touch()
            helper.touch()
            self.assertEqual(installer.find_helper(codex), helper)


if __name__ == "__main__":
    unittest.main()
