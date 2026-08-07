import importlib
from importlib import metadata
from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build.yml"


def _major_minor(version):
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise AssertionError(f"Unrecognised package version: {version}")
    return tuple(int(part) for part in match.groups())


def _yaml_block(workflow, header):
    lines = workflow.splitlines()
    try:
        start = lines.index(header)
    except ValueError as exception:
        raise AssertionError(f"Missing workflow block: {header.strip()}") from exception
    indentation = len(header) - len(header.lstrip())
    block = [header]
    for line in lines[start + 1 :]:
        if line.strip():
            line_indentation = len(line) - len(line.lstrip())
            if line_indentation <= indentation:
                break
        block.append(line)
    return "\n".join(block)


class RuntimeDependencyTests(unittest.TestCase):
    def test_requirements_pin_the_supported_runtime(self):
        requirements = {
            line.strip()
            for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("gradio==6.16.0", requirements)
        self.assertIn("opencv-python>=4.10,<5", requirements)
        self.assertIn("numpy>=2.0,<3", requirements)
        self.assertEqual(3, len(requirements))
        package_names = {
            re.match(r"^[A-Za-z0-9_.-]+", requirement).group(0).lower()
            for requirement in requirements
        }
        self.assertNotIn("matplotlib", package_names)
        for deep_learning_package in ("torch", "torchvision", "timm"):
            self.assertNotIn(deep_learning_package, package_names)

    def test_installed_runtime_matches_supported_versions(self):
        for module_name in ("gradio", "cv2", "numpy"):
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

        gradio_version = metadata.version("gradio")
        opencv_version = metadata.version("opencv-python")
        numpy_version = metadata.version("numpy")

        self.assertEqual("6.16.0", gradio_version)
        self.assertGreaterEqual(_major_minor(opencv_version), (4, 10))
        self.assertLess(_major_minor(opencv_version), (5, 0))
        self.assertGreaterEqual(_major_minor(numpy_version), (2, 0))
        self.assertLess(_major_minor(numpy_version), (3, 0))


class BuildWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_builds_and_tests_on_windows_and_macos_from_requirements(self):
        build_job = _yaml_block(self.workflow, "  build:")
        install_step = _yaml_block(
            self.workflow,
            "      - name: Install runtime and packaging dependencies",
        )
        research_install_step = _yaml_block(
            self.workflow,
            "      - name: Install research test dependency",
        )
        test_step = _yaml_block(self.workflow, "      - name: Run tests")
        self.assertRegex(
            build_job,
            r"os:\s*\[windows-latest,\s*macos-latest\]",
        )
        self.assertIn(
            "python -m pip install -r requirements.txt pyinstaller",
            install_step,
        )
        self.assertIn(
            'python -m pip install "matplotlib>=3.9,<4"',
            research_install_step,
        )
        self.assertIn(
            "python -m unittest discover -s tests -v",
            test_step,
        )
        for duplicate in ("gradio", "opencv-python", "numpy", "matplotlib"):
            self.assertNotIn(duplicate, install_step)

    def test_compiles_every_shipped_python_module(self):
        compile_step = _yaml_block(
            self.workflow,
            "      - name: Compile Python modules",
        )
        required_modules = (
            "app.py",
            "app_standalone.py",
            "analysis_service.py",
            "batch_service.py",
            "grid_detector.py",
            "web_controller.py",
            "web_launcher.py",
            "research/__init__.py",
            "research/annotate_inner_squares.py",
            "research/evaluate_cropping.py",
        )
        self.assertIn("python -m py_compile", compile_step)
        for module in required_modules:
            with self.subTest(module=module):
                self.assertIn(module, compile_step)

    def test_builds_separate_windows_website_and_desktop_onedir_apps(self):
        website_step = _yaml_block(
            self.workflow,
            "      - name: Build Windows offline website",
        )
        desktop_step = _yaml_block(
            self.workflow,
            "      - name: Build Windows desktop app",
        )
        self.assertIn(
            'pyinstaller --clean --noconfirm --onedir --collect-all gradio '
            '--add-data "web_styles.css;." --name StartWebsite web_launcher.py',
            website_step,
        )
        self.assertIn(
            "pyinstaller --clean --noconfirm --onedir --name "
            "CoagulationAnalysis app_standalone.py",
            desktop_step,
        )
        self.assertIn("if: runner.os == 'Windows'", website_step)
        self.assertIn("if: runner.os == 'Windows'", desktop_step)

    def test_windows_packaged_website_smoke_is_loopback_only_and_stopped(self):
        smoke_step = _yaml_block(
            self.workflow,
            "      - name: Smoke test packaged Windows website",
        )
        required_fragments = (
            "if: runner.os == 'Windows'",
            'dist/StartWebsite/StartWebsite.exe',
            '"--port", "7860", "--no-browser", "--results-root"',
            "http://127.0.0.1:7860",
            "StatusCode -ne 200",
            "Coagulation Analysis",
            "$forbiddenExternalReferences",
            "src|href",
            "fonts.googleapis.com",
            "cdnjs.cloudflare.com",
            "cdn.jsdelivr.net",
            "@import",
            "url(",
            "finally",
            "Stop-Process",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, smoke_step)
        self.assertNotIn("share=True", smoke_step)
        self.assertNotIn("--share", smoke_step)

    def test_windows_archive_contains_complete_web_and_desktop_onedir_apps(self):
        archive_step = _yaml_block(
            self.workflow,
            "      - name: Assemble Windows release archive",
        )
        required_fragments = (
            "if: runner.os == 'Windows'",
            "release/CoagulationAnalysis-Windows",
            '"$releaseRoot/Web"',
            '"$releaseRoot/Desktop"',
            'dist/StartWebsite/*',
            'dist/CoagulationAnalysis/*',
            "Copy-Item README.md",
            'Web/StartWebsite.exe',
            'Desktop/CoagulationAnalysis.exe',
            "Compress-Archive",
            '"$releaseRoot/*"',
            "CoagulationAnalysis-Windows.zip",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, archive_step)

    def test_single_release_job_uploads_both_platform_artifacts(self):
        build_job = _yaml_block(self.workflow, "  build:")
        release_job = _yaml_block(self.workflow, "  release:")
        release_step = _yaml_block(
            self.workflow,
            "      - name: Upload platform packages to release",
        )
        self.assertEqual(1, self.workflow.count("softprops/action-gh-release@v2"))
        self.assertNotIn("softprops/action-gh-release@v2", build_job)
        self.assertIn("needs: build", release_job)
        self.assertIn("actions/upload-artifact@v4", build_job)
        self.assertIn("actions/download-artifact@v4", release_job)
        self.assertIn("softprops/action-gh-release@v2", release_step)
        self.assertIn("CoagulationAnalysis-Windows.zip", release_step)
        self.assertIn("CoagulationAnalysis-Mac.zip", release_step)
        self.assertIn(
            "Web/StartWebsite.exe is the recommended offline entry",
            release_step,
        )


if __name__ == "__main__":
    unittest.main()
