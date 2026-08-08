import importlib
from importlib import metadata
from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "build.yml"
ACTION_REVISIONS = {
    "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "actions/download-artifact": "d3f86a106a0bac45b974a628896c90dbdf5c8093",
    "softprops/action-gh-release": "3bb12739c298aeb8a4eeaf626c5b8d85266b0e65",
}


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
            r"os:\s*\[windows-2022,\s*macos-14\]",
        )
        self.assertIn(
            'python -m pip install -r requirements.txt "pyinstaller==6.21.0"',
            install_step,
        )
        self.assertIn(
            'python -m pip install "matplotlib==3.11.1"',
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
            '--collect-data safehttpx --collect-data groovy '
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
            "TcpListener",
            '"--port", "$port", "--no-browser", "--results-root"',
            '"http://127.0.0.1:$port"',
            "StatusCode -ne 200",
            "Coagulation Analysis",
            "tests.test_grid_detector import make_fixture",
            "cv2.imencode",
            "gradio_client import Client, handle_file",
            "client.predict",
            'api_name="/lambda"',
            "packaged-functional-smoke.py",
            "$clientProcess.WaitForExit(60000)",
            'bundle.glob("cell_*.png")',
            '"*_results.csv"',
            '"*_results.json"',
            '"*_grid_overlay.png"',
            '"*_heatmap.png"',
            '"*_analysis.zip"',
            "$forbiddenExternalReferences",
            "src|href",
            "fonts.googleapis.com",
            "cdnjs.cloudflare.com",
            "cdn.jsdelivr.net",
            "@import",
            "url(",
            "finally",
            "taskkill.exe",
            "/T",
            "/F",
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
            "Copy-Item build-manifest.txt",
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
        release_action = (
            "softprops/action-gh-release@"
            + ACTION_REVISIONS["softprops/action-gh-release"]
        )
        self.assertEqual(1, self.workflow.count(release_action))
        self.assertNotIn(release_action, build_job)
        self.assertIn("needs: build", release_job)
        self.assertIn(
            "actions/upload-artifact@" + ACTION_REVISIONS["actions/upload-artifact"],
            build_job,
        )
        self.assertIn(
            "actions/download-artifact@"
            + ACTION_REVISIONS["actions/download-artifact"],
            release_job,
        )
        self.assertIn(release_action, release_step)
        self.assertIn("CoagulationAnalysis-Windows.zip", release_step)
        self.assertIn("CoagulationAnalysis-Mac.zip", release_step)
        self.assertIn(
            "Web/StartWebsite.exe is the recommended offline entry",
            release_step,
        )

    def test_release_runs_are_serialized_and_use_immutable_latest_tags(self):
        concurrency = _yaml_block(self.workflow, "concurrency:")
        release_step = _yaml_block(
            self.workflow,
            "      - name: Upload platform packages to release",
        )

        self.assertIn("group: coagulation-release-main", concurrency)
        self.assertIn("cancel-in-progress: true", concurrency)
        self.assertIn(
            "tag_name: build-${{ github.run_number }}-${{ github.run_attempt }}",
            release_step,
        )
        self.assertIn("make_latest: true", release_step)
        self.assertNotIn("tag_name: latest", release_step)

    def test_release_verifies_current_main_head_before_publishing(self):
        release_job = _yaml_block(self.workflow, "  release:")
        current_head_step = _yaml_block(
            self.workflow,
            "      - name: Verify build is current main head",
        )
        release_step = _yaml_block(
            self.workflow,
            "      - name: Upload platform packages to release",
        )

        self.assertIn("api.github.com/repos", current_head_step)
        self.assertIn("git/ref/heads/main", current_head_step)
        self.assertIn("GITHUB_SHA", current_head_step)
        self.assertIn("GITHUB_TOKEN", current_head_step)
        self.assertLess(
            release_job.index(current_head_step),
            release_job.index(release_step),
        )

    def test_workflow_uses_least_privilege_and_immutable_action_revisions(self):
        top_permissions = _yaml_block(self.workflow, "permissions:")
        build_job = _yaml_block(self.workflow, "  build:")
        release_job = _yaml_block(self.workflow, "  release:")
        checkout_step = _yaml_block(
            self.workflow,
            "      - name: Checkout source",
        )

        self.assertIn("contents: read", top_permissions)
        self.assertNotIn("contents: write", build_job)
        self.assertIn("contents: write", release_job)
        self.assertIn("persist-credentials: false", checkout_step)
        self.assertIn("runs-on: ubuntu-24.04", release_job)
        for action, revision in ACTION_REVISIONS.items():
            with self.subTest(action=action):
                self.assertIn(f"uses: {action}@{revision}", self.workflow)
        for line in self.workflow.splitlines():
            if "uses:" in line:
                self.assertRegex(line, r"uses:\s+[\w./-]+@[0-9a-f]{40}(?:\s+#.*)?$")

    def test_research_dependency_is_removed_before_packaging_and_manifested(self):
        remove_step = _yaml_block(
            self.workflow,
            "      - name: Remove research-only dependency before packaging",
        )
        manifest_step = _yaml_block(
            self.workflow,
            "      - name: Record packaging environment",
        )
        mac_archive_step = _yaml_block(
            self.workflow,
            "      - name: Assemble macOS desktop archive",
        )
        windows_archive_step = _yaml_block(
            self.workflow,
            "      - name: Assemble Windows release archive",
        )

        self.assertIn("python -m pip uninstall --yes matplotlib", remove_step)
        self.assertIn("find_spec", remove_step)
        self.assertIn("python -m pip freeze --all > build-manifest.txt", manifest_step)
        self.assertLess(self.workflow.index(remove_step), self.workflow.index(manifest_step))
        for build_name in (
            "      - name: Build Windows offline website",
            "      - name: Build Windows desktop app",
            "      - name: Build macOS desktop app",
        ):
            with self.subTest(build=build_name.strip()):
                self.assertLess(
                    self.workflow.index(remove_step),
                    self.workflow.index(build_name),
                )
        self.assertIn("build-manifest.txt", windows_archive_step)
        self.assertIn("build-manifest.txt", mac_archive_step)


if __name__ == "__main__":
    unittest.main()
