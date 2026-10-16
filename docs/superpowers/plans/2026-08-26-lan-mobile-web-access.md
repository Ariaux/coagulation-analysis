# LAN Mobile Web Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the packaged Windows website serve the computer and phones on the same trusted Wi-Fi, with a local QR code plus phone gallery and native-camera inputs, while preserving offline processing and host-only result storage.

**Architecture:** Add a pure `lan_access.py` boundary for private IPv4 discovery and connection metadata. Pass it from `web_launcher.py` into `app.create_app`, render a locally generated QR, and keep analysis behind the existing controller. Use two Gradio file uploads; local JavaScript adds `capture="environment"` to the camera input so LAN access does not depend on HTTPS-only webcam APIs.

**Tech Stack:** Python 3.12, Gradio 6.16.0, qrcode/Pillow, sockets, local JavaScript/CSS, unittest, PyInstaller, GitHub Actions.

---

## Files

- Create `lan_access.py` and `tests/test_lan_access.py`.
- Modify `web_launcher.py`, `app.py`, `web_styles.css`.
- Modify launcher, controller, and runtime-dependency tests.
- Modify runtime/build requirements, both platform lock files, build workflow, and README.

## Task 1: Private LAN discovery

**Files:**
- Create: `lan_access.py`
- Create: `tests/test_lan_access.py`

- [ ] **Step 1: Write failing tests**

Test filtering, de-duplication, preference, and fallback:

```python
def test_filters_and_prefers_private_route(self):
    info = discover_lan_access(
        8123,
        candidate_provider=lambda: [
            "127.0.0.1", "169.254.2.3", "8.8.8.8",
            "192.168.1.44", "10.0.0.7", "192.168.1.44",
        ],
        preferred_provider=lambda: "10.0.0.7",
    )
    self.assertEqual("http://10.0.0.7:8123", info.preferred_url)
    self.assertEqual(
        ("http://10.0.0.7:8123", "http://192.168.1.44:8123"),
        info.phone_urls,
    )

def test_no_lan_keeps_loopback(self):
    info = discover_lan_access(
        7860,
        candidate_provider=lambda: ["127.0.0.1"],
        preferred_provider=lambda: None,
    )
    self.assertEqual("http://127.0.0.1:7860", info.loopback_url)
    self.assertEqual((), info.phone_urls)
```

- [ ] **Step 2: Confirm RED**

Run: `python -m unittest tests.test_lan_access -v`

Expected: `ModuleNotFoundError: No module named 'lan_access'`.

- [ ] **Step 3: Implement the pure boundary**

Create immutable metadata and injected discovery providers:

```python
@dataclass(frozen=True)
class LanAccessInfo:
    loopback_url: str
    phone_urls: tuple[str, ...]
    preferred_url: str | None

def _usable_private_ipv4(value: str) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError:
        return False
    return (
        address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
    )

def discover_lan_access(
    port: int,
    *,
    candidate_provider=_candidate_ipv4_addresses,
    preferred_provider=_preferred_route_ipv4,
) -> LanAccessInfo:
    candidates = set(candidate_provider())
    preferred = preferred_provider()
    if preferred:
        candidates.add(preferred)
    usable = sorted(value for value in candidates if _usable_private_ipv4(value))
    if preferred in usable:
        usable.remove(preferred)
        usable.insert(0, preferred)
    urls = tuple(f"http://{address}:{port}" for address in usable)
    return LanAccessInfo(
        f"http://127.0.0.1:{port}",
        urls,
        urls[0] if urls else None,
    )
```

Use `socket.getaddrinfo(socket.gethostname(), ..., AF_INET)` for candidates. Use an un-sent UDP route probe only to select the preferred local address. Convert provider errors to an empty candidate set. Cover malformed, public, link-local, duplicate, and non-private preferred values.

- [ ] **Step 4: Verify and commit**

Run: `python -m unittest tests.test_lan_access tests.test_web_launcher -v`

Commit:

```bash
git add lan_access.py tests/test_lan_access.py
git commit -m "feat: discover private LAN access"
```

## Task 2: Connection panel and local QR

**Files:**
- Modify: `app.py:423-455`
- Modify: `web_styles.css`
- Modify: `tests/test_web_controller.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing tests**

Construct `LanAccessInfo` with `192.168.1.44`; assert the app config contains a `lan-connection-panel`, exact phone URL, trusted-network/no-password warning, and a QR data URL. Decode the QR in the test and assert its payload is exactly the phone URL. Test no-LAN instructions and absence of QR.

- [ ] **Step 2: Confirm RED**

Run: `python -m unittest tests.test_web_controller.WebApplicationTests -v`

Expected: factory signature and QR helper failures.

- [ ] **Step 3: Add exact runtime dependency and helper**

Add `qrcode==8.2` to `requirements.txt`. Implement:

```python
def _qr_data_url(value: str) -> str:
    image = qrcode.make(value)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")

def create_app(
    results_root: str | Path,
    lan_access: LanAccessInfo | None = None,
) -> gr.Blocks:
    ...
```

Render preferred QR, all private candidates, Wi-Fi restart instructions, and no-password warning before the tabs. QR contains URL only.

- [ ] **Step 4: Style, verify, commit**

Add stable `#lan-connection-panel` and `#lan-qr` selectors. Run the focused suite and `git diff --check`.

Commit:

```bash
git add app.py web_styles.css tests/test_web_controller.py requirements.txt
git commit -m "feat: show local mobile connection"
```

## Task 3: Gallery and native camera capture

**Files:**
- Modify: `app.py:144-455`
- Modify: `web_styles.css`
- Modify: `tests/test_web_controller.py`

- [ ] **Step 1: Write failing source tests**

Require two single-file filepath components with IDs `gallery-source` and `camera-source`, while batch remains multiple. Require local bootstrap text with `accept="image/*"` and `capture="environment"` only for camera. Reject `getUserMedia`, remote scripts, and ambiguous/missing sources.

```python
def test_single_source(self):
    self.assertEqual("a.png", _single_source("a.png", None))
    self.assertEqual("b.jpg", _single_source(None, "b.jpg"))
    with self.assertRaisesRegex(ValueError, "Choose or take"):
        _single_source(None, None)
    with self.assertRaisesRegex(ValueError, "only one"):
        _single_source("a.png", "b.jpg")
```

- [ ] **Step 2: Confirm RED**

Run: `python -m unittest tests.test_web_controller.WebApplicationTests -v`

- [ ] **Step 3: Implement dual sources**

Use two `gr.File(file_count="single", type="filepath")` components. Selecting either clears the other. Analyze calls `_single_source` before the existing adapter. Relabel folder buttons `Open folder on Windows PC`.

- [ ] **Step 4: Add local capture bootstrap**

Inject before the Gradio module:

```javascript
function configureCameraInput() {
  const input = document.querySelector('#camera-source input[type="file"]');
  if (!input) return false;
  input.setAttribute('accept', 'image/*');
  input.setAttribute('capture', 'environment');
  return true;
}
```

Use a bounded mutation observer on the app root. Do not use live webcam APIs.

- [ ] **Step 5: Add mobile CSS and verify**

At `max-width: 720px`, stack controls, make images full width, retain row-major three-column crop gallery, and confine tables to their own horizontal scroll area.

Run:

```bash
python -m unittest tests.test_web_controller tests.test_analysis_service -v
git diff --check
```

Commit `feat: add mobile image capture`.

## Task 4: LAN launcher and concurrent clients

**Files:**
- Modify: `web_launcher.py:223-304`
- Modify: `tests/test_web_launcher.py`
- Modify: `tests/test_web_controller.py`

- [ ] **Step 1: Write failing launcher tests**

Require:

```python
create_app.assert_called_once_with(resolved_root, access)
application.launch.assert_called_once_with(
    server_name="0.0.0.0",
    server_port=8123,
    share=False,
    inbrowser=False,
    prevent_thread_lock=True,
    show_error=True,
    allowed_paths=[str(resolved_root)],
    footer_links=[],
)
browser.assert_called_once_with("http://127.0.0.1:8123")
```

Also require console phone URLs, private-network/no-password warning, no-LAN fallback, and preservation of all cleanup/audit tests.

- [ ] **Step 2: Confirm RED**

Run: `python -m unittest tests.test_web_launcher -v`.

- [ ] **Step 3: Implement LAN binding**

Discover metadata for the selected port, call `create_app(root, access)`, bind `0.0.0.0`, keep `share=False` and `allowed_paths`, and always open `access.loopback_url` on the PC. Print computer/phone URLs and close-window instructions. Preserve lifecycle cleanup and sanitized audit behavior.

- [ ] **Step 4: Test concurrent sessions**

Use two clients to submit different synthetic fixtures concurrently. Assert two complete distinct result bundles and that a failure from one client cannot remove the other committed bundle.

- [ ] **Step 5: Verify and commit**

Run:

```bash
python -m unittest tests.test_lan_access tests.test_web_launcher   tests.test_web_controller tests.test_batch_service -v
```

Commit `feat: serve website on private LAN`.

## Task 5: Reproducible package and documentation

**Files:**
- Modify: `requirements-build.in`
- Regenerate: `requirements-windows.lock`, `requirements-macos.lock`
- Modify: `.github/workflows/build.yml`
- Modify: `tests/test_runtime_dependencies.py`
- Modify: `README.md:58-137`

- [ ] **Step 1: Write failing package tests**

Require `qrcode==8.2` in runtime and both hashed locks, `lan_access.py` in compile coverage, `--collect-all qrcode` in the Windows website build, and README terms for same Wi-Fi, QR, private network, no password, gallery, camera, and `Web\\StartWebsite.exe`.

- [ ] **Step 2: Confirm RED**

Run: `python -m unittest tests.test_runtime_dependencies -v`.

- [ ] **Step 3: Regenerate platform locks**

```bash
uv pip compile requirements-build.in --refresh --upgrade   --python-version 3.12 --python-platform x86_64-pc-windows-msvc   --generate-hashes --only-binary=:all:   --output-file requirements-windows.lock

uv pip compile requirements-build.in --refresh --upgrade   --python-version 3.12 --python-platform aarch64-apple-darwin   --generate-hashes --only-binary=:all:   --output-file requirements-macos.lock
```

Assert Windows retains `pywin32-ctypes`, macOS retains `macholib`, and both contain exact QR hashes.

- [ ] **Step 4: Update CI package/smoke**

Compile `lan_access.py`; package QR data; retain safehttpx/groovy data. Extend packaged smoke to verify loopback, private-only displayed phone URLs, local QR bytes, no external resources, synthetic nine-cell output, and full process-tree shutdown.

- [ ] **Step 5: Rewrite Windows instructions**

Document complete extraction, same trusted Wi-Fi, private-only firewall permission, QR/text URL, gallery/camera, host `Web\\results`, no password, restart after Wi-Fi change, no cloud, and close-window shutdown. State that anyone on the same network who knows the URL can open it.

- [ ] **Step 6: Verify and commit**

```bash
python -m unittest tests.test_runtime_dependencies -v
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/build.yml")'
rg -n --pcre2 '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' README.md
rg -n 'share=True|public website|public tunnel|Hugging Face'   README.md app.py web_launcher.py
git diff --check
```

Expected: tests/YAML pass; both searches print nothing.

Commit `build: package LAN mobile website`.

## Task 6: Full and physical acceptance

**Files:**
- Create locally, do not commit: `acceptance/lan-mobile-<run-id>/`
- Modify product files only after reproducing any acceptance failure in a test.

- [ ] **Step 1: Run the complete gate**

```bash
python -m unittest discover -s tests -q
python -m py_compile app.py app_standalone.py analysis_service.py   batch_service.py grid_detector.py lan_access.py web_controller.py web_launcher.py
git diff --check
git status --short
```

Expected: all pass and the worktree is clean.

- [ ] **Step 2: Real phone acceptance**

On the same private Wi-Fi, scan QR, load the page, verify warning, choose a gallery photo, capture a rear-camera photo, analyze while the PC page remains open, download ZIP, and confirm durable artifacts exist only under the Windows results root. Changing Wi-Fi should require restart; closing launcher should stop both clients.

- [ ] **Step 3: Physical fixture acceptance**

Use the user's original straight-on photograph, inset 5, threshold 60. Assert nine rows/crops and CSV/JSON/overlay/heatmap/ZIP. Verify every final box is strictly inside its detected box and visually inspect all crops for black fixture border. If the attachment expired, request it again; do not substitute CAD or synthetic data.

- [ ] **Step 4: CI and release ZIP**

Push a ready PR without assistant attribution, wait for all jobs, download the Windows archive, and run:

```bash
unzip -t CoagulationAnalysis-Windows.zip
shasum -a 256 CoagulationAnalysis-Windows.zip
unzip -l CoagulationAnalysis-Windows.zip |
  rg 'Web/StartWebsite\.exe|Desktop/CoagulationAnalysis\.exe'
```

- [ ] **Step 5: Deliver**

Report release URL, local ZIP path, size, SHA-256, Actions link, and concise PC/phone startup instructions.

