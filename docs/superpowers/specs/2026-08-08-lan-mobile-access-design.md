# LAN Mobile Access Design

## Purpose

Extend the packaged Windows offline website so a Windows computer and a phone
on the same trusted Wi-Fi network can use the analysis interface at the same
time. The Windows computer remains the only analysis host and the only durable
result store. No cloud service or internet connection is required.

## Confirmed user choices

- Access is limited to a phone and Windows computer on the same Wi-Fi network.
- The feature does not require access when the Windows computer is absent.
- No access password is required.
- Startup shows both a scannable QR code and a text URL for the phone.
- The computer continues to open the website automatically.
- Mobile single-image analysis supports both the camera and photo library.
- The existing offline Windows ZIP remains the delivery format.

## Architecture

`StartWebsite.exe` remains the recommended entry point. The launcher binds the
Gradio server to all local IPv4 interfaces so that both loopback and LAN clients
can reach the same application process. The computer browser opens a loopback
URL such as `http://127.0.0.1:7860`; phones use a private LAN URL such as
`http://192.168.1.20:7860`.

The launcher discovers usable private IPv4 addresses without contacting an
external host. It rejects loopback, link-local, multicast, unspecified, and
public addresses. When more than one private address is available, it presents
all candidates and marks the address associated with the default route as the
preferred address when that can be determined locally. A missing LAN address
does not prevent loopback use.

The application receives the preferred mobile URL as display-only connection
metadata. It generates the QR code locally from that URL using a pinned,
packaged dependency. The QR code contains only the LAN URL and no image,
measurement, path, or user data. The application continues to prohibit remote
fonts, scripts, analytics, CDN assets, and Gradio sharing services.

## User interface

The page header includes a compact connection panel containing:

- the preferred phone URL;
- a QR code for that URL;
- any additional valid LAN URLs;
- a notice that the site has no password and must be used only on a trusted
  private network;
- a reminder that changing Wi-Fi can invalidate the URL and requires restart.

Single-image analysis presents two explicit inputs: `Choose from gallery` and
`Take photo`. The gallery input is the normal file picker. The photo input uses
the mobile browser's native image-file capture control with the rear camera;
it does not use `getUserMedia` or live webcam streaming, because phone browsers
normally require HTTPS for those APIs on non-loopback addresses. Both inputs
produce a local temporary filepath and therefore preserve the existing
controller and analysis-service boundary. Selecting one clears the other so
that the source of the next analysis is unambiguous.

Batch analysis continues to use a multiple-file input. It supports selecting
multiple saved images from the phone library or file picker; direct camera
capture is intentionally limited to single-image analysis to avoid accidental
sample grouping.

The layout becomes responsive for narrow phone screens. Controls stack
vertically, the nine crop previews retain their row-major order, tables remain
readable without page-wide horizontal overflow, and download controls remain
available. The folder buttons are relabeled to `Open folder on Windows PC`
because they act on the host computer even when invoked from a phone. Mobile
users retrieve artifacts through the existing CSV and ZIP downloads.

## Data flow and concurrency

Phone images travel only across the current LAN to the Windows process. The
Windows process performs detection, inset cropping, measurement, heatmap
creation, and archive creation. Durable artifacts remain under
`Web/results/`; nothing is stored by a cloud service.

Computer and phone sessions can remain open concurrently and submit work to
the same application. Existing source identity, per-target locking, staging,
and atomic publication rules continue to isolate result bundles. A failure in
one request must clear only that request's outputs and must not remove a
successful result from another client.

## Security and privacy boundary

Binding to a LAN interface intentionally expands access beyond the host. With
authentication disabled by user choice, any device on the same network that
knows the URL can reach the site. The interface and README must state this
plainly. Users must select only Windows Firewall's private-network permission
and must not expose the program on a public network.

The launcher does not create firewall rules, request administrator privileges,
configure port forwarding, publish a public tunnel, or enable `share=True`.
Only the selected results root is served through Gradio's allowed-path
boundary. Existing result-folder containment and no-follow protections remain
in force.

## Failure handling

- No private address: launch loopback normally and show a reconnect-and-restart
  instruction instead of a QR code.
- Multiple adapters: show every valid private IPv4 candidate and a preferred
  candidate when available.
- Address changes after launch: the old phone address may stop working; the UI
  instructs the user to restart the launcher.
- Windows Firewall blocks the connection: retain local access and document how
  to allow the executable on private networks only.
- QR generation fails: keep the text phone URL usable and record a sanitized
  startup diagnostic without preventing analysis.
- Phone disconnects during upload or analysis: the server contains the failed
  request; any already committed bundle remains valid.

## Packaging and dependency policy

The QR generator is an exact runtime dependency in the platform lock files and
is installed with hashes in CI. PyInstaller explicitly includes any required QR
package data. The packaged smoke test launches the website on a private test
interface where available, verifies the loopback page, verifies that the
connection metadata contains no public URL, and performs a synthetic analysis
through the packaged application.

The Windows release layout remains:

```text
CoagulationAnalysis-Windows/
├── Web/
│   ├── StartWebsite.exe
│   └── _internal/
├── Desktop/
│   ├── CoagulationAnalysis.exe
│   └── _internal/
└── README.md
```

## Verification

Automated verification covers:

- private IPv4 filtering, preferred-address selection, multiple adapters, and
  no-network fallback;
- binding on the LAN while opening the computer's loopback URL;
- exact QR payload and entirely local QR generation;
- gallery selection and native file-capture sources for single-image analysis,
  including verification that LAN access does not depend on an HTTPS-only
  webcam API;
- multiple-file batch input and unchanged controller contracts;
- two clients submitting independent analyses without output collision;
- narrow viewport layout and artifact download controls;
- result-root containment and private-network warnings;
- packaged Windows startup, synthetic analysis, shutdown, and absence of
  external assets or public URLs.

Manual acceptance uses a Windows computer and a phone on the same private
Wi-Fi. It verifies QR scanning, gallery selection, native rear-camera capture,
simultaneous computer access, result downloads, Windows-hosted artifacts,
Wi-Fi-change instructions, and clean shutdown when the launcher window closes.

## Non-goals

- Public internet access or use without the Windows host.
- Cloud storage, accounts, authentication, or password management.
- Automatic router, firewall, or port-forwarding configuration.
- Installing the website as a standalone phone application or PWA.
- Direct camera capture into a multi-image batch.
