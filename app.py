"""Local Gradio interface for fixed nine-grid coagulation analysis."""

from __future__ import annotations

from base64 import b64encode, b85decode
from html import escape
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
import zlib

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import qrcode
from gradio import route_utils as gradio_route_utils
from gradio import routes as gradio_routes
from jinja2 import ChoiceLoader, DictLoader

from lan_access import LanAccessInfo
from web_controller import open_result_folder, run_batch_analysis, run_single_analysis


_SYSTEM_FONT_STACK = (
    "ui-sans-serif",
    "system-ui",
    "-apple-system",
    "BlinkMacSystemFont",
    "Segoe UI",
    "sans-serif",
)
_SYSTEM_MONO_STACK = (
    "ui-monospace",
    "SFMono-Regular",
    "Consolas",
    "monospace",
)
_LOCAL_THEME = gr.themes.Base(
    font=_SYSTEM_FONT_STACK,
    font_mono=_SYSTEM_MONO_STACK,
).set(
    checkbox_check="none",
    radio_circle="none",
)
_TEMPLATE_NAMES = ("frontend/index.html", "frontend/share.html")
_EXTERNAL_HEAD_TAG = re.compile(
    r"""
    (?is)
    [ \t]*<(?:link|meta)\b
    (?=[^>]*\b(?:href|content)\s*=\s*["']https?://)
    [^>]*>\s*
    |
    [ \t]*<script\b
    (?=[^>]*\bsrc\s*=\s*["']https?://)
    [^>]*>\s*</script>\s*
    """,
    re.VERBOSE,
)
_LOCAL_ASSET_TAG = re.compile(
    r"""(?is)<(?:script|link)\b(?=[^>]*(?:src|href)=["']\./assets/)[^>]*>"""
    r"(?:\s*</script>)?"
)
_ENGLISH_BOOTSTRAP = """\
<script data-offline-language>
document.documentElement.lang = "en";
window.gradio_config.root = window.location.origin;
Object.defineProperty(window.navigator, "language", {
  configurable: true,
  get: () => "en-US"
});
Object.defineProperty(window.navigator, "languages", {
  configurable: true,
  get: () => ["en-US", "en"]
});
</script>
"""
_MOBILE_CAPTURE_BOOTSTRAP = """\
<script data-mobile-capture>
function configureCameraInput() {
  const input = document.querySelector('#camera-source input[type="file"]');
  if (!input) return false;
  input.setAttribute('accept', 'image/*');
  input.setAttribute('capture', 'environment');
  return true;
}
if (!configureCameraInput()) {
  const observer = new MutationObserver(() => {
    if (configureCameraInput()) observer.disconnect();
  });
  observer.observe(document.documentElement, {childList: true, subtree: true});
  window.setTimeout(() => observer.disconnect(), 15000);
}
</script>
"""
_MOBILE_LAYOUT_STYLE = """\
<style data-mobile-layout>
@media (max-width: 720px) {
  html,
  body {
    max-width: 100%;
    overflow-x: hidden;
  }

  .gradio-container {
    box-sizing: border-box !important;
    width: 100vw !important;
    max-width: 100vw !important;
    padding: 12px !important;
  }

  .gradio-container,
  .gradio-container .main,
  .gradio-container .wrap,
  .gradio-container .contain,
  .gradio-container .column,
  .gradio-container .row,
  .gradio-container .tabs,
  .gradio-container .tabitem {
    box-sizing: border-box !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
  }

  .gradio-container .main {
    padding: 0 !important;
  }

  .gradio-container .row {
    flex-direction: column;
  }
}
</style>
"""
_TEMPLATE_LOCK = threading.RLock()
_TEMPLATE_LOADER_MARKER = "_coagulation_offline_templates"
_CONFIG_PATCH_LOCK = threading.RLock()
_CONFIG_PATCH_MARKER = "_coagulation_offline_app_ids"
_EMBEDDED_CSS = """:root {
  --navy: #182a45;
  --blue: #3f78b5;
  --red: #a42e3d;
  --surface: #ffffff;
  --background: #f5f7fa;
}

body {
  background: var(--background);
}

.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto;
  padding: 24px !important;
  color: var(--navy);
  background: var(--background);
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  color: var(--navy);
  letter-spacing: -0.015em;
}

#application-title h1 {
  color: var(--navy) !important;
}

#lan-connection-panel {
  border: 1px solid #b8c7d9;
  border-radius: 12px;
  background: #f6f9fc;
  padding: 14px;
}

#lan-qr {
  width: 168px;
  height: 168px;
  image-rendering: pixelated;
}

.gradio-container .block,
.gradio-container .form {
  border-color: rgba(24, 42, 69, 0.16);
  border-radius: 10px;
  background: var(--surface);
}

.gradio-container .gap {
  gap: 18px;
}

.gradio-container button.primary {
  background: var(--red) !important;
  border-color: var(--red) !important;
  color: var(--surface) !important;
  font-weight: 650;
}

.gradio-container button.primary:hover {
  background: var(--navy) !important;
  border-color: var(--navy) !important;
}

body .gradio-container div.status-field.block {
  border-left: 4px solid var(--blue) !important;
}

.gradio-container .status-field textarea,
.gradio-container .status-field input {
  color: var(--navy);
  font-weight: 600;
}

@media (max-width: 720px) {
  .gradio-container {
    box-sizing: border-box !important;
    width: 100vw !important;
    padding: 12px !important;
  }

  .gradio-container,
  .gradio-container .main,
  .gradio-container .wrap,
  .gradio-container .contain,
  .gradio-container .column,
  .gradio-container .row,
  .gradio-container .tabs,
  .gradio-container .tabitem {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
  }

  .gradio-container .row {
    flex-direction: column;
  }

  #lan-connection-panel,
  #lan-connection-panel section,
  .gradio-container img {
    max-width: 100%;
  }

  .gradio-container .table-wrap {
    max-width: 100%;
    overflow-x: auto;
  }
}
"""
_EMBEDDED_CSS = zlib.decompress(
    b85decode(
        "c-pmENsr^W6~5Q6V5%@cIgt$Qmfgr8$RWTSavJ0iG;*Ls%4SNI6iTio!Tk3;l3FN{vR%{NXrQ`^>s!CKPzPSu9s1t`Lav+R>J_&K^i*WUjuv0!V0+;O&Z!tJ3R0xAQMav&PE3UqlYBmPzI2qp9HlfQk(y)G3ycJL5c^b(46AlgvX6-xD_JG(qcPd%do_gfoL1tdj6_)&>Vxj7ex)3y_C=Tnv-Kl+J3>h^kOxz(;4%(&Q9N*Y!lZsXAP@Oz^Nu3%gC8^eyh|Mv#|{d82YFjB-RSH4^F1V%2hqx&q~F6aVVBl@FApey4YR@W2!q~Y@C<_SBQf~f!^5dNm(D|0lMkt_4CgOLUSBF$^NRWNpuG+H13Nc0?{L-0ePW06usw=1Yd5c}4ug2PJS4Bnnh*OyH}lO16|mYZ@R^k#2lPwTQSP7%b6Z@sZc8~U9OUArDZvgt+K!$b^jFZ`%inSStNi<Ku-}1m5PJNT9&3vJ@ynxwrtM2w?Nny<yW6t=vI=CB)tpf7!q9@$H%7YC0i9X(?0eo%4)Vhk<hkVNjNjdUsajyn^CPQ-O7W$u7n2PpL{bq1Lf=B^Nn{+MYo_DCYlAu1;%#2HN8Ct;;s1lQ!<B_6-=@%0U>yyuqtRXZFKGulcY%Xk^6i2(hcP`%x}F+dlS|&Y8CH5CtZhmRmQs{-DHL3?W92fi3Gh7Onv{#-e_z^;6(5o^>EGfi>SnZiqqLF=VU>HL?0AC7xc3^RBrt-mtIva^H1&9BaGDv7KLdrYCZ*p*-e5~$a!OelXq>oZoxfa~Pu-=I?hQ&;=K!ZK^HW$Q(9u0|Mzs6GNQq#O)bKa#0eX$QdM{Ei1GoBWb#D$D(3>bDtmMn!ls#yw84VIQBY1${0+jo)x<C%S^)#lQr?Xn&GnK6B`}S-5?6w&Vwz;L%E8}%_7C|7x<VdYjyhkOwg>eb5_X9Pk<Op=aSX-Bj^r8@lZ!Ch|@AorFtJN|DZkFPfkt7gdB#1-!5v3xS>{3Ivfkj5ULAItGn3P0sH@msn>~yw9s$*f>He2XJsbLSE{5OWfGypXA69?^M!Ga%bly9CW2Ta18NQy)dFG{if#0W8+KZpaX_YYBXR8S<qIZD<v^h6^P=e~tc#nc$34UGt`V(v;X1$jREVL<570e|r!Rf&*@I!{GH3H2c^h(91H+U0>nhpSFw2*d(agU5^iU9eOb?I<8qc{x|D415sVo;?utHZ1sN7zl)Xq~*ZiEnAOzBsEY?QC2q}NP*fECcV)BXEv?!FT+;-`~4XVB@y0?P)FZ8S4PsJ=AiLQ(@>t{mM&>K`8{!Cj8FESZZti?iAIGYEHCl7c@9${s`&NIK?$tcm~NFQQU!JlcIj>R=^7p-BOXme{w#t>HP>w>;>998i6ieSR#@P5Mwh=$w5wlPd%@+WEQ-^H;_ex+$L~!?+Q{5QrhF?12a)v-T2hf4i!^HHo&&L_XG*Z})4hXs02HxIp8dpzqd_0uZsTPf7>>uPmUhAn;9_sKZTJDsB6n#v$LkJ+^DG>GV~SN_YE!qYV|67&2cl%Jvc$n{Lm>g-Fi-=xUsj-Q8Il8&Lj!-JSm6&XvNVn0myE_in5}Z$7dJ}I{?0)WYzlUQ;lRe#;ed;-Pr@hWa0coEGZeSTlLa>abYI-iq%H3bq_pFBqYS;zvK}859^wj@A1!O$4({wr0cAAa@5{LtCrXq(6pXi>o1a)o^w3zaZpizVg@dOUn_)_7)B&T#$7YFeEpM={oebl8!7&78tpwlPkV1BWKw5p}bxHsH2qTaGQYZ>8JTM$bix?~NR2$E$`dKhx7G4&VvlXK~;DyHiIZnskr<wkWIm1wvmf)?Fz#zZSUKBIhdf9GnD#21u&66Zy7yg(f34aK|7ei0tUfSmbEyx?{>$jS}$l}9!c*<~>pY)OK_`9qNB+auhFY;Ws?J(~~7!Q^z5Z*tXb2TlS*TLV-v}gHOdEDG<=GUZbD@_EatZFXZ#N~u;Q1In19}@j*QII7=LB?r&DYY${e;CHLZNZ4(I0(jANXpEaet+9wdGskXPLJ%~(fiKshVLY|*;yum-{Q)q2`$o?*g{U%0U7N;k27eewFummdJLj9eD3GhELlz8wq`-sr73Gn+=9{azM97Omu1RUfC9noi9)54XIZqzpKzKpx#l!aMP8)EQd`p2dgJfUmI?&Z_s%V%kY_vEG5WCV?2Sp1He-t8Ae*#W40mni!d2pnVh2L85U(;c`K!oa|M~3tG7~ja@+V}fNH>zH`!ZUXxV{1V0{E!p&IvNcSeX1U5WJ;NT*NeAcMnNCDse_jTN6-DOx4I;&ENqg5;MY@7f6u(+I=(-V?jH|A)fZld3(VqPHH#<y?nr+u*(r{2e*mRmA-AS#^`Q^tk|8Ii(iAdyEB5d+w`PhFFMP?n5ECFq2L;jMmKOhz2^BgFTU#b6V-;_&$V}#=c_lIdnLHCdL~EFYCu6U3Ni*kCXiZ1N|4d!vq-Yws;F~tP^gqxR*Do1LS$W$D+>(oP$}qqor)TlPQ8GIS4(4)hkm157lkUzEGgcS&Q-<Ou<8~>+?}E=o0YalD`a^gfgH92X-W9r%C3nC;=y|74--F6?ar>c714LXhgHj!DX6IwjV;;kg47Q?i*bpjcOmsXspM@uf9f;dBQ8tIKb(0AyajJK(n&X34hH2^Z^`%C)%VS5OxleI5>BCBXNsMVI^BA?sxQ`AwDbkctk2xKmH)YK95=U)ao>k7nt9Ab$F-E&vQOm6@Y6IWYra00cqHEXuLL5qxHo&d-hI7YGwr&O)UWqK?4CEj;7Z*#y*Ank=kXX0@z5y0+m2Hx*=MhO*N@m146LK!^p(*!iz_{mJjYpl1DE{!#bY*!N?Zx=u6l_;mPkvhJeIGoSK2}v7L$D*WAbAf0(M>7IXVS~S$Ru?9$w<*3IE}Gm(Y)s`4ieWpzslJ7dVn~Y1q4H3sWM``pcOThS9U;`1?eDupmxfyp3tn9bR_fh}p8~#ps?VBJn`v4{swwM<~@!wzZUy>LU@xl>`q+9lcPCb^p(3+b>opX;`<|ZeXD~zt(vf?p`!^=ItHav0N>+YhgEO!TveQw-G<NNzgwes|QO9AI1k3i?1O@^;r;V62{^~g6Wfiar+n)$C2f;wRzt7T5t50rj_<LcufSN6SW`iTtpC?jhiy?7D5e%53#I#1i@|JG2i4AGsY{UDfZC9O8>B!h=)Z#5HE{DZ5!LU+QXpH>|U(bUh_1HtGc2Sd-ptbSiVkKJpS5GTX`~20MS2k&HZ$tCsE~E@&t94+46=#qv>s6jAQlKEre4!V}$5W!I^okT72VDa|Wa(vVxV_sigVA5NYZ3>)2%Wh|TO`XIG!Cd9whnHN3MV?Tc?fb~9+3)gzNyRgSfGi7g9o)+SJ&5Zcu>lnd7MeHLFJlPf51AsqlfrvS(G@K7xNUW+z=>`!lV@U+s+=n20vk^Wh}P1M^tF1g6(=1OMT{<$>-@!P}y0gVv(2L"
    )
).decode("utf-8").replace(
    "#product-bar > div", "#product-bar .prose"
).replace(
    "#hero-card > div", "#hero-card .prose"
).replace(
    "@media (max-width: 760px) {\n  .gradio-container {",
    "@media (max-width: 760px) {\n"
    "  body > gradio-app .gradio-container {\n"
    "    padding: 12px 12px 42px !important;\n"
    "  }\n\n"
    "  body > gradio-app .gradio-container .main {\n"
    "    padding: 0 !important;\n"
    "  }\n\n"
    "  .gradio-container {",
).replace(
    "body > gradio-app .gradio-container",
    ".gradio-container.gradio-container",
).replace(
    ".gradio-container.gradio-container .main",
    ".gradio-container .main.main",
).replace(
    "padding: 22px 28px 64px !important;",
    "padding: 22px clamp(12px, 2vw, 28px) 64px !important;",
)


def _offline_example_inputs():
    return None


def _load_local_styles() -> str:
    candidates = [Path(__file__).with_name("web_styles.css")]
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        packaged_path = Path(frozen_root) / "web_styles.css"
        if packaged_path not in candidates:
            candidates.append(packaged_path)

    for path in candidates:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
    return _EMBEDDED_CSS


def _strip_external_head_tags(source: str) -> str:
    return _EXTERNAL_HEAD_TAG.sub("", source)


def _offline_config_json(value):
    config = dict(value)
    config["root"] = ""
    return gradio_routes.toorjson(config)


def _offline_api_json(value):
    def without_code_snippets(item):
        if isinstance(item, dict):
            return {
                key: without_code_snippets(child)
                for key, child in item.items()
                if key != "code_snippets"
            }
        if isinstance(item, list):
            return [without_code_snippets(child) for child in item]
        return item

    return gradio_routes.toorjson(without_code_snippets(value))


def _register_offline_config(app_id: int) -> None:
    """Keep Gradio 6.16's live config local without embedding its origin."""
    with _CONFIG_PATCH_LOCK:
        update_root = gradio_route_utils.update_root_in_config
        offline_app_ids = getattr(update_root, _CONFIG_PATCH_MARKER, None)
        if offline_app_ids is None:
            offline_app_ids = set()
            original_update_root = update_root

            def update_offline_root(config, root):
                updated = original_update_root(config, root)
                if updated.get("app_id") not in offline_app_ids:
                    return updated

                def make_local(item):
                    if isinstance(item, dict):
                        return {key: make_local(value) for key, value in item.items()}
                    if isinstance(item, list):
                        return [make_local(value) for value in item]
                    if isinstance(item, str) and root and item.startswith(root):
                        return item[len(root) :]
                    return item

                local_config = make_local(updated)
                local_config["root"] = ""
                return local_config

            setattr(update_offline_root, _CONFIG_PATCH_MARKER, offline_app_ids)
            gradio_route_utils.update_root_in_config = update_offline_root

        offline_app_ids.add(app_id)


def _insert_before_module(source: str, content: str) -> str:
    module = re.search(
        r"""(?is)<script\b(?=[^>]*\btype=["']module["'])""",
        source,
    )
    if module is None:
        raise RuntimeError("The packaged Gradio template has no module entry point.")
    return source[: module.start()] + content + source[module.start() :]


def _install_offline_templates() -> None:
    environment = gradio_routes.templates.env
    with _TEMPLATE_LOCK:
        current_loader = environment.loader
        if getattr(current_loader, _TEMPLATE_LOADER_MARKER, False):
            return
        if current_loader is None:
            raise RuntimeError("The packaged Gradio template loader is unavailable.")

        packaged = {
            name: current_loader.get_source(environment, name)[0]
            for name in _TEMPLATE_NAMES
        }
        local_assets = "\n".join(_LOCAL_ASSET_TAG.findall(packaged[_TEMPLATE_NAMES[0]]))
        if not local_assets:
            raise RuntimeError("The packaged Gradio local assets are unavailable.")

        overrides: dict[str, str] = {}
        for name, source in packaged.items():
            offline_source = _strip_external_head_tags(source)
            offline_source = offline_source.replace(
                "{{ config | toorjson }}",
                "{{ config | toofflinejson }}",
            )
            offline_source = offline_source.replace(
                "{{ gradio_api_info | toorjson }}",
                "{{ gradio_api_info | toofflineapi }}",
            )
            if not _LOCAL_ASSET_TAG.search(offline_source):
                offline_source = offline_source.replace(
                    "</head>",
                    f"{local_assets}\n</head>",
                    1,
                )
            offline_source = _insert_before_module(
                offline_source,
                _ENGLISH_BOOTSTRAP
                + _MOBILE_LAYOUT_STYLE
                + _MOBILE_CAPTURE_BOOTSTRAP,
            )
            overrides[name] = offline_source

        environment.filters["toofflinejson"] = _offline_config_json
        environment.filters["toofflineapi"] = _offline_api_json
        offline_loader = ChoiceLoader([DictLoader(overrides), current_loader])
        setattr(offline_loader, _TEMPLATE_LOADER_MARKER, True)
        environment.loader = offline_loader
        environment.cache.clear()


def _cleanup_hosted_results(
    root: str | Path,
    max_age_seconds: int = 24 * 60 * 60,
    now: float | None = None,
) -> None:
    """Remove only expired, published result bundles from hosted storage."""
    cutoff = (time.time() if now is None else now) - max_age_seconds
    try:
        entries = list(Path(root).iterdir())
    except OSError:
        return
    for entry in entries:
        published_name = entry.name.startswith("batch_") or entry.name.endswith(
            "_analysis"
        )
        if entry.name.startswith(".") or not published_name:
            continue
        try:
            if (
                entry.is_dir()
                and not entry.is_symlink()
                and entry.stat().st_mtime < cutoff
            ):
                shutil.rmtree(entry)
        except OSError:
            continue


def _single_values(path, inset, threshold, root, hosted=False):
    if hosted:
        _cleanup_hosted_results(root)
    response = run_single_analysis(path, inset, threshold, root)
    return (
        response.crops,
        response.overlay_path,
        response.heatmap_path,
        response.rows,
        response.csv_path,
        response.zip_path,
        response.output_dir,
        response.status,
    )


def _batch_values(paths, inset, threshold, root, hosted=False):
    if hosted:
        _cleanup_hosted_results(root)
    response = run_batch_analysis(paths or [], inset, threshold, root)
    return (
        response.rows,
        response.summary_csv,
        response.failures_csv,
        response.zip_path,
        response.batch_dir,
        response.status,
    )


def _single_source(gallery_path, camera_path):
    selected = [path for path in (gallery_path, camera_path) if path]
    if not selected:
        raise ValueError("Choose or take one image before analysis.")
    if len(selected) != 1:
        raise ValueError("Use only one image source at a time.")
    return selected[0]


def _build_single_tab(root: Path, hosted: bool = False) -> None:
    with gr.Row(elem_id="single-workspace", elem_classes="workspace-grid"):
        with gr.Column(
            scale=2,
            elem_id="single-input-card",
            elem_classes=["workspace-card", "input-card"],
        ):
            gr.Markdown(
                "### Upload sample\nChoose one image from your device or take a photo.",
                elem_classes="section-heading",
            )
            gallery_source = gr.File(
                label="Choose from gallery or files",
                file_count="single",
                type="filepath",
                elem_id="gallery-source",
                elem_classes="upload-field",
            )
            camera_source = gr.File(
                label="Take photo",
                file_count="single",
                type="filepath",
                elem_id="camera-source",
                elem_classes="upload-field",
            )
            gr.Markdown(
                "### Analysis settings\nFine-tune only when your workflow needs it.",
                elem_classes="section-heading settings-heading",
            )
            with gr.Group(elem_classes="settings-group"):
                inset = gr.Slider(
                    0,
                    15,
                    value=5,
                    step=0.5,
                    label="Inner crop inset",
                )
                threshold = gr.Slider(
                    0,
                    255,
                    value=60,
                    step=1,
                    label="No-clot threshold",
                )
            analyze = gr.Button(
                "Analyze Image",
                variant="primary",
                elem_classes="primary-action",
            )
            status = gr.Textbox(
                label="Status",
                interactive=False,
                elem_classes="status-field",
            )
        with gr.Column(
            scale=3,
            elem_id="single-results-card",
            elem_classes=["workspace-card", "result-card"],
        ):
            gr.Markdown(
                "### Detected cells\nThe final 3×3 inner crops appear here.",
                elem_classes="section-heading",
            )
            crops = gr.Gallery(
                label="Final inner crops",
                columns=3,
                rows=3,
                elem_id="result-gallery",
            )
            gr.Markdown(
                "### Visual results\nReview the detected boundaries and heatmap.",
                elem_classes="section-heading results-heading",
            )
            with gr.Row():
                overlay = gr.Image(
                    label="Detected and final boundaries",
                    type="filepath",
                )
                heatmap = gr.Image(
                    label="Publication heatmap",
                    type="filepath",
                )
            gr.Markdown(
                "### Measurements\nCell-level values and confidence scores.",
                elem_classes="section-heading results-heading",
            )
            table = gr.Dataframe(
                headers=[
                    "Cell",
                    "Row",
                    "Column",
                    "Mean",
                    "Confidence",
                    "Recovered",
                ],
                interactive=False,
                label="Per-cell results",
                elem_classes="results-table",
            )
            with gr.Group(elem_classes="download-card"):
                gr.Markdown(
                    "### Download results\nKeep the measurements or the complete result bundle.",
                    elem_classes="section-heading",
                )
                with gr.Row(elem_classes="download-row"):
                    csv_file = gr.File(label="Download CSV", interactive=False)
                    zip_file = gr.File(
                        label="Download result ZIP", interactive=False
                    )
                result_dir = (
                    gr.State()
                    if hosted
                    else gr.Textbox(
                        label="Saved result folder", interactive=False
                    )
                )
                if not hosted:
                    open_folder = gr.Button(
                        "Open folder on Windows PC",
                        elem_classes="secondary-action",
                    )

    gallery_source.change(lambda: None, outputs=camera_source)
    camera_source.change(lambda: None, outputs=gallery_source)

    analyze.click(
        lambda gallery, camera, value, cutoff: _single_values(
            _single_source(gallery, camera), value, cutoff, root, hosted
        ),
        [gallery_source, camera_source, inset, threshold],
        [crops, overlay, heatmap, table, csv_file, zip_file, result_dir, status],
        api_name="analyze_single",
    )
    if not hosted:
        open_folder.click(
            lambda path: open_result_folder(path, root),
            result_dir,
            status,
        )



def _build_batch_tab(root: Path, hosted: bool = False) -> None:
    with gr.Row(elem_id="batch-workspace", elem_classes="workspace-grid"):
        with gr.Column(
            scale=2,
            elem_id="batch-input-card",
            elem_classes=["workspace-card", "input-card"],
        ):
            gr.Markdown(
                "### Upload samples\nChoose multiple complete 3×3 fixture images.",
                elem_classes="section-heading",
            )
            sources = gr.File(
                label="Complete 3×3 fixture images",
                file_count="multiple",
                type="filepath",
                elem_classes="upload-field",
            )
            gr.Markdown(
                "### Analysis settings\nThe same settings apply to every image.",
                elem_classes="section-heading settings-heading",
            )
            with gr.Group(elem_classes="settings-group"):
                inset = gr.Slider(
                    0,
                    15,
                    value=5,
                    step=0.5,
                    label="Inner crop inset",
                )
                threshold = gr.Slider(
                    0,
                    255,
                    value=60,
                    step=1,
                    label="No-clot threshold",
                )
            analyze = gr.Button(
                "Analyze Batch",
                variant="primary",
                elem_classes="primary-action",
            )
            status = gr.Textbox(
                label="Batch status",
                interactive=False,
                elem_classes="status-field",
            )
        with gr.Column(
            scale=3,
            elem_id="batch-results-card",
            elem_classes=["workspace-card", "result-card"],
        ):
            gr.Markdown(
                "### Measurements\nProgress and results for every uploaded image.",
                elem_classes="section-heading",
            )
            table = gr.Dataframe(
                headers=["Image", "Cells", "Status", "Reason", "Result"],
                interactive=False,
                label="Batch results",
                elem_classes="results-table",
            )
            with gr.Group(elem_classes="download-card"):
                gr.Markdown(
                    "### Download results\nExport summaries, failures, or the full batch.",
                    elem_classes="section-heading",
                )
                with gr.Row(elem_classes="download-row"):
                    summary = gr.File(
                        label="Batch summary CSV", interactive=False
                    )
                    failures = gr.File(
                        label="Failure report CSV", interactive=False
                    )
                    archive = gr.File(
                        label="Download batch ZIP", interactive=False
                    )
                batch_dir = (
                    gr.State()
                    if hosted
                    else gr.Textbox(
                        label="Saved batch folder", interactive=False
                    )
                )
                if not hosted:
                    open_folder = gr.Button(
                        "Open batch folder on Windows PC",
                        elem_classes="secondary-action",
                    )

    analyze.click(
        lambda paths, value, cutoff: _batch_values(
            paths, value, cutoff, root, hosted
        ),
        [sources, inset, threshold],
        [table, summary, failures, archive, batch_dir, status],
    )
    if not hosted:
        open_folder.click(
            lambda path: open_result_folder(path, root),
            batch_dir,
            status,
        )


def _qr_data_url(value: str) -> str:
    image = qrcode.make(value)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")


def _connection_panel(access: LanAccessInfo) -> str:
    if access.preferred_url is None:
        return (
            "<section><h2>Phone access unavailable</h2>"
            "<p>Connect this PC and phone to the same Wi-Fi, then restart "
            "StartWebsite.exe.</p></section>"
        )
    links = "".join(
        f'<li><code>{escape(url)}</code></li>' for url in access.phone_urls
    )
    return (
        "<section><h2>Open on your phone</h2>"
        f'<img id="lan-qr" src="{_qr_data_url(access.preferred_url)}" '
        'alt="Phone connection QR code">'
        f"<ul>{links}</ul>"
        "<p><strong>No password is enabled.</strong> Use only on a trusted "
        "private Wi-Fi network. Restart after changing Wi-Fi.</p></section>"
    )


def create_app(
    results_root: str | Path,
    lan_access: LanAccessInfo | None = None,
    hosted: bool = False,
) -> gr.Blocks:
    """Build the analysis application without starting its server."""
    _install_offline_templates()
    root = Path(results_root)
    css = _load_local_styles()
    with gr.Blocks(
        title="Coagulation Analysis",
        analytics_enabled=False,
        delete_cache=(3600, 86400) if hosted else None,
    ) as application:
        workspace_label = "Online workspace" if hosted else "Local workspace"
        gr.HTML(
            "<div class='brand-lockup'><span class='brand-mark'>C</span>"
            "<span class='brand-name'>Coagula</span></div>"
            f"<span class='environment-badge'><i></i>{workspace_label}</span>",
            elem_id="product-bar",
        )
        gr.HTML(
            "<div class='hero-copy'><p>Image analysis</p>"
            "<h1>Clear results. Confident decisions.</h1>"
            "<span>Upload a 3×3 sample image and get precise, "
            "cell-level measurements in seconds.</span></div>"
            "<div class='hero-visual' aria-hidden='true'>"
            "<i></i><i></i><i></i><i></i><i></i><i></i>"
            "<i></i><i></i><i></i></div>",
            elem_id="hero-card",
        )
        if lan_access is not None:
            gr.HTML(_connection_panel(lan_access), elem_id="lan-connection-panel")
        with gr.Tabs(elem_id="analysis-tabs"):
            with gr.Tab("Single image"):
                _build_single_tab(root, hosted=hosted)
            with gr.Tab("Batch processing"):
                _build_batch_tab(root, hosted=hosted)
    # Gradio 6 applies CSS at launch. Retaining it on both attributes keeps the
    # factory inspectable and lets any caller launch the returned Blocks object
    # without separately knowing the stylesheet path.
    application.css = css
    application._deprecated_css = css
    application.css_paths = []
    application.head_paths = []
    application.theme = _LOCAL_THEME
    application._deprecated_theme = _LOCAL_THEME
    application._set_html_css_theme_variables()
    application.footer_links = []
    for block in application.blocks.values():
        if hasattr(block, "example_inputs"):
            block.example_inputs = _offline_example_inputs
    application.config = application.get_config_file()
    _register_offline_config(application.config["app_id"])
    return application


def main() -> None:
    """Launch locally or with Hugging Face Space network settings."""
    if os.environ.get("SPACE_ID"):
        results_root = Path(tempfile.gettempdir()) / "coagulation-hugging-face-results"
        results_root.mkdir(parents=True, exist_ok=True)
        create_app(results_root, hosted=True).launch(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", "7860")),
            share=False,
            inbrowser=False,
            allowed_paths=[str(results_root)],
            footer_links=[],
        )
        return

    create_app(Path.cwd() / "results").launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
