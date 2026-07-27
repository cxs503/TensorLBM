#!/usr/bin/env python3
"""
Playwright browser test for TensorLBM SUBOFF 3D preview + parametric mode + flow field evolution.
Uses system Firefox via channel.
"""
import asyncio
import json
import os
import sys
from playwright.async_api import async_playwright

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "test_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

URL = "http://localhost:8004/"
RESULTS = {"errors": [], "findings": [], "param_values": {}, "canvas_checks": {}}


def log(msg):
    print(f"[TEST] {msg}")
    RESULTS["findings"].append(msg)


async def shot(page, name):
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    await page.screenshot(path=path, full_page=False)
    print(f"[SHOT] saved {path}")
    return path


async def main():
    async with async_playwright() as p:
        # Try system Firefox via channel
        try:
            browser = await p.firefox.launch(
                headless=True,
                channel="firefox",
                args=["--width=1600", "--height=1000"],
            )
            log("Launched system Firefox via channel")
        except Exception as e:
            log(f"Channel firefox failed: {e}, trying executable_path")
            browser = await p.firefox.launch(
                headless=True,
                executable_path="/usr/bin/firefox",
                args=["--width=1600", "--height=1000"],
            )
            log("Launched system Firefox via executable_path")

        context = await browser.new_context(
            viewport={"width": 1600, "height": 1000},
            ignore_https_errors=True,
        )
        page = await context.new_page()

        # Collect console errors
        console_msgs = []
        page.on("console", lambda msg: console_msgs.append({"type": msg.type, "text": msg.text}) if msg.type in ("error", "warning") else None)
        page.on("pageerror", lambda err: console_msgs.append({"type": "pageerror", "text": str(err)}))

        # Step 1: Navigate
        log("Step 1: Navigating to http://localhost:8004/")
        await page.goto(URL, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)

        # Step 2: Snapshot page structure
        log("Step 2: Taking snapshot of page structure (dashboard)")
        await shot(page, "01_dashboard")

        # Get the nav links
        nav_links = await page.query_selector_all("[data-tab]")
        nav_info = []
        for link in nav_links:
            tab = await link.get_attribute("data-tab")
            text = (await link.inner_text()).strip().replace("\n", " ")
            nav_info.append({"tab": tab, "text": text})
        log(f"Nav links found: {json.dumps(nav_info, indent=2)}")

        # Step 3: Click the "Generic CFD" tab
        log("Step 3: Clicking Generic CFD tab")
        generic_tab = await page.query_selector('[data-tab="generic"]')
        if generic_tab:
            await generic_tab.click()
            log("  Clicked [data-tab='generic']")
        else:
            log("  ERROR: Could not find [data-tab='generic'] element")
            RESULTS["errors"].append("Could not find Generic CFD tab")

        await page.wait_for_timeout(1500)

        # Step 4: Screenshot initial state
        log("Step 4: Screenshot of Generic CFD initial state")
        await shot(page, "02_generic_initial")

        # Check what's visible
        stl_radio = await page.query_selector("#generic-geo-stl")
        param_radio = await page.query_selector("#generic-geo-param")
        stl_checked = await stl_radio.is_checked() if stl_radio else None
        param_checked = await param_radio.is_checked() if param_radio else None
        log(f"  STL radio checked: {stl_checked}, Parametric radio checked: {param_checked}")

        # Step 5: Click the "Parametric" radio button
        log("Step 5: Clicking Parametric radio button")
        if param_radio:
            await param_radio.click()
            await page.wait_for_timeout(1000)
            log("  Clicked Parametric radio")
        else:
            log("  ERROR: Parametric radio not found")
            RESULTS["errors"].append("Parametric radio button not found")

        # Step 6: Screenshot parametric mode
        log("Step 6: Screenshot of parametric mode")
        await shot(page, "03_parametric_mode")

        # Check parametric options visible
        param_opts = await page.query_selector("#generic-param-opts")
        param_opts_display = await param_opts.get_attribute("style") if param_opts else None
        log(f"  Parametric opts style: {param_opts_display}")

        stl_section = await page.query_selector("#generic-stl-section")
        stl_section_display = await stl_section.get_attribute("style") if stl_section else None
        log(f"  STL section style: {stl_section_display}")

        # Step 7: Select "suboff" in shape dropdown
        log("Step 7: Selecting 'suboff' in shape dropdown")
        shape_sel = await page.query_selector("#generic-param-shape")
        if shape_sel:
            await shape_sel.select_option("suboff")
            await page.wait_for_timeout(2000)  # Give time for 3D render + defaults
            log("  Selected suboff")
        else:
            log("  ERROR: Shape dropdown not found")
            RESULTS["errors"].append("Shape dropdown not found")

        # Step 8: Screenshot SUBOFF 3D preview
        log("Step 8: Screenshot of SUBOFF 3D preview")
        await shot(page, "04_suboff_preview")

        # Check hull_type wrap visible
        hull_wrap = await page.query_selector("#generic-hull-type-wrap")
        hull_wrap_display = await hull_wrap.get_attribute("style") if hull_wrap else None
        log(f"  Hull type wrap style: {hull_wrap_display}")

        # Check param info text
        param_info = await page.query_selector("#generic-param-info")
        param_info_text = await param_info.inner_text() if param_info else None
        log(f"  Param info text: {param_info_text}")

        # Step 9: Check console for JS errors
        log("Step 9: Checking browser console for errors")
        await page.wait_for_timeout(500)
        error_msgs = [m for m in console_msgs if m["type"] in ("error", "pageerror")]
        warning_msgs = [m for m in console_msgs if m["type"] == "warning"]
        log(f"  Console errors: {len(error_msgs)}")
        for m in error_msgs:
            log(f"    ERROR: {m['text'][:300]}")
        log(f"  Console warnings: {len(warning_msgs)}")
        for m in warning_msgs:
            log(f"    WARN: {m['text'][:300]}")
        RESULTS["console_errors"] = error_msgs
        RESULTS["console_warnings"] = warning_msgs

        # Step 10: Try hull_type options
        hull_sel = await page.query_selector("#generic-hull-type")
        if hull_sel:
            for hull_val in ["bare_hull", "with_sail", "full"]:
                log(f"Step 10: Selecting hull_type='{hull_val}'")
                await hull_sel.select_option(hull_val)
                await page.wait_for_timeout(1500)
                await shot(page, f"05_hull_{hull_val}")
                # Check canvas still has content
                canvas_check = await page.evaluate("""() => {
                    const c = document.getElementById('generic-canvas');
                    if (!c) return {exists: false};
                    return {exists: true, width: c.width, height: c.height, clientW: c.clientWidth, clientH: c.clientHeight};
                }""")
                log(f"  Canvas check for {hull_val}: {json.dumps(canvas_check)}")
                RESULTS["canvas_checks"][hull_val] = canvas_check
        else:
            log("  ERROR: hull_type dropdown not found")
            RESULTS["errors"].append("hull_type dropdown not found")

        # Step 12: Check if 3D preview canvas shows submarine shape
        log("Step 12: Checking 3D preview canvas for submarine shape")
        canvas_info = await page.evaluate("""() => {
            const canvas = document.getElementById('generic-canvas');
            if (!canvas) return {exists: false};
            const wrap = document.getElementById('generic-canvas-wrap');
            const placeholder = document.getElementById('generic-placeholder');
            return {
                exists: true,
                canvasW: canvas.clientWidth,
                canvasH: canvas.clientHeight,
                wrapDisplay: wrap ? wrap.style.display : 'N/A',
                placeholderDisplay: placeholder ? placeholder.style.display : 'N/A',
                tagName: canvas.tagName,
            };
        }""")
        log(f"  Canvas info: {json.dumps(canvas_info, indent=2)}")
        RESULTS["canvas_checks"]["suboff_3d"] = canvas_info

        # Check WebGL renderer info
        webgl_info = await page.evaluate("""() => {
            const canvas = document.getElementById('generic-canvas');
            if (!canvas) return null;
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            if (!gl) return {error: 'No WebGL context'};
            const dbg = gl.getExtension('WEBGL_debug_renderer_info');
            return {
                vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
                renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
                version: gl.getParameter(gl.VERSION),
                drawingBufferWidth: gl.drawingBufferWidth,
                drawingBufferHeight: gl.drawingBufferHeight,
            };
        }""")
        log(f"  WebGL info: {json.dumps(webgl_info, indent=2)}")
        RESULTS["canvas_checks"]["webgl"] = webgl_info

        # Check mesh info
        mesh_info = await page.query_selector("#generic-mesh-info")
        mesh_info_text = await mesh_info.inner_text() if mesh_info else None
        log(f"  Mesh info text: {mesh_info_text}")

        # Step 13: Check pre-filled parameters
        log("Step 13: Checking pre-filled SUBOFF parameters")
        param_ids = {
            "generic-re": "Re",
            "generic-u-in": "u_in",
            "generic-steps": "steps",
            "generic-warmup": "warmup",
            "generic-viscosity": "viscosity",
            "generic-cs": "Cs",
            "generic-density": "density",
            "generic-param-length": "param_length",
            "generic-param-radius": "param_radius",
            "generic-nx": "nx",
            "generic-ny": "ny",
            "generic-nz": "nz",
        }
        expected = {
            "Re": "1000",
            "u_in": "0.06",
            "steps": "10000",
            "warmup": "5000",
        }
        for eid, label in param_ids.items():
            el = await page.query_selector(f"#{eid}")
            if el:
                val = await el.input_value()
                RESULTS["param_values"][label] = val
                match = ""
                if label in expected:
                    match = " [OK]" if str(val) == expected[label] else f" [MISMATCH: expected {expected[label]}]"
                log(f"  {label} = {val}{match}")
            else:
                log(f"  {label} = NOT FOUND")
                RESULTS["param_values"][label] = "NOT_FOUND"

        # Check collision select
        coll_sel = await page.query_selector("#generic-collision")
        if coll_sel:
            coll_val = await coll_sel.evaluate("el => el.value")
            log(f"  collision = {coll_val}")
            RESULTS["param_values"]["collision"] = coll_val

        # Check auto-domain checkbox
        auto_cb = await page.query_selector("#generic-auto-domain")
        if auto_cb:
            auto_checked = await auto_cb.is_checked()
            log(f"  auto-domain checked = {auto_checked}")
            RESULTS["param_values"]["auto_domain"] = auto_checked

        # Step 14: Check flow field evolution canvas
        log("Step 14: Checking flow field evolution canvas")
        evo_canvas = await page.query_selector("#generic-evo-canvas")
        evo_status = await page.query_selector("#generic-evo-status")
        evo_info = await page.query_selector("#generic-evo-info")
        if evo_canvas:
            evo_info_dict = await page.evaluate("""() => {
                const c = document.getElementById('generic-evo-canvas');
                return {
                    exists: true,
                    width: c.width,
                    height: c.height,
                    clientW: c.clientWidth,
                    clientH: c.clientHeight,
                };
            }""")
            log(f"  Evolution canvas: {json.dumps(evo_info_dict)}")
            RESULTS["canvas_checks"]["evo_canvas"] = evo_info_dict
        else:
            log("  ERROR: Flow field evolution canvas not found")
            RESULTS["errors"].append("Flow field evolution canvas not found")

        evo_status_text = await evo_status.inner_text() if evo_status else None
        evo_info_text = await evo_info.inner_text() if evo_info else None
        log(f"  Evo status: '{evo_status_text}'")
        log(f"  Evo info: '{evo_info_text}'")

        # Final full screenshot
        log("Taking final full-page screenshot")
        await shot(page, "06_final_state")

        # Also take a screenshot of just the 3D preview area
        canvas_wrap = await page.query_selector("#generic-canvas-wrap")
        if canvas_wrap:
            await canvas_wrap.screenshot(path=os.path.join(SCREENSHOT_DIR, "07_3d_preview_only.png"))
            log("Saved 3D preview only screenshot")

        # Check for Three.js scene objects via JS evaluation
        three_check = await page.evaluate("""() => {
            const hasThree = typeof THREE !== 'undefined';
            const canvas = document.getElementById('generic-canvas');
            let canvasOk = false;
            if (canvas) {
                const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                canvasOk = !!gl;
                if (gl) {
                    canvasOk = gl.drawingBufferWidth > 0 && gl.drawingBufferHeight > 0;
                }
            }
            return {hasThree, canvasOk};
        }""")
        log(f"  Three.js loaded: {three_check.get('hasThree')}, Canvas WebGL OK: {three_check.get('canvasOk')}")
        RESULTS["canvas_checks"]["three_js"] = three_check

        log(f"\n=== CONSOLE SUMMARY ===")
        log(f"Total error/pageerror messages: {len(error_msgs)}")
        log(f"Total warning messages: {len(warning_msgs)}")

        await browser.close()

    # Write results JSON
    results_path = os.path.join(SCREENSHOT_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\n[RESULTS] Written to {results_path}")
    print(f"\n=== SUMMARY ===")
    print(f"Errors: {len(RESULTS.get('errors', []))}")
    print(f"Console errors: {len(RESULTS.get('console_errors', []))}")
    print(f"Console warnings: {len(RESULTS.get('console_warnings', []))}")


if __name__ == "__main__":
    asyncio.run(main())
