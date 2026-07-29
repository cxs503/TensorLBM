#!/usr/bin/env python3
"""
Selenium browser test for TensorLBM SUBOFF 3D preview + parametric mode + flow field evolution.
Uses system Firefox + geckodriver on loongarch64.
"""
import json
import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "test_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

URL = "http://localhost:8004/"
RESULTS = {"errors": [], "findings": [], "param_values": {}, "canvas_checks": {}, "console_errors": []}


def log(msg):
    print(f"[TEST] {msg}")
    RESULTS["findings"].append(msg)


def shot(driver, name):
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    driver.save_screenshot(path)
    print(f"[SHOT] saved {path}")
    return path


def main():
    os.environ["DISPLAY"] = ":99"

    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--width=1600")
    opts.add_argument("--height=1000")
    service = Service(executable_path="/root/.cargo/bin/geckodriver")
    driver = webdriver.Firefox(service=service, options=opts)
    driver.set_page_load_timeout(30)
    driver.implicitly_wait(5)

    # Inject console error capture before page load
    # We'll use a CDP-like approach: inject a script that captures errors
    # First, navigate to about:blank and inject error capture
    driver.get("about:blank")
    driver.execute_script("""
        window.__consoleErrors = [];
        window.__consoleWarnings = [];
        const origError = console.error;
        console.error = function(...args) {
            window.__consoleErrors.push(args.join(' '));
            origError.apply(console, args);
        };
        const origWarn = console.warn;
        console.warn = function(...args) {
            window.__consoleWarnings.push(args.join(' '));
            origWarn.apply(console, args);
        };
        window.addEventListener('error', function(e) {
            window.__consoleErrors.push('JS Error: ' + e.message + ' at ' + e.filename + ':' + e.lineno);
        });
        window.addEventListener('unhandledrejection', function(e) {
            window.__consoleErrors.push('Unhandled rejection: ' + (e.reason && e.reason.message || e.reason));
        });
    """)

    # Step 1: Navigate
    log("Step 1: Navigating to http://localhost:8004/")
    driver.get(URL)
    time.sleep(3)  # Wait for page to fully load including JS

    # Step 2: Snapshot page structure
    log("Step 2: Taking snapshot of page structure (dashboard)")
    shot(driver, "01_dashboard")

    # Get the nav links
    nav_links = driver.find_elements(By.CSS_SELECTOR, "[data-tab]")
    nav_info = []
    for link in nav_links:
        tab = link.get_attribute("data-tab")
        text = link.text.strip().replace("\n", " ")
        nav_info.append({"tab": tab, "text": text})
    log(f"Nav links found: {json.dumps(nav_info, indent=2)}")

    # Step 3: Click the "Generic CFD" tab
    log("Step 3: Clicking Generic CFD tab")
    try:
        generic_tab = driver.find_element(By.CSS_SELECTOR, '[data-tab="generic"]')
        generic_tab.click()
        log("  Clicked [data-tab='generic']")
        time.sleep(1.5)
    except Exception as e:
        log(f"  ERROR: Could not find/click Generic CFD tab: {e}")
        RESULTS["errors"].append(f"Could not find Generic CFD tab: {e}")

    # Step 4: Screenshot initial state
    log("Step 4: Screenshot of Generic CFD initial state")
    shot(driver, "02_generic_initial")

    # Check what's visible
    stl_radio = driver.find_element(By.ID, "generic-geo-stl")
    param_radio = driver.find_element(By.ID, "generic-geo-param")
    stl_checked = stl_radio.is_selected()
    param_checked = param_radio.is_selected()
    log(f"  STL radio checked: {stl_checked}, Parametric radio checked: {param_checked}")

    # Step 5: Click the "Parametric" radio button
    log("Step 5: Clicking Parametric radio button")
    try:
        # Click the label for the radio button (Bootstrap btn-check pattern)
        param_label = driver.find_element(By.CSS_SELECTOR, 'label[for="generic-geo-param"]')
        param_label.click()
        time.sleep(1)
        log("  Clicked Parametric radio label")
    except Exception as e:
        log(f"  ERROR: Parametric radio not found: {e}")
        RESULTS["errors"].append(f"Parametric radio button not found: {e}")

    # Step 6: Screenshot parametric mode
    log("Step 6: Screenshot of parametric mode")
    shot(driver, "03_parametric_mode")

    # Check parametric options visible
    param_opts = driver.find_element(By.ID, "generic-param-opts")
    param_opts_display = param_opts.get_attribute("style")
    log(f"  Parametric opts style: {param_opts_display}")

    stl_section = driver.find_element(By.ID, "generic-stl-section")
    stl_section_display = stl_section.get_attribute("style")
    log(f"  STL section style: {stl_section_display}")

    # Step 7: Select "suboff" in shape dropdown
    log("Step 7: Selecting 'suboff' in shape dropdown")
    try:
        shape_sel = Select(driver.find_element(By.ID, "generic-param-shape"))
        shape_sel.select_by_value("suboff")
        time.sleep(2)  # Give time for 3D render + defaults
        log("  Selected suboff")
    except Exception as e:
        log(f"  ERROR: Shape dropdown not found: {e}")
        RESULTS["errors"].append(f"Shape dropdown not found: {e}")

    # Step 8: Screenshot SUBOFF 3D preview
    log("Step 8: Screenshot of SUBOFF 3D preview")
    shot(driver, "04_suboff_preview")

    # Check hull_type wrap visible
    try:
        hull_wrap = driver.find_element(By.ID, "generic-hull-type-wrap")
        hull_wrap_display = hull_wrap.get_attribute("style")
        log(f"  Hull type wrap style: {hull_wrap_display}")
    except:
        log("  Hull type wrap not found")

    # Check param info text
    try:
        param_info = driver.find_element(By.ID, "generic-param-info")
        param_info_text = param_info.text
        log(f"  Param info text: {param_info_text}")
    except:
        log("  Param info not found")

    # Step 9: Check console for JS errors
    log("Step 9: Checking browser console for errors")
    time.sleep(0.5)
    console_errors = driver.execute_script("return window.__consoleErrors || [];")
    console_warnings = driver.execute_script("return window.__consoleWarnings || [];")
    log(f"  Console errors: {len(console_errors)}")
    for m in console_errors:
        log(f"    ERROR: {m[:300]}")
    log(f"  Console warnings: {len(console_warnings)}")
    for m in console_warnings:
        log(f"    WARN: {m[:300]}")
    RESULTS["console_errors"] = console_errors
    RESULTS["console_warnings"] = console_warnings

    # Step 10: Try hull_type options
    try:
        hull_sel = Select(driver.find_element(By.ID, "generic-hull-type"))
        for hull_val in ["bare_hull", "with_sail", "full"]:
            log(f"Step 10: Selecting hull_type='{hull_val}'")
            hull_sel.select_by_value(hull_val)
            time.sleep(1.5)
            shot(driver, f"05_hull_{hull_val}")
            # Check canvas still has content
            canvas_check = driver.execute_script("""
                const c = document.getElementById('generic-canvas');
                if (!c) return {exists: false};
                return {exists: true, width: c.width, height: c.height, clientW: c.clientWidth, clientH: c.clientHeight};
            """)
            log(f"  Canvas check for {hull_val}: {json.dumps(canvas_check)}")
            RESULTS["canvas_checks"][hull_val] = canvas_check
    except Exception as e:
        log(f"  ERROR: hull_type dropdown not found: {e}")
        RESULTS["errors"].append(f"hull_type dropdown not found: {e}")

    # Step 12: Check if 3D preview canvas shows submarine shape
    log("Step 12: Checking 3D preview canvas for submarine shape")
    canvas_info = driver.execute_script("""
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
    """)
    log(f"  Canvas info: {json.dumps(canvas_info, indent=2)}")
    RESULTS["canvas_checks"]["suboff_3d"] = canvas_info

    # Check WebGL renderer info
    webgl_info = driver.execute_script("""
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
    """)
    log(f"  WebGL info: {json.dumps(webgl_info, indent=2)}")
    RESULTS["canvas_checks"]["webgl"] = webgl_info

    # Check mesh info
    try:
        mesh_info = driver.find_element(By.ID, "generic-mesh-info")
        mesh_info_text = mesh_info.text
        log(f"  Mesh info text: {mesh_info_text}")
    except:
        log("  Mesh info not found")

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
        try:
            el = driver.find_element(By.ID, eid)
            val = el.get_attribute("value")
            RESULTS["param_values"][label] = val
            match = ""
            if label in expected:
                match = " [OK]" if str(val) == expected[label] else f" [MISMATCH: expected {expected[label]}]"
            log(f"  {label} = {val}{match}")
        except:
            log(f"  {label} = NOT FOUND")
            RESULTS["param_values"][label] = "NOT_FOUND"

    # Check collision select
    try:
        coll_sel = driver.find_element(By.ID, "generic-collision")
        coll_val = coll_sel.get_attribute("value")
        log(f"  collision = {coll_val}")
        RESULTS["param_values"]["collision"] = coll_val
    except:
        log("  collision = NOT FOUND")

    # Check auto-domain checkbox
    try:
        auto_cb = driver.find_element(By.ID, "generic-auto-domain")
        auto_checked = auto_cb.is_selected()
        log(f"  auto-domain checked = {auto_checked}")
        RESULTS["param_values"]["auto_domain"] = auto_checked
    except:
        log("  auto-domain = NOT FOUND")

    # Step 14: Check flow field evolution canvas
    log("Step 14: Checking flow field evolution canvas")
    try:
        evo_canvas = driver.find_element(By.ID, "generic-evo-canvas")
        evo_info_dict = driver.execute_script("""
            const c = document.getElementById('generic-evo-canvas');
            return {
                exists: true,
                width: c.width,
                height: c.height,
                clientW: c.clientWidth,
                clientH: c.clientHeight,
            };
        """)
        log(f"  Evolution canvas: {json.dumps(evo_info_dict)}")
        RESULTS["canvas_checks"]["evo_canvas"] = evo_info_dict
    except:
        log("  ERROR: Flow field evolution canvas not found")
        RESULTS["errors"].append("Flow field evolution canvas not found")

    try:
        evo_status = driver.find_element(By.ID, "generic-evo-status")
        evo_info = driver.find_element(By.ID, "generic-evo-info")
        log(f"  Evo status: '{evo_status.text}'")
        log(f"  Evo info: '{evo_info.text}'")
    except:
        log("  Evo status/info not found")

    # Final full screenshot
    log("Taking final full-page screenshot")
    shot(driver, "06_final_state")

    # Also take a screenshot of just the 3D preview area
    try:
        canvas_wrap = driver.find_element(By.ID, "generic-canvas-wrap")
        canvas_wrap.screenshot(os.path.join(SCREENSHOT_DIR, "07_3d_preview_only.png"))
        log("Saved 3D preview only screenshot")
    except:
        log("  Could not screenshot 3D preview area")

    # Check for Three.js loaded
    three_check = driver.execute_script("""
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
    """)
    log(f"  Three.js loaded: {three_check.get('hasThree')}, Canvas WebGL OK: {three_check.get('canvasOk')}")
    RESULTS["canvas_checks"]["three_js"] = three_check

    # Check for toast notification (SUBOFF defaults applied)
    toast_text = driver.execute_script("""
        var toasts = document.querySelectorAll('.toast, .toast-body, [class*=\"toast\"]');
        var texts = [];
        toasts.forEach(function(t) { texts.push(t.textContent.trim()); });
        return texts;
    """)
    if toast_text:
        log(f"  Toast messages: {json.dumps(toast_text)}")

    # Final console error check
    final_errors = driver.execute_script("return window.__consoleErrors || [];")
    final_warnings = driver.execute_script("return window.__consoleWarnings || [];")
    log(f"\n=== CONSOLE SUMMARY ===")
    log(f"Total error messages: {len(final_errors)}")
    log(f"Total warning messages: {len(final_warnings)}")
    RESULTS["console_errors"] = final_errors
    RESULTS["console_warnings"] = final_warnings

    driver.quit()

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
    main()
