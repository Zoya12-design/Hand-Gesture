"""GestureX Streamlit dashboard.

Run from the project root:
    py -3.14 -m streamlit run src/dashboard/dashboard.py
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cursor_controller import CursorController
from src.core.gesture_engine import GestureEngine

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"


def read_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_settings(settings: dict[str, Any]) -> None:
    with SETTINGS_PATH.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)
        file.write("\n")


class GestureXRuntime:
    """Runs camera, recognition, and control away from Streamlit's UI thread."""

    def __init__(self) -> None:
        self.commands: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.snapshot: dict[str, Any] = {
            "running": False, "camera_ok": False, "frame": None,
            "gesture": "—", "stable_gesture": "—", "confidence": 0.0,
            "control_status": "DISABLED", "action": "Runtime not started",
            "error": None, "fps": 0.0,
        }

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="GestureXRuntime", daemon=True)
        self.thread.start()

    def send(self, command: str) -> None:
        self.commands.put(command)

    def stop(self) -> None:
        self.stop_event.set()
        self.commands.put("stop")
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)

    def get_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.snapshot)

    def _update(self, **values: Any) -> None:
        with self.lock:
            self.snapshot.update(values)

    def _run(self) -> None:
        camera = engine = controller = None
        frame_times: list[float] = []
        try:
            settings = read_settings()
            camera_index = int(settings.get("camera", {}).get("index", 0))
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError(
                    f"Could not open camera index {camera_index}. "
                    "Check that the webcam is connected and not in use."
                )

            camera_cfg = settings.get("camera", {})
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg.get("width", 1280)))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg.get("height", 720)))

            engine = GestureEngine(settings_path=SETTINGS_PATH)
            controller = CursorController(settings_path=SETTINGS_PATH)
            engine.start()
            self._update(
                running=True, camera_ok=True, error=None,
                action="Camera ready. Controls are disabled.",
            )

            while not self.stop_event.is_set():
                while True:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == "enable":
                        controller.enable()
                    elif command == "disable":
                        controller.disable()
                    elif command == "pause" and controller.enabled and not controller.paused:
                        controller.toggle_pause()
                    elif command == "resume" and controller.enabled and controller.paused:
                        controller.toggle_pause()
                    elif command == "stop":
                        self.stop_event.set()
                        break

                if self.stop_event.is_set():
                    break

                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames.")

                display, recognition = engine.process_frame(frame)
                status = controller.process(recognition)

                cv2.putText(display, f"Gesture: {recognition['gesture'].upper()}",
                            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display, f"Stable: {recognition['stable_gesture'].upper()}",
                            (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
                cv2.putText(display, f"Control: {status['status']}",
                            (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)

                encoded_ok, encoded = cv2.imencode(
                    ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 78]
                )
                if not encoded_ok:
                    raise RuntimeError("Could not encode webcam preview.")

                now = time.monotonic()
                frame_times.append(now)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 0.0
                if len(frame_times) >= 2 and frame_times[-1] > frame_times[0]:
                    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

                self._update(
                    frame=encoded.tobytes(),
                    gesture=recognition["gesture"],
                    stable_gesture=recognition["stable_gesture"],
                    confidence=float(recognition["confidence"]),
                    control_status=status["status"],
                    action=status["action"],
                    fps=fps, running=True, camera_ok=True, error=None,
                )

        except Exception as exc:
            self._update(error=f"{type(exc).__name__}: {exc}",
                         action="Runtime stopped because of an error.")
        finally:
            if controller is not None:
                controller.emergency_stop()
            if engine is not None:
                engine.close()
            if camera is not None:
                camera.release()
            self._update(running=False, camera_ok=False, control_status="DISABLED")


def get_runtime() -> GestureXRuntime:
    if "gesturex_runtime" not in st.session_state:
        st.session_state["gesturex_runtime"] = GestureXRuntime()
    return st.session_state["gesturex_runtime"]


def render_live(runtime: GestureXRuntime) -> None:
    snap = runtime.get_snapshot()
    if snap["frame"] is not None:
        st.image(snap["frame"], use_container_width=True)
    else:
        st.info("Start the camera to see the live preview.")
    cols = st.columns(4)
    cols[0].metric("Stable gesture", snap["stable_gesture"])
    cols[1].metric("Confidence", f"{snap['confidence'] * 100:.1f}%")
    cols[2].metric("Controls", snap["control_status"])
    cols[3].metric("Camera FPS", f"{snap['fps']:.1f}")
    st.caption(f"Current action: {snap['action']}")
    if snap["error"]:
        st.error(snap["error"])


st.set_page_config(page_title="GestureX | Touchless Control", page_icon="🖐️", layout="wide")
st.title("GestureX")
st.caption("AI-powered touchless computer control")

runtime = get_runtime()
snapshot = runtime.get_snapshot()

with st.sidebar:
    st.header("System")
    st.success("Camera runtime is running") if snapshot["running"] else st.warning("Camera runtime is stopped")

    if st.button("Start camera", type="primary", use_container_width=True,
                 disabled=snapshot["running"]):
        runtime.start()
        st.rerun()

    if st.button("Stop camera", use_container_width=True,
                 disabled=not snapshot["running"]):
        runtime.stop()
        st.rerun()

    st.divider()
    st.subheader("Computer controls")
    if st.button("Enable controls", type="primary", use_container_width=True,
                 disabled=not snapshot["running"] or snapshot["control_status"] == "ACTIVE"):
        runtime.send("enable")
    if st.button("Pause controls", use_container_width=True,
                 disabled=snapshot["control_status"] != "ACTIVE"):
        runtime.send("pause")
    if st.button("Resume controls", use_container_width=True,
                 disabled=snapshot["control_status"] != "PAUSED"):
        runtime.send("resume")
    if st.button("Disable controls", use_container_width=True,
                 disabled=snapshot["control_status"] == "DISABLED"):
        runtime.send("disable")
    st.caption("Computer control always starts disabled. Enable it only when ready.")

tab_live, tab_settings, tab_guide = st.tabs(["Live dashboard", "Settings", "Gesture guide"])

with tab_live:
    st.subheader("Live camera and recognition")
    if hasattr(st, "fragment"):
        @st.fragment(run_every="250ms")
        def live_panel():
            render_live(runtime)
        live_panel()
    else:
        st.warning("Automatic preview refresh requires Streamlit 1.37+. Upgrade Streamlit to enable it.")
        render_live(runtime)

    st.divider()
    st.subheader("Safety")
    st.write("Emergency stop immediately disables computer-control actions.")
    if st.button("EMERGENCY STOP", disabled=not snapshot["running"] and
                 snapshot["control_status"] == "DISABLED"):
        runtime.send("disable")
        st.warning("Disable command sent to the controller.")

with tab_settings:
    st.subheader("Control configuration")
    st.write("Saved changes apply after stopping and restarting the camera runtime.")
    current = read_settings()

    with st.form("gesturex_settings"):
        st.markdown("**Recognition**")
        confidence = st.slider("Confidence threshold", 0.50, 0.99,
                               float(current["recognition"]["confidence_threshold"]), 0.01)
        stable_frames = st.slider("Frames for stable gesture", 1, 5,
                                  int(current["recognition"]["stable_frames"]))

        st.markdown("**Cursor movement**")
        margin = st.slider("Camera movement margin", 0.00, 0.25,
                            float(current["cursor"]["camera_margin"]), 0.01)
        min_smoothing = st.slider("Minimum smoothing", 0.05, 0.90,
                                  float(current["cursor"]["min_smoothing"]), 0.01)
        max_smoothing = st.slider("Maximum smoothing", 0.10, 1.00,
                                  float(current["cursor"]["max_smoothing"]), 0.01)
        speed_reference = st.slider("Movement speed reference", 50, 500,
                                    int(current["cursor"]["speed_reference"]), 10)

        st.markdown("**Scrolling**")
        scroll_amount = st.slider("Scroll amount", 1, 12,
                                  int(current["scroll"]["amount"]))
        scroll_interval = st.slider("Scroll interval (seconds)", 0.02, 0.20,
                                    float(current["scroll"]["interval_seconds"]),
                                    0.005, format="%.3f")
        upper = st.slider("Upper scroll zone limit", 0.20, 0.49,
                          float(current["scroll"]["upper_limit"]), 0.01)
        lower = st.slider("Lower scroll zone limit", 0.51, 0.80,
                          float(current["scroll"]["lower_limit"]), 0.01)
        submitted = st.form_submit_button("Save settings", type="primary",
                                           use_container_width=True)

    if submitted:
        if min_smoothing > max_smoothing:
            st.error("Minimum smoothing cannot exceed maximum smoothing.")
        elif upper >= lower:
            st.error("Upper scroll limit must be smaller than lower scroll limit.")
        else:
            updated = read_settings()
            updated["recognition"]["confidence_threshold"] = confidence
            updated["recognition"]["stable_frames"] = stable_frames
            updated["cursor"].update({
                "camera_margin": margin,
                "min_smoothing": min_smoothing,
                "max_smoothing": max_smoothing,
                "speed_reference": float(speed_reference),
            })
            updated["scroll"].update({
                "amount": scroll_amount,
                "interval_seconds": scroll_interval,
                "upper_limit": upper,
                "lower_limit": lower,
            })
            write_settings(updated)
            st.success("Settings saved.")
            if runtime.get_snapshot()["running"]:
                st.info("Stop and restart the camera runtime to apply these settings.")

with tab_guide:
    st.subheader("Gesture mapping")
    st.table({
        "Gesture": ["One", "Fist", "Like", "Peace", "Two Up", "Palm", "No Gesture"],
        "Action": ["Move cursor", "Stop cursor", "Left-click", "Right-click",
                   "Position-based scrolling", "Pause / resume", "No action"],
    })
    st.info("Move your physical mouse to the top-left corner to trigger PyAutoGUI's fail-safe.")

st.divider()
st.caption("GestureX • Local webcam processing • Controls disabled by default")
