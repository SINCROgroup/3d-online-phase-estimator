"""
A simple viewer for CSV motion capture data files for SHARESPACE Health Scenario.

Install dependencies:
    pip install anytree numpy pandas pyqtgraph pyopengl pyqt6

Run the viewer:
    python data_viewer.py <data_directory>

    Where `<data_directory>` contains the motion capture data files in CSV format.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pyqtgraph.opengl as gl
from anytree import Node
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


RESOLUTION = (1280, 720)


MANNY = Node("pelvis", id=19, children=[
    Node("spine_01", id=26, children=[
        Node("spine_02", id=27, children=[
            Node("spine_03", id=28, children=[
                Node("spine_04", id=29, children=[
                    Node("spine_05", id=30, children=[
                        Node("neck_01", id=18, children=[
                            Node("neck_02", id=35, children=[
                                Node("head", id=36),
                            ]),
                        ]),
                        Node("clavicle_l", id=33, swap="clavicle_r", children=[
                            Node("upperarm_l", id=5, swap="upperarm_r", children=[
                                Node("lowerarm_l", id=7, swap="lowerarm_r", children=[
                                    Node("hand_l", id=9, swap="hand_r"),
                                ]),
                            ]),
                        ]),
                        Node("clavicle_r", id=34, swap="clavicle_l", children=[
                            Node("upperarm_r", id=6, swap="upperarm_l", children=[
                                Node("lowerarm_r", id=8, swap="lowerarm_l", children=[
                                    Node("hand_r", id=10, swap="hand_l"),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    Node("thigh_l", id=11, swap="thigh_r", children=[
        Node("calf_l", id=13, swap="calf_r", children=[
            Node("foot_l", id=15, swap="foot_r", children=[
                Node("ball_l", id=20, swap="ball_r")
            ]),
        ]),
    ]),
    Node("thigh_r", id=12, swap="thigh_l", children=[
        Node("calf_r", id=14, swap="calf_l", children=[
            Node("foot_r", id=16, swap="foot_l", children=[
                Node("ball_r", id=21, swap="ball_l")
            ]),
        ]),
    ]),
])



class TRC:
    """
    Represents generic motion data.

    Attributes:
        skeleton: The hierarchical structure of joints as an anytree Node.
        frame_rate: The frame rate of the motion data.
        camera_rate: The camera frame rate of the capture system. Default is the same as frame_rate. This can be higher
                    than the frame rate if the capture system records at a higher rate than the motion data, e.g., when
                    motion is computed for only every Nth frame, with N > 1.
        frames: A list of motion data frames. Each frame is a list of marker positions (x, y, z) in the same order as the marker names.
        num_markers: The number of markers in the motion data.
        num_frames: The number of frames in the motion data.
        marker_names: A list of marker names in the same order as the marker positions in each frame.
    """

    def __init__(
        self,
        frame_rate: float,
        camera_rate: Optional[float] = None,
        frames: List[Any] = [],
    ):
        camera_rate = camera_rate or frame_rate
        assert frame_rate > 0, "Frame rate must be positive."
        assert frame_rate <= camera_rate, (
            "Frame rate must be less than or equal to camera rate."
        )

        self.skeleton = MANNY
        self.frame_rate = frame_rate
        self.camera_rate = camera_rate

        # Initialize frames.
        self._frames_df = None
        self._frames = []
        for frame in frames:
            self._frames.append(frame)
        self.to_pandas()

    @staticmethod
    def fromfile(file_path: str) -> "TRC":
        # Parse header.
        header = pd.read_csv(
            file_path, sep=",", skiprows=1, header=None, nrows=2, encoding="ISO-8859-1"
        )
        header = dict(zip(header.iloc[0].tolist(), header.iloc[1].tolist()))

        # Parse marker information.
        markers = pd.read_csv(file_path, sep=",", skiprows=3, nrows=1)
        markers = markers.columns.tolist()[2:-1:3]
        # print("All markers:", markers)

        # Read motion data.
        filtered_columns = ["Frame#", "Time"] + [
            f"{m}_{axis}" for m in markers for axis in ["X", "Y", "Z"]
        ]
        data = pd.read_csv(
            file_path,
            sep=",",
            skiprows=5,
            index_col=False,
            header=None,
            names=filtered_columns,
        )

        # Remove markers not in the skeleton.
        skeleton_markers = [MANNY.name] + [m.name for m in MANNY.descendants]
        filtered_markers = [m for m in markers if m in skeleton_markers]
        # print("Filtered markers:", filtered_markers)
        filtered_columns = ["Frame#", "Time"] + [
            f"{m}_{axis}" for m in filtered_markers for axis in ["X", "Y", "Z"]
        ]
        data = data[filtered_columns]

        # Reshape data into frames (F, M, 3).
        frames = data.iloc[:, 2:].values.reshape(-1, len(filtered_markers), 3).tolist()

        return TRC(
            frame_rate=float(header["DataRate"]),
            frames=frames,
        )

    @property
    def frames(self):
        return self._frames

    @property
    def skeleton(self):
        return self._skeleton

    @skeleton.setter
    def skeleton(self, value):
        self._skeleton = value
        self._num_markers = 1 + len(value.descendants)

        marker_names = [value.name]
        for node in value.descendants:
            marker_names.append(node.name)
        self._marker_names = marker_names

    @property
    def marker_names(self):
        return self._marker_names

    @property
    def num_markers(self):
        return self._num_markers

    @property
    def num_frames(self):
        return len(self.frames)

    @property
    def duration(self):
        return self.num_frames / max(self.frame_rate, 1)

    def append(self, frame: Any) -> None:
        self._frames.append(frame)
        self._frames_df = None  # Invalidate the DataFrame cache.

    def to_pandas(self):
        """
        Returns a pandas DataFrame representation of the motion data.
        """
        if self._frames_df is not None:
            return self._frames_df

        data = []
        for i, frame in enumerate(self.frames):
            data.append([i, i / self.frame_rate] + np.array(frame).flatten().tolist())
        columns = ["Frame#", "Time"] + [
            f"{m}_{axis}" for m in self.marker_names for axis in ["X", "Y", "Z"]
        ]
        self._frames_df = pd.DataFrame(data, columns=columns)
        return self._frames_df

    def __getitem__(self, index):
        return self.frames[index]


def create_floor():
    grid = gl.GLGridItem()
    grid.setSize(x=10, y=10)
    grid.setSpacing(x=1, y=1)
    return grid


class MotionRenderer(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setBackgroundColor((0, 0, 139))  # Deep blue

        self._motion: TRC | None = None
        self._current_frame = 0
        self._frame_rate = 30.0

        self.floor = create_floor()
        self.addItem(self.floor)
        self._items = []

    def load_motion(self, motion: TRC):
        self.clear_items()
        self._motion = motion
        self._frame_rate = motion.frame_rate
        self._current_frame = 0

    def clear_items(self):
        for item in self._items:
            self.removeItem(item)
        self._items.clear()

    def render_next(self):
        if self._motion is None or self._current_frame >= self._motion.num_frames:
            return
        self.render_frame_at(self._current_frame)
        self._current_frame += 1

    def render_frame_at(self, index):
        self.clear_items()

        try:
            positions = np.array(self._motion.frames[index])
        except Exception as e:
            print(f"Frame {index} error: {e}")
            return

        scatter = gl.GLScatterPlotItem(pos=positions, size=10, color=(0, 0, 1, 1))
        self.addItem(scatter)
        self._items.append(scatter)

        nodes = [self._motion.skeleton] + list(self._motion.skeleton.descendants)
        if len(nodes) != len(positions):
            print("Warning: Node count mismatch.")
            return

        for node in nodes:
            if node.parent:
                ci = nodes.index(node)
                pi = nodes.index(node.parent)
                line = gl.GLLinePlotItem(
                    pos=np.vstack([positions[pi], positions[ci]]),
                    color=(0, 1, 1, 1),
                    width=2,
                    antialias=True,
                )
                self.addItem(line)
                self._items.append(line)


class RenderThread(QThread):
    render_available = pyqtSignal()

    def __init__(self, renderer: MotionRenderer):
        super().__init__()
        self.renderer = renderer
        self._running = True

    def run(self):
        while self._running:
            if (
                self.renderer._motion is None
                or self.renderer._current_frame >= self.renderer._motion.num_frames
            ):
                time.sleep(0.5)
                continue
            self.render_available.emit()
            time.sleep(1 / self.renderer._frame_rate)

    def stop(self):
        self._running = False


class MainWindow(QMainWindow):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.setWindowTitle("CSV Motion Viewer")
        self.resize(*RESOLUTION)

        self.renderer = MotionRenderer()
        self.render_thread = RenderThread(self.renderer)
        self.render_thread.render_available.connect(self.renderer.render_next)
        self.render_thread.start()

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_selected_file)

        for file in sorted(data_dir.glob("*.csv")):
            item = QListWidgetItem(file.name)
            item.setData(Qt.ItemDataRole.UserRole, str(file))
            self.file_list.addItem(item)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(self.file_list)
        sidebar = QWidget()
        sidebar.setLayout(sidebar_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(self.renderer)
        splitter.setSizes([200, 1080])

        container = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_selected_file(self, item: QListWidgetItem):
        filepath = item.data(Qt.ItemDataRole.UserRole)
        try:
            motion = TRC.fromfile(filepath)
            self.renderer.load_motion(motion)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(Path(args.data_dir))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
