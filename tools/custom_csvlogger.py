from __future__ import annotations
import contextlib
import csv
import datetime
import os
import re
from collections.abc import Sequence
from multiprocessing import Lock
from pathlib import Path
from typing import TYPE_CHECKING, Any
from gradio_client import utils as client_utils
import gradio as gr
from gradio import utils, wasm_utils

if TYPE_CHECKING:
    from gradio.components import Component
from gradio.flagging import FlaggingCallback
from threading import Lock

class CSVLogger_custom(FlaggingCallback):
    """
    The default implementation of the FlaggingCallback abstract class in gradio>=5.0. Each flagged
    sample (both the input and output data) is logged to a CSV file with headers on the machine running
    the gradio app. Unlike ClassicCSVLogger, this implementation is concurrent-safe and it creates a new
    dataset file every time the headers of the CSV (derived from the labels of the components) change. It also
    only creates columns for "username" and "flag" if the flag_option and username are provided, respectively.

    Example:
        import gradio as gr
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            flagging_callback=CSVLogger())
    Guides: using-flagging
    """

    def __init__(
        self,
        simplify_file_data: bool = True,
        verbose: bool = True,
        dataset_file_name: str | None = None,
    ):
        """
        Parameters:
            simplify_file_data: If True, the file data will be simplified before being written to the CSV file. If CSVLogger is being used to cache examples, this is set to False to preserve the original FileData class
            verbose: If True, prints messages to the console about the dataset file creation
            dataset_file_name: The name of the dataset file to be created (should end in ".csv"). If None, the dataset file will be named "dataset1.csv" or the next available number.
        """
        self.simplify_file_data = simplify_file_data
        self.verbose = verbose
        self.dataset_file_name = dataset_file_name
        self.lock = (
            Lock() if not wasm_utils.IS_WASM else contextlib.nullcontext()
        )  # The multiprocessing module doesn't work on Lite.

    def setup(
        self,
        components: Sequence[Component],
        flagging_dir: str | Path,
    ):
        self.components = components
        self.flagging_dir = Path(flagging_dir)
        self.first_time = True

    def _create_dataset_file(self, additional_headers: list[str] | None = None):
        os.makedirs(self.flagging_dir, exist_ok=True)

        if additional_headers is None:
            additional_headers = []
        headers = (
            [
                getattr(component, "label", None) or f"component {idx}"
                for idx, component in enumerate(self.components)
            ]
            + additional_headers
            + [
                "timestamp",
            ]
        )
        headers = utils.sanitize_list_for_csv(headers)
        dataset_files = list(Path(self.flagging_dir).glob("dataset*.csv"))

        if self.dataset_file_name:
            self.dataset_filepath = self.flagging_dir / self.dataset_file_name
        elif dataset_files:
            try:
                latest_file = max(
                    dataset_files, key=lambda f: int(re.findall(r"\d+", f.stem)[0])
                )
                latest_num = int(re.findall(r"\d+", latest_file.stem)[0])

                with open(latest_file, newline="", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    existing_headers = next(reader, None)

                if existing_headers != headers:
                    new_num = latest_num + 1
                    self.dataset_filepath = self.flagging_dir / f"dataset{new_num}.csv"
                else:
                    self.dataset_filepath = latest_file
            except Exception:
                self.dataset_filepath = self.flagging_dir / "dataset1.csv"
        else:
            self.dataset_filepath = self.flagging_dir / "dataset1.csv"

        if not Path(self.dataset_filepath).exists():
            with open(
                self.dataset_filepath, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(utils.sanitize_list_for_csv(headers))
            if self.verbose:
                print("Created dataset file at:", self.dataset_filepath)
        elif self.verbose:
            print("Using existing dataset file at:", self.dataset_filepath)

    def flag(
        self,
        flag_data: list[Any],
        flag_option: str | None = None,
        username: str | None = None,
    ) -> int:
        if self.first_time:
            additional_headers = []
            if flag_option is not None:
                additional_headers.append("flag")
            if username is not None:
                additional_headers.append("username")
            self._create_dataset_file(additional_headers=additional_headers)
            self.first_time = False

        csv_data = []
        for idx, (component, sample) in enumerate(
            zip(self.components, flag_data, strict=False)
        ):
            save_dir = (
                self.flagging_dir
                / client_utils.strip_invalid_filename_characters(
                    getattr(component, "label", None) or f"component {idx}"
                )
            )
            if utils.is_prop_update(sample):
                csv_data.append(str(sample))
            else:
                data = (
                    component.flag(sample, flag_dir=save_dir)
                    if sample is not None
                    else ""
                )
                if self.simplify_file_data:
                    data = utils.simplify_file_data_in_str(data)
                csv_data.append(data)

        if flag_option is not None:
            csv_data.append(flag_option)
        if username is not None:
            csv_data.append(username)
        csv_data.append(str(datetime.datetime.now()))

        with self.lock:
            with open(
                self.dataset_filepath, "a", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(utils.sanitize_list_for_csv(csv_data))
            with open(self.dataset_filepath, encoding="utf-8") as csvfile:
                line_count = len(list(csv.reader(csvfile))) - 1

        return line_count