#!/usr/bin/env python3
"""terminal_ai.py – Retro AI dialog terminal for a live‑action game

Features
--------
* Parses a YAML dialog graph (structure described in README).
* SQLite persistence – every node transition is flushed to disk.
* Passcode‑gated sessions with 2‑minute idle timeout.
* Simple ANSI/Rich‐powered CLI running entirely offline.
* Robust: catches exceptions, logs to error.log, never leaves the
  terminal in a broken state.

Usage
-----
$ python3 terminal_ai.py dialog.yaml passcodes.txt
"""



from __future__ import annotations


import signal
def handle_sigint(signum, frame):
    pass
signal.signal(signal.SIGINT, handle_sigint)

import argparse
import os
import random
import sqlite3
import sys
import textwrap
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml  # PyYAML
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

console = Console()
WRAP_WIDTH = 80
IDLE_TIMEOUT = 120  # seconds
DB_FILE = "state.db"
ERROR_LOG = "error.log"

################################################################################
#                               Logging helper                                 #
################################################################################

def log_error(message: str, exc: Exception | None = None, /, *, to_console: bool = False) -> None:
    """Append *message* (and optional traceback of *exc*) to ERROR_LOG.

    Parameters
    ----------
    message : str
        Human-readable description of the failure or debug event.
    exc : Exception | None, optional
        If supplied, full traceback is recorded after *message*.
    to_console : bool, default False
        Echo the same content to the terminal in red – handy while live-
        debugging but noisy for players, so keep disabled in production.
    """
    stamp = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    blob = f"[{stamp}] {message}\n"
    if exc is not None:
        blob += "".join(traceback.format_exception(exc))
    with open(ERROR_LOG, "a", encoding="utf-8") as fh:
        fh.write(blob)
    if to_console:
        console.print(f"[red]{blob}[/red]")

################################################################################
#                                 Data Model                                   #
################################################################################

class Node:
    """A single dialog node loaded from YAML."""

    def __init__(self, raw: dict):
        self.id: str = raw["id"]
        self.text: str = raw.get("text", "")
        self.edges: Dict[str, str] = raw.get("edges", {})
        self.pre_actions: List[dict] = raw.get("pre_actions", [])
        self.post_actions: List[dict] = raw.get("post_actions", [])

    def next_node_for(self, user_input: str) -> Optional[str]:
        key = user_input.strip().lower()
        return self.edges.get(key) or self.edges.get("*")


class DialogGraph:
    """Loads and validates the graph from YAML."""

    def __init__(self, yaml_path: Path):
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.nodes: Dict[str, Node] = {raw["id"]: Node(raw) for raw in data}
        self._validate()

    def _validate(self):
        # All edge targets exist
        unknown: List[Tuple[str, str]] = []
        for node in self.nodes.values():
            for target in node.edges.values():
                if target not in self.nodes:
                    unknown.append((node.id, target))
        if unknown:
            msgs = ", ".join(f"{n}->{t}" for n, t in unknown)
            raise ValueError(f"Edges point to unknown nodes: {msgs}")

        # Graph connectivity (weak) – simple DFS
        visited = set()
        start = next(iter(self.nodes))
        stack = [start]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            stack.extend(self.nodes[nid].edges.values())
        if len(visited) != len(self.nodes):
            isolated = set(self.nodes) - visited
            raise ValueError(f"Isolated nodes detected: {', '.join(isolated)}")

    def __getitem__(self, node_id: str) -> Node:
        return self.nodes[node_id]

################################################################################
#                               Persistence                                    #
################################################################################

class DB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self._ensure_schema()

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS players (
                passcode TEXT PRIMARY KEY,
                current_node TEXT NOT NULL,
                data       BLOB,
                updated_at REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    # CRUD --------------------------------------------------------------------
    def get_or_create_player(self, passcode: str, default_node: str) -> Tuple[str, dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT current_node, data FROM players WHERE passcode=?", (passcode,))
        row = cur.fetchone()
        if row:
            node, blob = row
            data = yaml.safe_load(blob) if blob else {}
        else:
            node = default_node
            data = {}
            cur.execute(
                "INSERT INTO players(passcode, current_node, data, updated_at) VALUES(?,?,?,?)",
                (passcode, node, yaml.safe_dump(data), time.time()),
            )
            self.conn.commit()
        return node, data

    def save_player(self, passcode: str, node: str, data: dict):
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE players SET current_node=?, data=?, updated_at=? WHERE passcode=?",
            (node, yaml.safe_dump(data), time.time(), passcode),
        )
        self.conn.commit()

################################################################################
#                             Action Handlers                                  #
################################################################################

def execute_pre_actions(actions: List[dict], player_data: dict):
    for act in actions:
        for k, v in act.items():
            if k == "set_random_survivors":
                player_data["SURVIVOR1"], player_data["SURVIVOR3"] = pick_random_survivors(v)
            elif k == "set_state":
                player_data["STATE"] = v


def execute_post_actions(actions: List[dict], player_data: dict):
    for act in actions:
        for k, v in act.items():
            if k == "set_state":
                player_data["STATE"] = v


def pick_random_survivors(n: int) -> Tuple[str, str]:
    pool = [
        "Erica Pike",
        "Jonas Trent",
        "Marisol Vega",
        "Elias Krohn",
        "Tanya Holt",
        "Dexter Shaw",
        "Bianca Frost",
        "Howard Lin",
        "Raj Patel",
        "Helena Marks",
        "Quinn Beal",
        "Xavier Ro",
        "Sasha Wynne",
        "Omar Cutler",
        "Yuri Belov",
        "Lena Ortiz",
        "Igor Molin",
        "Piper Jules",
    ]
    return tuple(random.sample(pool, k=n))  # type: ignore

################################################################################
#                               Engine                                         #
################################################################################

class Engine:
    def __init__(self, graph: DialogGraph, db: DB, passcodes: List[str]):
        self.graph = graph
        self.db = db
        self.passcodes = set(pc.strip() for pc in passcodes if pc.strip())
        self.player_code: Optional[str] = None
        self.player_node: Optional[str] = None
        self.player_data: dict = {}
        self.last_input_time: float = time.time()

    # ---------------------------- auth ---------------------------------------
    def authenticate(self):
        while True:
            self.clear()
            console.print(Panel(Text("EYE UPLINK – ENTER PASSCODE", style="bold green")))
            code = Prompt.ask("PASSCODE").strip()
            if code in self.passcodes:
                self.player_code = code
                node, data = self.db.get_or_create_player(code, "ONBOARDING_1")
                self.player_node = node
                self.player_data = data or {}
                break
            if code == "GETMEOUT":
                self.shutdown()
            console.print("[red]INVALID PASSCODE[/red]")
            time.sleep(1.5)

    # --------------------------- main loop -----------------------------------
    def run(self):
        try:
            while True:
                if not self.player_code:
                    self.authenticate()

                node = self.graph[self.player_node]  # type: ignore
                execute_pre_actions(node.pre_actions, self.player_data)
                self.render(node)

                user_input = self.read_input()
                target = node.next_node_for(user_input)

                if user_input.lower() == "exit":
                    if target:
                        self.player_node = target
                        self.db.save_player(self.player_code, self.player_node, self.player_data)
                    self.logout()
                    continue

                if not target:
                    console.print("[red]DOES NOT CONVERGE. TRY AGAIN.[/red]")
                    time.sleep(1.2)
                    continue

                execute_post_actions(node.post_actions, self.player_data)
                self.player_node = target
                self.db.save_player(self.player_code, self.player_node, self.player_data)
        except KeyboardInterrupt:
            console.print("[red]![/red]")
        except Exception as e:
            with open(ERROR_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"{time.asctime()}: {e}\n")
            self.shutdown()

    # ------------------------- helpers --------------------------------------
    def render(self, node: Node):
        self.clear()
        text = self.substitute_placeholders(node.text)
        console.print(text)

    def substitute_placeholders(self, s: str) -> str:
        out = s
        for key, val in self.player_data.items():
            placeholder = f"${key}$"
            out = out.replace(placeholder, str(val))
        return out

    def read_input(self) -> str:
        self.last_input_time = time.time()
        while True:
            if time.time() - self.last_input_time > IDLE_TIMEOUT:
                self.logout()
                return ""  # will prompt for auth again
            if console.is_terminal:
                try:
                    return console.input("\n> ")
                except (EOFError, KeyboardInterrupt):
                    self.logout()
            else:
                time.sleep(0.1)

    def logout(self):
        if self.player_code:
            self.db.save_player(self.player_code, self.player_node, self.player_data)
        self.player_code = None
        self.player_node = None
        self.player_data = {}
        console.print("\n[cyan]SESSION TERMINATED. REAUTHENTICATE.[/cyan]")
        time.sleep(1.5)

    @staticmethod
    def clear():
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def shutdown():
        console.print("\n[bright_yellow]SHUTTING DOWN…[/bright_yellow]")
        sys.exit(0)

################################################################################
#                                   CLI                                        #
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Retro AI dialog terminal")
    parser.add_argument("-d", type=Path, help="Path to dialog YAML file", default="dialog")
    parser.add_argument("-p", type=Path, help="File with one passcode per line", default="passcodes")
    args = parser.parse_args()

    graph = DialogGraph(args.d)
    db = DB(Path(DB_FILE))
    with args.p.open("r", encoding="utf-8") as f:
        codes = [ln.strip() for ln in f if ln.strip()]
    engine = Engine(graph, db, codes)
    try:
        engine.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

