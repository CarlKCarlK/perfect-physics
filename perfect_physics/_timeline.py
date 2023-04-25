# use sprepr to load from a file
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import cloudpickle as pickle
from ._circle import Circle
from ._wall import Wall


class Timeline:
    def __init__(self):
        self.events = []

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def load(path):
        path = Path(path)
        with path.open("rb") as f:
            return pickle.load(f)

    def append(self, event):
        self.events.append(event)

    def extend(self, timeline):
        self.events.extend(timeline.events)


@dataclass
class Collision:
    before: Any
    after: Any


@dataclass
class TimeNow:
    clock: float


@dataclass
class Move:
    before: Circle
    after: Circle
    span: float

    def moved(self):
        return self.before.x != self.after.x or self.before.y != self.after.y


@dataclass
class TimeReset:
    pass


@dataclass
class Reverse:
    pass


@dataclass
class MultipleCollisions:
    count: int


def billiards_audio(file, timeline: List, speed_up):
    from pydub import AudioSegment

    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)

    # for item in timeline.events:
    #     logging.info(f"audio1: {item}")

    # only keep moves that actually moved
    events = [x for x in timeline.events if not isinstance(x, Move) or x.moved()]

    # Find the time of the last event
    duration = 0
    last_time_now = 0
    for event in events:
        if isinstance(event, TimeNow):
            duration = event.clock
            last_time_now = duration
        elif isinstance(event, Move):
            duration = last_time_now + event.span

    data_root = Path(__file__).absolute().parent / "data"
    ball_ball = AudioSegment.from_file(data_root / "ball_ball.wav")
    stick_cue = AudioSegment.from_file(data_root / "stick_cue.wav")
    bumper = AudioSegment.from_file(data_root / "bumper.wav")
    reverse = AudioSegment.from_file(data_root / "reverse.wav")

    time = 0
    audio = AudioSegment.silent(duration * 1000 / speed_up)
    audio = audio.overlay(stick_cue, position=0)
    for event in events:
        if isinstance(event, TimeNow):
            time = event.clock
            continue
        elif isinstance(event, Collision):
            if isinstance(event.before[1], Wall):
                audio = audio.overlay(bumper, position=(time * 1000 / speed_up))
            else:
                assert isinstance(event.before[1], Circle)
                audio = audio.overlay(ball_ball, position=(time * 1000 / speed_up))
        elif isinstance(event, Reverse):
            audio = audio.overlay(reverse, position=(time * 1000 / speed_up))
        elif isinstance(event, Move):
            pass
            # trimmed = rolling[: int(event.time * 1000 / speed_up)]
            # audio = audio.overlay(trimmed, position=int(time * 1000 / speed_up))
        elif isinstance(event, MultipleCollisions) or isinstance(event, TimeReset):
            pass
        else:
            raise Exception(f"Unknown event: {event}")
    audio.export(file, format="wav")
