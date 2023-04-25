import bisect
import copy
import glob
import logging
import random
from pathlib import Path
from typing import List

import cloudpickle as pickle
import numpy as np

try:
    from pysnptools.util.mapreduce1 import map_reduce
except ImportError:

    def map_reduce(input_seq, mapper, runner=None):
        return [mapper(item) for item in input_seq]


from sympy import Rational, S, simplify, pi, cos, sin

from ._circle import Circle
from ._misc import plot
from ._physics import Physics
from ._timeline import Timeline, TimeNow, TimeReset, Reverse, billiards_audio
from ._wall import Wall


class World:
    def __init__(
        self,
        circle_list: List[Circle] = [],
        wall_list: List[Wall] = [],
        xlim=(-1, 21),
        ylim=(-5, 5),
        rng=0,
    ):
        self.circle_list = circle_list
        self.wall_list = wall_list
        self.xlim = xlim
        self.ylim = ylim
        self.clock = 0
        self.physics = Physics()
        # if rng is integer, use it as a seed for a random number generator
        if isinstance(rng, int):
            rng = random.Random(rng)
        self.rng = rng

    def figure(self, font_scale=1, label_fun=None, show_fun=None, **kwargs):
        return plot(
            show=False,
            circle_list=self.circle_list,
            wall_list=self.wall_list,
            clock=self.clock,
            font_scale=font_scale,
            xlim=self.xlim,
            ylim=self.ylim,
            label_fun=label_fun,
            show_fun=show_fun,
            **kwargs,
        )

    def show(self, font_scale=1, show=True, draw_radius=1.0, **kwargs):
        if not show:
            return
        return plot(
            show=True,
            circle_list=self.circle_list,
            wall_list=self.wall_list,
            clock=self.clock,
            font_scale=font_scale,
            xlim=self.xlim,
            ylim=self.ylim,
            draw_radius=draw_radius,
            **kwargs,
        )

    def _tick(self, timeline: List, hint_ssca_list, default_tick=1, runner=None):
        logging.info(f"tick start: clock={self.clock}")
        span, span_float, ca_list, hint_ssca_list = self.physics._move_to_collision(
            self.circle_list,
            self.wall_list,
            default_tick,
            timeline,
            hint_ssca_list,
            runner=runner,
        )
        if span is S.Zero:
            logging.info("tick finish: no time passed")
        else:
            logging.debug("tick simplify")
            self.clock = simplify(self.clock + span)
            logging.info(f"tick finish: clock={self.clock}")
            timeline.append(TimeNow(float(self.clock)))
        return (span, span_float, ca_list), hint_ssca_list

    def _tick_no_collision(self, span):
        logging.info(f"tick no collision start: clock={self.clock}")
        self.physics._move_no_collision(self.circle_list, span)
        if isinstance(span, float):
            self.clock = float(self.clock) + span
        else:
            self.clock = simplify(self.clock + span)
        logging.info(f"tick no collision finish: clock={self.clock}")

    def energy(self):
        return sum([circle.energy() for circle in self.circle_list])

    def _tock(self, ss_calist, timeline: List):
        ca_list = ss_calist[2]
        id_list = [(c.id, a.id) for c, a in ca_list]
        logging.info(f"tock start. {len(ca_list)} collision(s): {id_list}")
        energy_before = self.energy()
        energy_before_float = float(energy_before)
        self.physics._change_velocity(ss_calist, self.rng, timeline)
        energy_after = self.energy()
        energy_after_float = float(energy_after)
        if energy_before_float != energy_after_float:
            logging.warning(
                f"energy before {energy_before_float} != energy after {energy_after_float}"  # noqa E501
            )
            raise Exception("energy not conserved")
        logging.info("tock finish")

    def _world_file_to_preview_file(world_file):
        return World._world_file_to_other_file(world_file, "previews", ".png")

    def _world_file_to_world_with_extras_file(world_file):
        return World._world_file_to_other_file(
            world_file, "world_with_extras", ".wwe_cp"
        )

    def _world_file_to_timeline_file(world_file):
        return World._world_file_to_other_file(world_file, "timelines", ".timeline_cp")

    # def _world_file_to_ss_calist_file(world_file):
    #     return World._world_file_to_other_file(
    #         world_file, "ss_calists", ".ss_calist_cp"
    #     )

    def _world_file_to_other_file(world_file, dir, suffix):
        world_file = Path(world_file)
        other_file = world_file.with_suffix(suffix)
        other_parts = list(other_file.parts)
        other_parts[other_parts.index("worlds")] = dir
        other_file = Path(*other_parts)
        other_file.parent.mkdir(parents=True, exist_ok=True)
        return other_file

    def save(
        self,
        world_file,
        save_image=True,
    ):
        world_file = Path(world_file)
        world_file.parent.mkdir(parents=True, exist_ok=True)

        with open(world_file, "wb") as f:
            pickle.dump(self, f)

        if save_image:
            self.figure().savefig(World._world_file_to_preview_file(world_file))

        return self

    def save_with_extras(
        self,
        world_file,
        timeline,
        ss_calist,
        hint_ssca_list,
        save_image=True,
    ):
        world_with_extras_file = World._world_file_to_world_with_extras_file(world_file)

        with open(world_with_extras_file, "wb") as f:
            pickle.dump(
                (
                    self,
                    ss_calist,
                    hint_ssca_list,
                ),
                f,
            )

        timeline.save(World._world_file_to_timeline_file(world_file))
        self.save(world_file, save_image=save_image)
        return self

    def load(filename):
        with open(filename, "rb") as f:
            self = pickle.load(f)
        return self

    def load_with_extras(world_file):
        world_with_extras_file = World._world_file_to_world_with_extras_file(world_file)
        with world_with_extras_file.open("rb") as f:
            (
                world,
                ss_calist,
                hint_ssca_list,
            ) = pickle.load(f)

        return world, ss_calist, hint_ssca_list

    def inscribed(
        circle_of_circle_radius,
        circle_count,
        circle_of_wall_radius,
        wall_count,
        r=1,
        reverse_direction=False,
    ):
        circle_list = []
        circle_vertex_list = []
        for circle_index in range(circle_count):
            angle = 2 * pi * circle_index / circle_count
            x = cos(angle) * circle_of_circle_radius
            y = sin(angle) * circle_of_circle_radius
            circle_vertex_list.append((x, y))
        for circle_index in range(circle_count):
            x, y = circle_vertex_list[circle_index]
            circle_list.append(
                Circle(
                    x,
                    y,
                    r=r,
                    vx=x / circle_of_circle_radius * (-1 if reverse_direction else 1),
                    vy=y / circle_of_circle_radius * (-1 if reverse_direction else 1),
                    m=1,
                )
            )
            if circle_of_circle_radius == 0:
                circle_list[-1].vx = 1
                circle_list[-1].vy = 0

        wall_list = []
        wall_vertex_list = []
        for wall_index in range(wall_count):
            angle = 2 * pi * wall_index / wall_count + pi / 2
            x = cos(angle) * circle_of_wall_radius
            y = sin(angle) * circle_of_wall_radius
            wall_vertex_list.append((x, y))
        for wall_index in range(wall_count):
            x0, y0 = wall_vertex_list[wall_index]
            x1, y1 = wall_vertex_list[(wall_index + 1) % wall_count]
            wall_list.append(Wall(x0=x0, y0=y0, x1=x1, y1=y1))

        world = World(
            circle_list,
            wall_list,
            xlim=(-circle_of_wall_radius, circle_of_wall_radius),
            ylim=(-circle_of_wall_radius, circle_of_wall_radius),
        )
        return world

    def billiards(
        folder=None,
        rows=4,  # negative means no 1st ball
        rng=0,
        jitter=None,  # (0, 1, Rational(1, 1)),
        runner=None,
    ):
        world_path = World._world_file(folder, 0) if folder is not None else None
        if world_path is not None and world_path.exists():
            return World.load(world_path)

        if isinstance(rng, int):
            rng = random.Random(rng)

        length, width = S(92), S(42)  # inches
        hl = length / 2  # half length
        hw = width / 2  # half width

        w0 = Wall(x0=0, y0=0, x1=0, y1=1, id="w0")
        w1 = Wall(x0=length, y0=0, x1=length, y1=1, id="w1")
        w2 = Wall(x0=0, y0=0, x1=1, y1=0, id="w2")
        w3 = Wall(x0=0, y0=width, x1=1, y1=width, id="w3")

        r = (S(2) + Rational(1, 4)) / 2  # 2 ¼ inch diameter balls
        cue = Circle(x=hl / 2, y=hw, r=r, vx=0, vy=0, m=1, id="cue")

        world = World(
            circle_list=[cue],
            wall_list=[w0, w1, w2, w3],
            xlim=(-1, length + 1),
            ylim=(-1, width + 1),
            rng=rng,
        )

        def place_ball(y, r_factor, id):
            r1 = (
                r + 0
                if jitter is None
                else (rng.randint(jitter[0], jitter[1] - 1) * jitter[2])
            )

            b1 = Circle(
                x=length - r1, y=y + r_factor * r1, r=r1, vx=-1, vy=0, m=1, id=id
            )  # 1 ball
            world.circle_list.append(b1)
            world._tick(Timeline(), hint_ssca_list=[], runner=runner)
            world.clock = 0
            b1.vx = 0

        if abs(rows) > 1:
            r0 = (
                r + 0
                if jitter is None
                else rng.randint(jitter[0], jitter[1] - 1) * jitter[2]
            )
            b0 = Circle(
                x=Rational(length * 3 / 4), y=hw, r=r0, vx=0, vy=0, m=1, id="b1"
            )
            world.circle_list.append(b0)

        if abs(rows) >= 2:
            place_ball(hw, 1, id="b2")
            place_ball(hw, -1, id="b3")

        if abs(rows) >= 3:
            place_ball(hw, 2, id="b4")
            place_ball(hw, 0, id="b5")
            place_ball(hw, -2, id="b6")

        if abs(rows) >= 4:
            place_ball(hw, 3, id="b7")
            place_ball(hw, 1, id="b8")
            place_ball(hw, -1, id="b9")
            place_ball(hw, -3, id="b10")

        # negative means remove "b1"
        if rows < 0:
            world.circle_list.pop(1)

        for circle in world.circle_list[1:]:
            circle.r = r
        cue.vx = 1
        timeline = Timeline()
        timeline.append(TimeReset())

        if world_path is not None:
            world.save_with_extras(
                world_path,
                timeline=Timeline(),
                ss_calist=None,
                hint_ssca_list=None,
            )

        return world

    def tick_tock(self, timeline, hint_ssca_list, default_tick=1, runner=None):
        ss_calist, hint_ssca_list = self._tick(
            timeline, hint_ssca_list, default_tick, runner=runner
        )
        self._tock(ss_calist, timeline)
        return hint_ssca_list

    def reverse(self):
        for circle in self.circle_list:
            circle.vx = -circle.vx
            circle.vy = -circle.vy

    def _world_file(folder, frame_index):
        folder = Path(folder)
        file = (
            folder
            / "run"
            / "worlds"
            / (
                f"{frame_index:05d}.world_cp"
                if frame_index is not None
                else "*.world_cp"
            )
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        return file

    def run_to_file(self, folder, frame_count, reverse_frames={}):
        tick_remainder = 1

        last_file_index = 0
        for frame_index in range(frame_count):
            world_path = World._world_file(folder, frame_index)
            if world_path.exists():
                last_file_index = frame_index
            else:
                break

        for frame_index in range(last_file_index, frame_count):
            world_path = World._world_file(folder, frame_index)
            logging.info(f"Working on {world_path}")

            if frame_index in reverse_frames:
                tick_remainder = (frame_index + 1) % 2
                if not world_path.exists():
                    timeline = Timeline()
                    timeline.append(Reverse())
                    self.reverse()
                    self.save_with_extras(
                        world_path,
                        timeline=timeline,
                        ss_calist=None,
                        hint_ssca_list=[],
                    )
                else:
                    self, ss_calist, hint_ssca_list = World.load_with_extras(world_path)

            if frame_index == 0:
                if not world_path.exists():
                    self.save_with_extras(
                        world_path,
                        timeline=Timeline(),
                        ss_calist=None,
                        hint_ssca_list=None,
                    )
                ss_calist = None
                hint_ssca_list = []
            elif frame_index % 2 == tick_remainder:
                if not world_path.exists():
                    timeline = Timeline()
                    ss_calist, hint_ssca_list = self._tick(timeline, hint_ssca_list)
                    self.save_with_extras(
                        world_path,
                        timeline=timeline,
                        ss_calist=ss_calist,
                        hint_ssca_list=hint_ssca_list,
                    )
                else:
                    self, ss_calist, hint_ssca_list = World.load_with_extras(world_path)
            else:  # tock
                if not world_path.exists():
                    timeline = Timeline()
                    self._tock(ss_calist, timeline)
                    self.save_with_extras(
                        world_path,
                        timeline=timeline,
                        ss_calist=None,
                        hint_ssca_list=hint_ssca_list,
                    )
                else:
                    self, ss_calist, hint_ssca_list = World.load_with_extras(world_path)

    def _create_hint_file(folder, i):
        hint_file = World._hint_file(folder, i)
        if hint_file.exists():
            with hint_file.open("rb") as f:
                hint_ssca_list = pickle.load(f)
        else:
            hint_ssca_list = []
        return hint_ssca_list

    def _hint_file(folder, frame_index):
        folder = Path(folder)
        file = folder / "run" / "hints" / f"{frame_index:05d}.hint_cp"
        file.parent.mkdir(parents=True, exist_ok=True)
        return file

    def run_in_place(self, count, show=False, font_scale=1, **kwargs):
        self.show(show=show, font_scale=font_scale, **kwargs)
        hint_ssca_list = []
        for i in range(count):
            timeline = Timeline()
            logging.info(f"Working on {i}")
            ss_calist, hint_ssca_list = self._tick(timeline, hint_ssca_list)
            self.show(
                show=show and ss_calist[0] is not S.Zero,
                font_scale=font_scale,
                **kwargs,
            )
            self._tock(ss_calist, timeline)
            self.show(show=show, font_scale=font_scale, **kwargs)

    def render(
        folder,
        speed_up,
        fps=24,
        font_scale=4,
        audio=billiards_audio,
        slice=np.s_[:],
        round_time=True,
        talkie_codec="png",
        silent_codec="MJPG",
        prefix="",
        runner=None,
        **kwargs,
    ):
        import cv2

        folder = Path(folder)
        clock_list, world_list, timeline = World._cp_to_lists(
            folder, slice=slice, filter_same_time=True
        )

        frame_count = int((clock_list[-1] - clock_list[0]) * fps / speed_up) + 1
        logging.info(f"frame_count {frame_count}")

        render_folder = folder / "render"
        render_folder.mkdir(parents=True, exist_ok=True)

        misc_folder = render_folder / "misc"
        misc_folder.mkdir(parents=True, exist_ok=True)
        wav_file = misc_folder / f"soundtrack{prefix}.wav"

        if audio and not wav_file.exists():
            audio(wav_file, timeline, speed_up=speed_up)

        frame_folder = render_folder / f"frames{prefix}"
        frame_folder.mkdir(parents=True, exist_ok=True)

        def mapper(frame_index):
            frame_file = frame_folder / f"{frame_index:05d}.png"
            if not frame_file.exists():
                clock = frame_index * speed_up / fps + clock_list[0]
                # binary search in sorted list clock_list
                i = bisect.bisect_right(clock_list, clock) - 1
                previous = clock_list[i]
                logging.info(
                    f"{frame_index} of {frame_count}, clock={clock}>={previous}"
                )
                world = copy.deepcopy(world_list[i])
                world._tick_no_collision(float(clock - previous))
                if round_time:
                    world.clock = int(frame_index * speed_up / fps + clock_list[0])
                else:
                    world.clock = S(frame_index) * speed_up / fps + clock_list[0]
                world.figure(font_scale=font_scale, **kwargs).savefig(frame_file)
            return frame_file

        frame_files = map_reduce(range(frame_count), mapper=mapper, runner=runner)

        # https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
        silent_file = misc_folder / f"silent_video{prefix}.avi"
        if not silent_file.exists():
            out = None
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if out is None:
                    height, width, _ = frame.shape
                    out = cv2.VideoWriter(
                        str(silent_file),
                        cv2.VideoWriter_fourcc(*silent_codec),
                        fps,
                        (width, height),
                    )
                out.write(frame)
            out.release()

        talkie_file = render_folder / f"{folder.name}{prefix}.avi"
        if not talkie_file.exists():
            World._combine_audio(silent_file, wav_file, talkie_file, codec=talkie_codec)

    def _cp_to_lists(folder, slice=np.s_[:], filter_same_time=True):
        world_files = sorted(glob.glob(str(World._world_file(folder, None))))
        world_files = np.array(world_files)[slice]
        previous_clock = float("-inf")
        clock_list = []
        world_list = []
        timeline = Timeline()
        for world_file in world_files:
            world_file = Path(world_file)
            sub_timeline = Timeline.load(World._world_file_to_timeline_file(world_file))
            # If nothing happened (and not the first frame), skip it
            if len(world_list) > 0 and len(sub_timeline.events) == 0:
                continue
            timeline.extend(sub_timeline)
            world = World.load(world_file)
            if isinstance(world.clock, int):
                clock_float = world.clock
            else:
                clock_float = float(world.clock)
            logging.debug(f"{world_file}, {clock_float}")
            if clock_float > previous_clock:
                previous_clock = clock_float
                clock_list.append(clock_float)
                world_list.append(world)
            elif clock_float == previous_clock:
                if filter_same_time:
                    world_list[-1] = world
                else:
                    clock_list.append(clock_float)
                    world_list.append(world)
            else:
                raise ValueError("Clock is not monotonically increasing")
        return clock_list, world_list, timeline

    def render_events(
        folder,
        seconds_per_event=2,
        fps=24,
        font_scale=4,
        slice=np.s_[:],
        talkie_codec="png",
        silent_codec="MJPG",
        prefix="",
        **kwargs,
    ):
        from pydub import AudioSegment
        import cv2

        frames_per_half_event = seconds_per_event * fps / 2
        assert frames_per_half_event == int(
            frames_per_half_event
        ), "seconds_per_event * fps must be even"

        folder = Path(folder)
        render_folder = folder / "render_events"
        render_folder.mkdir(parents=True, exist_ok=True)

        clock_list, world_list, _timeline = World._cp_to_lists(
            folder, filter_same_time=False, slice=slice
        )

        frame_folder = render_folder / "frames"
        frame_folder.mkdir(parents=True, exist_ok=True)

        message_folder = render_folder / "messages"
        message_folder.mkdir(parents=True, exist_ok=True)

        for world_index, clock in enumerate(clock_list):
            plain_file = frame_folder / f"{world_index:05d}.plain.png"
            message_file = frame_folder / f"{world_index:05d}.message.png"
            if not plain_file.exists():
                world = world_list[world_index]
                figure = world.figure(font_scale=font_scale, **kwargs)
                figure.savefig(plain_file)

                if world_index == 0:
                    message = ""
                else:
                    span_float = clock - clock_list[world_index - 1]
                    if span_float == 0:
                        message = "UPDATE vx's & vy's"
                    else:
                        span = simplify(
                            world_list[world_index].clock
                            - world_list[world_index - 1].clock
                        )
                        message = f"MOVE TIME {span}"
                figure.text(
                    0.5,
                    0.75,
                    message[:100],
                    color="red",
                    ha="center",
                    transform=figure.axes[0].transAxes,
                    fontsize=20,
                )
                figure.savefig(message_file)
                message_file = message_folder / f"{world_index:05d}.message_txt"
                with open(message_file, "w") as f:
                    f.write(message)

        misc_folder = render_folder / "misc"
        misc_folder.mkdir(parents=True, exist_ok=True)
        wav_file = misc_folder / f"soundtrack{prefix}.wav"

        duration = len(clock_list) * seconds_per_event
        if not wav_file.exists():
            audio = AudioSegment.silent(duration * 1000)
            data_root = Path(__file__).absolute().parent / "data"
            skip_sound = AudioSegment.from_file(data_root / "Electronic Button.mp3")
            update_sound = AudioSegment.from_file(data_root / "Flash Beep.mp3")
            for world_index, _clock in enumerate(clock_list):
                message_file = message_folder / f"{world_index:05d}.message_txt"
                position = world_index * seconds_per_event * 1000
                with open(message_file) as f:
                    message = f.read()
                if message == "":
                    pass
                elif message.startswith("UPDATE"):
                    audio = audio.overlay(update_sound, position=position)
                else:
                    audio = audio.overlay(skip_sound, position=position)
            audio.export(wav_file, format="wav")

        # https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
        silent_file = misc_folder / f"silent_video{prefix}.avi"
        frames_per_half_event = seconds_per_event * fps / 2
        assert frames_per_half_event == int(
            frames_per_half_event
        ), "event_span * fps must be even"
        frames_per_half_event = int(frames_per_half_event)
        if not silent_file.exists():
            out = None
            for world_index, _clock in enumerate(clock_list):
                plain_file = frame_folder / f"{world_index:05d}.plain.png"
                message_file = frame_folder / f"{world_index:05d}.message.png"
                plain_frame = cv2.imread(str(plain_file))
                message_frame = cv2.imread(str(message_file))
                for frame_index in range(frames_per_half_event * 2):
                    if out is None:
                        height, width, _ = plain_frame.shape
                        out = cv2.VideoWriter(
                            str(silent_file),
                            cv2.VideoWriter_fourcc(*silent_codec),
                            fps,
                            (width, height),
                        )
                    if frame_index < frames_per_half_event:
                        frame = message_frame
                    else:
                        frame = plain_frame
                    out.write(frame)
            out.release()

        talkie_file = render_folder / f"{folder.name}{prefix}.events.avi"
        if not talkie_file.exists():
            World._combine_audio(silent_file, wav_file, talkie_file, codec=talkie_codec)

    def _combine_audio(video_file, audio_file, out_file, codec="png"):
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.video.io.VideoFileClip import VideoFileClip

        video_clip = VideoFileClip(str(video_file))
        audio_clip = AudioFileClip(str(audio_file))
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(str(out_file), fps=video_clip.fps, codec=codec)
