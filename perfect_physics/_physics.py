import logging
from pathlib import Path
from typing import Any, List

import numpy as np

try:
    from pysnptools.util.mapreduce1 import map_reduce
except ImportError:

    def map_reduce(input_seq, mapper, runner=None):
        return [mapper(item) for item in input_seq]


from sympy import S, im, oo, simplify

from ._circle import Circle
from ._misc import load
from ._timeline import Collision, Move, MultipleCollisions
from ._wall import Wall


class Physics:
    def __init__(self):
        data_root = Path(__file__).absolute().parent / "data"

        self._circle_circle_span = load(data_root / "cc_time_solutions.sympy")
        self._circle_wall_span = load(data_root / "cw_time_solutions.sympy")
        self._circle_circle_bounce = load(data_root / "cc_velocity_solution.sympy")
        self._circle_wall_bounce = load(data_root / "cw_velocity_limits.sympy")
        self._point_point_speed_formula = load(data_root / "instant_speed.sympy")
        self._point_wall_speed_formula = load(data_root / "instant_speed_wall.sympy")

    def _circle_circle_instant_speed(self, a: Circle, b: Circle):
        speed = self._point_point_speed_formula.subs(
            {
                "a_x": a.x,
                "a_y": a.y,
                "a_vx": a.vx,
                "a_vy": a.vy,
                "b_x": b.x,
                "b_y": b.y,
                "b_vx": b.vx,
                "b_vy": b.vy,
            }
        )

        return speed

    def _is_moving_toward(self, a: Circle, b: Any):
        eps = 0.00001
        if isinstance(b, Circle):
            speed = self._circle_circle_instant_speed(a, b)
        elif isinstance(b, Wall):
            speed = self._circle_wall_instant_speed(a, b)
        else:
            raise TypeError("2nd object must be a circle or a wall")

        complex_speed = complex(speed)
        if not (-eps < complex_speed.imag < eps):
            return False
        float_speed = complex_speed.real
        if float_speed < -eps:
            return False
        # bug: Does not handle the case where speed is still complex
        # bug: see 11/16/2022 8:52AM infinite_precision 5,10
        if float_speed > eps:
            return True
        return simplify(speed) > 0

    def _circle_wall_instant_speed(self, a: Circle, w: Wall):
        speed = self._point_wall_speed_formula.subs(
            {
                "a_x": a.x,
                "a_y": a.y,
                "a_vx": a.vx,
                "a_vy": a.vy,
                "x_0": w.x0,
                "y_0": w.y0,
                "x_1": w.x1,
                "y_1": w.y1,
            }
        )

        return speed

    def _circle_circle_spans(self, a: Circle, b: Circle):
        result_list = [
            time_solution.subs(
                [
                    ("a_x", a.x),
                    ("a_y", a.y),
                    ("a_r", a.r),
                    ("a_vx", a.vx),
                    ("a_vy", a.vy),
                    ("b_x", b.x),
                    ("b_y", b.y),
                    ("b_r", b.r),
                    ("b_vx", b.vx),
                    ("b_vy", b.vy),
                ]
            )
            for time_solution in self._circle_circle_span
        ]
        return result_list

    def _circle_wall_spans(self, a: Circle, w: Wall):
        result_list = [
            time_solution.subs(
                [
                    ("a_x", a.x),
                    ("a_y", a.y),
                    ("a_r", a.r),
                    ("a_vx", a.vx),
                    ("a_vy", a.vy),
                    ("x_0", w.x0),
                    ("y_0", w.y0),
                    ("x_1", w.x1),
                    ("y_1", w.y1),
                ]
            )
            for time_solution in self._circle_wall_span
        ]
        return result_list

    def _spans(self, a: Circle, b: Any):
        if isinstance(b, Circle):
            return self._circle_circle_spans(a, b)
        elif isinstance(b, Wall):
            return self._circle_wall_spans(a, b)
        else:
            raise TypeError("2nd object must be a circle or a wall")

    def _circle_circle_velocities(self, a: Circle, b: Circle):
        result_list = self._circle_circle_bounce.subs(
            [
                ("a_x", a.x),
                ("a_y", a.y),
                ("a_r", a.r),
                ("a_vx", a.vx),
                ("a_vy", a.vy),
                ("a_m", a.m),
                ("b_x", b.x),
                ("b_y", b.y),
                ("b_r", b.r),
                ("b_vx", b.vx),
                ("b_vy", b.vy),
                ("b_m", b.m),
            ]
        )
        return result_list

    def _circle_wall_velocities(self, a: Circle, w: Wall):
        result_list = self._circle_wall_bounce.subs(
            [
                ("a_x", a.x),
                ("a_y", a.y),
                ("a_r", a.r),
                ("a_vx", a.vx),
                ("a_vy", a.vy),
                ("a_m", a.m),
                ("w_x0", w.x0),
                ("w_y0", w.y0),
                ("w_x1", w.x1),
                ("w_y1", w.y1),
            ]
        )
        return result_list

    def _velocities(self, a: Circle, b: Any):
        if isinstance(b, Circle):
            return self._circle_circle_velocities(a, b)
        elif isinstance(b, Wall):
            return self._circle_wall_velocities(a, b)
        else:
            raise TypeError("2nd object must be a circle or a wall")

    def _fix_ssca(ssca, circle_list, wall_list):
        # Because of multi_proc, next may contain copies rather than original
        # objects, so, we need to find the original objects.
        span, span_float, circle, any = ssca
        circle2 = circle.find_circle(circle_list)
        if isinstance(any, Circle):
            any2 = any.find_circle(circle_list)
        else:
            any2 = any.find_wall(wall_list)
        return (span, span_float, circle2, any2)

    def _move_to_collision(
        self,
        circle_list: List[Circle],
        wall_list: List[Any],
        default_tick,
        timeline,
        hint_ssca_list,
        runner,
    ):
        ssca_list = self._all_collisions_float_sorted(
            circle_list, wall_list, hint_ssca_list, runner
        )
        span, span_float, ca_list = self._find_first_collisions(ssca_list, default_tick)

        new_ssca_list = self._new_hint_list(
            circle_list, wall_list, ssca_list, span, span_float, ca_list
        )

        self._apply_span(circle_list, timeline, span, span_float)
        return span, span_float, ca_list, new_ssca_list

    def _new_hint_list(
        self, circle_list, wall_list, ssca_list, span, span_float, ca_list
    ):
        new_ssca_list = []
        assert len(set([a.id for a in circle_list + wall_list])) == len(
            circle_list
        ) + len(wall_list), "Expect unique ids"
        remove_ids = {c.id for c, _ in ca_list} | {
            a.id for _, a in ca_list if isinstance(a, Circle)
        }
        for span2, span2_float, circle, any in ssca_list:
            if circle.id in remove_ids or any.id in remove_ids:
                continue
            new_ssca_list.append((span2 - span, span2_float - span_float, circle, any))
        return new_ssca_list

    def _apply_span(self, circle_list, timeline, span, span_float):
        if span is S.Zero:
            return
        for circle in circle_list:
            before = circle.float_clone()
            circle.tick(span)
            timeline.append(
                Move(before=before, after=circle.float_clone(), span=span_float)
            )

    def _find_first_collisions(self, ssca_list, default_tick):
        eps = 0.00001
        last_span = oo
        last_span_float = np.inf
        ca_list = []
        for span, span_float, circle, any in ssca_list:
            if -eps < span_float < eps and span == 0:
                span = S.Zero

            if span_float > last_span_float + eps:
                break
            if span < last_span:
                last_span = span
                last_span_float = span_float
                ca_list = [(circle, any)]
            elif span == last_span:
                ca_list.append((circle, any))

        if last_span is oo:
            last_span, last_span_float = default_tick, float(default_tick)
            ca_list = []
        return last_span, last_span_float, ca_list

    def _combo_count(circle_list, wall_list):
        circle_count = len(circle_list)
        wall_count = len(wall_list)
        return (circle_count**2 - circle_count) / 2 + circle_count * wall_count

    def _all_collisions_float_sorted(
        self, circle_list, wall_list, hint_ssca_list, runner
    ):
        circle_id_hint_set = {ssca[2].id for ssca in hint_ssca_list} | {
            ssca[3].id for ssca in hint_ssca_list if isinstance(ssca[3], Circle)
        }
        logging.info(f"{circle_id_hint_set=}")
        pairs = []
        for index1, circle in enumerate(circle_list):
            for any in circle_list[index1 + 1 :] + wall_list:
                if circle.id in circle_id_hint_set and (
                    isinstance(any, Wall) or any.id in circle_id_hint_set
                ):
                    continue
                if (
                    circle.vx in [S.Zero, 0.0, 0]
                    and circle.vy in [S.Zero, 0.0, 0]
                    and any.vx in [S.Zero, 0.0, 0]
                    and any.vy in [S.Zero, 0.0, 0]
                ):
                    continue
                pairs.append((circle, any))
        logging.info(
            f"Looking at {len(pairs)} pairs out of {Physics._combo_count(circle_list, wall_list)} possible pairs"  # noqa E501
        )

        ssca_list = map_reduce(
            pairs, lambda pair: self._find_span(*pair), runner=runner
        )
        logging.debug("About to _fix_ssca")
        ssca_list = [
            Physics._fix_ssca(ssca, circle_list, wall_list) for ssca in ssca_list
        ] + hint_ssca_list
        logging.debug("About to sort ssa_list on floats")
        ssca_list.sort(key=lambda x: x[1])
        return ssca_list

    def _find_span(self, circle, any):
        logging.debug(f"_spans {circle.id} and {any.id}")
        eps = 0.00001
        ssca = (oo, np.inf, circle, any)
        spans = self._spans(circle, any)
        # logging.debug(f"compare {len(spans)} spans with last_span")
        for span in spans:
            span_complex = complex(span)
            if not (-eps < span_complex.imag < eps):
                continue
            span_float = span_complex.real
            if span_float < -eps:
                continue
            if span_float > ssca[1] + eps:
                continue
            if not self._is_moving_toward(circle, any):
                continue
            logging.debug(f"About to simplify in _find_span {circle.id} {any.id}")
            span = simplify(span)
            logging.debug(f"Finished simplify in _find_span {circle.id} {any.id}")
            # don't use .is_real https://github.com/sympy/sympy/issues/21892
            if im(span) != 0:
                continue
            if eps < span_float < ssca[1] - eps:
                ssca = (span, span_float, circle, any)
                continue
            logging.debug(
                f"About to compare span with zero. Its float value is {span_float}"
            )
            if span < 0:
                continue
            logging.debug(
                f"About to less than compare span with ssca[0]. Their float values are {span_float} and {ssca[1]}"  # noqa E501
            )
            if span < ssca[0]:
                ssca = (span, span_float, circle, any)
                continue
            logging.debug(
                f"About to equality compare span with ssca[0]. Their float values are {span_float} and {ssca[1]}"  # noqa E501
            )
            if span == ssca[0]:
                pass
        return ssca

    def _move_no_collision(self, circle_list: List[Circle], span):
        for circle in circle_list:
            circle.tick(span)

    # https://www.myphysicslab.com/engine2D/collision-methods-en.html
    # https://physics.stackexchange.com/questions/296767/multiple-colliding-balls
    # https://www.physicsforums.com/threads/variation-on-3-ball-elastic-collision.851560
    # https://www.physicsforums.com/threads/elastic-collision-of-3-balls.720078/

    def _change_velocity(self, ss_calist, rng, timeline):
        span, span_float, ca_list = ss_calist
        if len(ca_list) > 1:
            logging.warning(f"Multiple collisions: {len(ca_list)}")
            rng.shuffle(ca_list)
            timeline.append(MultipleCollisions(len(ca_list)))
        for circle, any in ca_list:
            velocity = self._velocity_or_none(circle, any)
            if velocity:
                before = (circle.float_clone(), any.float_clone())
                circle.vx = velocity[0]
                circle.vy = velocity[1]
                if isinstance(any, Circle):
                    any.vx = velocity[2]
                    any.vy = velocity[3]
                after = (circle.float_clone(), any.float_clone())
                timeline.append(Collision(before, after))

    def _velocity_or_none(self, circle, any):
        velocity = self._velocities(circle, any)

        eps = 0.00001
        cvx_float = float(circle.vx)
        cvy_float = float(circle.vy)
        avx_float = float(any.vx) if not isinstance(any, Wall) else 0
        avy_float = float(any.vy) if not isinstance(any, Wall) else 0

        velocity_float = [float(v) for v in velocity]
        if (
            abs(velocity_float[0] - cvx_float) > eps
            or abs(velocity_float[1] - cvy_float) > eps
        ):
            return [simplify(v) for v in velocity]
        if isinstance(any, Circle) and (
            abs(velocity_float[2] - avx_float) > eps
            or abs(velocity_float[3] - avy_float) > eps
        ):
            return [simplify(v) for v in velocity]

        velocity = [simplify(v) for v in velocity]
        if velocity[:2] == [circle.vx, circle.vy] and (
            isinstance(any, Wall) or velocity[2:] == [any.vx, any.vy]
        ):
            return None
        else:
            return velocity
