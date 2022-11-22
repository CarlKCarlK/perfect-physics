import logging

from sympy import Rational, S, sqrt

from perfect_physics import Circle, Physics, Wall, World


def test_circle_circle_spans():
    physics = Physics()
    times = physics._circle_circle_spans(
        Circle(x=0, y=0, r=1, vx=1, vy=0, m=1), Circle(x=10, y=0, r=1, vx=0, vy=0, m=1)
    )
    assert times == [12, 8]

    times = physics._circle_circle_spans(
        Circle(x=sqrt(3) / 2, y=0, r=1, vx=sqrt(14) / 2, vy=sqrt(14) / 2, m=1),
        Circle(
            x=sqrt(3) / 2, y=sqrt(Rational(1, 2)), r=Rational(3, 11), vx=0, vy=0, m=1
        ),
    )
    assert times == [sqrt(7) / 14 + sqrt(4641) / 154, -sqrt(4641) / 154 + sqrt(7) / 14]


def _circle_wall_spans():
    physics = Physics()
    times = physics._circle_wall_spans(
        Circle(x=0, y=0, r=1, vx=1, vy=0, m=1), Wall(x0=10, y0=0, x1=10, y1=1)
    )
    assert times == [9, 11]

    times = physics._circle_wall_spans(
        Circle(x=sqrt(3) / 2, y=0, r=1, vx=sqrt(14) / 2, vy=sqrt(14) / 2, m=1),
        Wall(x0=10, y0=0, x1=7, y1=1),
    )
    assert [float(f) for f in times] == [
        float(f)
        for f in [
            sqrt(14) * (-2 * sqrt(10) - sqrt(3) + 20) / 56,
            sqrt(14) * (-sqrt(3) + 2 * sqrt(10) + 20) / 56,
        ]
    ]


def test_spans():
    physics = Physics()

    times = physics._spans(
        Circle(x=sqrt(3) / 2, y=0, r=1, vx=sqrt(14) / 2, vy=sqrt(14) / 2, m=1),
        Wall(x0=10, y0=0, x1=7, y1=1),
    )
    assert [float(f) for f in times] == [
        float(f)
        for f in [
            sqrt(14) * (-2 * sqrt(10) - sqrt(3) + 20) / 56,
            sqrt(14) * (-sqrt(3) + 2 * sqrt(10) + 20) / 56,
        ]
    ]

    times = physics._spans(
        Circle(x=sqrt(3) / 2, y=0, r=1, vx=sqrt(14) / 2, vy=sqrt(14) / 2, m=1),
        Wall(x0=10, y0=0, x1=7, y1=1),
    )
    assert [float(f) for f in times] == [
        float(f)
        for f in [
            sqrt(14) * (-2 * sqrt(10) - sqrt(3) + 20) / 56,
            sqrt(14) * (-sqrt(3) + 2 * sqrt(10) + 20) / 56,
        ]
    ]


def test_circle_circle_velocities():
    physics = Physics()
    velocities = physics._circle_circle_velocities(
        Circle(x=0, y=0, r=1, vx=1, vy=0, m=1),
        Circle(x=2, y=2, r=1, vx=0, vy=0, m=9_999),
    )
    assert velocities == (S(1) / 10000, S(-9999) / 10000, S(1) / 10000, S(1) / 10000)

    velocities = physics._circle_circle_velocities(
        Circle(x=sqrt(3) / 2, y=0, r=1, vx=sqrt(14) / 2, vy=sqrt(14) / 2, m=1),
        Circle(
            x=sqrt(3) / 2, y=sqrt(Rational(1, 2)), r=Rational(3, 11), vx=0, vy=0, m=1
        ),
    )
    assert velocities == (sqrt(14) / 2, 0, 0, sqrt(14) / 2)


def test_circle_wall_velocities():
    physics = Physics()
    velocities = physics._circle_wall_velocities(
        Circle(x=0, y=0, r=1, vx=1, vy=0, m=1), Wall(x0=1, y0=0, x1=1, y1=1)
    )
    print(velocities)
    assert velocities == (-1, 0)

    velocities = physics._circle_wall_velocities(
        Circle(x=0, y=0, r=1, vx=1, vy=0, m=1), Wall(x0=0, y0=0, x1=1, y1=0)
    )
    print(velocities)
    assert velocities == (1, 0)

    velocities = physics._circle_wall_velocities(
        Circle(x=sqrt(3) / 2, y=0, r=1, vx=sqrt(14) / 2, vy=sqrt(14) / 2, m=1),
        Wall(x0=10, y0=0, x1=7, y1=1),
    )
    print(velocities)
    assert velocities == (sqrt(14) / 10, -7 * sqrt(14) / 10)


def test_velocities():
    physics = Physics()

    velocities = physics._velocities(
        Circle(x=sqrt(3) / 2, y=0, r=1, vx=sqrt(14) / 2, vy=sqrt(14) / 2, m=1),
        Circle(
            x=sqrt(3) / 2, y=sqrt(Rational(1, 2)), r=Rational(3, 11), vx=0, vy=0, m=1
        ),
    )
    assert velocities == (sqrt(14) / 2, 0, 0, sqrt(14) / 2)

    velocities = physics._velocities(
        Circle(x=0, y=0, r=1, vx=1, vy=0, m=1), Wall(x0=1, y0=0, x1=1, y1=1)
    )
    print(velocities)
    assert velocities == (-1, 0)


def test_bounce():
    c1 = Circle(x=0, y=0, r=1, vx=1, vy=-1, m=1)
    w1 = Wall(x0=20, y0=0, x1=20, y1=1)
    world = World([c1], [w1])
    timeline = []
    hint_ssca_list = []
    world.tick_tock(timeline, hint_ssca_list)
    assert c1.vx == -1


def test_tick():
    c1 = Circle(x=0, y=0, r=1, vx=1, vy=-1, m=1)
    w1 = Wall(x0=20, y0=0, x1=20, y1=1)
    world = World([c1], [w1])
    timeline = []
    hint_ssca_list = []
    (_, span_float, _), _ = world._tick(timeline, hint_ssca_list)
    assert span_float == 19


def test_plot(tmp_path):
    c1 = Circle(x=19, y=0, r=1, vx=1, vy=-1, m=1)
    w1 = Wall(x0=20, y0=0, x1=20, y1=1)
    world = World([c1], [w1])
    figure = world.figure()
    figure.savefig(tmp_path / "del_me.png")


def test_circle_circle_speed():
    c1 = Circle(x=0, y=0, r=1, vx=1, vy=0, m=1)
    c2 = Circle(x=2, y=2, r=1, vx=0, vy=0, m=9_999)
    physics = Physics()
    assert physics._circle_circle_instant_speed(c1, c2) == sqrt(2) / 2


def test_billiards(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    folder = tmp_path / "test_billiards/test_billiards"
    world = World.billiards(folder, rows=2, rng=1)
    world.run_to_file(folder, 6)
    World.render(folder, speed_up=10)


def test_speed_again():
    a = Circle(x=4, y=0, r=1, vx=0, vy=0, m=1, id="cue")
    b = Circle(x=6, y=0, r=1, vx=1, vy=0, m=1, id="b1")
    physics = Physics()
    assert physics._circle_circle_instant_speed(a, b) == -1
