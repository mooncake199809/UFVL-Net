_base_ = [
    './chess.py'
]

scene='fire'
data = dict(
    train=dict(
        scene=scene),
    val=dict(
        scene=scene),
    test=dict(
        scene=scene))

