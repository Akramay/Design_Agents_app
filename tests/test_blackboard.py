import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "adaptive_tutor")
    )
)

from blackboard import Blackboard


def test_write_and_read():

    bb = Blackboard(save_path="temp.json")

    bb.write("name", "Mostafa")

    assert bb.read("name") == "Mostafa"