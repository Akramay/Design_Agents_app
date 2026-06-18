from adaptive_tutor.blackboard import Blackboard


def test_write_and_read():

    bb = Blackboard(save_path="temp.json")

    bb.write("name", "Mostafa")

    assert bb.read("name") == "Mostafa"