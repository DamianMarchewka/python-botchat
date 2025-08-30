from train import open_json

def test_open_json():
    data = open_json("intents.json")
    assert isinstance(data, dict)
    assert "intents" in data
    assert len(data["intents"]) > 0