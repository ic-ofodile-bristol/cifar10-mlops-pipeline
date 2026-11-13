
from src.config import load_config


def test_load_config(tmp_path):
    config_file = tmp_path / "test.yaml"

    config_file.write_text(
        "seed: 42\n"
        "batch_size: 64\n"
        "learning_rate: 0.001\n"
    )

    config = load_config(config_file)

    assert config["seed"] == 42
    assert config["batch_size"] == 64
    assert config["learning_rate"] == 0.001
