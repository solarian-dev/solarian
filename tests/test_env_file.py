import os

from solarian.utils.env_file import load_env_file


def test_an_env_file_sets_what_the_shell_did_not(tmp_path, monkeypatch):
    monkeypatch.setenv("FROM_SHELL", "shell")
    monkeypatch.delenv("PLAIN", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("EXPORTED", raising=False)
    path = tmp_path / ".env"
    path.write_text("# secrets\nPLAIN=a=b\nQUOTED = \"q\"\nexport EXPORTED=e\nFROM_SHELL=file\n\nnot a pair\n")

    assert load_env_file(path) == ["PLAIN", "QUOTED", "EXPORTED"]

    assert (os.environ["PLAIN"], os.environ["QUOTED"], os.environ["EXPORTED"]) == ("a=b", "q", "e")
    assert os.environ["FROM_SHELL"] == "shell"
