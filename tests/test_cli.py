from llm_teams.cli import main


def test_help_runs(capsys):
    assert main([]) == 0
    assert "llm-teams" in capsys.readouterr().out
