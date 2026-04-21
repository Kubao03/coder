import pytest
from coder.permissions import parse_rule, match_rule, load_rules_from_settings


class TestParseRule:
    def test_tool_only(self):
        r = parse_rule("Bash")
        assert r.tool_name == "Bash"
        assert r.content is None

    def test_tool_with_content(self):
        r = parse_rule("Bash(git *)")
        assert r.tool_name == "Bash"
        assert r.content == "git *"

    def test_tool_with_path(self):
        r = parse_rule("Read(src/**)")
        assert r.tool_name == "Read"
        assert r.content == "src/**"

    def test_escaped_parens(self):
        r = parse_rule(r"Bash(python -c \"print\(1\)\")")
        assert "print(1)" in r.content

    def test_invalid_rule(self):
        with pytest.raises(ValueError):
            parse_rule("123invalid")

    def test_str_roundtrip(self):
        r = parse_rule("Bash(git *)")
        assert str(r) == "Bash(git *)"

    def test_str_no_content(self):
        r = parse_rule("Read")
        assert str(r) == "Read"


class TestMatchRule:
    def test_tool_only_matches_all(self):
        r = parse_rule("Bash")
        assert match_rule(r, "Bash", {"command": "ls"})
        assert match_rule(r, "Bash", {"command": "rm -rf /"})

    def test_tool_only_wrong_tool(self):
        r = parse_rule("Bash")
        assert not match_rule(r, "Read", {"file_path": "/tmp"})

    def test_bash_prefix(self):
        r = parse_rule("Bash(git *)")
        assert match_rule(r, "Bash", {"command": "git status"})
        assert match_rule(r, "Bash", {"command": "git push"})
        assert not match_rule(r, "Bash", {"command": "npm install"})

    def test_bash_exact(self):
        r = parse_rule("Bash(npm install)")
        assert match_rule(r, "Bash", {"command": "npm install"})
        assert match_rule(r, "Bash", {"command": "npm install --save foo"})
        assert not match_rule(r, "Bash", {"command": "npm run build"})

    def test_file_glob(self):
        r = parse_rule("Read(src/**)")
        assert match_rule(r, "Read", {"file_path": "src/main.py"})
        assert match_rule(r, "Read", {"file_path": "src/a/b.py"})
        assert not match_rule(r, "Read", {"file_path": "tests/test.py"})

    def test_write_glob(self):
        r = parse_rule("FileWrite(*.py)")
        assert match_rule(r, "FileWrite", {"file_path": "main.py"})
        assert not match_rule(r, "FileWrite", {"file_path": "main.js"})


class TestLoadRulesFromSettings:
    def test_load(self):
        rules = load_rules_from_settings(
            allow_list=["Bash(git *)", "Read"],
            deny_list=["Bash(rm -rf *)"],
            source="user",
        )
        deny_rules = [r for r in rules if r.behavior == "deny"]
        allow_rules = [r for r in rules if r.behavior == "allow"]
        assert len(deny_rules) == 1
        assert len(allow_rules) == 2

    def test_invalid_rules_skipped(self):
        rules = load_rules_from_settings(
            allow_list=["123bad", "Read"],
            deny_list=[],
            source="user",
        )
        assert len(rules) == 1
        assert rules[0].value.tool_name == "Read"
