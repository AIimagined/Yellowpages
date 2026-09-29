"""Generate README.md: a categorized index of a GitHub user's starred repos.

Usage:  python build.py            fetch stars, classify, write README.md
        python build.py --selftest run the built-in checks

Env:    GITHUB_TOKEN / GH_TOKEN    optional, raises the API rate limit
        OPENROUTER_API_KEY         optional, classifies repos the rules miss
        OPENROUTER_MODEL           optional, model used for that (default below)
        STARS_USER                 whose stars to index (default below)
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER = os.environ.get("STARS_USER", "AIimagined")
MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5.5")
ROOT = Path(__file__).parent
RULES_FILE = ROOT / "categories.json"
OVERRIDES_FILE = ROOT / "overrides.json"
UNCATEGORIZED = ("Uncategorized", "Unsorted")
LLM_CHUNK = 100  # repos per request, keeps the JSON reply small

# Topic tag -> display name. Only tags the author chose, so these are central to the repo.
FRAMEWORKS = {
    "react": "React", "nextjs": "Next.js", "vue": "Vue", "svelte": "Svelte", "angular": "Angular",
    "astro": "Astro", "nuxt": "Nuxt", "tailwindcss": "Tailwind", "electron": "Electron", "tauri": "Tauri",
    "react-native": "React Native", "flutter": "Flutter", "nodejs": "Node.js", "bun": "Bun", "deno": "Deno",
    "fastapi": "FastAPI", "django": "Django", "flask": "Flask", "laravel": "Laravel", "rails": "Rails",
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "langchain": "LangChain", "threejs": "Three.js",
    "docker": "Docker", "kubernetes": "Kubernetes", "postgresql": "PostgreSQL", "sqlite": "SQLite",
    "redis": "Redis", "ffmpeg": "FFmpeg", "playwright": "Playwright", "webgl": "WebGL", "webgpu": "WebGPU",
}


def fetch_stars(user):
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "yellowpages"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    repos, page = [], 1
    while True:
        url = f"https://api.github.com/users/{user}/starred?per_page=100&page={page}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
            batch = json.load(r)
        if not batch:
            return repos
        repos += batch
        page += 1


def normalize(text):
    return re.sub(r"[-_/\s]+", " ", text.lower())


def compile_rules(rules):
    # ponytail: first matching rule wins, so order in categories.json is priority.
    # Switch to scoring if purpose-before-platform ordering stops being enough.
    return [
        (
            (rule["category"], rule["group"]),
            re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(normalize(k)) for k in rule["match"])),
        )
        for rule in rules
    ]


def classify(repo, compiled, overrides):
    if repo["full_name"] in overrides:
        return tuple(overrides[repo["full_name"]])
    text = normalize(" ".join([repo["name"], repo.get("description") or "", *repo.get("topics", [])]))
    for target, pattern in compiled:
        if pattern.search(text):
            return target
    return UNCATEGORIZED


def llm_classify(repos, rules, key):
    """Ask an LLM (via OpenRouter) to place repos the rules missed. Returns {full_name: [category, group]}."""
    targets = {f"{r['category']} / {r['group']}": [r["category"], r["group"]] for r in rules}
    targets[" / ".join(UNCATEGORIZED)] = list(UNCATEGORIZED)
    schema = {
        "type": "object",
        "properties": {
            "repos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "group": {"type": "string", "enum": list(targets)},
                    },
                    "required": ["name", "group"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["repos"],
        "additionalProperties": False,
    }
    result = {}
    for i in range(0, len(repos), LLM_CHUNK):
        chunk = [
            {k: r.get(k) for k in ("full_name", "description", "topics", "language")}
            for r in repos[i : i + LLM_CHUNK]
        ]
        body = {
            "model": MODEL,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "classification", "strict": True, "schema": schema},
            },
            "messages": [{
                "role": "user",
                "content": "Assign each GitHub repo to the group that best matches its main purpose. "
                "Group repos that solve the same problem together. Use the Uncategorized group only "
                "when nothing fits. Return one entry per repo, using its full_name as name.\n\n"
                + json.dumps(chunk, ensure_ascii=False),
            }],
        }
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as r:
            reply = json.load(r)
        wanted = {r["full_name"] for r in chunk}
        for item in json.loads(reply["choices"][0]["message"]["content"])["repos"]:
            target = targets.get(item["group"])
            if item["name"] in wanted and target:
                result[item["name"]] = target  # "Uncategorized" is saved too, so it is asked only once
    return result


def cell(text):
    """Make text safe inside a markdown table cell."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace("|", "\\|")
    return text if len(text) <= 160 else text[:157].rstrip() + "..."


def stars(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def license_name(repo):
    spdx = (repo.get("license") or {}).get("spdx_id")
    return {None: "None", "NOASSERTION": "Custom"}.get(spdx, spdx)


def stack(repo):
    frameworks = [FRAMEWORKS[t] for t in repo.get("topics", []) if t in FRAMEWORKS][:2]
    parts = [repo.get("language") or "", ", ".join(frameworks)]
    return cell(" · ".join(p for p in parts if p)) or "-"


def updated(repo, today):
    pushed = (repo.get("pushed_at") or "")[:10]
    if repo.get("archived"):
        return f"📦 {pushed}"
    if not pushed:
        return "-"
    age = (today - datetime.strptime(pushed, "%Y-%m-%d").date()).days
    return f"{'🟢' if age < 90 else '🟡' if age < 365 else '🔴'} {pushed}"


def anchor(heading):
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def render(user, repos, placed, order, today):
    tree = {}
    for repo in repos:
        category, group = placed[repo["full_name"]]
        tree.setdefault(category, {}).setdefault(group, []).append(repo)
    rank = {target: i for i, target in enumerate(order)}
    categories = sorted(tree, key=lambda c: min(rank.get((c, g), len(rank)) for g in tree[c]))

    out = [
        "# Yellowpages",
        "",
        f"Categorized index of {len(repos)} repositories starred by "
        f"[@{user}](https://github.com/{user}?tab=stars). Repos with the same purpose sit in the same group, "
        "sorted by stars. Nothing is cloned here: every entry links to the original repository.",
        "",
        "> Generated by `build.py`. Do not edit by hand: change `categories.json` or `overrides.json` instead.",
        "",
        "**Last updated:** 🟢 under 3 months ago · 🟡 under 1 year · 🔴 over 1 year · 📦 archived  ",
        "**License:** MIT, Apache-2.0, BSD = free for commercial use · GPL, AGPL = changes must be "
        "open-sourced · None, Custom = check with the author before using  ",
        "**🔗** = project website or demo",
        "",
        "## Contents",
        "",
    ]
    for category in categories:
        count = sum(len(v) for v in tree[category].values())
        out.append(f"- [{category}](#{anchor(category)}) ({count})")
    for category in categories:
        out += ["", f"## {category}"]
        groups = sorted(tree[category], key=lambda g: rank.get((category, g), len(rank)))
        for group in groups:
            rows = sorted(tree[category][group], key=lambda r: -r["stargazers_count"])
            out += [
                "",
                f"### {group} ({len(rows)})",
                "",
                "| Repository | Details | Stack | License | Stars | Last updated |",
                "|---|---|---|---|---|---|",
            ]
            for r in rows:
                name = f"[{cell(r['full_name'])}]({r['html_url']})"
                home = r.get("homepage") or ""
                if re.fullmatch(r"https?://[^\s()|<>]+", home):
                    name += f" [🔗]({home})"
                out.append(
                    f"| {name} | {cell(r.get('description')) or '-'} | {stack(r)} | {license_name(r)} "
                    f"| {stars(r['stargazers_count'])} | {updated(r, today)} |"
                )
    return "\n".join(out) + "\n"


def main():
    rules = json.loads(RULES_FILE.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8")) if OVERRIDES_FILE.exists() else {}
    compiled = compile_rules(rules)

    repos = fetch_stars(USER)
    if not repos:
        sys.exit(f"no starred repos returned for {USER}, README left untouched")

    placed = {r["full_name"]: classify(r, compiled, overrides) for r in repos}
    # Repos that are deliberately left unsorted in overrides.json are not re-sent to the LLM.
    missed = [r for r in repos if placed[r["full_name"]] == UNCATEGORIZED and r["full_name"] not in overrides]

    key = os.environ.get("OPENROUTER_API_KEY")
    if missed and key:
        try:
            found = llm_classify(missed, rules, key)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            # Network, HTTP, or malformed-reply failure: keep going, those repos stay uncategorized.
            print(f"warning: LLM step failed ({type(e).__name__}: {e}), skipping", file=sys.stderr)
        else:
            overrides.update(found)
            placed.update({name: tuple(target) for name, target in found.items()})
            OVERRIDES_FILE.write_text(
                json.dumps(overrides, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    order = [(r["category"], r["group"]) for r in rules]
    today = datetime.now(timezone.utc).date()
    (ROOT / "README.md").write_text(render(USER, repos, placed, order, today), encoding="utf-8")
    left = sum(1 for t in placed.values() if t == UNCATEGORIZED)
    print(f"{len(repos)} repos, {left} uncategorized")


def selftest():
    compiled = compile_rules([
        {"category": "A", "group": "Memory", "match": ["agent-memory", "memory"]},
        {"category": "B", "group": "Agents", "match": ["ai-agents"]},
    ])
    repo = {"full_name": "o/mem", "name": "mem", "description": "Agent_Memory store", "topics": ["ai-agents"]}
    assert classify(repo, compiled, {}) == ("A", "Memory"), "earlier rule must win"
    assert classify(repo, compiled, {"o/mem": ["X", "Y"]}) == ("X", "Y"), "override must win"
    repo = {"full_name": "o/x", "name": "x", "description": "memoryless cache", "topics": []}
    assert classify(repo, compiled, {}) == UNCATEGORIZED, "must match whole words only"
    assert cell("a | b\n<img>") == "a \\| b &lt;img>"
    assert len(cell("x" * 500)) == 160
    assert stars(999) == "999" and stars(12345) == "12.3k"
    assert license_name({"license": None}) == "None"
    assert license_name({"license": {"spdx_id": "NOASSERTION"}}) == "Custom"
    assert anchor("Voice & Audio") == "voice--audio"
    assert stack({"language": "TypeScript", "topics": ["react", "ai", "nextjs", "vue"]}) == "TypeScript · React, Next.js"
    assert stack({"language": None, "topics": []}) == "-"
    today = datetime(2026, 9, 29).date()
    assert updated({"pushed_at": "2026-09-01T00:00:00Z"}, today) == "🟢 2026-09-01"
    assert updated({"pushed_at": "2026-01-01T00:00:00Z"}, today) == "🟡 2026-01-01"
    assert updated({"pushed_at": "2020-01-01T00:00:00Z"}, today) == "🔴 2020-01-01"
    assert updated({"pushed_at": "2026-09-01T00:00:00Z", "archived": True}, today) == "📦 2026-09-01"
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
