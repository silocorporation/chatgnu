import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * Command Interpreter & Pseudocode Brain – single‑file React app
 *
 * What it does
 * 1) Takes a user command and refines keywords using a built‑in dictionary.
 * 2) Builds a structured, verbose interpretation using templates.
 * 3) Runs lightweight logic passes to enhance/normalize the interpretation (one‑time execution per run).
 * 4) Computes similarity spectrum (0–1000) for: exact same, similar, different, opposite vs. prior items & a tiny snippet library.
 * 5) Every 9 minutes, auto‑synthesizes a fresh pseudo‑code plan from all accumulated commands ("the brain").
 * 6) Stores everything in localStorage (serves as a mini in‑browser DB). No server required.
 *
 * Notes
 * - Pure React + Tailwind CSS classes for styling (Tailwind not required to run, but classes included).
 * - All logic is implemented in this file; replace/extend dictionaries and snippet library as desired.
 */

// -----------------------------
// Mini "DB" (in‑memory + localStorage persistence)
// -----------------------------
const LS_KEYS = {
  commands: "cmdbrain.commands.v1",
  brainRuns: "cmdbrain.brainruns.v1",
  snippets: "cmdbrain.snippets.v1",
  dict: "cmdbrain.dictionary.v1",
};

function loadLS(key, fallback) {
  try {
    const s = localStorage.getItem(key);
    return s ? JSON.parse(s) : fallback;
  } catch {
    return fallback;
  }
}
function saveLS(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

// -----------------------------
// Dictionary (synonyms, antonyms, stopwords)
// Extend as needed. Keep short for demo purposes.
// -----------------------------
const DEFAULT_DICTIONARY = {
  synonyms: {
    build: ["create", "make", "construct", "generate"],
    website: ["webapp", "site", "web site", "frontend"],
    command: ["instruction", "prompt", "order"],
    interpret: ["parse", "understand", "analyze"],
    refine: ["improve", "polish", "clarify"],
    keywords: ["terms", "tokens", "tags"],
    template: ["schema", "blueprint", "pattern"],
    logic: ["rules", "reasoning", "inference"],
    code: ["program", "source", "implementation"],
    snippet: ["example", "sample", "fragment"],
    schedule: ["cron", "interval", "timer"],
  },
  antonyms: {
    build: ["destroy"],
    similar: ["different", "opposite"],
    create: ["delete"],
    include: ["exclude"],
    allow: ["forbid", "deny"],
  },
  stop: new Set([
    "the","a","an","and","or","but","of","to","in","on","for","with","by","is","are","be","it","this","that",
  ]),
  replacements: [
    // one‑time enhancement / normalization passes
    { pattern: /\bkind of\b/gi, replace: "somewhat" },
    { pattern: /\bsort of\b/gi, replace: "partially" },
    { pattern: /\bvery\b/gi, replace: "highly" },
    { pattern: /\b\s+and\s+and\b/gi, replace: " and " },
    { pattern: /\s{2,}/g, replace: " " },
    { pattern: /\s+([,.;:])/g, replace: "$1" },
  ],
};

// -----------------------------
// Tiny cross‑language code snippet library (tagged)
// Extend richly in real projects
// -----------------------------
const DEFAULT_SNIPPETS = [
  {
    id: "py-requests-get",
    language: "python",
    title: "HTTP GET via requests",
    tags: ["http","get","network","fetch"],
    snippet: `import requests\nresp = requests.get(url, timeout=10)\nprint(resp.text)`
  },
  {
    id: "js-fetch-get",
    language: "javascript",
    title: "HTTP GET via fetch",
    tags: ["http","get","network","fetch"],
    snippet: `const resp = await fetch(url);\nconst text = await resp.text();\nconsole.log(text);`
  },
  {
    id: "py-sqlite",
    language: "python",
    title: "SQLite query",
    tags: ["db","sqlite","query","select"],
    snippet: `import sqlite3\ncon = sqlite3.connect('app.db')\ncur = con.cursor()\nfor row in cur.execute('SELECT * FROM items'):\n    print(row)`
  },
  {
    id: "js-sqlite-wasm",
    language: "javascript",
    title: "SQLite (WASM) demo",
    tags: ["db","sqlite","query","select"],
    snippet: `// using sql.js (WASM) \n// const db = new SQL.Database();\n// const res = db.exec('SELECT 1');\n// console.log(res);`
  },
  {
    id: "py-regex",
    language: "python",
    title: "Regex substitution",
    tags: ["regex","replace","text"],
    snippet: `import re\ntext = re.sub(r"foo","bar", text)`
  },
  {
    id: "js-regex",
    language: "javascript",
    title: "Regex replacement",
    tags: ["regex","replace","text"],
    snippet: `const result = text.replace(/foo/g, 'bar');`
  },
];

// -----------------------------
// Utilities
// -----------------------------
function normalize(str) {
  return (str || "")
    .toLowerCase()
    .replace(/[^a-z0-9_\-\s]/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}
function tokenize(str, stopset) {
  const toks = normalize(str).split(" ").filter(Boolean);
  return toks.filter((t) => !stopset.has(t));
}
function uniq(arr) {
  return Array.from(new Set(arr));
}
function expandSynonyms(keywords, dict) {
  const out = new Set(keywords);
  for (const k of keywords) {
    const syns = dict.synonyms[k];
    if (syns) syns.forEach((s) => out.add(s));
  }
  return Array.from(out);
}
function antonymsOf(keywords, dict) {
  const out = new Set();
  for (const k of keywords) {
    const ants = dict.antonyms[k];
    if (ants) ants.forEach((a) => out.add(a));
  }
  return Array.from(out);
}
function vectorize(tokens) {
  const freq = new Map();
  tokens.forEach((t) => freq.set(t, (freq.get(t) || 0) + 1));
  return freq;
}
function cosineSim(aTokens, bTokens) {
  const a = vectorize(aTokens);
  const b = vectorize(bTokens);
  const all = new Set([...a.keys(), ...b.keys()]);
  let dot = 0, a2 = 0, b2 = 0;
  all.forEach((t) => {
    const av = a.get(t) || 0;
    const bv = b.get(t) || 0;
    dot += av * bv;
  });
  a.forEach((v) => (a2 += v * v));
  b.forEach((v) => (b2 += v * v));
  if (a2 === 0 || b2 === 0) return 0;
  return dot / (Math.sqrt(a2) * Math.sqrt(b2));
}
function spectrum01k(value) {
  const v = Math.max(0, Math.min(1, value));
  return Math.round(v * 1000);
}

function scoreSnippet(snippet, tokens) {
  const tagVec = vectorize(snippet.tags);
  const tokVec = vectorize(tokens);
  // simple overlap score
  let overlap = 0;
  snippet.tags.forEach((t) => {
    if (tokVec.get(t)) overlap += 1;
  });
  // add language/keyword nudges
  if (tokens.includes(snippet.language)) overlap += 0.5;
  return overlap;
}

function enhanceOnce(text, dict) {
  let out = text;
  for (const r of dict.replacements) {
    out = out.replace(r.pattern, r.replace);
  }
  return out.trim();
}

function nowISO() { return new Date().toISOString(); }

// -----------------------------
// Pseudocode generator (from accumulated commands)
// -----------------------------
function synthesizePseudocode(allCommands, refinedDict) {
  if (!allCommands.length) return "# No commands yet. Add one above.";
  const last = allCommands[allCommands.length - 1];
  const tokens = tokenize(last.raw, refinedDict.stop);
  const expanded = expandSynonyms(tokens, refinedDict);

  const steps = [
    "Parse incoming command and detect intent, entities, constraints.",
    "Refine keywords using dictionary (synonyms/antonyms, stopword removal).",
    "Compute similarity against library and prior commands.",
    "Assemble verbose interpretation using templates.",
    "Run one-time enhancement passes (text normalization).",
    "Select best-matching code snippet(s) by tag overlap.",
    "Emit final response + suggested multi-language snippets.",
  ];

  return [
    "# PSEUDOCODE PLAN (auto-generated)",
    `# Last command at ${last.createdAt}`,
    "",
    "Input:",
    `  command = """${last.raw.replace(/\n/g, " ")}"""`,
    "",
    "Derived Keywords:",
    "  - " + expanded.join("\n  - "),
    "",
    "Algorithm Steps:",
    ...steps.map((s, i) => `  ${i+1}. ${s}`),
    "",
    "Output:",
    "  - verbose_interpretation",
    "  - similarity_spectrum (0..1000)",
    "  - recommended_snippets",
  ].join("\n");
}

// -----------------------------
// Verbose interpretation template builder
// -----------------------------
function buildVerboseTemplate({ raw, tokens, expanded, antonyms, similarScore, differentScore }) {
  const facts = [
    { k: "Command", v: raw },
    { k: "Primary Keywords", v: tokens.join(", ") || "(none)" },
    { k: "Expanded Keywords", v: expanded.join(", ") || "(none)" },
    { k: "Opposite Terms", v: antonyms.join(", ") || "(none)" },
  ];

  const template = [
    "# Interpretation",
    "## Facts",
    ...facts.map((f) => `- **${f.k}:** ${f.v}`),
    "",
    "## Intent",
    "- The user likely wants the system to parse, refine, and reason about the command to produce an actionable summary.",
    "",
    "## Logic",
    `- Similarity → ${similarScore}/1000 (higher = closer)`,
    `- Difference → ${differentScore}/1000 (higher = more different)`,
    "- Use antonyms to probe opposites and ensure coverage of negative space.",
    "",
    "## Output Template",
    "- Summary: <one paragraph summary>",
    "- Steps: <bullet list of actions>",
    "- Snippets: <ranked list of language/library examples>",
  ].join("\n");
  return template;
}

// -----------------------------
// Main App
// -----------------------------
export default function App() {
  const [dictionary, setDictionary] = useState(() => loadLS(LS_KEYS.dict, DEFAULT_DICTIONARY));
  const [snippets, setSnippets] = useState(() => loadLS(LS_KEYS.snippets, DEFAULT_SNIPPETS));
  const [commands, setCommands] = useState(() => loadLS(LS_KEYS.commands, []));
  const [brainRuns, setBrainRuns] = useState(() => loadLS(LS_KEYS.brainRuns, []));

  const [input, setInput] = useState("");
  const [interpretation, setInterpretation] = useState("");
  const [enhancedInterpretation, setEnhancedInterpretation] = useState("");
  const [similarList, setSimilarList] = useState([]);
  const [oppositeList, setOppositeList] = useState([]);
  const [differentList, setDifferentList] = useState([]);
  const [snippetPicks, setSnippetPicks] = useState([]);
  const [similarityScore, setSimilarityScore] = useState(0);
  const [differenceScore, setDifferenceScore] = useState(0);
  const [pseudocode, setPseudocode] = useState("");

  const intervalRef = useRef(null);

  // persist DB
  useEffect(() => saveLS(LS_KEYS.commands, commands), [commands]);
  useEffect(() => saveLS(LS_KEYS.brainRuns, brainRuns), [brainRuns]);
  useEffect(() => saveLS(LS_KEYS.snippets, snippets), [snippets]);
  useEffect(() => saveLS(LS_KEYS.dict, dictionary), [dictionary]);

  // brain scheduler: every 9 minutes
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      runBrain();
    }, 9 * 60 * 1000);
    return () => clearInterval(intervalRef.current);
  }, [commands, dictionary]);

  // helpers
  const computeSimilarityAgainstHistory = (tokens) => {
    if (!commands.length) return { sim: 0, diff: 1000, similar: [], different: [], opposite: [] };

    const all = commands.map((c) => ({
      ...c, tokens: tokenize(c.raw, dictionary.stop),
    }));

    const withScores = all.map((c) => ({
      ...c,
      score: cosineSim(tokens, c.tokens),
    }));

    const sorted = [...withScores].sort((a, b) => b.score - a.score);
    const simTop = sorted.slice(0, 5);
    const diffTop = sorted.slice(-5).reverse();

    // Opposite: use antonyms overlap heuristic
    const antonyms = antonymsOf(tokens, dictionary);
    const oppList = all
      .map((c) => ({
        ...c,
        opp: cosineSim(antonyms, c.tokens),
      }))
      .sort((a, b) => b.opp - a.opp)
      .slice(0, 5);

    return {
      sim: spectrum01k(sorted[0]?.score || 0),
      diff: spectrum01k(1 - (sorted[0]?.score || 0)),
      similar: simTop,
      different: diffTop,
      opposite: oppList,
    };
  };

  const runInterpretation = () => {
    const createdAt = nowISO();
    const raw = input.trim();
    if (!raw) return;

    const tokens = tokenize(raw, dictionary.stop);
    const expanded = expandSynonyms(tokens, dictionary);
    const antonyms = antonymsOf(tokens, dictionary);

    const simData = computeSimilarityAgainstHistory(tokens);

    // Build template
    const templ = buildVerboseTemplate({
      raw, tokens, expanded, antonyms,
      similarScore: simData.sim, differentScore: simData.diff,
    });

    // Enhance (one-time logic)
    const enhanced = enhanceOnce(templ, dictionary);

    // Snippet ranking
    const picks = [...snippets]
      .map((s) => ({ s, score: scoreSnippet(s, expanded) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map((x) => x.s);

    // Save command
    const cmd = { id: crypto.randomUUID(), raw, createdAt };
    const nextCommands = [...commands, cmd];
    setCommands(nextCommands);

    setInterpretation(templ);
    setEnhancedInterpretation(enhanced);
    setSnippetPicks(picks);
    setSimilarityScore(simData.sim);
    setDifferenceScore(simData.diff);
    setSimilarList(simData.similar);
    setDifferentList(simData.different);
    setOppositeList(simData.opposite);

    // Also refresh pseudocode preview based on accumulated commands
    setPseudocode(synthesizePseudocode(nextCommands, dictionary));
  };

  const runBrain = () => {
    const createdAt = nowISO();
    const pseudo = synthesizePseudocode(commands, dictionary);
    setPseudocode(pseudo);
    const run = { id: crypto.randomUUID(), createdAt, pseudo };
    const nextRuns = [run, ...brainRuns].slice(0, 30);
    setBrainRuns(nextRuns);
  };

  const exportJSON = () => {
    const blob = new Blob([
      JSON.stringify({ commands, brainRuns, snippets, dictionary }, null, 2),
    ], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cmdbrain-export-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const addSnippet = () => {
    const title = prompt("Snippet title?");
    if (!title) return;
    const language = prompt("Language (e.g., python, javascript)?") || "text";
    const tags = (prompt("Comma‑separated tags?") || "").split(",").map((t) => normalize(t)).filter(Boolean);
    const snippet = prompt("Paste snippet:") || "";
    const s = { id: crypto.randomUUID(), title, language: normalize(language), tags, snippet };
    const next = [...snippets, s];
    setSnippets(next);
  };

  const removeCommand = (id) => {
    setCommands(commands.filter((c) => c.id !== id));
  };

  // -----------------------------
  // UI
  // -----------------------------
  return (
    <div className="min-h-screen w-full bg-slate-50 text-slate-900 p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Command Interpreter & Pseudocode Brain</h1>
            <p className="text-sm text-slate-600">Refine → Interpret → Reason → Rank Similarity (0–1000) → Suggest Snippets → Auto‑plan every 9 minutes</p>
          </div>
          <div className="flex gap-2">
            <button onClick={exportJSON} className="px-3 py-2 rounded-2xl bg-white shadow hover:shadow-md border">Export JSON</button>
            <button onClick={addSnippet} className="px-3 py-2 rounded-2xl bg-white shadow hover:shadow-md border">Add Snippet</button>
            <button onClick={runBrain} className="px-3 py-2 rounded-2xl bg-indigo-600 text-white shadow hover:shadow-md">Run Brain Now</button>
          </div>
        </header>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Input & History */}
          <div className="col-span-1 space-y-4">
            <div className="bg-white rounded-2xl shadow border p-4">
              <h2 className="font-semibold mb-2">1) Enter a Command</h2>
              <textarea
                className="w-full rounded-xl border p-3 min-h-[120px] focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="e.g., make a website that refines keywords, builds a verbose interpretation, and finds similar code snippets"
                value={input}
                onChange={(e) => setInput(e.target.value)}
              />
              <div className="flex gap-2 justify-end mt-3">
                <button onClick={runInterpretation} className="px-3 py-2 rounded-2xl bg-indigo-600 text-white shadow hover:shadow-md">Refine & Interpret</button>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow border p-4">
              <h3 className="font-semibold mb-2">Command History</h3>
              <div className="space-y-2 max-h-72 overflow-auto pr-1">
                {commands.length === 0 && (
                  <p className="text-sm text-slate-500">No commands yet.</p>
                )}
                {commands.slice().reverse().map((c) => (
                  <div key={c.id} className="border rounded-xl p-2 flex items-start justify-between gap-2">
                    <div>
                      <div className="text-xs text-slate-500">{new Date(c.createdAt).toLocaleString()}</div>
                      <div className="text-sm whitespace-pre-wrap">{c.raw}</div>
                    </div>
                    <button onClick={() => removeCommand(c.id)} className="text-xs px-2 py-1 bg-slate-100 rounded-lg hover:bg-rose-50 hover:text-rose-600">delete</button>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Middle: Interpretation & Spectrum */}
          <div className="col-span-1 space-y-4">
            <div className="bg-white rounded-2xl shadow border p-4">
              <h2 className="font-semibold mb-2">2) Verbose Interpretation (Template)</h2>
              <pre className="whitespace-pre-wrap text-sm bg-slate-50 rounded-xl p-3 border max-h-[320px] overflow-auto">{interpretation || "(Run an interpretation to see output)"}</pre>
            </div>
            <div className="bg-white rounded-2xl shadow border p-4">
              <h2 className="font-semibold mb-2">3) After One‑Time Logic Enhancements</h2>
              <pre className="whitespace-pre-wrap text-sm bg-slate-50 rounded-xl p-3 border max-h-[240px] overflow-auto">{enhancedInterpretation || "(No enhanced output yet)"}</pre>
            </div>
            <div className="bg-white rounded-2xl shadow border p-4">
              <h2 className="font-semibold">Similarity Spectrum</h2>
              <div className="mt-2">
                <div className="text-xs text-slate-600">0 (different) → 1000 (same)</div>
                <div className="w-full h-3 bg-slate-200 rounded-full overflow-hidden mt-1">
                  <div
                    className="h-3 bg-indigo-500"
                    style={{ width: `${similarityScore / 10}%` }}
                    title={`Similarity: ${similarityScore}/1000`}
                  />
                </div>
                <div className="flex justify-between text-xs mt-1">
                  <span>Similarity: {similarityScore}</span>
                  <span>Difference: {differenceScore}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Snippets & Comparisons */}
          <div className="col-span-1 space-y-4">
            <div className="bg-white rounded-2xl shadow border p-4">
              <h2 className="font-semibold mb-2">4) Suggested Snippets</h2>
              {snippetPicks.length === 0 ? (
                <p className="text-sm text-slate-500">(Will appear after an interpretation)</p>
              ) : (
                <div className="space-y-3 max-h-[260px] overflow-auto pr-1">
                  {snippetPicks.map((s) => (
                    <div key={s.id} className="border rounded-xl p-3">
                      <div className="text-sm font-medium">{s.title} <span className="text-xs text-slate-500">({s.language})</span></div>
                      <div className="text-xs text-slate-500">tags: {s.tags.join(", ")}</div>
                      <pre className="whitespace-pre-wrap text-xs bg-slate-50 rounded-lg p-2 border mt-1">{s.snippet}</pre>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-white rounded-2xl shadow border p-4">
              <h2 className="font-semibold mb-2">5) Similar / Opposite / Different (from history)</h2>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div>
                  <div className="font-semibold mb-1">Similar</div>
                  <ul className="space-y-1 max-h-40 overflow-auto pr-1">
                    {similarList.map((x) => (
                      <li key={x.id} className="border rounded-lg p-2">{x.raw}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <div className="font-semibold mb-1">Opposite</div>
                  <ul className="space-y-1 max-h-40 overflow-auto pr-1">
                    {oppositeList.map((x) => (
                      <li key={x.id} className="border rounded-lg p-2">{x.raw}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <div className="font-semibold mb-1">Different</div>
                  <ul className="space-y-1 max-h-40 overflow-auto pr-1">
                    {differentList.map((x) => (
                      <li key={x.id} className="border rounded-lg p-2">{x.raw}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow border p-4">
              <h2 className="font-semibold mb-2">6) Brain (every 9 minutes)</h2>
              <pre className="whitespace-pre-wrap text-xs bg-slate-50 rounded-lg p-2 border max-h-56 overflow-auto">{pseudocode || "(No plan yet)"}</pre>
              <div className="text-xs text-slate-500 mt-2">History</div>
              <div className="space-y-2 max-h-44 overflow-auto pr-1">
                {brainRuns.map((r) => (
                  <details key={r.id} className="border rounded-lg p-2">
                    <summary className="text-xs cursor-pointer select-none">{new Date(r.createdAt).toLocaleString()}</summary>
                    <pre className="whitespace-pre-wrap text-xs bg-slate-50 rounded-lg p-2 border mt-2">{r.pseudo}</pre>
                  </details>
                ))}
              </div>
            </div>
          </div>
        </section>

        <footer className="text-xs text-slate-500 pt-4 border-t">
          <p>All data persists locally in your browser (localStorage). Extend dictionaries & snippets inside the source. © {new Date().getFullYear()}</p>
        </footer>
      </div>
    </div>
  );
}
