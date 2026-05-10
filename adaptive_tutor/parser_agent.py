"""
parser_agent.py
───────────────
Responsible for:
  1. Extracting raw text from uploaded lecture (.ppt/.pdf/.docx)
  2. Running spaCy NLP to find candidate concepts
  3. Calling LLM to build ordered concept dependency graph
  4. Writing concept_graph to the blackboard

Perceives:  file_path  (on blackboard)
Acts:       concept_graph, current_concept  (written to blackboard)
"""

import json
import re
from collections import Counter

import spacy

from base_agent   import BaseAgent
from llm_provider import call_llm, parse_json


class ParserAgent(BaseAgent):

    def __init__(self, blackboard):
        super().__init__("ParserAgent", blackboard)

        # load spaCy English model for NLP
        print("  [ParserAgent] Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("  Falling back to lightweight text heuristics.")
            self.nlp = None

    # ── PERCEIVE ─────────────────────────────────────────────
    def perceive(self):
        """Read the file path from the blackboard."""
        file_path = self.blackboard.read("file_path")
        print(f"  [ParserAgent] PERCEIVE → file path: {file_path}")
        self.blackboard.log_thinking(
            "ParserAgent",
            f"I see a lecture file at: {file_path}. I will extract its concepts."
        )
        return {"file_path": file_path}

    # ── REASON ───────────────────────────────────────────────
    def reason(self, perception: dict) -> list:
        """
        Full pipeline:
          raw text → slides → noun phrases → clean candidates → LLM concept graph
        """
        path = perception["file_path"]

        # ── Step 1: extract raw text ──────────────────────
        print("\n  [ParserAgent] REASON Step 1: Extracting text from file...")
        raw_text = self._extract_text(path)
        print(f"  [ParserAgent] Extracted {len(raw_text)} characters of text")
        self.blackboard.log_thinking(
            "ParserAgent",
            f"Extracted {len(raw_text)} characters. Splitting into slides..."
        )

        # ── Step 2: split into slides ─────────────────────
        slides = self._split_into_slides(raw_text)
        print(f"  [ParserAgent] Found {len(slides)} slides/sections")

        # ── Step 3: spaCy extracts noun phrases ──────────
        print("\n  [ParserAgent] REASON Step 2: Running spaCy NLP...")
        candidates = self._extract_noun_phrases(slides)
        print(f"  [ParserAgent] Raw noun phrases found: {len(candidates)}")

        # ── Step 4: clean and rank ────────────────────────
        clean_concepts = self._clean_candidates(candidates)
        print(f"  [ParserAgent] After cleaning: {len(clean_concepts)} concepts")
        print(f"  [ParserAgent] Candidates → {clean_concepts}")
        self.blackboard.log_thinking(
            "ParserAgent",
            f"NLP found {len(clean_concepts)} candidate concepts: {clean_concepts[:5]}..."
        )

        # ── Step 5: LLM builds dependency graph ──────────
        print("\n  [ParserAgent] REASON Step 3: Asking LLM to build concept graph...")
        concept_graph = self._build_concept_graph(clean_concepts, raw_text[:3000])
        print(f"  [ParserAgent] Concept graph built with {len(concept_graph)} concepts")

        return concept_graph

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: list):
        """Write the concept graph and set the first concept."""
        self.blackboard.write("concept_graph",   decision)
        self.blackboard.write("current_concept", decision[0]["concept"])

        print("\n  [ParserAgent] ACT → Writing to blackboard:")
        print(f"  {'─'*40}")
        for i, c in enumerate(decision):
            deps = c.get("depends_on", [])
            dep_str = f" (needs: {deps})" if deps else " (no prerequisites)"
            print(f"  {i+1}. [{c['difficulty']}/5] {c['concept']}{dep_str}")
        print(f"  {'─'*40}")
        print(f"  [ParserAgent] Starting concept: {decision[0]['concept']}")

        self.blackboard.log_thinking(
            "ParserAgent",
            f"Concept graph ready. Found {len(decision)} concepts. "
            f"Starting with: '{decision[0]['concept']}' (easiest first)."
        )

    # ── PRIVATE HELPERS ───────────────────────────────────────

    def _extract_text(self, path: str) -> str:
        """Extract raw text from any supported file format."""
        ext = path.lower().split(".")[-1]

        if ext in ["ppt", "pptx"]:
            return self._extract_from_pptx(path)
        elif ext == "pdf":
            return self._extract_from_pdf(path)
        elif ext in ["docx", "doc"]:
            return self._extract_from_docx(path)
        elif ext == "txt":
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: .{ext}")

    def _extract_from_pptx(self, path: str) -> str:
        """Extract text from PowerPoint using python-pptx."""
        try:
            from pptx import Presentation
            prs = Presentation(path)
            slides_text = []
            for slide in prs.slides:
                slide_parts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_parts.append(shape.text.strip())
                if slide_parts:
                    slides_text.append("\f".join(slide_parts))
            return "\f".join(slides_text)
        except ImportError:
            # fallback to markitdown
            return self._extract_with_markitdown(path)

    def _extract_from_pdf(self, path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            return "\f".join(pages)
        except ImportError:
            return self._extract_with_markitdown(path)

    def _extract_from_docx(self, path: str) -> str:
        """Extract text from Word document."""
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except ImportError:
            return self._extract_with_markitdown(path)

    def _extract_with_markitdown(self, path: str) -> str:
        """Universal fallback using markitdown library."""
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(path)
            return result.text_content
        except ImportError:
            raise ImportError(
                "Install markitdown: pip install markitdown\n"
                "Or install specific parsers:\n"
                "  pip install python-pptx  (for .pptx)\n"
                "  pip install pymupdf      (for .pdf)\n"
                "  pip install python-docx  (for .docx)"
            )

    def _split_into_slides(self, raw_text: str) -> list:
        """Split text into slide/page sections."""
        # form feed character \f separates slides in most extractors
        slides = raw_text.split("\f")
        # also split on double newlines as backup
        if len(slides) < 3:
            slides = re.split(r"\n{3,}", raw_text)
        # filter out very short sections (slide numbers, blank slides)
        slides = [s.strip() for s in slides if len(s.strip()) > 30]
        return slides

    def _extract_noun_phrases(self, slides: list) -> list:
        """Use spaCy to extract meaningful noun phrases from all slides."""
        if self.nlp is None:
            return self._extract_candidate_phrases_fallback(slides)
        all_phrases = []
        for slide in slides:
            doc = self.nlp(slide[:1000])  # limit per slide for speed
            for chunk in doc.noun_chunks:
                text = chunk.text.lower().strip()
                # keep only multi-word phrases (more likely to be concepts)
                if len(text.split()) >= 2 and len(text) > 5:
                    all_phrases.append(text)
        return all_phrases
    def _extract_candidate_phrases_fallback(self, slides: list) -> list:
        """
        Heuristic fallback when the spaCy model is unavailable.
        It prefers short heading-like lines and repeated 2-4 word phrases.
        """
        phrases = []

        for slide in slides:
            for raw_line in slide.splitlines():
                line = re.sub(r"[^A-Za-z0-9\-\s]", " ", raw_line).strip()
                if not line:
                    continue

                words = [w for w in line.lower().split() if len(w) > 2]
                if 2 <= len(words) <= 6:
                    phrases.append(" ".join(words))

                for n in (2, 3, 4):
                    for i in range(len(words) - n + 1):
                        gram = words[i:i + n]
                        if all(word.isalpha() for word in gram):
                            phrases.append(" ".join(gram))

        return phrases

    
    def _clean_candidates(self, phrases: list) -> list:
        """
        Clean, deduplicate, and rank candidate concept phrases.
        Keep only frequent ones — frequency = importance in the lecture.
        """
        # count frequency
        freq = Counter(phrases)

        # remove generic stop-phrases that are never real concepts
        stopwords = {
            "the agent", "the system", "the user", "a number",
            "the process", "this part", "the problem", "the result",
            "the following", "the same", "the next", "the first",
            "the world", "the environment", "the set"
        }

        # keep if appears ≥ 2 times and not a stopword
        clean = [
            phrase for phrase, count in freq.most_common(30)
            if count >= 1 and phrase not in stopwords
        ]

        # remove substrings (e.g., if "agent" and "reactive agent" both appear, keep "reactive agent")
        final = []
        for phrase in clean:
            is_substring = any(
                phrase != other and phrase in other
                for other in clean
            )
            if not is_substring:
                final.append(phrase)

        return final[:20]  # max 20 candidates

    def _build_concept_graph(self, candidates: list, lecture_context: str) -> list:
        """
        Ask the LLM to select real teachable concepts from the candidate list
        and build an ordered dependency graph with proper summaries.
        """
        prompt = f"""You are an expert curriculum designer analyzing a university lecture.

Here are candidate topics extracted from the lecture (may include noise):
{candidates}

Here is the lecture content (first 3000 characters):
\"\"\"
{lecture_context}
\"\"\"

Your task: select 5-8 REAL, teachable concepts from this lecture.

STRICT RULES:
1. Each concept must be a proper academic topic (e.g. "Natural Language Processing", "Text Corpora", "Tokenization").
2. NEVER include: professor names, emails, course codes, "agenda", "contents", sentence fragments, or anything that is not a topic.
3. Order concepts from simplest to most complex (concept 1 has no prerequisites).
4. The "summary" field must be 1-2 sentences explaining what the concept IS, taken from the lecture content.
5. "difficulty" must be 1 (easiest) to 5 (hardest).
6. "depends_on" must be concept names that must be understood first (empty list for first concept).

Return ONLY valid JSON, no markdown:
[
  {{
    "concept": "Natural Language",
    "difficulty": 1,
    "depends_on": [],
    "summary": "A natural language is one developed by humans through natural use and communication, as opposed to artificial programming languages."
  }},
  ...
]"""

        try:
            raw = call_llm(prompt, max_tokens=900)
            graph = parse_json(raw)

            # Validate: each item must have a real concept name AND a real summary
            forbidden_in_concept = {"dr.", "prof", "email", "@", "university", "agenda",
                                     "contents", "maryam", "noha", "nourhan", "copyright",
                                     ".edu", ".com", "csc"}
            forbidden_in_summary = {"@", ".edu", ".com", "http", "dr.", "prof",
                                     "miuegypt", "csc0275"}
            cleaned = [
                c for c in graph
                if isinstance(c.get("concept"), str)
                and isinstance(c.get("summary"), str)
                and len(c["concept"]) <= 60
                and len(c["concept"].split()) <= 7
                and not c["concept"].endswith((",", "."))
                and not any(f in c["concept"].lower() for f in forbidden_in_concept)
                and len(c["summary"]) >= 20
                and " " in c["summary"]
                and not any(f in c["summary"].lower() for f in forbidden_in_summary)
            ]

            if len(cleaned) < 3:
                raise ValueError(f"Too few valid concepts after filtering: {cleaned}")

            return cleaned

        except Exception as e:
            print(f"  [ParserAgent] LLM concept graph failed ({e}), using fallback...")
            return self._fallback_graph_from_text(lecture_context)

    def _fallback_graph_from_text(self, text: str) -> list:
        """
        Heuristic fallback: extract slide headings as concepts.
        Produces real summaries by grabbing the first clean sentence after each heading.
        """
        # Anything matching these patterns is NEVER a concept name or a summary
        garbage_patterns = [
            "dr.", "prof", "email", "@", ".edu", ".com", ".org",
            "contents", "agenda", "lecture", "university", "copyright",
            "page", "slide", "textbook", "marks", "attendance",
            "midterm", "final exam", "csc", "miuegypt", "eng.",
        ]

        def is_garbage(s: str) -> bool:
            low = s.lower()
            return any(g in low for g in garbage_patterns)

        def looks_like_sentence(s: str) -> bool:
            """True if s is a real descriptive sentence, not an email/code/fragment."""
            if is_garbage(s):
                return False
            if len(s) < 25:
                return False
            # Must contain at least one space (i.e. multiple words)
            if " " not in s:
                return False
            # Must start with a letter, not a bullet symbol or digit
            if not s[0].isalpha():
                # Allow bullet prefix like "• word ..." by stripping it
                s = s.lstrip("•–-– ").strip()
                if not s or not s[0].isalpha():
                    return False
            return True

        lines = text.splitlines()
        headings = []
        seen = set()

        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line or len(line) < 5 or len(line) > 60:
                continue
            if is_garbage(line):
                continue
            if line.endswith(",") or not line[0].isupper():
                continue

            word_count = len(line.split())
            # Require at least 2 words so single words like "Processing" from
            # a title slide are skipped (they're never real concept headings)
            if word_count < 2 or word_count > 6:
                continue
            if line in seen:
                continue
            seen.add(line)

            # Grab the first clean descriptive line after this heading
            summary = ""
            for j in range(i + 1, min(i + 8, len(lines))):
                candidate = lines[j].strip().lstrip("•–-– ").strip()
                if looks_like_sentence(candidate):
                    summary = candidate[:180]
                    break

            if not summary:
                summary = f"{line} is a core concept covered in this lecture."

            headings.append((line, summary))

        # Skip the first 2 entries (usually course title / instructor slide)
        valid = headings[2:10] if len(headings) > 4 else headings

        return [
            {
                "concept":    h,
                "difficulty": min(i + 1, 5),
                "depends_on": [],
                "summary":    s,
            }
            for i, (h, s) in enumerate(valid)
        ]