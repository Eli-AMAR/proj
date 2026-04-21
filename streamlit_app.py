from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import streamlit as st
from openai import OpenAI
import streamlit.components.v1 as components

APP_TITLE = "Creative Agent | Guided Venture & Campaign Workflow"
APP_HEADER = "Creative Agent"
APP_INTRO = (
    "A guided workflow for choosing the right idea, not just generating ideas.\n\n"
    "Flow: define problem → generate 5 ideas → vote → shortlist 3 → compare → choose 1 → plan → handoff to Super."
)

BRIEF_TEMPLATE = """## Problem
What are we trying to solve?

## User / Audience
Who is this for?

## Market / Geography
Which country / city / market matters most?

## Goal
What outcome matters most?

## Budget
What budget range do we have?

## Timeline
How quickly do we need to launch?

## Constraints
What must we avoid or respect?
"""

HELP_TEXT = """### How this app works
1. Define the problem.
2. Generate 5 ideas from 5 creative ideators.
3. Let 5 evaluator agents vote, then shortlist 3 ideas.
4. Compare the 3 ideas on pros/cons, price signals, complexity, and rough money potential.
5. Let evaluators vote again, then choose 1 idea.
6. Generate an execution plan.
7. Hand off to Super.

At Step 2, Step 3, and Step 4 you can generate a fresh new Step 2 if you want a new direction.
"""

EVALUATOR_AGENTS = [
    "Strategist",
    "Creative Director",
    "Performance Marketer",
    "Brand Builder",
    "Operator / Feasibility Reviewer",
]

IDEATION_AGENTS = [
    "Trend Hunter",
    "Cultural Storyteller",
    "Campaign Provocateur",
    "Social Native Creator",
    "Experience Designer",
]

LANGUAGE_GUIDE = {
    "english": "Write the full response in English.",
    "arabic": "Write the full response in Arabic. Use clear modern Arabic; Saudi flavor is welcome when helpful.",
    "bilingual": "Write the response in both English and Arabic when helpful. Put English first, then Arabic. Keep it concise and readable.",
}

SUPER_URL = "https://super.cycls.ai/"


def _get_api_key() -> str | None:
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def _client() -> OpenAI:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it in Streamlit secrets or environment variables.")
    return OpenAI(api_key=api_key)


def _model() -> str:
    try:
        return st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    except Exception:
        return os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


def _response_text(prompt: str, *, use_web_search: bool = False) -> str:
    kwargs: dict[str, Any] = {
        "model": _model(),
        "input": prompt,
    }
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search_preview"}]
    response = _client().responses.create(**kwargs)
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    raise RuntimeError("Model did not return output_text.")


def _extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        for end in range(len(text), start, -1):
            chunk = text[start:end]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:1000]}")


def _response_json(prompt: str, *, use_web_search: bool = False) -> Any:
    return _extract_json(_response_text(prompt, use_web_search=use_web_search))


def _language_line(language: str) -> str:
    return LANGUAGE_GUIDE.get(language, LANGUAGE_GUIDE["english"])


def _problem_analysis_prompt(message: str, language: str) -> str:
    return f"""
You are checking whether the user gave enough detail to define a business / campaign / product problem.
{_language_line(language)}

Return JSON only with this exact shape:
{{
  "is_complete": true or false,
  "summary": "normalized 3-6 line problem brief",
  "missing_details": ["short missing detail question", "..."],
  "next_step_message": "a short user-facing message"
}}

Be permissive.
Move to idea generation whenever you can make reasonable assumptions.
Do NOT restart the intake just because one or two details are fuzzy.
Only keep is_complete=false if a missing detail would materially change the ideas.
In most cases, proceed when you know:
- what kind of product / offer this is at a broad level
- the audience
- the market / geography
- the main goal
If budget, timeline, channel, or exact age band are fuzzy, usually proceed and make explicit assumptions.
If a user adds more detail after an earlier incomplete answer, treat it as additive context, not a new problem.
Ask for at most 3 focused missing details.

User input / accumulated context:
{message}
""".strip()


def _generate_ideas_prompt(state: dict[str, Any]) -> str:
    return f"""
You are running Step 2 of a guided ideation workflow.
{_language_line(state['language'])}

Problem brief:
{state['problem_brief']}

Generate exactly 5 ideas, each created by a different ideation persona:
{json.dumps(IDEATION_AGENTS, ensure_ascii=False)}

Return JSON only with this exact shape:
{{
  "problem_summary": "short summary",
  "ideas": [
    {{
      "id": 1,
      "generator": "one of the 5 ideation personas",
      "name": "idea name",
      "one_liner": "single sentence",
      "why_it_could_work": ["point", "point"],
      "target_fit": "why this fits the audience",
      "first_move": "what we would do first"
    }}
  ]
}}

Rules:
- Make the 5 ideas genuinely different.
- Think like 5 creative people, not one person with 5 minor rewrites.
- Keep each idea practical enough to test.
- If the problem is in Arabic, stay naturally Arabic-first.
""".strip()


def _vote_top_two_prompt(state: dict[str, Any]) -> str:
    return f"""
You are running the evaluator vote for Step 2.
{_language_line(state['language'])}

Problem brief:
{state['problem_brief']}

Ideas:
{json.dumps(state['ideas'], ensure_ascii=False, indent=2)}

Evaluators:
{json.dumps(EVALUATOR_AGENTS, ensure_ascii=False)}

Return JSON only:
{{
  "votes": [
    {{
      "evaluator": "Strategist",
      "top_two": [1, 3],
      "reason": "short reason"
    }}
  ]
}}
""".strip()


def _compare_shortlist_prompt(state: dict[str, Any]) -> str:
    shortlisted = [idea for idea in state["ideas"] if idea["id"] in state["shortlisted"]]
    return f"""
You are running Step 3: compare 3 shortlisted ideas.
{_language_line(state['language'])}

Problem brief:
{state['problem_brief']}

Shortlisted ideas:
{json.dumps(shortlisted, ensure_ascii=False, indent=2)}

Do light web-informed reasoning for price / cost / competitor signals when useful.
Give simple rough estimates, not finance-grade models.
Interpret "search price" broadly:
- likely build / launch cost
- relevant competitor or category pricing signals
- likely media / ad cost signals
- tooling / implementation cost signals if relevant

Return JSON only with this exact shape:
{{
  "comparisons": [
    {{
      "id": 1,
      "name": "idea name",
      "advantages": ["point", "point"],
      "disadvantages": ["point", "point"],
      "price_search": {{
        "build_cost_signal": "rough range with short note",
        "competitor_or_market_signal": "rough range with short note",
        "media_or_distribution_signal": "rough range with short note",
        "tooling_signal": "rough range with short note"
      }},
      "complexity": "Low / Medium / High",
      "money_estimate": "simple rough money potential estimate",
      "why_money_estimate": "1-2 short sentences",
      "risk_level": "Low / Medium / High"
    }}
  ]
}}
""".strip()


def _vote_best_prompt(state: dict[str, Any]) -> str:
    return f"""
You are running the evaluator vote for Step 3.
{_language_line(state['language'])}

Problem brief:
{state['problem_brief']}

Comparisons:
{json.dumps(state['comparisons'], ensure_ascii=False, indent=2)}

Evaluators:
{json.dumps(EVALUATOR_AGENTS, ensure_ascii=False)}

Return JSON only:
{{
  "votes": [
    {{
      "evaluator": "Strategist",
      "best": 2,
      "reason": "short reason"
    }}
  ]
}}
""".strip()


def _plan_prompt(state: dict[str, Any]) -> str:
    chosen = state.get("selected_idea") or {}
    comparison = next((x for x in state["comparisons"] if x.get("id") == chosen.get("id")), {})
    return f"""
You are running Step 4: build the execution plan.
{_language_line(state['language'])}

Problem brief:
{state['problem_brief']}

Chosen idea:
{json.dumps(chosen, ensure_ascii=False, indent=2)}

Comparison notes:
{json.dumps(comparison, ensure_ascii=False, indent=2)}

Write a clear markdown plan with these sections:
1. Selected idea
2. Why this idea won
3. Goal
4. Success metrics
5. 30-day execution plan
6. Team / roles needed
7. Budget buckets
8. Biggest risks + mitigation
9. First 3 actions to do this week

Keep it practical and operator-ready.
""".strip()


def _super_prompt(state: dict[str, Any]) -> str:
    chosen = state.get("selected_idea") or {}
    comparison = next((x for x in state["comparisons"] if x.get("id") == chosen.get("id")), {})
    return f"""
You are my execution agent. Start working immediately on this chosen idea.

Problem brief:
{state['problem_brief']}

Chosen idea:
{json.dumps(chosen, ensure_ascii=False, indent=2)}

Decision notes:
{json.dumps(comparison, ensure_ascii=False, indent=2)}

Execution plan:
{state['plan_markdown']}

Your job now:
1. Turn this into a concrete work plan.
2. Break it into milestones and tasks.
3. Identify assumptions that must be validated first.
4. Produce the first deliverables needed to start execution immediately.
5. Be proactive and practical.

Start with:
- a concise execution summary
- the first milestone
- the first 5 tasks in priority order
""".strip()


def _count_votes(vote_payload: list[dict[str, Any]], key: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in vote_payload:
        for idx in item.get(key, []):
            counts[idx] = counts.get(idx, 0) + 1
    return counts


def _init_state() -> None:
    defaults = {
        "language": "english",
        "stage": "define_problem",
        "problem_summary": "",
        "problem_brief": "",
        "raw_problem_input": "",
        "missing_details": [],
        "ideas": [],
        "round1_votes": {},
        "round1_reasoning": [],
        "shortlisted": [],
        "comparisons": [],
        "round2_votes": {},
        "round2_reasoning": [],
        "selected_idea": None,
        "plan_markdown": "",
        "super_prompt": "",
        "step1_answer": "",
        "error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_workflow(keep_language: bool = True) -> None:
    language = st.session_state.get("language", "english") if keep_language else "english"
    for k in list(st.session_state.keys()):
        if k.startswith("select_"):
            del st.session_state[k]
    _init_state()
    st.session_state.language = language
    st.session_state.stage = "define_problem"
    st.session_state.problem_summary = ""
    st.session_state.problem_brief = ""
    st.session_state.raw_problem_input = ""
    st.session_state.missing_details = []
    st.session_state.ideas = []
    st.session_state.round1_votes = {}
    st.session_state.round1_reasoning = []
    st.session_state.shortlisted = []
    st.session_state.comparisons = []
    st.session_state.round2_votes = {}
    st.session_state.round2_reasoning = []
    st.session_state.selected_idea = None
    st.session_state.plan_markdown = ""
    st.session_state.super_prompt = ""
    st.session_state.step1_answer = ""
    st.session_state.error = ""


def _regenerate_step2() -> None:
    state = st.session_state
    ideas_payload = _response_json(_generate_ideas_prompt(state))
    state.problem_summary = ideas_payload.get("problem_summary", state.problem_summary)
    state.ideas = ideas_payload.get("ideas", [])
    vote_payload = _response_json(_vote_top_two_prompt(state))
    state.round1_reasoning = vote_payload.get("votes", [])
    state.round1_votes = _count_votes(state.round1_reasoning, "top_two")
    state.shortlisted = []
    state.comparisons = []
    state.round2_votes = {}
    state.round2_reasoning = []
    state.selected_idea = None
    state.plan_markdown = ""
    state.super_prompt = ""
    state.stage = "choose_three"


def _copy_open_html(prompt: str) -> str:
    encoded = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    return f"""<div style=\"display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0 12px 0;\">
  <button onclick=\"navigator.clipboard.writeText(atob('{encoded}')).then(() => this.innerText='Copied!')\" style=\"padding:10px 16px;border:none;border-radius:10px;background:#10a37f;color:white;font-weight:600;cursor:pointer;\">Copy Prompt</button>
  <a href=\"{SUPER_URL}\" target=\"_blank\" rel=\"noopener noreferrer\" onclick=\"navigator.clipboard.writeText(atob('{encoded}'))\" style=\"padding:10px 16px;border-radius:10px;background:#2563eb;color:white;text-decoration:none;font-weight:600;display:inline-block;\">Open Super</a>
</div>
<p><strong>Recommended flow:</strong> click <strong>Copy Prompt</strong>, then <strong>Open Super</strong>, paste, and press Enter.</p>
<p><em>The prompt is auto-copied again when you click Open Super.</em></p>"""


def _render_step2() -> None:
    st.markdown("## Step 1 — Define the problem")
    st.markdown(f"**Problem summary:** {st.session_state.problem_summary}")
    st.info("The problem is now defined clearly enough to move forward.")
    st.markdown("## Step 2 — 5 ideas from 5 creative ideators")
    for idea in sorted(st.session_state.ideas, key=lambda x: x["id"]):
        with st.container(border=True):
            st.markdown(f"### {idea['id']}) {idea['name']} — {st.session_state.round1_votes.get(idea['id'], 0)} vote(s)")
            st.markdown(f"**Generated by:** {idea['generator']}")
            st.markdown(f"**One-liner:** {idea['one_liner']}")
            st.markdown("**Why it could work:**")
            for point in idea.get("why_it_could_work", []):
                st.markdown(f"- {point}")
            st.markdown(f"**Target fit:** {idea.get('target_fit', '')}")
            st.markdown(f"**First move:** {idea.get('first_move', '')}")

    st.markdown("### Evaluator votes")
    for vote in st.session_state.round1_reasoning:
        st.markdown(f"- **{vote['evaluator']}** → {vote['top_two']} — {vote['reason']}")

    c1, c2 = st.columns([2, 1])
    with c1:
        picks = st.multiselect(
            "Choose exactly 3 ideas",
            options=[idea["id"] for idea in st.session_state.ideas],
            format_func=lambda n: next(i["name"] for i in st.session_state.ideas if i["id"] == n),
            key="select_three",
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate new Step 2", use_container_width=True):
            _regenerate_step2()
            st.rerun()

    if st.button("Continue to Step 3", type="primary"):
        if len(picks) != 3:
            st.error("Please choose exactly 3 ideas.")
        else:
            st.session_state.shortlisted = picks
            comparison_payload = _response_json(_compare_shortlist_prompt(st.session_state), use_web_search=True)
            st.session_state.comparisons = comparison_payload.get("comparisons", [])
            vote_payload = _response_json(_vote_best_prompt(st.session_state))
            st.session_state.round2_reasoning = vote_payload.get("votes", [])
            st.session_state.round2_votes = _count_votes([
                {"best": [item.get("best")]} for item in st.session_state.round2_reasoning if item.get("best") is not None
            ], "best")
            st.session_state.stage = "choose_one"
            st.rerun()


def _render_step3() -> None:
    st.markdown("## Step 3 — Decide between the 3 shortlisted ideas")
    for comp in st.session_state.comparisons:
        with st.container(border=True):
            st.markdown(f"### {comp['id']}) {comp['name']} — {st.session_state.round2_votes.get(comp['id'], 0)} vote(s)")
            st.markdown("**Advantages**")
            for item in comp.get("advantages", []):
                st.markdown(f"- {item}")
            st.markdown("**Disadvantages**")
            for item in comp.get("disadvantages", []):
                st.markdown(f"- {item}")
            price = comp.get("price_search", {})
            st.markdown("**Price search**")
            st.markdown(f"- Build / launch: {price.get('build_cost_signal', '')}")
            st.markdown(f"- Competitor / market: {price.get('competitor_or_market_signal', '')}")
            st.markdown(f"- Media / distribution: {price.get('media_or_distribution_signal', '')}")
            st.markdown(f"- Tooling: {price.get('tooling_signal', '')}")
            st.markdown(f"**Complexity:** {comp.get('complexity', '')}")
            st.markdown(f"**Rough money estimate:** {comp.get('money_estimate', '')}")
            st.markdown(f"**Why this estimate:** {comp.get('why_money_estimate', '')}")
            st.markdown(f"**Risk level:** {comp.get('risk_level', '')}")

    st.markdown("### Evaluator votes")
    for vote in st.session_state.round2_reasoning:
        st.markdown(f"- **{vote['evaluator']}** → Idea {vote['best']} — {vote['reason']}")

    c1, c2 = st.columns([2, 1])
    with c1:
        choice = st.selectbox(
            "Choose 1 final idea",
            options=[c["id"] for c in st.session_state.comparisons],
            format_func=lambda n: next(c["name"] for c in st.session_state.comparisons if c["id"] == n),
            key="select_one",
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Back to new Step 2", use_container_width=True):
            _regenerate_step2()
            st.rerun()

    if st.button("Continue to Step 4", type="primary"):
        st.session_state.selected_idea = next((idea for idea in st.session_state.ideas if idea["id"] == choice), None)
        st.session_state.plan_markdown = _response_text(_plan_prompt(st.session_state))
        st.session_state.super_prompt = _super_prompt(st.session_state)
        st.session_state.stage = "delivered"
        st.rerun()


def _render_step4_and_5() -> None:
    st.markdown("## Step 4 — Plan")
    st.markdown(st.session_state.plan_markdown)
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Back to new Step 2", use_container_width=True):
            _regenerate_step2()
            st.rerun()
    with c2:
        st.info("If you do not like this direction, generate a brand-new Step 2 from the same Step 1 problem.")

    st.markdown("## Step 5 — Delivery")
    st.markdown(f"**Selected idea:** {(st.session_state.selected_idea or {}).get('name', 'N/A')}")
    components.html(_copy_open_html(st.session_state.super_prompt), height=120)
    st.text_area("Ready-to-paste Super prompt", st.session_state.super_prompt, height=320)
    st.link_button("Open Super", SUPER_URL)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💡", layout="wide")
    _init_state()

    st.title(APP_HEADER)
    st.write(APP_INTRO)

    with st.sidebar:
        st.header("Controls")
        language = st.selectbox("Language", ["english", "arabic", "bilingual"], index=["english", "arabic", "bilingual"].index(st.session_state.language))
        st.session_state.language = language
        if st.button("Reset workflow", use_container_width=True):
            _reset_workflow()
            st.rerun()
        with st.expander("Brief template"):
            st.code(BRIEF_TEMPLATE)
        with st.expander("Help"):
            st.markdown(HELP_TEXT)

    api_key = _get_api_key()
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Community Cloud → App settings → Secrets, or locally in `.streamlit/secrets.toml`.")
        st.code('OPENAI_API_KEY = "your_key_here"')
        return

    try:
        if st.session_state.stage == "define_problem":
            st.markdown("## Step 1 — Define the problem")
            if not st.session_state.raw_problem_input:
                st.markdown("I have your context. I only need a few missing details before I generate ideas.")
                st.markdown("1. What product, service, campaign, or business problem are you working on?")
                st.markdown("2. Who is the target audience and in which market or geography?")
                st.markdown("3. What is the main goal you want to achieve?")
                st.markdown("You can answer in one paragraph or paste the template below.")
                user_problem = st.text_area("Step 1 input", value=st.session_state.step1_answer, height=220, placeholder=BRIEF_TEMPLATE)
                st.session_state.step1_answer = user_problem
                if st.button("Continue from Step 1", type="primary"):
                    if not user_problem.strip():
                        st.error("Please provide the problem context first.")
                    else:
                        st.session_state.raw_problem_input = user_problem.strip()
                        st.session_state.missing_details = [
                            "What product, service, campaign, or business problem are you working on?",
                            "Who is the target audience and in which market or geography?",
                            "What is the main goal you want to achieve?",
                        ]
                        st.rerun()
                return

            st.markdown("I have your context. I only need a few missing details before I generate ideas.")
            missing = st.session_state.missing_details or [
                "What product, service, campaign, or business problem are you working on?",
                "Who is the target audience and in which market or geography?",
                "What is the main goal you want to achieve?",
            ]
            for i, q in enumerate(missing[:5], start=1):
                st.markdown(f"{i}. {q}")
            st.markdown("You can answer in one paragraph or paste the template below.")
            extra = st.text_area("Step 1 details", height=220, placeholder=BRIEF_TEMPLATE)
            if st.button("Generate Step 2", type="primary"):
                combined_input = f"{st.session_state.raw_problem_input}\n\nAdditional user input:\n{extra.strip()}"
                analysis = _response_json(_problem_analysis_prompt(combined_input, st.session_state.language))
                st.session_state.raw_problem_input = combined_input
                st.session_state.problem_summary = analysis.get("summary", "")
                st.session_state.problem_brief = analysis.get("summary", "") or combined_input
                st.session_state.missing_details = analysis.get("missing_details", [])
                if not analysis.get("is_complete", False):
                    st.warning("I still need a little more detail before generating ideas.")
                    st.rerun()
                _regenerate_step2()
                st.rerun()
            return

        if st.session_state.stage == "choose_three":
            _render_step2()
            return

        if st.session_state.stage == "choose_one":
            _render_step3()
            return

        if st.session_state.stage == "delivered":
            _render_step4_and_5()
            return

    except Exception as exc:
        st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
