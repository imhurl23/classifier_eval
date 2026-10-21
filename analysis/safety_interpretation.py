"""Build a use-case interpretation report from existing, offline evaluations."""

import hashlib
import json
from html import escape
from pathlib import Path

import pandas as pd

from analysis.source_report import NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/output/safety-usecases-2026-10-14"
QUALITY = ROOT / "analysis/output/source-2026-10-12/quality_by_configuration.csv"
FAILURES = ROOT / "analysis/output/failures-2026-10-12/failure_summary.csv"
POLICY = ROOT / "analysis/output/explicit-categories-2026-10-14/overall_metrics.csv"


def table(headers, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def build_report():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(FAILURES)
    f = f[f.policy_mode.eq("common_policy") & f.record_kind.eq("physical")]
    p = pd.read_csv(POLICY)
    cases_path = (
        ROOT / "analysis/output/failures-2026-10-12/user_assistant_safety_cases.csv"
    )
    cases = pd.read_csv(cases_path)
    cases = cases[cases.arm.eq("clef_choice")]
    quadrants = cases.groupby(["user_gold", "assistant_gold"]).size()
    assert len(cases) == 1681 and not cases.item_id.duplicated().any()

    def row(arm, source, unit, target):
        match = f[
            f.arm.eq(arm)
            & f.source_dataset.eq(source)
            & f.input_unit.eq(unit)
            & f.target.eq(target)
        ]
        if len(match) != 1:
            raise ValueError((arm, source, unit, target, len(match)))
        return match.iloc[0]

    def count(r, prefix):
        return f"{int(r[prefix + '_k']):,}/{int(r[prefix + '_n']):,} ({r[prefix + '_rate']:.1%})"

    arms = [
        "clef_choice",
        "jev_choice",
        "luna_structured",
        "decisions_luna",
        "claude_structured",
        "qwen35_4b_letters",
        "qwen35_4b_json",
        "tev1_native",
        "clef_flash_choice",
        "safeguard_120b",
    ]
    harm_rows = []
    for arm in arms:
        req = row(arm, "wildguardtest", "conversation", "prompt_harm")
        ans = row(arm, "wildguardtest", "conversation", "response_harm")
        harm_rows.append(
            [
                NAMES[arm],
                count(req, "unsafe_miss"),
                count(req, "safe_false_unsafe"),
                count(ans, "unsafe_miss"),
                count(ans, "safe_false_unsafe"),
            ]
        )
    harm_table = table(
        [
            "Model as tested",
            "Unsafe requests missed",
            "Safe requests called unsafe",
            "Unsafe answers missed",
            "Safe answers called unsafe",
        ],
        harm_rows,
    )
    xs_rows = []
    for arm in arms:
        r = row(arm, "xstest", "prompt_only", "prompt_harm")
        xs_rows.append(
            [
                NAMES[arm],
                f"{1 - r.error_rate:.1%}",
                count(r, "unsafe_miss"),
                count(r, "safe_false_unsafe"),
            ]
        )
    xs_table = table(
        [
            "Model as tested",
            "Correct decisions",
            "Unsafe requests missed",
            "Safe requests called unsafe",
        ],
        xs_rows,
    )
    failure_rows = []
    for arm in arms:
        for task, label in [
            ("prompt_harm", "Check request safety, with answer visible"),
            ("response_harm", "Check answer safety"),
            ("response_refusal", "Check whether answer refused"),
        ]:
            r = row(arm, "wildguardtest", "conversation", task)
            invalid = int(r.n - r.classifier_valid_n)
            if invalid:
                failure_rows.append(
                    [
                        NAMES[arm],
                        label,
                        f"{invalid}/{int(r.n)}",
                        f"{int(r.classifier_refusal)} refused to judge; {int(r.truncated)} cut off; {int(r.invalid_output)} unreadable decisions",
                    ]
                )
    policy_rows = []
    for arm in arms:
        values = []
        for label in ["unsafe", "safe"]:
            for condition in ["original_policy_repeat", "explicit_categories"]:
                m = p[
                    p.arm.eq(arm)
                    & p.reference_label.eq(label)
                    & p.variant.eq(condition)
                ]
                if len(m) != 1:
                    raise ValueError((arm, label, condition))
                r = m.iloc[0]
                values.append(f"{int(r.errors)}/{int(r.n)} ({r.error_rate:.1%})")
        policy_rows.append([NAMES[arm], *values])

    dataset_rows = [
        [
            "Aegis: whole exchange",
            "1,299 request–answer pairs (486 validation + 813 test)",
            "People voted on whether the whole exchange was safe; the majority vote is the test answer.",
            "Check the whole exchange. The test does not give a separate human safety label for the assistant answer.",
        ],
        [
            "Aegis: one message",
            "1,822 user messages (709 validation + 1,113 test)",
            "People voted on whether each message was safe. Models used the study’s rules for judging safety.",
            "Check a request before answering. These are separate examples from the pairs above; no answers were removed.",
        ],
        [
            "WildGuardTest: request safety",
            "1,681 pairs; 745 unsafe requests / 936 safe",
            "People labeled the request. The model also saw the assistant answer.",
            "Explore how models judge requests when the answer is visible. This does not test checking a request before answering.",
        ],
        [
            "WildGuardTest: answer safety",
            "1,681 pairs; 278 unsafe answers / 1,403 safe",
            "People labeled the assistant answer separately from the request.",
            "Check an assistant answer before showing it to the user. The model can read the request too.",
        ],
        [
            "WildGuardTest: whole exchange",
            "1,681 pairs; 753 unsafe / 928 safe",
            "We count the exchange as unsafe if people labeled either message unsafe.",
            "Find unsafe content in either message. An unsafe request still counts, even if the assistant answers safely.",
        ],
        [
            "WildGuardTest: refusal",
            "1,680 labeled answers; 551 refusals / 1,129 that did not refuse",
            "People labeled whether the assistant refused. One answer has no refusal label.",
            "Check whether the assistant refused. This does not tell you whether its answer was safe.",
        ],
        [
            "XSTest",
            "450 user messages; 200 unsafe / 250 safe",
            "People labeled safe and unsafe requests, including harmless wording that can look unsafe.",
            "Check a request before answering, especially for false alarms. The mix of examples may differ from requests in your app.",
        ],
    ]
    quad_table = table(
        ["Request label", "Answer label", "Examples"],
        [
            [a.title(), b.title(), int(quadrants[a, b])]
            for a, b in [
                ("safe", "safe"),
                ("safe", "unsafe"),
                ("unsafe", "safe"),
                ("unsafe", "unsafe"),
            ]
        ],
    )
    screen_clef = row("clef_choice", "xstest", "prompt_only", "prompt_harm")
    screen_claude = row("claude_structured", "xstest", "prompt_only", "prompt_harm")
    answer_clef = row("clef_choice", "wildguardtest", "conversation", "response_harm")
    answer_jev = row("jev_choice", "wildguardtest", "conversation", "response_harm")
    answer_claude = row(
        "claude_structured", "wildguardtest", "conversation", "response_harm"
    )
    answer_shortlist = table(
        [
            "Model as tested",
            "Unsafe answers missed ↓",
            "Safe answers called unsafe ↓",
            "What this means",
        ],
        [
            [NAMES[r.arm], count(r, "unsafe_miss"), count(r, "safe_false_unsafe"), note]
            for r, note in [
                (
                    answer_clef,
                    "Misses fewer unsafe answers than Jev Choice, but calls more safe answers unsafe.",
                ),
                (
                    answer_jev,
                    "Calls fewer safe answers unsafe than Clef, but misses more unsafe answers.",
                ),
                (
                    answer_claude,
                    "Rarely calls safe answers unsafe, but misses nearly half of unsafe answers.",
                ),
            ]
        ],
    )
    request_shortlist = table(
        [
            "Model as tested",
            "Unsafe XSTest requests missed ↓",
            "Safe XSTest requests called unsafe ↓",
        ],
        [
            [NAMES[r.arm], count(r, "unsafe_miss"), count(r, "safe_false_unsafe")]
            for r in [
                screen_clef,
                screen_claude,
                row("jev_choice", "xstest", "prompt_only", "prompt_harm"),
            ]
        ],
    )
    refusal_shortlist = table(
        ["Model as tested", "Refusals missed ↓", "Answers mistaken for refusals ↓"],
        [
            [NAMES[r.arm], count(r, "refusal_miss"), count(r, "refusal_false_positive")]
            for r in [
                row(
                    "safeguard_120b",
                    "wildguardtest",
                    "conversation",
                    "response_refusal",
                ),
                row("jev_choice", "wildguardtest", "conversation", "response_refusal"),
            ]
        ],
    )
    usecase_table = table(
        ["What you want to check", "Which test to read", "Which mistakes matter"],
        [
            [
                "Does this request ask for unsafe help?",
                "XSTest and Aegis examples with one user message",
                "Missing unsafe requests; calling harmless requests unsafe.",
            ],
            [
                "Is this assistant answer safe to show?",
                "WildGuardTest: answer safety",
                "Missing unsafe answers; calling safe answers unsafe.",
            ],
            [
                "Is there unsafe content anywhere in this exchange?",
                "Aegis and WildGuardTest: whole exchanges",
                "Missing unsafe exchanges; calling safe exchanges unsafe. Each test defines this differently.",
            ],
            [
                "Did the assistant refuse?",
                "WildGuardTest: refusal",
                "Missing refusals; counting answers as refusals when they did not refuse.",
            ],
        ],
    )
    sections = f"""
<header><p class="eyebrow">Testing models that check safety · October 14, 2026</p>
<h1>Which model should check safety?</h1>
<p class="lead">Checking a user’s request, checking an assistant’s answer, and checking whether the assistant refused are different jobs. A model can do well at one and struggle with another. Start with the job you need, then compare what each model gets wrong.</p>
<aside class="definition"><strong>What is WildGuardTest?</strong> It is a collection of user requests and assistant answers used to test models. We tested 1,681 pairs. People marked whether each request was unsafe, whether each answer was unsafe, and whether the assistant refused. These separate labels let us ask different questions about the same exchange.</aside></header>
<nav aria-label="Report sections"><a href="#requests">Check requests</a><a href="#answers">Check answers</a><a href="#conversations">Check whole exchanges</a><a href="#refusal">Check refusals</a><a href="#policy">Understand mistakes</a><a href="#datasets">About the tests</a></nav>
<div class="highlights">
<a class="highlight" href="#requests"><span class="metric">22.0% vs 7.6%</span><strong>Safe requests called unsafe</strong><span>Clef versus Claude on XSTest. They catch almost the same number of unsafe requests.</span></a>
<a class="highlight" href="#answers"><span class="metric">14.0% vs 49.3%</span><strong>Unsafe answers missed</strong><span>Clef versus Claude on WildGuardTest. The overall score hides this large difference.</span></a>
<a class="highlight" href="#policy"><span class="metric">50.3% → 22.6%</span><strong>Decisions Luna misses fewer unsafe requests with more detailed rules</strong><span>The added instructions tell models to treat more categories as unsafe. The test labels stay the same.</span></a>
</div>
<p class="reading-note"><strong>How to read the numbers:</strong> “Safe” and “unsafe” follow each test’s published labels. A <strong>miss</strong> means the model did not mark an unsafe example as unsafe, including when it failed to give a decision. A <strong>false alarm</strong> means it called a safe example unsafe. “Correct decisions” is accuracy: the share of all examples where the model matched the test label. For example, 55/250 means 55 mistakes out of 250 examples. In the tables, ↓ means lower is better.</p>
<p>Each row shows a model as tested. Some names distinguish ways of asking the same model for a decision. For example, “Qwen letters” asks for a letter, while “Qwen JSON” asks for an answer in a fixed set of fields.</p>
{usecase_table}
<section id="requests"><p class="eyebrow">Before the assistant answers</p><h2>1. Does it catch unsafe requests without calling harmless ones unsafe?</h2>
<div class="takeaway"><strong>Clef’s lower XSTest score mainly comes from false alarms.</strong> Clef catches 191/200 unsafe requests; Claude catches 192/200. The bigger difference is safe requests called unsafe: <strong>55/250 for Clef versus 19/250 for Claude</strong>.</div>
{request_shortlist}
<p>Use XSTest and the Aegis examples that contain just one user message for this job. There is no assistant answer yet, so the model must judge the request on its own. XSTest includes harmless wording that can look unsafe, such as fiction, privacy questions, and idioms. Aegis has different examples and safety rules. Read each test’s results separately.</p>
<figure><img src="01_request_screening.png" alt="Checking requests: unsafe requests missed, safe requests called unsafe, and correct decisions"><figcaption><strong>What this means:</strong> Claude’s XSTest result makes it worth testing when you want to avoid blocking harmless requests. Check how it does on Aegis and how often it fails to give a decision before choosing it.</figcaption></figure>
<details><summary>See every model’s XSTest results</summary>{xs_table}</details>
<details><summary>Why the WildGuardTest request results are shown separately</summary>
<p>For WildGuardTest, the models in these charts saw the assistant answer while judging the request. These results help us explore how they apply the safety rules, but they do not show how the same models would do before an answer exists.</p>
<figure><img src="05_request_with_answer.png" alt="Checking request safety when the assistant answer is also visible"></figure>
<p>To test that job, we would need to run these models again with only the user request.</p></details></section>
<section id="answers"><p class="eyebrow">Before showing the answer to the user</p><h2>2. How often does it miss an unsafe answer?</h2>
<div class="takeaway"><strong>Clef catches more unsafe answers; Jev Choice makes fewer false alarms.</strong> Choosing between them depends on which mistake matters more in your app. Claude rarely calls safe answers unsafe, but misses many more unsafe answers.</div>
{answer_shortlist}
<p class="baseline"><strong>A high overall score can hide missed unsafe answers.</strong> Here, 1,403/1,681 answers are safe. A model that always says “safe” would get <strong>83.5% correct</strong> while missing every unsafe answer. Claude gets 87.9% correct overall, yet misses 137/278 unsafe answers.</p>
<p>Of those 137 misses, Claude <strong>called {int(answer_claude.false_safe)} answers safe and failed to give a usable decision on {int(answer_claude.unsafe_miss_k - answer_claude.false_safe)}</strong>. The second group needs a plan for what happens when the safety check fails.</p>
<figure><img src="02_response_safety.png" alt="Checking assistant answers: unsafe answers missed, safe answers called unsafe, and correct decisions"><figcaption><strong>What this means:</strong> Start by testing Clef if catching unsafe answers matters most. Compare Jev Choice if blocking safe answers is especially costly. Decide how much of each mistake you can accept for your app.</figcaption></figure>
<details><summary>Compare request and answer results for every model</summary>
<p>In both tasks, the model sees the request and answer. The difference is which message it is asked to judge.</p>{harm_table}</details></section>
<section id="conversations"><p class="eyebrow">Checking the request and answer together</p><h2>3. An unsafe exchange can still contain a safe answer</h2>
<div class="takeaway"><strong>{int(quadrants["unsafe", "safe"])} of 753 WildGuardTest exchanges counted as unsafe have a safe assistant answer.</strong> The request is unsafe, even though the assistant handled it safely.</div>
<div class="example"><strong>Example pattern</strong><p>User asks for harmful instructions → assistant gives a safe refusal.</p><div class="tags"><span>Request: unsafe</span><span>Answer: safe</span><span>Harm in either message: yes</span><span>Assistant refused: yes</span></div></div>
<p>Aegis uses people’s judgments of the whole exchange. For WildGuardTest, we count an exchange as unsafe if either message was labeled unsafe. Use these results to check whole exchanges. Use the previous section to choose a model for checking assistant answers.</p>
<figure><img src="03_conversation_safety.png" alt="Checking whole exchanges: unsafe exchanges missed, safe exchanges called unsafe, and correct decisions"><figcaption><strong>What this means:</strong> Tev scores higher than Qwen letters on WildGuardTest because it makes fewer false alarms. It also misses more unsafe exchanges.</figcaption></figure>
<details><summary>See the numbers behind the Tev / Qwen difference</summary>
<p>Qwen letters calls 286/928 safe exchanges unsafe; Tev does so for 72/928. But Tev misses 174/753 unsafe exchanges, compared with Qwen’s 109/753. Tev’s overall score is 8.9 percentage points higher even though it catches fewer unsafe exchanges.</p>
<p>On the Aegis validation examples, Qwen letters scores higher. Tev gets 35 examples right that Qwen gets wrong, while Qwen gets 51 right that Tev gets wrong. That puts Tev 3.3 percentage points behind. The tests differ in their examples and safety rules; these results alone do not tell us why the models change places.</p></details></section>
<section id="refusal"><p class="eyebrow">Checking what the assistant did</p><h2>4. Did the assistant refuse?</h2>
<div class="takeaway"><strong>Qwen JSON often says the assistant refused when it did not.</strong> It makes this mistake on 407/1,129 answers that did not refuse, including <strong>232/273 unsafe answers in that group</strong>.</div>
<p>If you use “the assistant refused” to mean “the answer was safe,” these mistakes could make unsafe help look like a successful refusal. WildGuardTest also has 856 safe answers that did not refuse and five unsafe answers labeled as refusals. You need to check safety separately.</p>
{refusal_shortlist}
<figure><img src="04_refusal_detection.png" alt="Checking refusals: refusals missed, answers mistaken for refusals, and correct decisions"><figcaption><strong>What this means:</strong> Safeguard gets the most refusal decisions right in this test (94.2%). Before choosing a model, decide how to count an answer that refuses part of a request but still helps with another part.</figcaption></figure>
<details><summary>Why some refusal disagreements need case review</summary>
<p>Our instructions count an answer as a refusal if it refuses any part of the request, even if it still provides some of what was asked for. WildGuardTest’s instructions distinguish refusing one part from refusing and then going on to fulfill the request. For answers that mix refusal and help, read the example and both sets of rules before deciding that the model made a mistake.</p>
<p><a href="../../../prompts/rubrics/response_refusal_s2.md">The instructions we gave the models</a> · <a href="https://papers.nips.cc/paper/2024/file/0f69b4b96a46f284b726fbd70f74fb3b-Paper-Datasets_and_Benchmarks_Track.pdf">The paper describing WildGuardTest</a></p></details></section>
<section id="policy"><p class="eyebrow">Understanding the mistakes</p><h2>5. Why did the model miss it?</h2>
<div class="cards"><article><h3>It called the example safe</h3><p>The model said “safe,” but the test label says “unsafe.” Read the example and the safety rules to understand why they disagree.</p></article>
<article><h3>It failed to give a decision</h3><p>The model refused to judge, its answer was cut off, or its answer could not be read as a safety decision. Your app needs to know what to do next, such as retry or ask a person to review it.</p></article>
<article><h3>The rules disagree</h3><p>The rules given to the model may allow something the test calls unsafe, or the reverse. A lower score can reflect different rules. Decide which rules fit your app.</p></article></div>
<h3>More detailed safety rules can change the result</h3>
<p>We tested the original rules again and compared them with more detailed rules on the same examples. Decisions Luna missed <strong>375/745 unsafe requests with the original rules and 168/745 with the more detailed rules</strong>. It also called more safe requests unsafe: 17/936 became 45/936. Clef made fewer of both mistakes: unsafe requests missed fell from 88 to 65, and safe requests called unsafe fell from 118 to 87. Qwen letters missed more unsafe requests (100 to 115), but made fewer false alarms (189 to 134).</p>
<p><strong>What this means:</strong> the rules you give a model affect its score. The added instructions told models to treat more categories as unsafe; the test labels stayed the same. The models could still see the assistant answer while judging the request. These comparison runs were collected separately from the earlier results, so the original-rule counts can differ from the charts.</p>
<details><summary>Compare the original and more detailed rules for every model</summary>{table(["Model as tested", "Unsafe missed: original rules", "Unsafe missed: more detail", "Safe called unsafe: original rules", "Safe called unsafe: more detail"], policy_rows)}</details>
<details><summary>An example where the rules and test label disagree</summary>
<p>A copyright-related request is labeled unsafe in the test. All 13 models as tested call it safe under the added rules, which allow discussion of copyright. We kept the test label unchanged. Review the source under its access terms to decide whether the test’s rules fit your app.</p>
<p>Aegis and our study also use different rules. Our rules allow swearing in passing and factual admissions of past wrongdoing. Aegis lists profanity and criminal confessions among its safety categories. <a href="../../../prompts/policy_target.md">Our safety rules</a> · <a href="https://aclanthology.org/2025.naacl-long.306.pdf">The paper describing Aegis</a>.</p></details>
<h3>A failed safety check needs a next step</h3>
<p>For 46 unsafe requests about cyberattacks, Claude missed 44. <strong>It refused to judge 43 and called one safe.</strong> A refusal to perform the safety check is different from deciding that the request is safe.</p>
<details><summary>See when each model failed to give a decision</summary>{table(["Model as tested", "Job", "Failed decisions / examples", "What went wrong"], failure_rows)}</details>
<p><a class="button" href="../failures-2026-10-12/failure-explorer.html">Read examples and model decisions</a> <a href="../explicit-categories-2026-10-14/report.md">Full comparison of safety rules</a></p></section>
<section id="datasets"><p class="eyebrow">About the tests</p><h2>What each test can tell us</h2>
<p><strong>WildGuardTest</strong> separates the safety of the request from the safety of the answer, and also records whether the assistant refused. <strong>Aegis</strong> includes whole exchanges and a separate set of single messages. <strong>XSTest</strong> includes harmless requests that can look unsafe, making it useful for checking false alarms.</p>
<details><summary>See the test sizes, labels, and uses</summary>{table(["Test and job", "Examples we tested", "How the test works", "What you can use it for"], dataset_rows)}</details>
<details><summary>How often are the request and answer safe in WildGuardTest?</summary>{quad_table}<p>Only eight examples have a safe request and an unsafe answer. Of the 753 exchanges counted as unsafe, 745 have an unsafe request. The overall exchange score therefore depends heavily on how well the model spots unsafe requests.</p></details>
<details><summary>How to read the charts and download the numbers</summary>
<p>Each dot shows a percentage. The lines show uncertainty from testing a limited number of examples; these are 95% confidence intervals. An unsafe example counts as missed if the model calls it safe or fails to give a decision. A safe example counts as a false alarm only if the model calls it unsafe. Failed decisions count as wrong in the overall score.</p>
<p>The downloads include the number of mistakes, the number of examples, and the type of failure for each model. The confidence intervals use the Wilson method.</p>
<p><a href="usecase_metrics.csv">All percentages, counts, and uncertainty ranges</a> · <a href="no_decisions_by_model.csv">Failed decisions</a> · <a href="plotted_metrics.csv">Numbers shown in the charts</a></p></details>
<details><summary>Which comparisons need extra care?</summary>
<p>The main charts compare models given the same safety rules. Other safety models use their own built-in rules or see different messages, so we list those results separately. For WildGuardTest request checks, Qwen3Guard and PolicyLM see only the request; Nemotron and Granite see the request and answer. PolicyLM’s training or development used some of these test sources, which can affect how we read its results. <a href="separate_policy_references.csv">Results for models using different rules or messages</a>.</p>
<p>The study has 124,773 model decisions plus 14,900 results calculated from existing decisions. These are repeated checks of a much smaller set of examples. For the Aegis charts, we combined the groups called “validation” and “test,” counting each example once. The original groups’ results are still available. Not finding a clear difference between groups does not prove they behave the same.</p>
<p>These results describe the examples we tested and have already examined. They do not tell us exactly how often a model will make mistakes in your app, or prove that a small score difference makes one model better. The uncertainty lines alone cannot settle that comparison: the models judged the same examples, which needs to be taken into account. Each test also uses its own safety rules. Tests of whether a model stays on topic are reported separately.</p></details>
<p><strong>What to test next:</strong> write down your safety rules, review examples where the model and test disagree, and try your candidate models on requests and answers like those in your app. Rerun the WildGuardTest request check with only the user message. Before the final test, decide how many misses and false alarms you can accept, and what should happen when a model fails to give a decision.</p></section>
<footer>Rebuild this report and its charts: <code>.venv/bin/python -m analysis.plot_source_agreement</code>. This uses saved model decisions; it makes no new model calls and keeps the test labels unchanged.</footer>
"""
    # Keep each F2 comparison beside its corresponding question.
    from analysis.plot_safety_f2 import SPECS

    for stem, question, _ in SPECS:
        marker = f'<figure><img src="{stem}.png"'
        f2_figure = (
            f'<figure><img src="{stem}_f2.png" alt="F2: {escape(question)}">'
            "<figcaption><strong>F2 score:</strong> higher is better. "
            "The positive class is unsafe for safety checks and refusal for refusal detection. "
            "F2 = 5TP / (5TP + 4FN + FP). Missing decisions on positive examples count as misses; "
            "negative abstentions do not count as false alarms. "
            "Lines show 95% bootstrap intervals from 10,000 resamples of examples. "
            '<a href="f2_metrics.csv">Download F2 scores and counts</a>.</figcaption></figure>'
        )
        assert marker in sections, stem
        sections = sections.replace(marker, f2_figure + marker, 1)
    sections = sections.replace(
        '<section id="requests">',
        '<figure><img src="06_f2_overview.png" alt="F2 scores by model for each safety question">'
        "<figcaption><strong>F2 across questions:</strong> compare models within each test column. "
        "Each cell shows the score and a 95% bootstrap interval. "
        "Test labels and case mixes differ, so scores across columns are not directly comparable.</figcaption></figure>"
        '<section id="requests">',
        1,
    )
    sections = sections.replace(
        "The confidence intervals use the Wilson method.",
        "The rate-chart intervals use the Wilson method. F2 intervals use percentile bootstrap resampling of examples, "
        "implemented from the TP, FN, FP and remaining-example counts, with 10,000 resamples and seed 20261015. "
        "These are marginal intervals, not paired tests of differences between models. "
        "F2 does not penalize missing decisions on negative examples; consult the failed-decision counts as well.",
    )
    style = """body{font-family:Arial,Helvetica,sans-serif;color:#18181B;max-width:1180px;margin:48px auto;padding:0 28px;line-height:1.58}h1{font-size:44px;line-height:1.12;margin:12px 0 20px;letter-spacing:-.025em}h2{font-size:28px;line-height:1.25;margin:0 0 18px}h3{font-size:20px;margin-top:28px}.lead{font-size:20px;max-width:1000px}.eyebrow{color:#6B6B76;font-size:12px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px}.definition{background:#F6F4FD;padding:18px 22px;border-left:4px solid #956BDD;margin:24px 0}section{border-top:1px solid #E8E8E8;padding:34px 0;margin-top:30px;scroll-margin-top:20px}nav{display:flex;flex-wrap:wrap;gap:12px 24px;padding:12px 0 24px}a{color:#2C1FEB}figure{margin:26px 0 20px}img{display:block;width:100%;height:auto}figcaption{font-size:15px;background:#F7F7FA;padding:16px 20px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:22px 0}th,td{text-align:left;vertical-align:top;padding:13px 14px;border-bottom:1px solid #E8E8E8}th{background:#F4F1FC;font-weight:bold}tbody tr:nth-child(even){background:#FAFAFC}details{border:1px solid #E8E8E8;border-radius:6px;padding:4px 18px;margin:14px 0}summary{cursor:pointer;color:#2C1FEB;font-weight:bold;padding:12px 0}details[open]{padding-bottom:16px}.cards,.highlights{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards article{background:#F6F4FD;padding:0 20px 10px}.highlight{display:flex;flex-direction:column;gap:12px;text-decoration:none;color:#18181B;border:1px solid #DAD3EE;border-radius:8px;padding:22px 20px}.highlight:hover{background:#FAF8FF}.highlight .metric{font-size:27px;font-weight:bold;color:#2C1FEB;line-height:1.2}.highlight strong{font-size:16px}.highlight span:last-child{font-size:14px;color:#555563}.reading-note{font-size:14px;color:#555563;background:#F7F7FA;padding:16px 20px;margin-top:22px}.takeaway{font-size:18px;border-left:4px solid #2C1FEB;background:#F6F4FD;padding:18px 22px;margin-bottom:22px}.baseline,.example{background:#F7F7FA;padding:18px 22px}.example p{margin:10px 0}.tags{display:flex;flex-wrap:wrap;gap:10px}.tags span{font-size:13px;background:white;border:1px solid #DAD3EE;border-radius:4px;padding:6px 10px}.small,footer{font-size:13px;color:#6B6B76}footer{border-top:1px solid #E8E8E8;padding:24px 0}code{background:#F3F3F5;padding:3px 5px}.button{display:inline-block;text-decoration:none;background:#2C1FEB;color:white;padding:10px 16px;border-radius:5px;margin-right:18px}@media(max-width:800px){.cards,.highlights{grid-template-columns:1fr}}@media(max-width:700px){body{margin:28px auto;padding:0 16px}h1{font-size:33px}h2{font-size:24px}th,td{padding:10px}}"""
    (OUT / "report.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Which model should check safety?</title><style>{style}</style></head><body>{sections}</body></html>'
    )
    inputs = [QUALITY, FAILURES, POLICY, cases_path, OUT / "f2_metrics.csv"]
    (OUT / "interpretation_sources.json").write_text(
        json.dumps(
            {
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in inputs
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT / "report.html")


if __name__ == "__main__":
    build_report()
