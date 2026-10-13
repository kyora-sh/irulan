"""Authorship and language filter tests. Every fixture is invented."""

import collections
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irulan import extract_cc, filters  # noqa: E402

TYPED = "kk sick\n\ncan u rq check the queue?\n\nfeels p slow rn, wdyt"


def check(text, **cfg):
    return filters.author_check(text, cfg)


class TypedTextTests(unittest.TestCase):
    def test_own_typing_is_untouched(self):
        for text in (TYPED, "yo following thing\n\nthe bbox lags on scroll -> can fix?",
                     "Okay so the deploy went fine\n\nnow lets do the docs pr"):
            out, reason, info = check(text)
            self.assertIsNone(reason, text)
            self.assertEqual(out, text)
            self.assertEqual(info["stripped"], [])

    def test_shorthand_lowers_the_score(self):
        formal = "The cache is warm now. The queue drains in order. Nothing else changed."
        casual = "the cache is warm now tho. queue drains in order rn. nothing else changed ig."
        self.assertGreater(filters.paragraph_score(formal)[0], filters.paragraph_score(casual)[0])


class PasteStripTests(unittest.TestCase):
    def test_quoted_assistant_paragraph_is_stripped(self):
        text = ("The widget service now retries twice before failing. I've also added a short backoff, "
                "so a flaky upstream no longer takes the page down. Your call on whether to ship it today.\n\n"
                "nah ship it tmrw\n\nbut pls add a test for the backoff first")
        out, reason, info = check(text)
        self.assertIsNone(reason)
        self.assertEqual(out, "nah ship it tmrw\n\nbut pls add a test for the backoff first")
        self.assertEqual(info["stripped"], ["register"])

    def test_markdown_report_is_stripped(self):
        text = ("## Summary\n- Moved the parser into its own module\n- Added three fake fixtures\n\n"
                "lgtm, can u open the pr rq?")
        out, reason, info = check(text)
        self.assertIsNone(reason)
        self.assertEqual(out, "lgtm, can u open the pr rq?")
        self.assertIn("llm_output", info["stripped"])

    def test_log_paste_is_stripped(self):
        logs = "\n".join("@fake/web build: step %d of 4 ok" % i for i in range(4))
        out, reason, _ = check(logs + "\n\nwhy does the build hang after this?")
        self.assertIsNone(reason)
        self.assertEqual(out, "why does the build hang after this?")
        trace = "fakepkg.errors.WidgetError: widget 7 not found\n[SQL: SELECT id FROM widgets]"
        out, reason, _ = check(trace + "\n\ncan rq migrate the local db?")
        self.assertEqual(out, "can rq migrate the local db?")

    def test_quote_markers_are_stripped(self):
        text = "▎ the fake runner restarts every node at once\n\nis that true?? feels wrong"
        self.assertEqual(check(text)[0], "is that true?? feels wrong")
        text = "- the fake job keeps its lock until the end;\n\nwhy would it need that lock"
        self.assertEqual(check(text)[0], "why would it need that lock")

    def test_paste_with_too_little_left_is_dropped(self):
        text = ("Here's the plan. First we rename the fake module. Then we update every caller "
                "and run the suite again.\n\ngo")
        self.assertEqual(check(text)[1], "not_typed_register")

    def test_lead_in_sentence_goes_with_the_paste(self):
        text = ("Help me tidy these fake settings.\n\n"
                "For each one, explain the change first. Then ask before running anything. "
                "Never edit global config without asking.")
        self.assertEqual(check(text)[1], "not_typed_register")


class WholeMessageTests(unittest.TestCase):
    def test_transcript(self):
        text = ("Speaker 1: so how was the trip\nSpeaker 2: Um, it was fine. Yeah.\n"
                "Speaker 1: Okay. And the fake demo?\nSpeaker 2: Mm-hmm. It went well, I think.")
        self.assertEqual(check(text)[1], "not_typed_transcript")
        filler = " ".join(["Okay. Um, so the thing is, uh, we tried it. Yeah."] * 6)
        self.assertEqual(check(filler)[1], "not_typed_transcript")

    def test_profile(self):
        text = ("View Jane Example’s profile\nJane Example • 2nd\nFake Corp logo\n"
                "Engineer\nFake Corp · Full-time\nJan 2020 - Present · 5 yrs\n500+ connections")
        self.assertEqual(check(text)[1], "not_typed_profile")

    def test_email(self):
        text = "From: someone@example.test\nTo: other@example.test\nSubject: fake invoice\n\nplease pay"
        self.assertEqual(check(text)[1], "not_typed_email")

    def test_length_outlier(self):
        self.assertEqual(check("word " * 700)[1], "not_typed_length_outlier")
        self.assertIsNone(check("word " * 700, max_words=1000)[1])

    def test_report_lines(self):
        text = ("Goal: serve markdown to fake agents\n\nIssue: the fake site sends html only\n\n"
                "Fix: add a markdown route\n\ncan u add that via a pr?")
        out, reason, info = check(text)
        self.assertEqual(out, "can u add that via a pr?")
        self.assertIn("report", info["stripped"])


class ConfigTests(unittest.TestCase):
    QUOTE = ("The fake index rebuilds every night at two. It skips archived rows. Nothing else reads it today.\n\n"
             "ok so we can drop the cron?")

    def test_threshold_is_configurable(self):
        self.assertEqual(check(self.QUOTE)[0], "ok so we can drop the cron?")
        self.assertEqual(check(self.QUOTE, par_threshold=99)[0], self.QUOTE)

    def test_disabled(self):
        self.assertEqual(check(self.QUOTE, enabled=False)[0], self.QUOTE)

    def test_load_filter_config_rejects_unknown_keys(self):
        cfg = extract_cc.load_filter_config(None, {"par_threshold": 5, "max_words": None})
        self.assertEqual(cfg["par_threshold"], 5)
        self.assertEqual(cfg["max_words"], filters.AUTHOR_DEFAULTS["max_words"])


class TemplateTests(unittest.TestCase):
    def test_recurring_lines_across_sessions(self):
        line = "Review the attached fake diff and list every risky change you find"
        recs = [{"session_id": "s%d" % i, "text": line + "\n\nok"} for i in range(4)]
        recs.append({"session_id": "s9", "text": "can rq review the fake diff and list the risky bits pls"})
        hot = filters.template_lines(recs, min_words=8, min_sessions=4)
        self.assertEqual(hot, {filters._norm_line(line)})
        out, reason, info = filters.author_check(line + "\n\nwhat do u think about the second one?", hot_lines=hot)
        self.assertEqual(out, "what do u think about the second one?")
        self.assertIn("template", info["stripped"])

    def test_own_stock_phrases_are_not_templates(self):
        line = "can rq fire up a fake reviewer on this pr before we merge"
        recs = [{"session_id": "s%d" % i, "text": line} for i in range(6)]
        self.assertEqual(filters.template_lines(recs, min_words=8, min_sessions=4), set())


class LanguageTests(unittest.TestCase):
    def test_mainly_german_is_dropped(self):
        text = "ich glaube das ist noch nicht fertig, aber wir können es morgen machen"
        self.assertEqual(filters.language_check(text), (None, "non_english", 0))

    def test_mainly_english_with_a_few_german_words_stays(self):
        text = "kk the fake deploy is done, passt, now lets check the logs and the queue"
        self.assertEqual(filters.language_check(text), (text, None, 0))

    def test_german_paragraph_is_stripped(self):
        text = ("ich glaube das ist noch nicht fertig, aber wir können es morgen machen\n\n"
                "does this read ok? want to send it to the fake team")
        out, reason, n = filters.language_check(text)
        self.assertIsNone(reason)
        self.assertEqual(out, "does this read ok? want to send it to the fake team")
        self.assertEqual(n, 1)

    def test_umlauts_count_as_german(self):
        en, de = filters.language_counts("schön grüße")
        self.assertEqual((en, de), (0, 2))


class AuthorPassTests(unittest.TestCase):
    def rec(self, text, sid="s1"):
        return {"project": "p", "session_id": sid, "timestamp": "t", "text": text}

    def test_drops_strips_and_logs(self):
        german = "ich glaube das ist noch nicht fertig, aber wir können es morgen machen"
        quoted = ("The fake cache expires every hour on the hour. It refills on demand. Nothing breaks when it does."
                  "\n\nso we can skip warmup?")
        records = [self.rec(TYPED), self.rec(german), self.rec(quoted)]
        dropped, counts, log = collections.Counter(), collections.Counter(), []
        cfg = dict(filters.AUTHOR_DEFAULTS)
        out = extract_cc.author_pass(records, cfg, False, dropped, counts, log)
        self.assertEqual([r["text"] for r in out], [TYPED, "so we can skip warmup?"])
        self.assertEqual(dropped["non_english"], 1)
        self.assertEqual(counts["author_messages_stripped"], 1)
        self.assertEqual(sorted((e["action"], e["reason"]) for e in log),
                         [("drop", "non_english"), ("strip", "stripped_register")])

    def test_keep_non_english(self):
        german = "ich glaube das ist noch nicht fertig, aber wir können es morgen machen"
        dropped, counts, log = collections.Counter(), collections.Counter(), []
        out = extract_cc.author_pass([self.rec(german)], dict(filters.AUTHOR_DEFAULTS), True, dropped, counts, log)
        self.assertEqual(len(out), 1)
        self.assertEqual(log, [])


if __name__ == "__main__":
    unittest.main()
