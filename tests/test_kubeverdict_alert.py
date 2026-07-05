"""KubeVerdict alert sink (M7) — push anomalous signals to the alert webhook.

Unit-level: severity filtering, payload shape, best-effort vs raise, and the
stdlib urllib POST. No network — post_alerts / urlopen are mocked.
"""
import json

import pytest

import kb.alert as alert_mod
from connectors import build
from kb.alert import KubeVerdictAlertSink, post_alerts
from kb.signal import SignalRecord


def _sig(severity, score, entity="Pod/prod/api-1", metric="cpu"):
    return SignalRecord(
        entity_uid=entity,
        metric_name=metric,
        ts=1_000,
        severity=severity,
        score=score,
        method="zscore",
    )


def test_only_anomalous_signals_are_pushed(monkeypatch):
    sent = []
    monkeypatch.setattr(
        alert_mod, "post_alerts", lambda ep, alerts, **kw: sent.append((ep, alerts))
    )
    sink = KubeVerdictAlertSink("http://kv/alerts")  # default min_severity=warning
    sink.write([_sig("normal", 0.1), _sig("warning", 3.2), _sig("critical", 5.0)])

    assert len(sent) == 1
    endpoint, alerts = sent[0]
    assert endpoint == "http://kv/alerts"
    assert [a["severity"] for a in alerts] == ["warning", "critical"]
    assert alerts[0]["metric_name"] == "cpu"
    assert "text" in alerts[0]  # AnomalyResult-aligned narrative


def test_min_severity_critical_filters_warnings(monkeypatch):
    sent = []
    monkeypatch.setattr(
        alert_mod, "post_alerts", lambda ep, alerts, **kw: sent.append(alerts)
    )
    sink = KubeVerdictAlertSink("http://kv", min_severity="critical")
    sink.write([_sig("warning", 3.0), _sig("critical", 9.0)])

    assert len(sent) == 1
    assert [a["severity"] for a in sent[0]] == ["critical"]


def test_no_anomalies_no_post(monkeypatch):
    called = []
    monkeypatch.setattr(
        alert_mod, "post_alerts", lambda *a, **k: called.append(a)
    )
    KubeVerdictAlertSink("http://kv").write([_sig("normal", 0.0)])
    assert called == []  # nothing worth alerting on -> no webhook call


def test_webhook_failure_is_best_effort(monkeypatch):
    def _boom(*a, **k):
        raise ConnectionError("kv down")

    monkeypatch.setattr(alert_mod, "post_alerts", _boom)
    # default raise_on_error=False -> swallowed
    KubeVerdictAlertSink("http://kv").write([_sig("critical", 5.0)])


def test_webhook_failure_can_raise(monkeypatch):
    def _boom(*a, **k):
        raise ConnectionError("kv down")

    monkeypatch.setattr(alert_mod, "post_alerts", _boom)
    sink = KubeVerdictAlertSink("http://kv", raise_on_error=True)
    with pytest.raises(ConnectionError):
        sink.write([_sig("critical", 5.0)])


def test_invalid_min_severity_rejected():
    with pytest.raises(ValueError):
        KubeVerdictAlertSink("http://kv", min_severity="fatal")


def test_registered_and_describes():
    sink = build("kubeverdict-alert", endpoint="http://kv", min_severity="critical")
    assert isinstance(sink, KubeVerdictAlertSink)
    d = sink.describe()
    assert d["type"] == "KubeVerdictAlertSink"
    assert d["endpoint"] == "http://kv"
    assert d["min_severity"] == "critical"


def test_post_alerts_builds_authorized_json_request(monkeypatch):
    captured = {}

    class _Resp:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data)
        captured["ctype"] = req.get_header("Content-type")
        captured["auth"] = req.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr(alert_mod.urllib.request, "urlopen", _fake_urlopen)
    status = post_alerts("http://kv/alerts", [{"severity": "critical"}], token="s3cr3t")

    assert status == 202
    assert captured["url"] == "http://kv/alerts"
    assert captured["method"] == "POST"
    assert captured["body"] == {"alerts": [{"severity": "critical"}]}
    assert captured["ctype"] == "application/json"
    assert captured["auth"] == "Bearer s3cr3t"
