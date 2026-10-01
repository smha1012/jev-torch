"""Fast tests with no model or dataset download:  pytest -q"""

import math

import pytest
import torch
import torch.nn.functional as F

from jev.collate import KIND_SLOTS, JEVCollator, JEVDataset, LengthGroupedBatchSampler
from jev.config import load_config
from jev.losses import fit_temperature, jev_loss, kl_per_example, rps_per_example
from jev.schema import KINDS, JEVExample


class CharTokenizer:
    """One token per character; enough to test prompt assembly and truncation."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def ex(kind="choice", options=("a", "b", "c"), target=None, label=None, state="s"):
    return JEVExample(state=state, question="q?", options=list(options), kind=kind, target=target, label=label)


# -- schema --------------------------------------------------------------------


def test_target_normalized_and_label_derived():
    e = ex(target=[1, 3, 0])
    assert e.target == [0.25, 0.75, 0.0] and e.label == 1


@pytest.mark.parametrize("bad", [dict(options=["a"], label=0), dict(label=5), dict(target=[1, 0]), dict()])
def test_schema_rejects_invalid(bad):
    kw = {"options": ["a", "b", "c"], **bad}
    with pytest.raises(ValueError):
        JEVExample(state="", question="q", **kw)


def test_unknown_fields_go_to_meta():
    e = JEVExample.from_dict({"state": "", "question": "q", "options": ["a", "b"], "label": 0, "family": "x"})
    assert e.get("family") == "x"


# -- config --------------------------------------------------------------------


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  name: foo\ntrain:\n  num_gpus: 2\n")
    cfg = load_config(p, ["train.num_gpus=8", "data.include={source: [openjev_v2]}"])
    assert cfg.model.name == "foo" and cfg.train.num_gpus == 8
    assert cfg.data.include == {"source": ["openjev_v2"]}
    with pytest.raises(ValueError):
        load_config(p, ["train.not_a_field=1"])


# -- collator ------------------------------------------------------------------


def test_slots_follow_kind():
    items = [JEVDataset([e])[0] for e in (ex("noul", ["false", "true"], label=1),
                                          ex("score", list("012345"), label=3),
                                          ex("choice", list("abcd"), label=0))]
    b = JEVCollator(CharTokenizer())(items)
    assert b["slot_index"][0, :2].tolist() == [0, 1]
    assert b["slot_index"][1, :6].tolist() == [2, 3, 4, 5, 6, 7]
    assert b["slot_index"][2, :4].tolist() == [8, 9, 10, 11]
    assert b["option_mask"].sum(1).tolist() == [2, 6, 4]
    assert b["kind"].tolist() == [KINDS.index(k) for k in ("noul", "score", "choice")]
    last = b["attention_mask"].sum(1) - 1  # every prompt ends with the decision marker
    assert all(b["input_ids"][i, last[i]] == ord(":") for i in range(3))


def test_long_state_keeps_head_and_tail():
    state = "H" * 500 + "M" * 1000 + "T" * 500
    col = JEVCollator(CharTokenizer(), max_length=300)
    b = col([JEVDataset([ex(state=state, label=0)])[0]])
    text = "".join(map(chr, b["input_ids"][0].tolist()))
    assert len(text) == 300 and text.endswith("[decision]:")
    assert "H" in text and "T" in text  # 60% head / 40% tail of the state budget survive


def test_choice_limit_enforced():
    with pytest.raises(ValueError):
        JEVDataset([ex("choice", [str(i) for i in range(KIND_SLOTS["choice"][1] + 1)], label=0)])


def test_shuffle_keeps_target_aligned():
    e = ex("choice", ["a", "b", "c", "d"], target=[0.1, 0.2, 0.3, 0.4])
    ds = JEVDataset([e], shuffle_prob=1.0)
    for _ in range(20):
        it = ds[0]
        assert dict(zip(it["options"], it["target"])) == dict(zip(e.options, e.target))
        assert it["options"][it["label"]] == "d"


def test_score_options_never_shuffled():
    ds = JEVDataset([ex("score", list("012345"), label=2)], shuffle_prob=1.0)
    assert all(ds[0]["options"] == list("012345") for _ in range(10))


# -- losses --------------------------------------------------------------------


def test_kl_equals_ce_for_one_hot():
    logits = torch.randn(4, 3)
    target = F.one_hot(torch.tensor([0, 2, 1, 1]), 3).float()
    mask = torch.ones(4, 3, dtype=torch.bool)
    ce = F.cross_entropy(logits, target.argmax(1), reduction="none")
    assert torch.allclose(kl_per_example(logits, target, mask), ce, atol=1e-5)


def test_padding_is_ignored():
    logits = torch.tensor([[1.0, 2.0, float("-inf")]])
    target = torch.tensor([[0.3, 0.7, 0.0]])
    mask = torch.tensor([[True, True, False]])
    kl = kl_per_example(logits, target, mask)
    assert torch.isfinite(kl).all()
    assert torch.allclose(kl, kl_per_example(logits[:, :2], target[:, :2], mask[:, :2]))


def test_rps_zero_when_exact_and_ordinal():
    target = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    near = torch.tensor([[0.0, 0.0, 0.0, 9.0]])  # predicts 3 when truth is 2
    far = torch.tensor([[9.0, 0.0, 0.0, 0.0]])  # predicts 0
    assert rps_per_example(target.log(), target, mask).item() < 1e-6
    assert rps_per_example(near, target, mask) < rps_per_example(far, target, mask)


def test_rps_only_on_score_rows():
    logits, target = torch.randn(2, 3), torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    choice = torch.tensor([KINDS.index("choice")] * 2)
    score = torch.tensor([KINDS.index("score")] * 2)
    assert jev_loss(logits, target, mask, choice, 0.5) == jev_loss(logits, target, mask, choice, 0.0)
    assert jev_loss(logits, target, mask, score, 0.5) > jev_loss(logits, target, mask, score, 0.0)


def test_temperature_recovers_and_never_hurts():
    torch.manual_seed(0)
    true_logits = torch.randn(4000, 4) * 2
    target = torch.softmax(true_logits, -1)
    mask = torch.ones_like(target, dtype=torch.bool)
    t = fit_temperature(true_logits * 3, target, mask)  # overconfident by 3x
    assert math.isclose(t, 3.0, rel_tol=0.05)
    assert fit_temperature(true_logits, target, mask) == pytest.approx(1.0, rel=0.02)


# -- sampler -------------------------------------------------------------------


def test_sampler_shards_disjoint_equal_and_resumable():
    lengths = list(range(1000))
    shards = [LengthGroupedBatchSampler(lengths, 8, seed=1, rank=r, world_size=4, multiple_of=2)
              for r in range(4)]
    batches = [list(s) for s in shards]
    assert len({len(b) for b in batches}) == 1 and len(batches[0]) % 2 == 0
    seen = [i for b in batches for batch in b for i in batch]
    assert len(seen) == len(set(seen))
    s = shards[0]
    s.set_epoch(0, start=5)
    assert list(s) == batches[0][5:]
    s.set_epoch(1)
    assert list(s) != batches[0]  # new order each epoch


def test_loss_weights_downweight_rows():
    e_keep = JEVExample(state="", question="q", options=["false", "true"], kind="noul", target=[0.5, 0.5],
                        meta={"source": "yuri_v1"})
    e_other = JEVExample(state="", question="q", options=["false", "true"], kind="noul", label=1,
                         meta={"source": "yuri_v3"})
    ds = JEVDataset([e_keep, e_other], loss_weights={"source": {"yuri_v1": 0.1}})
    b = JEVCollator(CharTokenizer())([ds[0], ds[1]])
    assert b["weight"].tolist() == pytest.approx([0.1, 1.0])
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.0]])  # row 0 disagrees with its [0.5, 0.5] target
    full = jev_loss(logits, b["target"], b["option_mask"], b["kind"], 0.0)
    weighted = jev_loss(logits, b["target"], b["option_mask"], b["kind"], 0.0, b["weight"])
    assert weighted < full


# -- hub / checkpoints -----------------------------------------------------------


def test_resolve_checkpoint_local_and_errors(tmp_path):
    from jev.model import resolve_checkpoint

    ckpt = tmp_path / "best"
    ckpt.mkdir()
    with pytest.raises(FileNotFoundError):  # a directory without jev_config.json
        resolve_checkpoint(ckpt)
    (ckpt / "jev_config.json").write_text("{}")
    assert resolve_checkpoint(ckpt) == ckpt
    with pytest.raises(FileNotFoundError):  # neither a directory nor "owner/name"
        resolve_checkpoint(tmp_path / "missing" / "deep" / "path")


def test_model_card_includes_usage_and_metrics(tmp_path):
    import json

    import yaml

    from jev.push_to_hub import build_model_card

    run = tmp_path / "run"
    ckpt = run / "best"
    ckpt.mkdir(parents=True)
    (ckpt / "jev_config.json").write_text(json.dumps({
        "model": {"name": "Qwen/Qwen3.5-9B", "lora_r": 16, "lora_alpha": 32},
        "temperature": {"noul": 1.0, "choice": 0.98, "score": 1.01}}))
    (run / "config.yaml").write_text(yaml.safe_dump({"train": {"max_steps": 4750, "global_batch_size": 128,
                                                               "lr": 1e-4, "head_lr": 2e-4}}))
    m = {"n": 100, "acc": 0.9, "kl": 0.02, "ece": 0.001}
    (run / "report.json").write_text(json.dumps({"test_set_30k": {"calibrated": {**m, "by_kind": {"noul": m}}}}))
    card = build_model_card(ckpt, "someone/jev-9b")
    front = yaml.safe_load(card.split("---")[1])
    assert front["base_model"] == "Qwen/Qwen3.5-9B" and "jev" in front["tags"]
    assert 'JEVPredictor("someone/jev-9b")' in card
    assert "| test_set_30k / noul | 100 | 0.900 |" in card
    assert "choice 0.980" in card


# -- Hub uploads -------------------------------------------------------------------


def test_resolve_repo_rules():
    from jev.hub import resolve_repo

    assert resolve_repo(None, "me", [], "jev-9b") is None
    assert resolve_repo("auto", "me", [], "jev-9b") == "me/jev-9b"
    assert resolve_repo("myorg/x", "me", ["myorg"], "jev-9b") == "myorg/x"
    with pytest.raises(ValueError):
        resolve_repo("otherorg/x", "me", ["myorg"], "jev-9b")  # no write access
    with pytest.raises(ValueError):
        resolve_repo("not-a-repo-id", "me", [], "jev-9b")


class FakeApi:
    def __init__(self, files=(), fail_upload=False):
        self.files, self.fail_upload, self.calls = list(files), fail_upload, []

    def whoami(self):
        return {"name": "me", "orgs": [{"name": "myorg"}]}

    def create_repo(self, repo, private, exist_ok):
        self.calls.append(("create_repo", repo, private))

    def list_repo_files(self, repo):
        return self.files

    def upload_folder(self, repo_id, folder_path, allow_patterns, commit_message):
        if self.fail_upload:
            raise RuntimeError("network down")
        import os

        self.calls.append(("upload", repo_id, commit_message, sorted(os.listdir(folder_path))))
        return type("Info", (), {"oid": "abc123"})()

    def delete_tag(self, repo, tag):
        raise RuntimeError("no such tag")

    def create_tag(self, repo, tag, revision):
        self.calls.append(("tag", tag, revision))


@pytest.fixture
def fake_hub(monkeypatch):
    import huggingface_hub

    api = FakeApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    return api


def test_hub_setup_auto_skips_without_token(fake_hub, monkeypatch):
    from jev.hub import HubUploader

    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert HubUploader.setup("auto", True, "jev-9b") is None
    with pytest.raises(SystemExit):  # an explicit target without a token stops the run
        HubUploader.setup("me/jev-9b", True, "jev-9b")


def test_hub_setup_checks_access_and_creates_private_repo(fake_hub, monkeypatch):
    from jev.hub import HubUploader

    monkeypatch.setenv("HF_TOKEN", "hf_x")
    hub = HubUploader.setup("auto", True, "jev-9b")
    assert hub.repo == "me/jev-9b" and ("create_repo", "me/jev-9b", True) in fake_hub.calls
    with pytest.raises(SystemExit):
        HubUploader.setup("strangers/jev", True, "jev-9b")


def _fake_ckpt(tmp_path):
    ckpt = tmp_path / "best"
    (ckpt / "adapter").mkdir(parents=True)
    for f in ("jev_config.json", "head.pt", "adapter/adapter_model.safetensors", "adapter/README.md"):
        (ckpt / f).write_text("x")
    return ckpt


def test_hub_push_uploads_files_and_tags(fake_hub, monkeypatch, tmp_path):
    from jev.hub import HubUploader

    monkeypatch.setenv("HF_TOKEN", "hf_x")
    hub = HubUploader.setup("auto", True, "jev-9b")
    assert hub.push(_fake_ckpt(tmp_path), "# card", "epoch 1 (step 5)", tag="epoch-1")
    upload = next(c for c in fake_hub.calls if c[0] == "upload")
    assert upload[3] == ["README.md", "adapter", "head.pt", "jev_config.json"]
    assert ("tag", "epoch-1", "abc123") in fake_hub.calls


def test_hub_push_failure_does_not_raise(monkeypatch, tmp_path):
    import huggingface_hub

    from jev.hub import HubUploader

    api = FakeApi(fail_upload=True)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    hub = HubUploader.setup("auto", True, "jev-9b")
    assert hub.push(_fake_ckpt(tmp_path), "# card", "epoch 1") is False


def test_load_env_does_not_override(tmp_path, monkeypatch):
    from jev.env import load_env

    f = tmp_path / ".env.local"
    f.write_text("# comment\nHF_TOKEN=from_file\nWANDB_API_KEY=\nJEV_TEST_X='quoted'\n")
    monkeypatch.setenv("HF_TOKEN", "from_env")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("JEV_TEST_X", raising=False)
    load_env(f)
    import os

    assert os.environ["HF_TOKEN"] == "from_env"  # real environment wins
    assert "WANDB_API_KEY" not in os.environ  # empty template line means unset
    assert os.environ["JEV_TEST_X"] == "quoted"
