"""Native radios and checkboxes are activated the way a person does: through the browser's own label
association, never forced, with the resulting state verified. Local Chromium on an inline page; no model."""
import pytest

from src.cua.models import ActionError, Locator

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

PAGE = """<!doctype html><html><head><style>
  .stack { position: relative; height: 40px; }
  .stack input { position: absolute; left: 0; top: 0; width: 24px; height: 24px; margin: 0; }
  .stack label { position: absolute; left: 0; top: 0; width: 160px; height: 30px; background: #eef; }
  #overlay { position: absolute; left: 0; top: 0; width: 400px; height: 60px; background: rgba(0,0,0,.05); }
  .blocked { position: relative; height: 60px; }
  .blocked input { position: absolute; left: 0; top: 0; width: 24px; height: 24px; margin: 0; }
  .blocked label { position: absolute; left: 200px; top: 0; }
  .group { position: relative; height: 40px; }
  .group input { position: absolute; left: 0; top: 0; width: 24px; height: 24px; margin: 0; }
  .group input + label.under-b { position: absolute; left: 0; top: 0; width: 120px; height: 30px; }
  .group .over-a { position: absolute; left: 0; top: 0; width: 200px; height: 34px; background: #fee; z-index: 2; }
  .group input:nth-of-type(2) { left: 240px; }
</style></head><body>
<h1>Choices</h1>
<div class="stack"><input type="radio" id="plan-a" name="plan"><label for="plan-a">Plan A</label></div>
<div><label id="wrap"><input type="radio" name="plan2"> Wrapped plan</label></div>
<div><input type="radio" name="plan3" aria-label="Bare plan"></div>
<div><input type="checkbox" id="agree"><label for="agree">Agree to terms</label></div>
<div class="stack"><input type="checkbox" id="terms"><label for="terms">Covered terms</label></div>
<div><input type="radio" id="pre" name="plan4" checked><label for="pre">Preselected</label></div>
<div><input type="radio" id="off" name="plan5" disabled><label for="off">Disabled plan</label></div>
<div><input type="checkbox" id="off2" disabled><label for="off2">Disabled box</label></div>
<div class="blocked"><input type="radio" id="under" name="plan6"><label for="under">Under overlay</label>
  <div id="overlay"></div></div>
<button id="press" onclick="log('button:press')">Press me</button>
<!-- keyboard fallback: B's label covers both A's input and A's label; same native radio group -->
<form id="ship"><fieldset class="group">
  <input type="radio" id="ship-a" name="ship"><label for="ship-a" class="under-b">Ship A</label>
  <input type="radio" id="ship-b" name="ship"><label for="ship-b" class="over-a">Ship B</label>
</fieldset></form>
<div class="group"><input type="checkbox" id="opt-a" name="opts"><label for="opt-a" class="under-b">Option A</label>
  <input type="checkbox" id="opt-b" name="opts"><label for="opt-b" class="over-a">Option B</label></div>
<div class="group"><input type="radio" id="cov-a" name="cov"><label for="cov-a" class="under-b">Covered A</label>
  <div class="over-a" id="stranger">unrelated</div></div>
<div class="group" inert><input type="radio" id="inert-a" name="inert"><label for="inert-a" class="under-b">Inert A</label>
  <input type="radio" id="inert-b" name="inert"><label for="inert-b" class="over-a">Inert B</label></div>
<div class="group"><input type="radio" id="blur-a" name="blur"><label for="blur-a" class="under-b">Blur A</label>
  <input type="radio" id="blur-b" name="blur"><label for="blur-b" class="over-a">Blur B</label></div>
<div class="group"><input type="radio" id="swap-a" name="swap"><label for="swap-a" class="under-b">Swap A</label>
  <input type="radio" id="swap-b" name="swap"><label for="swap-b" class="over-a">Swap B</label></div>
<div class="group"><input type="radio" id="late-a" name="late"><label for="late-a" class="under-b">Late A</label>
  <input type="radio" id="late-b" name="late"><label for="late-b" class="over-a">Late B</label></div>
<dialog id="modal"><p>Modal</p></dialog>
<script>
  window.__events = [];
  const log = (s) => window.__events.push(s);
  document.getElementById('blur-a').addEventListener('focus', (e) => e.target.blur());
  document.getElementById('swap-a').addEventListener('focus', (e) => e.target.replaceWith(e.target.cloneNode()));
  document.getElementById('late-a').addEventListener('change', (e) => e.target.replaceWith(e.target.cloneNode()));
  for (const el of document.querySelectorAll('input')) {
    el.addEventListener('click', () => log('click:' + (el.id || el.getAttribute('aria-label') || 'wrapped')));
    el.addEventListener('change', () => log('change:' + (el.id || el.getAttribute('aria-label') || 'wrapped')));
    el.addEventListener('input', () => log('input:' + (el.id || el.getAttribute('aria-label') || 'wrapped')));
  }
</script></body></html>"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("choices") / "choices.html"
    page.write_text(PAGE)
    live.navigate(page.as_uri())
    yield live
    live.close()


@pytest.fixture
def fresh(surface):
    surface.navigate(surface._page.url)
    return surface


def control(observation, role, name):
    [match] = [e for e in observation.elements if (e.role, e.name) == (role, name)]
    return match


def checked(surface, selector) -> bool:
    return surface._page.evaluate("(selector) => document.querySelector(selector).checked", selector)


def page_events(surface) -> list[str]:
    return surface._page.evaluate("() => window.__events")


def test_a_radio_covered_by_its_for_label_is_selected_through_the_label(fresh):
    radio = control(fresh.observe(), "radio", "Plan A")
    assert radio.source == "ax" and radio.ref.endswith("input:nth-of-type(1)") and radio.states["checked"] == "false"
    fresh.click(radio)
    assert checked(fresh, "#plan-a") is True
    assert page_events(fresh) == ["click:plan-a", "input:plan-a", "change:plan-a"]     # real user-style events
    assert control(fresh.observe(), "radio", "Plan A").states["checked"] == "true"


def test_a_wrapping_label_and_a_bare_radio_both_work(fresh):
    fresh.click(control(fresh.observe(), "radio", "Wrapped plan"))
    assert checked(fresh, "#wrap input") is True and "change:wrapped" in page_events(fresh)
    fresh.click(control(fresh.observe(), "radio", "Bare plan"))
    assert checked(fresh, "[aria-label='Bare plan']") is True and "change:Bare plan" in page_events(fresh)


def test_a_checkbox_toggles_each_time(fresh):
    box = control(fresh.observe(), "checkbox", "Agree to terms")
    fresh.click(box)
    assert checked(fresh, "#agree") is True
    fresh.click(control(fresh.observe(), "checkbox", "Agree to terms"))
    assert checked(fresh, "#agree") is False
    assert page_events(fresh).count("change:agree") == 2


def test_an_already_selected_radio_is_left_alone(fresh):
    fresh.click(control(fresh.observe(), "radio", "Preselected"))
    assert checked(fresh, "#pre") is True and page_events(fresh) == []          # satisfied without a click


def test_disabled_choice_controls_are_refused_before_acting(fresh):
    for role, name in (("radio", "Disabled plan"), ("checkbox", "Disabled box")):
        with pytest.raises(ActionError, match="is disabled") as refused:
            fresh.click(control(fresh.observe(), role, name))
        assert refused.value.performed == "no"
    assert page_events(fresh) == []


def test_an_unrelated_overlay_is_never_used_as_a_proxy_or_forced_through(fresh):
    with pytest.raises(ActionError, match="was not performed") as refused:
        fresh.click(control(fresh.observe(), "radio", "Under overlay"))
    assert refused.value.performed == "no" and "unrelated element covers the control" in str(refused.value)
    assert checked(fresh, "#under") is False and page_events(fresh) == []


def test_ordinary_buttons_keep_the_existing_click_path(fresh):
    button = control(fresh.observe(), "button", "Press me")
    assert button.source == "dom+ax"
    fresh.click(button)
    assert page_events(fresh) == ["button:press"]


def test_an_accessibility_only_choice_control_resolves_by_its_structural_reference(fresh):
    radio = control(fresh.observe(), "radio", "Plan A")
    resolved = fresh.resolve(Locator(strategies=[{"kind": "css", "selector": radio.ref}]))
    assert resolved is not None and resolved.ref == radio.ref
    fresh.click(resolved)                                                        # the css rung alone is enough
    assert checked(fresh, "#plan-a") is True


# ---------- phases decide the performed state, never the wording ----------

def test_a_failed_trial_is_not_performed_and_a_failed_real_click_is_unknown(fresh, monkeypatch):
    from playwright.sync_api import Locator, TimeoutError as PlaywrightTimeout
    real = Locator.click

    def wordy_real_click(self, *args, **kwargs):
        if kwargs.get("trial"):
            return real(self, *args, **kwargs)
        raise PlaywrightTimeout("Timeout 2500ms exceeded.\nwaiting for element to be visible, enabled and stable\n"
                                "<div> intercepts pointer events\nretrying click action")

    monkeypatch.setattr(Locator, "click", wordy_real_click)
    with pytest.raises(ActionError) as failed:
        fresh.click(control(fresh.observe(), "button", "Press me"))
    assert failed.value.performed == "unknown"                           # the phase decided, the wording did not
    assert str(failed.value).endswith("Timeout 2500ms exceeded.")         # only a first-line diagnostic is kept
    monkeypatch.undo()
    with pytest.raises(ActionError) as refused:                          # the trial itself fails: nothing dispatched
        fresh.click(control(fresh.observe(), "radio", "Under overlay"))
    assert refused.value.performed == "no" and page_events(fresh) == []


def test_a_fill_that_times_out_is_unknown_even_with_actionability_wording(fresh, monkeypatch):
    from playwright.sync_api import Locator, TimeoutError as PlaywrightTimeout
    from src.cua.models import Element
    fresh._page.evaluate("() => document.body.insertAdjacentHTML('beforeend', '<input id=note aria-label=Note>')")
    box = control(fresh.observe(), "textbox", "Note")
    monkeypatch.setattr(Locator, "fill", lambda self, *a, **k: (_ for _ in ()).throw(
        PlaywrightTimeout("Timeout exceeded.\nwaiting for element to be visible, enabled and editable")))
    with pytest.raises(ActionError) as failed:
        fresh.type(box, "x")
    assert failed.value.performed == "unknown"
    monkeypatch.undo()
    with pytest.raises(ActionError) as refused:                          # the preflight fails: not performed
        fresh.type(Element("textbox", "Nowhere", ref="input#nowhere"), "x")
    assert refused.value.performed == "no"


# ---------- a dynamic page: detached or replaced controls ----------

def test_a_control_detached_before_activation_is_not_performed(fresh):
    radio = control(fresh.observe(), "radio", "Plan A")
    fresh._page.evaluate("() => document.getElementById('plan-a').remove()")
    with pytest.raises(ActionError) as refused:
        fresh.click(radio)
    assert refused.value.performed == "no" and page_events(fresh) == []


def test_a_control_replaced_between_trial_and_real_activation_is_unknown(fresh, monkeypatch):
    from playwright.sync_api import ElementHandle
    real = ElementHandle.click
    page = fresh._page

    def detach_after_trial(self, *args, **kwargs):
        outcome = real(self, *args, **kwargs)
        if kwargs.get("trial"):
            page.evaluate("() => { const a = document.getElementById('plan-a'); a.replaceWith(a.cloneNode()); "
                          "document.querySelector('label[for=plan-a]').remove(); }")
        return outcome

    monkeypatch.setattr(ElementHandle, "click", detach_after_trial)
    with pytest.raises(ActionError) as failed:
        fresh.click(control(fresh.observe(), "radio", "Plan A"))
    assert failed.value.performed == "unknown"


def test_a_control_replaced_right_after_activation_is_unknown_not_a_false_success(fresh):
    with pytest.raises(ActionError) as failed:
        fresh.click(control(fresh.observe(), "radio", "Late A"))
    assert failed.value.performed == "unknown" and "left the document" in str(failed.value)
    assert "change:late-a" in page_events(fresh)                          # the activation really happened


# ---------- the guarded keyboard fallback ----------

def test_a_radio_obstructed_by_its_groupmates_label_is_selected_by_keyboard(fresh):
    radio = control(fresh.observe(), "radio", "Ship A")
    fresh.click(radio)
    assert checked(fresh, "#ship-a") is True and checked(fresh, "#ship-b") is False
    assert page_events(fresh) == ["click:ship-a", "input:ship-a", "change:ship-a"]   # native keyboard activation
    assert control(fresh.observe(), "radio", "Ship A").states["checked"] == "true"


def test_an_independent_same_named_checkbox_never_authorizes_the_keyboard_path(fresh):
    # Option B's label covers Option A and its label. Checkboxes sharing a name are independent choices,
    # so the keyboard path is refused before any key is sent and neither box changes.
    with pytest.raises(ActionError, match="unrelated element covers the control") as refused:
        fresh.click(control(fresh.observe(), "checkbox", "Option A"))
    assert refused.value.performed == "no"
    assert checked(fresh, "#opt-a") is False and checked(fresh, "#opt-b") is False and page_events(fresh) == []


def test_a_checkbox_covered_by_its_own_label_toggles_through_the_label(fresh):
    box = control(fresh.observe(), "checkbox", "Covered terms")
    fresh.click(box)
    assert checked(fresh, "#terms") is True
    assert page_events(fresh) == ["click:terms", "input:terms", "change:terms"]
    fresh.click(control(fresh.observe(), "checkbox", "Covered terms"))
    assert checked(fresh, "#terms") is False and page_events(fresh).count("change:terms") == 2


def test_the_adapter_never_forces_scripts_or_guesses_a_click():
    import re
    from pathlib import Path
    source = Path("src/cua/surface.py").read_text()
    assert "force=True" not in source and "force = True" not in source
    scripts = re.findall(r'r?"""\n?\(el\) => \{.*?\}\n"""', source, re.S) + [
        m for m in re.findall(r"[A-Z_]+_JS = r?\"\"\"(.*?)\"\"\"", source, re.S)]
    for js in scripts:
        assert ".click()" not in js and "dispatchEvent" not in js and re.search(r"\.checked\s*=[^=]", js) is None
    activation = source[source.index("def _activate_choice"):source.index("def type(")]
    assert "mouse.click" not in activation and "coords" not in activation and "evaluate(\"(el) => el.click" not in activation


def test_the_keyboard_never_bypasses_an_unrelated_overlay(fresh):
    with pytest.raises(ActionError, match="unrelated element covers the control") as refused:
        fresh.click(control(fresh.observe(), "radio", "Covered A"))
    assert refused.value.performed == "no" and checked(fresh, "#cov-a") is False and page_events(fresh) == []


def test_no_keyboard_fallback_behind_a_modal_or_inside_an_inert_subtree(fresh):
    from src.cua.models import Element
    fresh._page.evaluate("() => document.getElementById('modal').showModal()")
    with pytest.raises(ActionError, match="modal dialog is open elsewhere") as refused:
        fresh.click(Element("radio", "Ship A", ref="input#ship-a"))      # perception hides the page behind a modal
    assert refused.value.performed == "no" and checked(fresh, "#ship-a") is False
    fresh._page.evaluate("() => document.getElementById('modal').close()")
    inert = fresh._page.evaluate("() => { const a = document.getElementById('inert-a'); const r = a.getBoundingClientRect();"
                                 " return [r.x + r.width / 2, r.y + r.height / 2]; }")
    assert inert[0] > 0                                                   # the inert control exists and is laid out
    target = Element("radio", "Inert A", ref="input#inert-a")
    with pytest.raises(ActionError) as refused:
        fresh.click(target)
    assert refused.value.performed == "no" and checked(fresh, "#inert-a") is False and page_events(fresh) == []


def test_focus_failure_and_replacement_before_the_key_are_not_performed(fresh):
    with pytest.raises(ActionError, match="did not take focus") as refused:
        fresh.click(control(fresh.observe(), "radio", "Blur A"))
    assert refused.value.performed == "no" and checked(fresh, "#blur-a") is False
    with pytest.raises(ActionError) as replaced:
        fresh.click(control(fresh.observe(), "radio", "Swap A"))
    assert replaced.value.performed == "no" and not any(e.startswith("change") for e in page_events(fresh))
