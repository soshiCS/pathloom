"""Perception rules that only a real page can prove; runs Chromium on an inline page and skips if it is absent."""
import pytest

from src.cua.models import Locator

pytestmark = pytest.mark.browser

PAGE = """
<html><body>
  <div id="header">
    <a class="shopping_cart_link" data-test="shopping-cart-link" href="#">
      <span class="shopping_cart_badge" data-test="shopping-cart-badge">1</span>
    </a>
    <a class="cart_button" role="button" data-test="cart-button" style="display:inline-block;width:40px;height:40px">
      <span class="cart_badge">2</span>
    </a>
    <a id="menu-open" href="#" style="display:inline-block;width:24px;height:24px"></a>
    <a href="/inventory-item.html?id=4" role="button"
       aria-label="View details for Sauce Labs Backpack">Sauce Labs Backpack</a>
    <button data-test="add-to-cart-sauce-labs-backpack">Add to cart</button>
    <span class="title">Products</span>
  </div>
</body></html>
"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True)
    except Exception as error:      # no browser binary in this environment
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("page") / "shop.html"
    page.write_text(PAGE)
    live.navigate(page.as_uri())
    yield live
    live.close()


def test_icon_only_controls_are_named_by_their_test_id_or_id(surface):
    elements = {(e.role, e.name): e for e in surface.observe().elements}
    assert ("link", "shopping cart link") in elements          # the badge count "1" is not its name
    assert ("link", "menu open") in elements                   # an icon-only link named by its id
    assert ("button", "cart button") in elements               # an <a role="button"> without href, by its test id
    assert elements[("button", "cart button")].text == "2"
    assert ("button", "Add to cart") in elements
    # Regression: a real anchor with an href stays a link even when it carries role="button". Artifacts
    # recorded it as a link, and perception must not change between discovery and replay.
    assert ("link", "View details for Sauce Labs Backpack") in elements
    assert ("button", "View details for Sauce Labs Backpack") not in elements


def test_text_inside_a_control_is_not_a_separate_element(surface):
    observation = surface.observe()
    texts = [e.text for e in observation.elements if e.role == "text"]
    assert "Products" in texts and "1" not in texts and "2" not in texts   # badges belong to their controls
    cart = surface.resolve(Locator(strategies=[{"kind": "role", "role": "link", "name": "shopping cart link"}]))
    assert cart is not None and cart.text == "1"


def test_dom_paths_match_the_page_scripts_structural_path():
    """Offline: the CDP tree is mapped to the same tag:nth-of-type paths the page script emits."""
    from src.cua.surface import dom_paths
    element = lambda tag, node_id, children=(): {"nodeType": 1, "localName": tag, "backendNodeId": node_id,
                                                 "children": list(children)}
    tree = {"nodeType": 9, "children": [
        {"nodeType": 10, "nodeName": "html"},                       # the doctype is not an element
        element("html", 1, [
            element("head", 2),
            element("body", 3, [
                element("div", 4, [element("a", 5), {"nodeType": 3, "nodeValue": "text"}, element("a", 6)]),
                element("div", 7, [element("input", 8)]),
            ]),
        ]),
    ]}
    paths, order = dom_paths(tree)
    assert paths == {2: "head:nth-of-type(1)", 3: "body:nth-of-type(1)",
                     4: "body:nth-of-type(1) > div:nth-of-type(1)",
                     5: "body:nth-of-type(1) > div:nth-of-type(1) > a:nth-of-type(1)",
                     6: "body:nth-of-type(1) > div:nth-of-type(1) > a:nth-of-type(2)",
                     7: "body:nth-of-type(1) > div:nth-of-type(2)",
                     8: "body:nth-of-type(1) > div:nth-of-type(2) > input:nth-of-type(1)"}
    assert [order[paths[i]] for i in (2, 3, 4, 5, 6, 7, 8)] == [0, 1, 2, 3, 4, 5, 6]   # document order
