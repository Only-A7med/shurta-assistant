"""Tests for the service selection table.

A mistake here is quiet: a duplicate identifier drops a service from the corpus
without any error, and a typo in a category slug leaves a service uncategorised.
So the table checks itself before a collection starts, and that check is tested.
"""

from __future__ import annotations

import pytest

from shurta.collect import selection


class TestTheTableIsWellFormed:
    def test_the_check_passes(self) -> None:
        selection.check()

    def test_the_catalogue_is_the_intended_size(self) -> None:
        assert 30 <= len(selection.SELECTED) <= 50

    def test_identifiers_are_unique(self) -> None:
        identifiers = [entry.id for entry in selection.SELECTED]
        assert len(set(identifiers)) == len(identifiers)

    def test_slugs_are_unique(self) -> None:
        slugs = [entry.slug for entry in selection.SELECTED]
        assert len(set(slugs)) == len(slugs)

    def test_identifiers_are_readable_slugs(self) -> None:
        """They appear in citations, so they have to be legible."""
        for entry in selection.SELECTED:
            assert entry.id == entry.id.lower()
            assert " " not in entry.id
            assert entry.id.replace("-", "").isalnum()

    def test_every_category_is_one_the_site_publishes(self) -> None:
        for entry in selection.SELECTED:
            assert entry.category in selection.CATEGORY_SLUGS

    def test_every_exclusion_records_a_reason(self) -> None:
        for slug, reason in selection.EXCLUDED.items():
            assert reason.strip(), f"{slug} is excluded without a reason"

    def test_lookup_by_slug_covers_the_whole_table(self) -> None:
        assert len(selection.selected_by_slug()) == len(selection.SELECTED)


class TestTheCheckCatchesMistakes:
    def test_a_duplicate_identifier_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        duplicate = selection.Selected("same", "services/one", "traffic")
        other = selection.Selected("same", "services/two", "traffic")
        monkeypatch.setattr(selection, "SELECTED", (duplicate, other))
        with pytest.raises(ValueError, match="duplicate identifiers"):
            selection.check()

    def test_a_duplicate_slug_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = selection.Selected("one", "services/same", "traffic")
        second = selection.Selected("two", "services/same", "traffic")
        monkeypatch.setattr(selection, "SELECTED", (first, second))
        with pytest.raises(ValueError, match="duplicate slugs"):
            selection.check()

    def test_an_unknown_category_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = selection.Selected("one", "services/one", "not-a-category")
        monkeypatch.setattr(selection, "SELECTED", (entry,))
        with pytest.raises(ValueError, match="unknown category slugs"):
            selection.check()

    def test_a_slug_cannot_be_both_selected_and_excluded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = selection.Selected("one", "services/one", "traffic")
        monkeypatch.setattr(selection, "SELECTED", (entry,))
        monkeypatch.setattr(selection, "EXCLUDED", {"services/one": "contradictory"})
        with pytest.raises(ValueError, match="both selected and excluded"):
            selection.check()
