"""Apriori association rule mining for symptom-to-disease reasoning.

The implementation uses mlxtend to discover frequent itemsets and association
rules, then filters them so only symptom antecedents that predict diseases are
returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.preprocess import frame_to_transactions, preprocess_training_data
from src.utils import TARGET_COLUMN, humanize_label, normalize_symptom_name

DISEASE_PREFIX = "disease_"
DEFAULT_MIN_SUPPORT = 0.02
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MIN_LIFT = 1.0
DEFAULT_MAX_LEN = 3


@dataclass(frozen=True)
class AprioriResult:
    """Container for the mined Apriori rules and the encoded item matrix."""

    rules: pd.DataFrame
    encoded_transactions: pd.DataFrame


def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    """Transform transaction lists into a boolean item matrix."""

    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(encoded_array, columns=encoder.columns_)


def mine_association_rules(
    transactions: list[list[str]],
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_lift: float = DEFAULT_MIN_LIFT,
    max_len: int = DEFAULT_MAX_LEN,
) -> AprioriResult:
    """Generate Apriori rules from the training transactions."""

    encoded_transactions = encode_transactions(transactions)
    frequent_itemsets = apriori(
        encoded_transactions,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    if frequent_itemsets.empty:
        empty_rules = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
        return AprioriResult(rules=empty_rules, encoded_transactions=encoded_transactions)

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )
    if rules.empty:
        filtered_rules = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
        return AprioriResult(rules=filtered_rules, encoded_transactions=encoded_transactions)

    rules = rules[rules["lift"] >= min_lift].copy()
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules = rules.reset_index(drop=True)
    return AprioriResult(rules=rules, encoded_transactions=encoded_transactions)


def prepare_apriori_from_training(file_name: str = "Training.csv") -> AprioriResult:
    """Load the dataset, convert it to transactions, and mine association rules."""

    preprocessed = preprocess_training_data(file_name)
    transactions = frame_to_transactions(preprocessed.frame)
    return mine_association_rules(transactions)


def _is_disease_itemset(itemset: frozenset[str]) -> bool:
    """Check whether an itemset contains at least one disease item."""

    return any(item.startswith(DISEASE_PREFIX) for item in itemset)


def _antecedent_symptoms(antecedent: frozenset[str]) -> set[str]:
    """Extract only symptom items from an antecedent itemset."""

    return {
        item
        for item in antecedent
        if not item.startswith(DISEASE_PREFIX)
    }


def _disease_name_from_itemset(itemset: frozenset[str]) -> str:
    """Convert the disease-prefixed item into a display label."""

    disease_items = [item for item in itemset if item.startswith(DISEASE_PREFIX)]
    if not disease_items:
        return "Unknown"
    disease_item = disease_items[0].replace(DISEASE_PREFIX, "", 1)
    return humanize_label(disease_item)


def build_symptom_to_disease_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Keep only rules that predict disease items from symptom antecedents."""

    if rules.empty:
        return rules.copy()

    filtered_rules = rules.copy()
    disease_mask = filtered_rules["consequents"].apply(_is_disease_itemset)
    filtered_rules = filtered_rules.loc[disease_mask].copy()
    antecedent_mask = filtered_rules["antecedents"].apply(
        lambda itemset: len(_antecedent_symptoms(itemset)) > 0
    )
    filtered_rules = filtered_rules.loc[antecedent_mask].copy()
    filtered_rules["antecedent_size"] = filtered_rules["antecedents"].apply(
        lambda itemset: len(_antecedent_symptoms(itemset))
    )
    filtered_rules = filtered_rules.sort_values(
        by=["confidence", "lift", "support", "antecedent_size"],
        ascending=[False, False, False, True],
    )
    filtered_rules = filtered_rules.reset_index(drop=True)
    return filtered_rules


def format_rules_for_display(rules: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Apriori rules into a human-readable tabular format."""

    if rules.empty:
        return pd.DataFrame(
            columns=["antecedents", "consequent", "support_pct", "confidence_pct", "lift"]
        )

    display_frame = rules.copy()
    display_frame["antecedents"] = display_frame["antecedents"].apply(
        lambda itemset: ", ".join(sorted(humanize_label(item) for item in itemset))
    )
    display_frame["consequent"] = display_frame["consequents"].apply(
        lambda itemset: _disease_name_from_itemset(itemset)
    )
    display_frame["support_pct"] = (display_frame["support"] * 100).round(2)
    display_frame["confidence_pct"] = (display_frame["confidence"] * 100).round(2)
    display_frame["lift"] = display_frame["lift"].round(2)
    return display_frame[["antecedents", "consequent", "support_pct", "confidence_pct", "lift"]]


def recommend_diseases_from_symptoms(
    selected_symptoms: Iterable[str],
    rules: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the strongest disease rules that match the selected symptoms."""

    normalized_symptoms = {
        normalize_symptom_name(symptom) for symptom in selected_symptoms
    }
    if rules.empty or not normalized_symptoms:
        return pd.DataFrame(
            columns=["antecedents", "consequent", "support_pct", "confidence_pct", "lift"]
        )

    exact_matches: list[pd.Series] = []
    partial_matches: list[pd.Series] = []
    for _, rule in rules.iterrows():
        symptom_items = _antecedent_symptoms(rule["antecedents"])
        overlap_size = len(symptom_items.intersection(normalized_symptoms))
        if not symptom_items or overlap_size == 0:
            continue
        if symptom_items.issubset(normalized_symptoms):
            exact_matches.append(rule)
        else:
            rule = rule.copy()
            rule["match_size"] = overlap_size
            partial_matches.append(rule)

    if exact_matches:
        matched_rules = pd.DataFrame(exact_matches)
        matched_rules["match_size"] = matched_rules["antecedents"].apply(
            lambda itemset: len(_antecedent_symptoms(itemset))
        )
    elif partial_matches:
        matched_rules = pd.DataFrame(partial_matches)
    else:
        return pd.DataFrame(
            columns=["antecedents", "consequent", "support_pct", "confidence_pct", "lift"]
        )

    matched_rules = matched_rules.sort_values(
        by=["match_size", "confidence", "lift", "support"],
        ascending=[False, False, False, False],
    ).head(top_n)
    return format_rules_for_display(matched_rules)


def recommend_diseases_from_selection(
    selected_symptoms: Iterable[str],
    rules: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compatibility wrapper that forwards to the matching recommendation logic."""

    return recommend_diseases_from_symptoms(selected_symptoms, rules, top_n=top_n)


def build_apriori_pipeline(file_name: str = "Training.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience helper returning display rules and the encoded matrix."""

    result = prepare_apriori_from_training(file_name)
    symptom_rules = build_symptom_to_disease_rules(result.rules)
    display_rules = format_rules_for_display(symptom_rules)
    return display_rules, result.encoded_transactions
