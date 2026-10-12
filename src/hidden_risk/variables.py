"""The harmonised variables: one entry per column of the processed dataset.

This list is the single source of truth for step 1. Each entry says which NHANES
variable it comes from in each cycle, which codes mean "refused / don't know",
and how values are made comparable across cycles. docs/codebook.md is generated
from it, so add or change variables here, never by hand in the codebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd

YES_NO = {1: "yes", 0: "no"}


@dataclass(frozen=True)
class Var:
    name: str                          # column name in the processed dataset
    module: str                        # NHANES module, e.g. "HUQ"
    source: Union[str, Dict[str, str]]  # NHANES variable, or {cycle letters: variable} if renamed
    label: str
    kind: str = "category"             # binary | category | count | continuous
    values: Dict[int, str] = field(default_factory=dict)  # value labels after harmonisation
    missing: Tuple[int, ...] = ()      # codes for refused / don't know -> missing
    min_age: Optional[int] = None      # set missing outside the age range asked in every cycle
    max_age: Optional[int] = None
    # (values, cycle letter, all raw columns of that cycle) -> harmonised values
    transform: Optional[Callable[[pd.Series, str, pd.DataFrame], pd.Series]] = None
    note: str = ""

    def source_in(self, cycle: str) -> Optional[str]:
        """NHANES variable name in the given cycle (file suffix), or None if not collected."""
        if isinstance(self.source, str):
            return self.source
        for cycles, name in self.source.items():
            if cycle in cycles:
                return name
        return None

    @property
    def dtype(self) -> str:
        return "float64" if self.kind == "continuous" else "Int16"


# --------------------------------------------------------------------------- #
# Cross-cycle transforms
# --------------------------------------------------------------------------- #
def topcode_age(s: pd.Series, cycle: str, raw: pd.DataFrame) -> pd.Series:
    # 2005-06 top-codes age at 85, later cycles at 80
    return s.clip(upper=80)


# HUQ051 (2013+) has finer categories that nest exactly into HUQ050's
HUQ051_TO_HUQ050 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5}


def harmonise_visits(s: pd.Series, cycle: str, raw: pd.DataFrame) -> pd.Series:
    return s.map(HUQ051_TO_HUQ050) if cycle in "HIJ" else s


def harmonise_sleep_hours(s: pd.Series, cycle: str, raw: pd.DataFrame) -> pd.Series:
    # SLD010H (to 2014): whole hours, 12 = "12 or more"; SLD012 (2015+): half hours, 2-14.5
    return s.clip(lower=2, upper=12)


# DMDEDUC3 (school grade, ages 6-19) -> DMDEDUC2 categories (ages 20+)
DMDEDUC3_TO_DMDEDUC2 = {
    **{grade: 1 for grade in range(0, 9)}, 55: 1, 66: 1,  # up to 8th grade, "less than 5th/9th"
    **{grade: 2 for grade in range(9, 13)},               # 9th-12th grade, no diploma
    13: 3, 14: 3,                                          # high school graduate, GED
    15: 4,                                                 # more than high school
}


def education_with_youth(s: pd.Series, cycle: str, raw: pd.DataFrame) -> pd.Series:
    youth = raw["DMDEDUC3"].map(DMDEDUC3_TO_DMDEDUC2).where(raw["RIDAGEYR"] < 20)
    return s.fillna(youth)


def yes_no(s: pd.Series, cycle: str, raw: pd.DataFrame) -> pd.Series:
    return s.map({1: 1, 2: 0})


def diabetes_yes_no(s: pd.Series, cycle: str, raw: pd.DataFrame) -> pd.Series:
    return s.map({1: 1, 2: 0, 3: 0})  # 3 = borderline, not a diagnosis


def cholesterol_yes_no(s: pd.Series, cycle: str, raw: pd.DataFrame) -> pd.Series:
    s = yes_no(s, cycle, raw)
    if cycle in "DEF":  # until 2010 only asked if cholesterol was ever checked (BPQ060)
        s = s.mask(raw["BPQ060"] == 2, 0)
    return s


# --------------------------------------------------------------------------- #
# Variable list
# --------------------------------------------------------------------------- #
def _binary(name, module, source, label, **kw) -> Var:
    return Var(name, module, source, label, kind="binary", values=YES_NO,
               missing=(7, 9), transform=yes_no, **kw)


PHQ_VALUES = {0: "not at all", 1: "several days", 2: "more than half the days", 3: "nearly every day"}
PHQ_ITEMS = [
    "little interest or pleasure in doing things",
    "feeling down, depressed or hopeless",
    "trouble sleeping, or sleeping too much",
    "feeling tired or having little energy",
    "poor appetite or overeating",
    "feeling bad about yourself",
    "trouble concentrating",
    "moving or speaking slowly, or being fidgety",
    "thoughts of being better off dead or of hurting yourself",
]

VARIABLES = [
    # Demographics and survey design
    Var("age", "DEMO", "RIDAGEYR", "Age in years", kind="count", transform=topcode_age,
        note="Top-coded at 80 in all cycles (2005-06 used 85)."),
    Var("sex", "DEMO", "RIAGENDR", "Sex", values={1: "male", 2: "female"}),
    Var("ethnicity", "DEMO", "RIDRETH1", "Race / Hispanic origin",
        values={1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                4: "Non-Hispanic Black", 5: "Other race, incl. multi-racial"},
        note="RIDRETH1 is the only race variable available in every cycle."),
    Var("education", "DEMO", "DMDEDUC2", "Highest education completed",
        values={1: "less than 9th grade", 2: "9-11th grade", 3: "high school / GED",
                4: "some college / AA degree", 5: "college graduate or above"},
        missing=(7, 9), transform=education_with_youth,
        note="DMDEDUC2 is asked from age 20; for ages 18-19 it is derived from DMDEDUC3."),
    Var("marital_status", "DEMO", "DMDMARTL", "Marital status",
        values={1: "married", 2: "widowed", 3: "divorced", 4: "separated",
                5: "never married", 6: "living with partner"},
        missing=(77, 99), min_age=20,
        note="Asked from age 20 (2005-06 from 14); set missing under 20 in every cycle."),
    Var("income_poverty_ratio", "DEMO", "INDFMPIR", "Family income to poverty ratio",
        kind="continuous", note="Capped at 5 by NHANES."),
    Var("weight_interview", "DEMO", "WTINT2YR", "Interview weight, 2-year", kind="continuous"),
    Var("weight_mec", "DEMO", "WTMEC2YR", "Examination (MEC) weight, 2-year", kind="continuous",
        note="Use this weight: the PHQ-9 is administered in the MEC."),
    Var("strata", "DEMO", "SDMVSTRA", "Masked variance pseudo-stratum"),
    Var("psu", "DEMO", "SDMVPSU", "Masked variance pseudo-PSU"),

    # PHQ-9 (outcome = item 9)
    *[Var(f"phq_{i}", "DPQ", f"DPQ0{i}0", f"PHQ-9 item {i}: {text}", values=PHQ_VALUES,
          missing=(7, 9)) for i, text in enumerate(PHQ_ITEMS, start=1)],
    Var("phq_difficulty", "DPQ", "DPQ100", "Difficulty these problems have caused",
        values={0: "not at all", 1: "somewhat", 2: "very", 3: "extremely"}, missing=(7, 9),
        note="Only asked if any PHQ item is above 0."),

    # Healthcare use
    Var("self_rated_health", "HUQ", "HUQ010", "General health condition",
        values={1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor"}, missing=(7, 9)),
    Var("healthcare_visits", "HUQ", {"DEFG": "HUQ050", "HIJ": "HUQ051"},
        "Times received healthcare in past 12 months",
        values={0: "none", 1: "1", 2: "2-3", 3: "4-9", 4: "10-12", 5: "13 or more"},
        missing=(77, 99), transform=harmonise_visits,
        note="HUQ051 (2013+) categories collapsed to HUQ050's. HUQ050 includes emergency-room "
             "and home visits, HUQ051 excludes them."),
    _binary("hospital_stay", "HUQ", "HUQ071", "Overnight hospital patient in past 12 months"),
    _binary("mental_health_visit", "HUQ", "HUQ090", "Seen mental health professional in past 12 months"),

    # Chronic conditions (MCQ asked from age 20)
    _binary("asthma", "MCQ", "MCQ010", "Ever told had asthma"),
    _binary("arthritis", "MCQ", "MCQ160A", "Ever told had arthritis"),
    _binary("heart_failure", "MCQ", "MCQ160B", "Ever told had congestive heart failure"),
    _binary("coronary_heart_disease", "MCQ", "MCQ160C", "Ever told had coronary heart disease"),
    _binary("angina", "MCQ", "MCQ160D", "Ever told had angina"),
    _binary("heart_attack", "MCQ", "MCQ160E", "Ever told had heart attack"),
    _binary("stroke", "MCQ", "MCQ160F", "Ever told had stroke"),
    _binary("emphysema", "MCQ", "MCQ160G", "Ever told had emphysema"),
    _binary("chronic_bronchitis", "MCQ", "MCQ160K", "Ever told had chronic bronchitis"),
    _binary("copd", "MCQ", {"HIJ": "MCQ160O"}, "Ever told had COPD",
            note="Only asked from 2013; use emphysema / chronic bronchitis for all cycles."),
    _binary("liver_condition", "MCQ", "MCQ160L", "Ever told had a liver condition"),
    _binary("thyroid_problem", "MCQ", "MCQ160M", "Ever told had a thyroid problem"),
    _binary("cancer", "MCQ", "MCQ220", "Ever told had cancer or malignancy"),
    Var("diabetes", "DIQ", "DIQ010", "Doctor told you have diabetes", kind="binary",
        values=YES_NO, missing=(7, 9), transform=diabetes_yes_no,
        note="Borderline (code 3) counted as no."),
    _binary("hypertension", "BPQ", "BPQ020", "Ever told had high blood pressure"),
    Var("high_cholesterol", "BPQ", "BPQ080", "Ever told had high cholesterol", kind="binary",
        values=YES_NO, missing=(7, 9), transform=cholesterol_yes_no, min_age=20,
        note="Until 2010 only asked if cholesterol was ever checked; never checked (BPQ060 = no) "
             "is set to no, as from 2011 everyone is asked. Asked from age 20 until 2012, "
             "from 18 afterwards; set missing under 20."),

    # Sleep
    _binary("sleep_trouble", "SLQ", "SLQ050", "Ever told doctor had trouble sleeping"),
    Var("sleep_hours", "SLQ", {"DEFGH": "SLD010H", "IJ": "SLD012"},
        "Usual sleep hours on weekdays or workdays", kind="continuous", missing=(77, 99),
        transform=harmonise_sleep_hours,
        note="New sleep module from 2015: SLD012 (half hours, 2-14.5) replaces SLD010H "
             "(whole hours, 12 = 12 or more). Clipped to 2-12 in all cycles."),

    # Substance use
    _binary("smoked_100_cigarettes", "SMQ", "SMQ020", "Smoked at least 100 cigarettes in life",
            min_age=20, note="Asked from age 20 until 2012, from 18 afterwards; set missing under 20."),
    Var("smoking_now", "SMQ", "SMQ040", "Do you now smoke cigarettes",
        values={1: "every day", 2: "some days", 3: "not at all"}, missing=(7, 9), min_age=20,
        note="Only asked if smoked_100_cigarettes = yes, so it follows the same age rule."),
    _binary("ever_marijuana", "DUQ", "DUQ200", "Ever used marijuana or hashish",
            min_age=20, max_age=59,
            note="Asked to ages 20-59 (2005-08) or 18-59 (2009+); set missing under 20."),
    _binary("ever_hard_drugs", "DUQ", "DUQ240", "Ever used cocaine, heroin or methamphetamine",
            min_age=20, max_age=59,
            note="Asked to ages 20-59 (2005-06), 20-69 (2007-08) or 18-69 (2009+); kept for 20-59."),
    _binary("ever_injected_drugs", "DUQ", "DUQ370", "Ever used a needle to inject illegal drugs",
            min_age=20, max_age=59,
            note="Asked to ages 20-59 (2005-06), 20-69 (2007-08) or 18-69 (2009+); kept for 20-59."),

    # Social
    Var("work_status", "OCQ", "OCD150", "Type of work done last week",
        values={1: "working", 2: "with a job, not at work", 3: "looking for work", 4: "not working"},
        missing=(7, 9)),
    _binary("insured", "HIQ", "HIQ011", "Covered by health insurance"),

    # Body measures
    Var("bmi", "BMX", "BMXBMI", "Body mass index (kg/m2)", kind="continuous"),

    # Prescriptions (person level; drug names are in the prescriptions table)
    _binary("rx_any", "RXQ_RX", "RXDUSE", "Taken prescription medicine in past month"),
    Var("rx_count", "RXQ_RX", "RXDCOUNT", "Number of prescription medicines taken", kind="count",
        transform=lambda s, cycle, raw: s.mask(raw["RXDUSE"] == 2, 0),
        note="Set to 0 when rx_any = no."),
]

BY_NAME = {v.name: v for v in VARIABLES}

# Columns added by the build (not read from NHANES), for the codebook
DERIVED = {
    "seqn": "NHANES respondent ID",
    "cycle": "Survey cycle, e.g. 2005-2006",
    "split": "train (2005-2014) or test (2015-2018)",
    "weight_mec_pooled": "weight_mec / 7, for population estimates pooled over the 7 cycles",
    "phq8_score": "Sum of PHQ items 1-8 (0-24): depression score without the outcome item",
    "phq9_score": "Sum of PHQ items 1-9 (0-27). Contains the outcome item (phq_9), so it "
                  "leaks the label if used as a predictor",
    "phq8_positive": "1 if phq8_score >= 10 (positive screen on the other eight items)",
    "phq9_positive": "1 if phq9_score >= 10 (positive depression screen as used in practice)",
    "si": "Outcome: 1 if phq_9 >= 1 (several days or more)",
    "si_strict": "Sensitivity outcome: 1 if phq_9 >= 2 (more than half the days or more)",
}
