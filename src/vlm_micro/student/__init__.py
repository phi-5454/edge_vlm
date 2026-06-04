"""Compact student training components."""

from vlm_micro.student.data import CompactVocabulary, StudentDataModule, TallyQAStudentDataModule
from vlm_micro.student.model import StudentBaseline

__all__ = [
    "CompactVocabulary",
    "StudentBaseline",
    "StudentDataModule",
    "TallyQAStudentDataModule",
]
