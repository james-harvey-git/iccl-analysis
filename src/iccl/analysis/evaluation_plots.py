"""Reconstruct standard evaluation figures from portable result rows."""

from iccl.reporting.figures import evaluation_figures, write_html_figures

full_evaluation_figures = evaluation_figures

__all__ = ["full_evaluation_figures", "write_html_figures"]
