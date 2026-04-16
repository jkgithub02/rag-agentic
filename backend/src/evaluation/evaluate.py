from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import get_settings
from src.evaluation.ragas import (
    export_question_bank,
    run_ragas_evaluation,
    write_markdown_summary,
)


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--output-dir",
        default=str(Path("reports")),
        help="Directory for generated reports",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_report = output_dir / "ragas_report.json"
    md_report = output_dir / "RAGAS_EVALUATION.md"
    bank_export = output_dir / "ragas_question_bank.json"

    report = run_ragas_evaluation(
        api_url=args.api_url,
        output_path=json_report,
        ollama_base_url=settings.ollama_base_url,
        ollama_embedding_model=settings.ollama_embedding_model,
        aws_region=settings.aws_region,
        bedrock_chat_model_id=settings.bedrock_chat_model_id,
    )
    write_markdown_summary(report, output_path=md_report)
    export_question_bank(output_path=bank_export)

    print(f"Wrote JSON report: {json_report}")
    print(f"Wrote markdown summary: {md_report}")
    print(f"Wrote question bank: {bank_export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
