"""Run a small BrowseComp smoke with GPT-5.5 and native web search.

The dataset loader, decryption routine, query format, and grading prompt come
from ``browsecomp_eval.py`` in this repository.  This wrapper only supplies a
Responses API sampler with the hosted ``web_search`` tool and a deterministic
oracle mode for validating the grader before paying for search runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI

from .browsecomp_eval import BrowseCompEval, QUERY_TEMPLATE, decrypt
from .types import MessageList, SamplerBase, SamplerResponse


class ResponsesTextSampler(SamplerBase):
    """Minimal simple-evals sampler backed by the Responses API."""

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str,
        max_output_tokens: int,
        web_search: bool,
    ) -> None:
        self.client = OpenAI(timeout=1800.0, max_retries=1)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.web_search = web_search

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "input": message_list,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
        }
        if self.web_search:
            request["tools"] = [{"type": "web_search"}]
            request["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.responses.create(**request)
                usage = response.usage.model_dump() if response.usage else None
                output_types = [getattr(item, "type", None) for item in response.output]
                return SamplerResponse(
                    response_text=response.output_text,
                    actual_queried_message_list=message_list,
                    response_metadata={
                        "response_id": response.id,
                        "usage": usage,
                        "web_search_calls": output_types.count("web_search_call"),
                    },
                )
            except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError) as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"Responses API failed after 3 attempts: {last_error}")


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("oracle", "agent"), required=True)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--grader-model", default="gpt-5.5")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="high",
    )
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grader = ResponsesTextSampler(
        args.grader_model,
        reasoning_effort="low",
        max_output_tokens=4096,
        web_search=False,
    )
    evaluation = BrowseCompEval(grader_model=grader, num_examples=args.examples)
    agent = ResponsesTextSampler(
        args.model,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        web_search=True,
    )

    payload: dict[str, Any] = {
        "benchmark": "BrowseComp",
        "mode": args.mode,
        "model": args.model if args.mode == "agent" else "oracle",
        "grader_model": args.grader_model,
        "examples": args.examples,
        "sample_seed": 0,
        "started_at": datetime.now(UTC).isoformat(),
        "results": [],
    }
    _write_results(args.output, payload)

    for index, row in enumerate(evaluation.examples):
        problem = decrypt(row["problem"], row["canary"])
        answer = decrypt(row["answer"], row["canary"])
        question_hash = hashlib.sha256(problem.encode()).hexdigest()

        if args.mode == "oracle":
            response_text = (
                "Explanation: Oracle reference answer supplied to validate the grader.\n"
                f"Exact Answer: {answer}\n"
                "Confidence: 100%"
            )
            response_metadata: dict[str, Any] = {}
        else:
            messages: MessageList = [
                {
                    "role": "developer",
                    "content": (
                        "You are being evaluated on BrowseComp. Use web search persistently, "
                        "cross-check candidate answers, and follow the requested final-answer format."
                    ),
                },
                {
                    "role": "user",
                    "content": QUERY_TEMPLATE.format(Question=problem),
                },
            ]
            sampled = agent(messages)
            response_text = sampled.response_text
            response_metadata = sampled.response_metadata

        grade = evaluation.grade_sample(problem, answer, response_text)
        result = {
            "sample_index": index,
            "question_sha256": question_hash,
            "score": grade.strip().lower() in {"yes", "correct: yes"},
            "grade": grade,
            "response": response_text if args.mode == "agent" else None,
            "response_metadata": response_metadata,
        }
        payload["results"].append(result)
        payload["accuracy"] = sum(item["score"] for item in payload["results"]) / len(
            payload["results"]
        )
        _write_results(args.output, payload)
        print(f"sample {index + 1}/{args.examples}: {grade}", flush=True)

    payload["completed_at"] = datetime.now(UTC).isoformat()
    _write_results(args.output, payload)
    print(f"accuracy={payload['accuracy']:.3f}")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
