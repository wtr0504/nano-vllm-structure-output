from dataclasses import dataclass, field


@dataclass
class StructuredOutputsParams:
    json: str | dict | None = None
    _backend: str = "xgrammar"
    
    

    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    structured_outputs: StructuredOutputsParams | None = None

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"


