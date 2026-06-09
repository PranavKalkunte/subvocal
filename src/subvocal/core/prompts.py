"""Prompt templates and version manager for intent reconstruction.
"""


from subvocal.shorthand.vocab import COMMANDS

PROMPT_v1 = """You are the core intent reconstruction decoder for a silent speech subvocal interface neckband.
The user is speaking silently. sEMG sensors capture signals and classify them into phonetic/mouth-shape shorthand.
Due to sEMG noise, character insertions, deletions, and substitutions occur.

TARGET COMMAND VOCABULARY:
{vocab_desc}

SHORTHAND RULES applied by user:
1. Commands are written as abbreviations (e.g. GOTO -> gt, SEARCH -> srch, CLICK -> clk, TYPE -> typ).
2. Words have subsequent vowels and 'h' omitted, double consonants collapsed.

INPUT TO RECONSTRUCT:
Noisy Shorthand Stream: "{noisy_input}"
Heuristic Alignment Recommendation: "{heuristic_recommendation}"

USER CONTEXT DATA:
- Web Page Context (Visible elements): {web_context}
- Calendar Events: {calendar}
- Contacts List: {contacts}
- Conversation History: {history}

Your goal: Output ONLY the correct reconstructed command and arguments.
Reconstruct the full words, restore deleted vowels, correct phonetic substitutions.
The output MUST start with one of the target commands (e.g., GOTO, SEARCH, CLICK, TYPE, etc.).
If the command requires arguments (like GOTO a URL, SEARCH a query, TYPE a word), complete them based on the noisy stream and user context.

Output format: Return ONLY the raw reconstructed command string (e.g. "GOTO google.com" or "CLICK Sign In"). No explanations, no markdown formatting, no JSON wrappers.
"""

PROMPT_v2 = """System: You are the intent reconstruction layer of the Subvocal SDK.
A neckband classifier maps throat electromyography to phonetic shorthand. 
Your task is to decode this noisy stream into a valid command and arguments.

Target Actions:
{vocab_desc}

Active Contexts:
- On-Screen UI Elements: {web_context}
- Calendar Schedule: {calendar}
- Contact Directory: {contacts}
- Conversation Dialogue: {history}

Inputs:
- Input Shorthand: "{noisy_input}"
- Heuristic Search Suggestion: "{heuristic_recommendation}"

Instructions:
1. Decode the input shorthand. Map it to one of the Target Actions.
2. Select target arguments from the Active Contexts if they match phonetically.
3. Output ONLY the resolved command string (e.g. "GOTO reddit.com" or "SEARCH machine learning"). Do not wrap in markdown block or quotes.
"""


class PromptManager:
    """Manages versioned prompt templates for LLM intent decoding."""

    def __init__(self, default_version: str = "v1"):
        self.templates: dict[str, str] = {
            "v1": PROMPT_v1,
            "v2": PROMPT_v2,
        }
        self.default_version = default_version

        # Build vocabulary description once
        self.vocab_desc = "\n".join(
            [f"- {cmd}: {details['description']}" for cmd, details in COMMANDS.items()]
        )

    def register_template(self, version: str, template: str) -> None:
        """Register a new custom prompt template version."""
        self.templates[version] = template

    def get_template(self, version: str | None = None) -> str:
        """Retrieve a raw prompt template string by version."""
        ver = version or self.default_version
        if ver not in self.templates:
            raise KeyError(f"Prompt template version '{ver}' not found.")
        return self.templates[ver]

    def format_prompt(
        self,
        noisy_input: str,
        heuristic_recommendation: str,
        web_context: str,
        calendar: str,
        contacts: str,
        history: str,
        version: str | None = None,
    ) -> str:
        """Formats the selected prompt template with active inputs."""
        template = self.get_template(version)
        return template.format(
            vocab_desc=self.vocab_desc,
            noisy_input=noisy_input,
            heuristic_recommendation=heuristic_recommendation,
            web_context=web_context,
            calendar=calendar,
            contacts=contacts,
            history=history,
        )
