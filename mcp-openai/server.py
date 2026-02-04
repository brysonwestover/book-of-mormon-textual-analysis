#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp[cli]>=1.0.0",
#     "openai>=1.10.0",
# ]
# ///
"""
MCP server for OpenAI API access.
Enables Claude to send prompts to GPT models and receive responses.
Supports both Chat Completions API (GPT-4) and Responses API (GPT-5.2 Pro).
Enhanced for research-level analysis with reasoning traces and structured outputs.
"""

import os
import json
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("openai")

_client = None

# Models that require the Responses API
RESPONSES_API_MODELS = {"gpt-5.2-pro", "gpt-5.2", "gpt-5-pro", "gpt-5"}

# Valid reasoning effort levels for Responses API (GPT-5.2-pro only supports medium/high/xhigh)
VALID_REASONING_EFFORTS = {"medium", "high", "xhigh"}


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _client = OpenAI(api_key=api_key)
    return _client


def uses_responses_api(model: str) -> bool:
    """Check if a model requires the Responses API."""
    return model in RESPONSES_API_MODELS or model.startswith("gpt-5.2-pro")


def call_responses_api(
    prompt: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
    include_reasoning: bool = False,
) -> dict | str:
    """Call the Responses API for GPT-5.2 Pro and similar models."""
    client = get_client()

    params = {
        "model": model,
        "input": prompt,
        "instructions": system_prompt,
        "max_output_tokens": max_tokens,
    }

    if reasoning_effort:
        params["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**params)

    # Extract main text from response output
    main_text = None
    if hasattr(response, 'output_text'):
        main_text = response.output_text
    elif hasattr(response, 'output') and response.output:
        for item in response.output:
            if hasattr(item, 'content') and item.content:
                for content_item in item.content:
                    if hasattr(content_item, 'text'):
                        main_text = content_item.text
                        break
            if main_text:
                break

    if not main_text:
        main_text = str(response)

    # If reasoning traces requested, extract and include them
    if include_reasoning:
        result = {"response": main_text}

        # Extract reasoning/thinking traces if available
        reasoning_traces = []
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                item_type = getattr(item, 'type', None)
                if item_type == 'reasoning':
                    if hasattr(item, 'summary') and item.summary:
                        for summary_item in item.summary:
                            if hasattr(summary_item, 'text'):
                                reasoning_traces.append(summary_item.text)

        if reasoning_traces:
            result["reasoning_summary"] = reasoning_traces

        # Include usage metadata if available
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            result["usage"] = {
                "input_tokens": getattr(usage, 'input_tokens', None),
                "output_tokens": getattr(usage, 'output_tokens', None),
                "reasoning_tokens": getattr(usage, 'reasoning_tokens', None),
            }

        return result

    return main_text


def call_chat_completions_api(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the Chat Completions API for GPT-4 and similar models."""
    client = get_client()

    is_new_model = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")

    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    if is_new_model:
        params["max_completion_tokens"] = max_tokens
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_tokens

    response = client.chat.completions.create(**params)
    return response.choices[0].message.content


@mcp.tool()
def ask_gpt(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gpt-5.2-pro",
    temperature: float = 0.7,
    max_tokens: int = 16384,
    reasoning_effort: str = "high",
    include_reasoning: bool = False,
) -> dict | str:
    """
    Send a prompt to an OpenAI GPT model and return the response.

    Args:
        prompt: The user message to send
        system_prompt: System instructions for the model
        model: Model to use (gpt-5.2-pro, gpt-4o, gpt-4-turbo, etc.)
        temperature: Sampling temperature (0-2), only used for non-Responses API models
        max_tokens: Maximum response length (default 16384 for research tasks)
        reasoning_effort: Reasoning effort level for GPT-5.2 Pro (medium, high, xhigh). Default is 'high' for research quality.
        include_reasoning: If True, returns dict with response, reasoning traces, and token usage
    """
    if uses_responses_api(model):
        if reasoning_effort not in VALID_REASONING_EFFORTS:
            reasoning_effort = "high"
        return call_responses_api(prompt, system_prompt, model, max_tokens, reasoning_effort, include_reasoning)
    else:
        return call_chat_completions_api(prompt, system_prompt, model, temperature, max_tokens)


def ensure_additional_properties_false(schema: dict) -> dict:
    """
    Recursively ensure all object types in the schema have additionalProperties: false.
    This is required by OpenAI's structured output feature.
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    # If this is an object type, add additionalProperties: false
    if result.get("type") == "object":
        result["additionalProperties"] = False
        # Recursively process properties
        if "properties" in result:
            result["properties"] = {
                k: ensure_additional_properties_false(v)
                for k, v in result["properties"].items()
            }

    # Handle array items
    if result.get("type") == "array" and "items" in result:
        result["items"] = ensure_additional_properties_false(result["items"])

    # Handle anyOf, oneOf, allOf
    for key in ["anyOf", "oneOf", "allOf"]:
        if key in result:
            result[key] = [ensure_additional_properties_false(item) for item in result[key]]

    return result


@mcp.tool()
def ask_gpt_json(
    prompt: str,
    json_schema: str,
    system_prompt: str = "You are a helpful assistant. Always respond with valid JSON matching the provided schema.",
    model: str = "gpt-5.2-pro",
    max_tokens: int = 16384,
    reasoning_effort: str = "high",
) -> str:
    """
    Send a prompt to GPT and get a structured JSON response.
    Useful for extracting specific data points from text analysis.

    Args:
        prompt: The user message to send
        json_schema: JSON schema string describing the expected output structure
        system_prompt: System instructions for the model
        model: Model to use (gpt-5.2-pro, gpt-4o, etc.)
        max_tokens: Maximum response length
        reasoning_effort: Reasoning effort level for GPT-5.2 Pro (medium, high, xhigh)
    """
    client = get_client()

    # Parse the schema to validate it
    try:
        schema = json.loads(json_schema)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON schema: {e}"})

    # Ensure additionalProperties: false is set (required by OpenAI)
    schema = ensure_additional_properties_false(schema)

    # For Responses API models, use structured output
    if uses_responses_api(model):
        params = {
            "model": model,
            "input": prompt,
            "instructions": system_prompt,
            "max_output_tokens": max_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
            },
        }

        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}

        response = client.responses.create(**params)

        if hasattr(response, 'output_text'):
            return response.output_text
        elif hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            return content_item.text
        return str(response)
    else:
        # For Chat Completions API, use response_format
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_tokens": max_tokens,
        }
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content


@mcp.tool()
def debate_methodology(
    topic: str,
    position_a: str,
    position_b: str,
    model: str = "gpt-5.2-pro",
    reasoning_effort: str = "high",
) -> str:
    """
    Ask GPT to evaluate a methodological debate between two positions.

    Args:
        topic: The methodological question being debated
        position_a: First position/argument
        position_b: Second position/argument
        model: Model to use (defaults to gpt-5.2-pro for rigorous analysis)
        reasoning_effort: Reasoning effort for GPT-5.2 Pro (medium, high, xhigh)
    """
    system = """You are a methodological critic evaluating research frameworks.
Your role is to identify weaknesses, biases, and improvements in proposed methodologies.
Be direct and specific. Point out genuine problems rather than offering generic praise."""

    prompt = f"""METHODOLOGICAL QUESTION:
{topic}

POSITION A:
{position_a}

POSITION B:
{position_b}

Please evaluate:
1. Which position is stronger and why?
2. What weaknesses exist in each position?
3. What would improve this methodology?
4. Are there biases or blind spots in either position?"""

    return ask_gpt(prompt, system_prompt=system, model=model, reasoning_effort=reasoning_effort)


@mcp.tool()
def research_analysis(
    text: str,
    analysis_type: str,
    context: str = "",
    model: str = "gpt-5.2-pro",
    reasoning_effort: str = "xhigh",
    include_reasoning: bool = True,
) -> dict | str:
    """
    Perform deep research-level textual analysis with maximum reasoning.
    Designed for scholarly analysis of texts.

    Args:
        text: The text to analyze
        analysis_type: Type of analysis (e.g., 'stylometric', 'chiastic', 'authorship', 'linguistic', 'thematic', 'structural')
        context: Additional context about the text or specific research questions
        model: Model to use (defaults to gpt-5.2-pro)
        reasoning_effort: Defaults to 'xhigh' for maximum analytical depth
        include_reasoning: Whether to include reasoning traces and token usage (default True)
    """
    analysis_prompts = {
        "stylometric": """Perform a detailed stylometric analysis of this text. Examine:
- Vocabulary richness and word frequency patterns
- Sentence length and structure variation
- Use of function words (articles, prepositions, conjunctions)
- Distinctive phrases or word combinations
- Any shifts in style that might indicate different authorial voices""",

        "chiastic": """Analyze this text for chiastic structures (inverted parallelism). Look for:
- Simple chiasmus (ABBA patterns)
- Extended chiastic structures
- Parallel constructions that may form larger patterns
- Note the literary significance of any patterns found""",

        "authorship": """Analyze this text for authorship indicators. Consider:
- Distinctive vocabulary choices
- Grammatical patterns and preferences
- Thematic preoccupations
- Rhetorical strategies
- Compare stylistic consistency throughout the passage""",

        "linguistic": """Perform a linguistic analysis examining:
- Syntactic patterns and complexity
- Semantic fields and word choice
- Discourse markers and connectives
- Register and formality levels
- Any notable linguistic features or anomalies""",

        "thematic": """Analyze the thematic content:
- Primary and secondary themes
- How themes are developed and interconnected
- Symbolic or metaphorical language
- Narrative or argumentative progression""",

        "structural": """Analyze the structural organization:
- Overall structure and divisions
- Transitional elements
- Repetition and parallelism
- How structure supports meaning""",
    }

    base_prompt = analysis_prompts.get(
        analysis_type.lower(),
        f"Perform a detailed {analysis_type} analysis of the following text."
    )

    system = """You are a scholarly textual analyst with expertise in literary criticism,
linguistics, and historical document analysis. Provide rigorous, evidence-based analysis.
Cite specific examples from the text. Acknowledge uncertainty where appropriate.
Avoid overclaiming or reading too much into the evidence."""

    full_prompt = f"""{base_prompt}

{f'ADDITIONAL CONTEXT: {context}' if context else ''}

TEXT TO ANALYZE:
{text}

Provide a thorough analysis with specific textual evidence for each observation."""

    return ask_gpt(
        full_prompt,
        system_prompt=system,
        model=model,
        reasoning_effort=reasoning_effort,
        include_reasoning=include_reasoning,
    )


if __name__ == "__main__":
    mcp.run()
