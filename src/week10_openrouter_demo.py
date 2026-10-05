"""
Week 10: AI API Fundamentals via OpenRouter
================================================

Setup before running:
    1. Sign up free (no card needed): https://openrouter.ai/keys
    2. Add to your .env file:  OPENROUTER_API_KEY=sk-or-...
    3. Install the OpenAI SDK -- OpenRouter is a drop-in replacement for it:
           uv add openai

Why OpenRouter: it's a single gateway to hundreds of models (Gemini, OpenAI,
Anthropic, Llama, DeepSeek, ...) behind ONE API key and the standard OpenAI
Python client. The client setup and call shape never change -- only the
`model` string does. That's what makes this "any provider you want."
"""

# %% [markdown]
# # Our demo will cover:
# 1. Install the OpenAI SDK and add your OpenRouter API key to your .env file.
# 2. Run a simple prompt through the default model.
# 3. Tuning temperature and token parameters
# 4. System prompts and structured output
# 5. Turning your data cube into a prompt


# %% [markdown]
# ### 1. Install the OpenAI SDK and add your OpenRouter API key to your .env file.

# %%
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Standard OpenAI connection client to connect to most providers like OpenRouter. 
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# %% [markdown]
# ### 2. Run a simple prompt through the default model.

# %%
# Default model to use for the demo -- you can swap this string for any other
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# %%
# Let's try a simple API call to the default model.
# Insert your "user" prompt in the "content" field below.
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "user", "content": "Hello! Can you tell me a joke?"}
    ],
)
print(response.choices[0].message.content)

# %%
# Run the exact same prompt through three different providers.
# Models ending in ":free" cost nothing and need no card 
models_to_try = [
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free"
]

for model in models_to_try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "In one sentence, what is a data cube?"}],
    )
    print(f"--- {model} ---")
    print(response.choices[0].message.content)
    print()

# %% [markdown]
# ### 3. Tuning temperature and token parameters
# %%
# temperature and max_tokens are standard OpenAI-style params: they work
# the same way across every model routed through OpenRouter.
# temperaturate controls randomness (0 = deterministic, 1 = creative)
# max_tokens controls the maximum length of the output (in tokens, not words).

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[{"role": "user", "content": "Tell me one short but creative joke for actuarial and risk analyst students in NTU Singapore."}],
    temperature=0,
    max_tokens=1000,
    seed=42
)
print(response.choices[0].message.content)

# try dial up or down temperature and max_tokens to see how it affects the output



# %% [markdown]
# ### 4. System prompts and structured output

# %%
# 4a. System role is a way to give the model a persona or a set of instructions.
# It's a way to give the model a "context" for how to respond.
# Notice in the message list, we have passed in a system message first, then a user message. 
# The model will respond in the context of the system message.

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a senior risk analyst at a consumer finance company. "
                "You answer concisely and professionally, and you only discuss "
                "topics related to portfolio risk and credit analytics."
            ),
        },
        {"role": "user", "content": "Can you tell me a joke?"},
    ],
)
print(response.choices[0].message.content)

# %%
# 4b. Structured output. We can specify the type of output that we want,
# in this case, a JSON object with a specific structure schema:
# First field: trend. One-sentence summary of the trend.
# Second filed: risk_direction. One of three values: "improving", "stable", or "worsening".
# Third field: recommendation. One actionable recommendation.
# The model will respond with a JSON object that matches this schema.

structured_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "portfolio_insight",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "trend": {"type": "string", "description": "One-sentence summary of the trend"},
                "risk_direction": {"type": "string", "enum": ["improving", "stable", "worsening"]},
                "recommendation": {"type": "string", "description": "One actionable recommendation"},
            },
            "required": ["trend", "risk_direction", "recommendation"],
            "additionalProperties": False,
        },
    },
}

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "system", "content": "You are a senior risk analyst at a consumer finance company."},
        {"role": "user", "content": "Applications rose 8% this month while the default rate held steady at 6.2%."},
    ],
    response_format=structured_schema,
)
print(response.choices[0].message.content)

# %% [markdown]
# ### 5. Turning your data cube into a prompt

# %%
# Summarise data cube to produce last 6 months of metrics

DATA_PATH = Path(__file__).resolve().parent.parent / "data"/  "processed_data_cube.csv"
df_cube = pd.read_csv(DATA_PATH)

def build_data_summary(df: pd.DataFrame, n_rows: int = 6) -> str:
    """Turn the most recent rows of the cube into a compact text summary for a prompt."""
    monthly = (
        df.groupby(["YEAR_APPLIED", "MONTH_APPLIED"])
        .agg(
            total_applications=("total_applications", "sum"),
            total_defaults=("total_defaults", "sum"),
            total_credit=("total_credit", "sum"),
        )
        .reset_index()
        .sort_values(["YEAR_APPLIED", "MONTH_APPLIED"])
    )
    monthly["default_rate"] = (monthly["total_defaults"] / monthly["total_applications"]).round(4)
    return monthly.tail(n_rows).to_string(index=False)

data_summary = build_data_summary(df_cube)

print(data_summary)

# %%
# pass the data summary to the LLM model to reason over the data and provide a concise summary
response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior risk analyst at a consumer finance company. "
                    "Keep the tone professional and analytical."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Review the following monthly portfolio data.\n\n"
                    f"{data_summary}\n\n"
                    "Provide a concise, 3-paragraph executive summary covering:\n"
                    "1. Key trends in application volume and credit origination.\n"
                    "2. Whether portfolio risk (default rate) is increasing or decreasing.\n"
                    "3. One strategic recommendation based on this trend."
                ),
            },
        ],
        response_format=structured_schema,
    )
print(response.choices[0].message.content)

