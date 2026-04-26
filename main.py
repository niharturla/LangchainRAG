from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from nba_data import build_betting_context

model = OllamaLLM(model="llama3.2")

template = """
You are an NBA betting research assistant.
Use only the provided nba_api context.
Never claim guaranteed outcomes.
If parsed player or line is missing, you MUST output "Lean: Pass".
You MUST follow this deterministic lean signal: {model_signal}
If signal is OVER_SIGNAL -> Lean must be Over.
If signal is UNDER_SIGNAL -> Lean must be Under.
If signal is PASS_SIGNAL -> Lean must be Pass.

User question: {question}

Live stats context:
{context}

Return this structure:
1) Lean: Over / Under / Pass
2) Rationale (3 bullets using matchup + recent form + player trend). Please list the statistics that support the rationale.
   Only reference the inferred team and inferred opponent from context. Do not introduce a different opponent.
3) Risk factors (2 bullets)
4) Confidence: Low / Medium / High
5) XGBoost:
   - Predicted stat value: {xgb_pred}
   - Probability of over: {xgb_proba_over}
   - Model signal: {xgb_signal}
   Explain how this impacts the final lean.
6) Responsible note: one sentence reminding this is not financial advice
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("\n\n..............................")
    question = input("Ask your NBA betting question. (q to Quit): ")
    print("\n\n")
    if question == "q":
        break

    parsed = build_betting_context(question)
    print(f"Inferred player: {parsed['player_name'] or 'N/A'}")
    print(f"Inferred team: {parsed['team'] or 'N/A'}")
    print(f"Inferred opponent: {parsed['opponent'] or 'N/A'}")
    print(f"Inferred market: {parsed['market']}")
    print(f"Inferred line: {parsed['line'] if parsed['line'] is not None else 'N/A'}")

    if parsed["low_confidence"]:
        print("Low-confidence parse detected (missing player or line).")

    print(f"Model signal: {parsed['model_signal']}")
    print(f"XGBoost predicted value: {parsed['xgb_pred'] if parsed['xgb_pred'] is not None else 'N/A'}")
    print(
        "XGBoost probability over: "
        f"{parsed['xgb_proba_over'] if parsed['xgb_proba_over'] is not None else 'N/A'}"
    )
    print(f"XGBoost signal: {parsed['xgb_signal']}")
    if parsed.get("xgb_status") != "ok":
        print(f"XGBoost note: {parsed.get('xgb_message', 'XGBoost unavailable.')}")

    res = chain.invoke(
        {
            "question": question,
            "context": parsed["context"],
            "model_signal": parsed["model_signal"],
            "xgb_pred": parsed["xgb_pred"] if parsed["xgb_pred"] is not None else "N/A",
            "xgb_proba_over": (
                parsed["xgb_proba_over"] if parsed["xgb_proba_over"] is not None else "N/A"
            ),
            "xgb_signal": parsed["xgb_signal"],
        }
    )
    print(res)