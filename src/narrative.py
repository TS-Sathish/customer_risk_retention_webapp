import os
from dotenv import load_dotenv
load_dotenv()

def generate_narrative(probability, risk, factors, segment):
    key=os.getenv("OPENAI_API_KEY")
    names = [str(item.get("feature", "")) for item in factors[:3] if item.get("feature")]
    fallback = (f"Estimated churn probability is {probability:.1%} ({risk} risk), segment {segment}. "
                f"Model factors include {', '.join(names) if names else 'the available profile'}. "
                "Review the customer context and consider a relevant action for human review. "
                "This estimate is not a guarantee of churn or retention outcome.")
    if not key:
        return fallback
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=15.0, max_retries=1)
        prompt = ("Write a concise, cautious business note using only these non-identifying model results. "
                  "Do not invent facts or claim causality. Present suggestions for human review and do not "
                  "guarantee retention. "
                  f"Risk={risk}; probability={probability:.3f}; segment={segment}.")
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=180)
        content = response.choices[0].message.content
        return content.strip() if content else fallback
    except Exception:
        return fallback
