import google.generativeai as genai

class GeminiAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Use Gemini 2.5
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def summarize(self, text: str) -> str:
        try:
            response = self.model.generate_content(
                f"Summarize the following student performance data in simple, interpretable language:\n{text}"
            )
            return response.text
        except Exception as e:
            return f"Error while summarizing using Gemini: {e}"
