# AI-agent_CrewAI
AIエージェントのお勉強：CrewAIフレームワークを使ってみる

### 必要なパッケージ

```bash
pip install crewai
pip install crewai[google-genai]   # Google Gemini API を利用する場合
pip install crewai[litellm]        # ローカルの Llama-cpp server を利用したい場合 
pip install python-dotenv          # GEMINI_API_KEY=[Google Gemini のAPIキー] セットのため
pip install crewai[tools]          # ツール拡張など、機能込みこみでインストールする場合
```

### 01_crewai.py
CrewAIの基本的な使い方を示すサンプルコード

### 02_crewai_tools.py
CrewAIのツール機能の使い方を示すサンプルコード

