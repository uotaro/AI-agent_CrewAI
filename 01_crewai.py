# 01_crewai_01.py
# ===================================================================
# CrewAIの基本的な使い方を示すサンプルコード
# 事前準備：
# 1. crewaiパッケージをインストールする
#    pip install crewai
#    pip install crewai[google-genai]   # Google Gemini API を利用する場合
#    pip install crewai[litellm]        # ローカルの Llama-cpp server を利用したい場合 
# 2. LLMのAPIサーバーを用意する（例：llama.cppのAPIサーバー、またはGoogle Gemini API）
# 3. LLMの設定を行う（下記コード内のコメントを参照）
# ===================================================================
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()  # LLM(model="gemini/gemini-2.5-flash") を呼ぶ前に .env を読み込んでAPIキーを環境変数から読み込む

use_local_llm = True  # True: LAN内の llama.cpp APIサーバーを使用する, False: Google Gemini APIを使用する

#===================================================================
# LLMの設定 2通りの例を示す。どちらか一方を選択して使用すること。
#===================================================================
if use_local_llm:
    # ----- LLMの設定（LAN 内の Llama.cpp APIサーバー を使用する場合）-----
    # LangChainのChatOpenAIは使わず、CrewAIのLLMクラスを使用する
    # model 引数の頭に openai/ と付けることで、CrewAI内部（LiteLLM）に対して「OpenAI互換のAPIサーバー（llama.cpp など）を使うよ」と認識させることができる.
    llm = LLM(
        model="openai/Llama-3.3-8B-Instruct.Q4_K_S.gguf", # openai/ を頭につける
        base_url="http://localhost:8080/v1",
        api_key=os.getenv("LLAMA_API_TOKEN"),
        temperature=0.7
    )
else:
    # ----- LLMの設定（Google Gemini API（例は無料枠） を使用する場合）-----
    # 無料枠: 1分間に15回（Flashモデルの場合）ただし混みあってるときはさらに制限される可能性あり
    # model 引数の頭に gemini/ を付けることで、GeminiのAPIを呼び出す。
    # 事前に Google Gemini のAPIキーを取得し、環境変数 GEMINI_API_KEY に設定しておく必要がある。
    # APIキー取得場所: [Google AI Studio](https://aistudio.google.com/) からGoogleアカウントでログイン
    # APIキーは環境変数（GEMINI_API_KEY）から自動的に読み込まれる。
    llm = LLM(
        model="gemini/gemini-2.5-flash",  # gemini/ を頭につける
        temperature=0.7
    )

#===================================================================
# エージェントの定義
#===================================================================
researcher = Agent(
    role='Senior Researcher',
    goal='最新のAI技術トレンドを調査する',
    backstory='10年以上のAI研究経験を持つ専門家',
    verbose=True,
    llm=llm
)

writer = Agent(
    role='Tech Writer',
    goal='調査結果をわかりやすい記事にまとめる',
    backstory='技術記事の執筆経験が豊富なライター',
    verbose=True,
    llm=llm
)

#===================================================================
# タスクの定義
#===================================================================
research_task = Task(
    description='2026年のLLMエージェント技術のトレンドを調査してください',
    agent=researcher,
    expected_output='箇条書きのトレンドリスト'
)

write_task = Task(
    description='調査結果をブログ記事形式にまとめてください',
    agent=writer,
    expected_output='500文字程度のブログ記事'
)

#===================================================================
# Crewの作成と実行
#===================================================================
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # 順次実行
    verbose=True
)

result = crew.kickoff()
print(result)
