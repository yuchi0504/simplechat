# lambda/index.py

import json
import os
import boto3
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from botocore.exceptions import ClientError

# FastAPI インスタンス作成
app = FastAPI()

# CORS対応
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

# Bedrockクライアント（グローバルに初期化）
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1")  # Lambdaがないため明示指定
)

# リクエストボディ用のモデル
class ChatRequest(BaseModel):
    message: str
    conversationHistory: Optional[List[dict]] = []

# Bedrock形式のメッセージ整形
def build_bedrock_messages(messages: List[dict]):
    bedrock_messages = []
    for msg in messages:
        if msg["role"] == "user":
            bedrock_messages.append({
                "role": "user",
                "content": [{"text": msg["content"]}]
            })
        elif msg["role"] == "assistant":
            bedrock_messages.append({
                "role": "assistant",
                "content": [{"text": msg["content"]}]
            })
    return bedrock_messages

# メインのチャットAPIエンドポイント
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print("Received request:", request.json() if hasattr(request, "json") else request)

        message = request.message
        conversation_history = request.conversationHistory or []

        print("Processing message:", message)
        print("Using model:", MODEL_ID)

        # 会話履歴にユーザーのメッセージを追加
        messages = conversation_history.copy()
        messages.append({
            "role": "user",
            "content": message
        })

        # Bedrock 用にリクエストペイロードを構築
        bedrock_messages = build_bedrock_messages(messages)
        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 512,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }

        print("Calling Bedrock invoke_model API with payload:", json.dumps(request_payload))

        # モデル推論
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_payload),
            contentType="application/json"
        )

        # 結果を読み取り・解析
        response_body = json.loads(response['body'].read())
        print("Bedrock response:", json.dumps(response_body, default=str))

        # 応答メッセージ取得
        if not response_body.get('output') or not response_body['output'].get('message') or not response_body['output']['message'].get('content'):
            raise Exception("No response content from the model")

        assistant_response = response_body['output']['message']['content'][0]['text']

        # アシスタントの返答を履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })

        # レスポンス返却
        return {
            "success": True,
            "response": assistant_response,
            "conversationHistory": messages
        }

    except Exception as error:
        print("Error:", str(error))
        return {
            "success": False,
            "error": str(error)
        }
