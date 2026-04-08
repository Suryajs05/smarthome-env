import asyncio
import os
import json
import textwrap
import httpx
from typing import List, Optional
from openai import OpenAI

# Required Buildathon Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
# Environment tracking
TASK_NAME = os.getenv("ENV_TASK", "easy") # Change to medium or hard to test others
BENCHMARK = "smarthome_optimizer"
ENV_URL = os.getenv("ENV_URL", "https://suryajs05-smarthome-env.hf.space")
MAX_STEPS = 8

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI Smart Home Energy Manager.
    Your goal is to manage devices based on the time of day to maximize energy efficiency.
    You will receive JSON describing the environment.
    You MUST output EXACTLY ONE JSON object representing your action, with no markdown formatting or extra text.
    Format: {"device_id": "string", "command": "turn_on|turn_off|set_temp", "value": float_or_null}
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs: dict) -> dict:
    user_prompt = f"Current State: {json.dumps(obs)}. What is your next action JSON?"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, # Low temp for structured output
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Clean up possible markdown block
        text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Hugging Face API Error: {exc}") # <--- ADD THIS LINE
        return {"device_id": "none", "command": "error", "value": None}

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    async with httpx.AsyncClient() as http:
        # Reset Env
        try:
            res = await http.post(f"{ENV_URL}/reset", json={"task": TASK_NAME})
            state = res.json()
        except Exception as e:
            log_step(0, "reset", 0.0, True, str(e))
            log_end(False, 0, 0.0, [])
            return

        obs = state["observation"]
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done: break
            
            action_dict = get_model_action(client, obs)
            action_str = json.dumps(action_dict)
            
            # Step Env
            try:
                res = await http.post(f"{ENV_URL}/step", json=action_dict)
                result = res.json()
                
                obs = result.get("observation", obs)
                reward = float(result.get("reward", 0.0))
                done = result.get("done", False)
                error = result.get("error")
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        # In this env, the final reward represents the completion percentage (0.0 to 1.0)
        score = rewards[-1] if rewards else 0.0
        success = score >= 0.99

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())