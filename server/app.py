from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

app = FastAPI()

# --- Typed Models ---
class DeviceState(BaseModel):
    id: str
    type: str
    status: str
    temperature: Optional[float] = None

class Observation(BaseModel):
    time_of_day: str
    devices: List[DeviceState]
    total_power_watts: float
    feedback: str

class Action(BaseModel):
    device_id: str
    command: str 
    value: Optional[float] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    error: Optional[str] = None

class ResetRequest(BaseModel):
    task: str = "easy"

# --- In-Memory State ---
session_state = {}

def get_initial_state(task: str) -> Dict:
    if task == "easy":
        return {
            "task": "easy",
            "step": 0,
            "time_of_day": "14:00 (Daytime)",
            "devices": [
                {"id": "light_living_room", "type": "light", "status": "on", "power": 60},
                {"id": "light_kitchen", "type": "light", "status": "on", "power": 60},
            ]
        }
    elif task == "medium":
        return {
            "task": "medium",
            "step": 0,
            "time_of_day": "15:00",
            "devices": [
                {"id": "hvac_main", "type": "hvac", "status": "on", "temperature": 70.0, "power": 3500},
            ]
        }
    else:
        return {
            "task": "hard",
            "step": 0,
            "time_of_day": "18:00 (Peak Hours)",
            "devices": [
                {"id": "pool_pump", "type": "appliance", "status": "on", "power": 1500},
                {"id": "hvac_main", "type": "hvac", "status": "on", "temperature": 72.0, "power": 3000},
            ]
        }

def build_observation(state: Dict, feedback: str = "System ready.") -> Observation:
    devs = [DeviceState(id=d["id"], type=d["type"], status=d["status"], temperature=d.get("temperature")) for d in state["devices"]]
    total_power = sum(d["power"] for d in state["devices"] if d["status"] == "on")
    return Observation(time_of_day=state["time_of_day"], devices=devs, total_power_watts=total_power, feedback=feedback)

# --- Endpoints ---
@app.post("/reset", response_model=StepResult)
async def reset_env(req: Optional[ResetRequest] = None, task: Optional[str] = None):
    # This checks BOTH the JSON body and the URL (?task=medium)
    task_name = "easy"
    if task:
        task_name = task
    elif req and req.task:
        task_name = req.task

    if task_name not in ["easy", "medium", "hard"]:
        task_name = "easy"
    
    global session_state
    session_state = get_initial_state(task_name)
    obs = build_observation(session_state)
    
    # Returning a score strictly between 0 and 1
    return StepResult(observation=obs, reward=0.01, done=False)

@app.get("/state", response_model=Observation)
async def get_state():
    if not session_state:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return build_observation(session_state)

@app.post("/step", response_model=StepResult)
async def step_env(action: Action):
    global session_state
    if not session_state:
        raise HTTPException(status_code=400, detail="Environment not reset.")
    
    session_state["step"] += 1
    feedback = f"Action '{action.command}' applied to '{action.device_id}'."
    reward = 0.01
    done = False
    
    device = next((d for d in session_state["devices"] if d["id"] == action.device_id), None)
    if not device:
        return StepResult(observation=build_observation(session_state, "Error: Device not found"), reward=0.01, done=False, error="Device not found")

    if action.command in ["turn_on", "turn_off"]:
        device["status"] = action.command.split("_")[1]
    elif action.command == "set_temp" and action.value is not None and device["type"] == "hvac":
        device["temperature"] = action.value
    else:
        feedback = "Invalid command for device type."

    # Graders / Partial Reward Logic (Now strictly between 0.01 and 0.99)
    task = session_state["task"]
    if task == "easy":
        lights_on = sum(1 for d in session_state["devices"] if d["type"] == "light" and d["status"] == "on")
        reward = 0.99 - (lights_on * 0.40) # 0.99 if all off, 0.59 if one on, 0.19 if two on
        if lights_on == 0:
            done = True
            
    elif task == "medium":
        hvac = session_state["devices"][0]
        dist = abs(78.0 - hvac["temperature"])
        reward = max(0.01, 0.99 - (dist * 0.10)) 
        if hvac["temperature"] == 78.0:
            done = True
            
    elif task == "hard":
        pump = next(d for d in session_state["devices"] if d["id"] == "pool_pump")
        hvac = next(d for d in session_state["devices"] if d["id"] == "hvac_main")
        
        r_pump = 0.49 if pump["status"] == "off" else 0.01
        dist = abs(78.0 - hvac["temperature"])
        r_hvac = max(0.01, 0.50 - (dist * 0.05))
        
        reward = r_pump + r_hvac
        if pump["status"] == "off" and hvac["temperature"] == 78.0:
            done = True

    # Final Boundary Check to satisfy strictly (0, 1)
    reward = max(0.01, min(0.99, reward))

    if session_state["step"] >= 8:
        done = True

    obs = build_observation(session_state, feedback)
    return StepResult(observation=obs, reward=reward, done=done)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
